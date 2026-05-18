#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
INNATE3D Training Script for Taylor-Green Vortex 3D

Eğitim:
- State evolution: IC → num_steps adım → trajectory
- NS Residual Loss: |du/dt - RHS|² (du/dt = (u_new - u_old)/dt)
- Physics losses: Continuity, energy conservation

Author: INNATE TGV3D Training (2024)
"""

import math
import time
import os
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model import INNATE3D_TGV, create_model, FluidState3D


# =============================================================================
# CONFIG
# =============================================================================

CONFIG = {
    # Model (INNATE nöronlarından katmanlı ağ)
    "resolution": 32,
    "nu": 0.001,
    "num_layers": 6,
    "neurons_per_layer": 800,
    "target_params": 10000,

    # Training
    "epochs": 15_000,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "batch_size": 1,
    "num_steps": 20,
    "dt": 0.002,

    # Time
    "T_final": 0.8,
    "num_snapshots": 10,

    # Loss weights
    "ns_residual_weight": 1.0,
    "continuity_weight": 100.0,
    "energy_weight": 20.0,
    "enstrophy_weight": 10.0,

    # Curriculum
    "use_curriculum": True,
    "phase_A_end": 3000,
    "phase_B_end": 8000,
    "phase_C_end": 15000,

    # Checkpointing
    "checkpoint_every": 1000,
    "print_every": 200,
    "eval_every": 500,

    # Paths
    "results_dir": "results_innate_tgv3d",
    "checkpoint_prefix": "innate_tgv_",
}


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

def compute_pre_projection_consistency_loss(
    model: INNATE3D_TGV,
    state_prev: FluidState3D,
    state_curr_unprojected: FluidState3D,
    dt: float
) -> torch.Tensor:
    """
    Pre-projection consistency loss (basınç projeksiyonu öncesi ön-tutarlılık).

    DİKKAT — TERMİNOLOJİ:
    Bu fonksiyon **TAM NS residual'ı DEĞİL**, fractional-step şemasındaki
    `−∇p` terimi eksik bir kısmi residual'dır. Tezde "pre-projection
    consistency loss" veya "layer-physics consistency loss" diye adlandır.

    Hesaplanan:
        res = du/dt − (−u·∇u + ν∇²u)
            = ∂u/∂t + u·∇u − ν∇²u
            ≡ −∇p   (tam NS denkleminden)

    Yani aslında implicit basınç gradyanının büyüklüğünü cezalandırıyoruz.
    state_curr UNPROJECTED olduğu için projeksiyon adımının düzelteceği
    div-free olmayan kısım buraya yansır; loss bu farkı küçültmeye çalışır.

    Argümanlar
        model: INNATE3D_TGV — spectral operatörler için
        state_prev: t adımındaki (önceki) state
        state_curr_unprojected: t+dt adımındaki **projeksiyonsuz** state
        dt: zaman adımı (boyutsuz)

    Boyut analizi
        du/dt: [-]/[-] = [-]
        RHS (advection+diffusion): [-]
        residual: [-], loss = ⟨residual²⟩: [-]² ✓
    """
    ops = model.spectral_ops

    # du/dt ≈ (u_new_unprojected - u_old) / dt
    du_dt = (state_curr_unprojected.u - state_prev.u) / dt
    dv_dt = (state_curr_unprojected.v - state_prev.v) / dt
    dw_dt = (state_curr_unprojected.w - state_prev.w) / dt

    # RHS hesapla: -u·∇u + ν∇²u (state_prev ile)
    u, v, w = state_prev.u, state_prev.v, state_prev.w

    # Advection: u·∇u
    du_dx, du_dy, du_dz = ops.gradient(u)
    dv_dx, dv_dy, dv_dz = ops.gradient(v)
    dw_dx, dw_dy, dw_dz = ops.gradient(w)

    adv_u = u * du_dx + v * du_dy + w * du_dz
    adv_v = u * dv_dx + v * dv_dy + w * dv_dz
    adv_w = u * dw_dx + v * dw_dy + w * dw_dz

    # Diffusion: ν∇²u
    nu = model.nu
    lap_u = ops.laplacian(u)
    lap_v = ops.laplacian(v)
    lap_w = ops.laplacian(w)

    diff_u = nu * lap_u
    diff_v = nu * lap_v
    diff_w = nu * lap_w

    # RHS = -advection + diffusion
    rhs_u = -adv_u + diff_u
    rhs_v = -adv_v + diff_v
    rhs_w = -adv_w + diff_w

    # Residual = du/dt - RHS
    res_u = du_dt - rhs_u
    res_v = dv_dt - rhs_v
    res_w = dw_dt - rhs_w

    # L2 loss
    loss = (res_u**2 + res_v**2 + res_w**2).mean()

    return loss


# Geriye uyumluluk: eski isim → yeni fonksiyona alias
# Tezde "NS residual" ifadesi yanıltıcıydı (∇p eksik).
# Yeni adlandırma: pre-projection consistency loss.
def compute_ns_residual_loss(
    model: INNATE3D_TGV,
    state_prev: FluidState3D,
    state_curr_unprojected: FluidState3D,
    dt: float,
) -> torch.Tensor:
    """[DEPRECATED] compute_pre_projection_consistency_loss için alias."""
    return compute_pre_projection_consistency_loss(
        model, state_prev, state_curr_unprojected, dt
    )


def compute_continuity_loss(model: INNATE3D_TGV, state: FluidState3D) -> torch.Tensor:
    """
    Süreklilik: ∇·u = 0 (saf L2)

    Boyut: ⟨(∇·u)²⟩, [-]² (tutarlı tek-norm).

    NOT: Eski sürümde `+ 0.1 * |div|.mean()` (L1) terimi vardı; iki farklı
    boyut sınıfını (L1 vs L2) doğrudan toplamak boyutsal olarak yanlıştı ve
    div ~ 1e-6 mertebesinde olduğu için L1 her zaman baskındı (L2 işlevsiz).
    Saf L2 daha temiz ve gradient akışı tek skala üzerinde.
    """
    div = model.spectral_ops.divergence(state.u, state.v, state.w)
    return (div**2).mean()


def compute_energy_loss(
    state_prev: FluidState3D,
    state_curr: FluidState3D,
    nu: float,
    dt: float
) -> torch.Tensor:
    """
    Enerji dengesi: dE/dt + 2ν * enstrophy = 0

    E = 0.5 * <u²>
    Enstrophy = 0.5 * <ω²>

    dE/dt = -2ν * enstrophy (viskoz sönüm)
    """
    E_prev = state_prev.kinetic_energy()
    E_curr = state_curr.kinetic_energy()

    dE_dt = (E_curr - E_prev) / dt

    # Enstrophy (zaten 0.5 içeriyor)
    enstrophy = state_curr.enstrophy()

    # Enerji dengesi: dE/dt + 2ν * enstrophy ≈ 0
    # (FluidState3D.enstrophy = 0.5*<ω²>, dolayısıyla 2*enstrophy = <ω²>)
    residual = dE_dt + 2 * nu * enstrophy

    return residual**2


def compute_enstrophy_loss(
    state_prev: FluidState3D,
    state_curr: FluidState3D
) -> torch.Tensor:
    """Enstrophy spike cezası - aşırı artışı engelle."""
    Z_prev = state_prev.enstrophy()
    Z_curr = state_curr.enstrophy()

    ratio = Z_curr / (Z_prev + 1e-8)
    # 10x'den fazla artış cezalandırılır
    penalty = torch.relu(ratio - 10.0)**2

    return penalty.mean()


def get_curriculum_weights(epoch: int, config: Dict) -> Dict[str, float]:
    """Curriculum learning weights."""
    if not config["use_curriculum"]:
        return {
            'ns_residual': config["ns_residual_weight"],
            'continuity': config["continuity_weight"],
            'energy': config["energy_weight"],
            'enstrophy': config["enstrophy_weight"],
        }

    if epoch < config["phase_A_end"]:
        # Phase A: Temel fizik
        return {
            'ns_residual': 1.0,
            'continuity': 20.0,
            'energy': 0.0,
            'enstrophy': 0.0,
        }
    elif epoch < config["phase_B_end"]:
        # Phase B: Ramp up
        progress = (epoch - config["phase_A_end"]) / (config["phase_B_end"] - config["phase_A_end"])
        return {
            'ns_residual': 1.0,
            'continuity': 20.0 + progress * 80.0,
            'energy': progress * config["energy_weight"],
            'enstrophy': progress * config["enstrophy_weight"],
        }
    else:
        # Phase C: Full
        return {
            'ns_residual': config["ns_residual_weight"],
            'continuity': config["continuity_weight"],
            'energy': config["energy_weight"],
            'enstrophy': config["enstrophy_weight"],
        }


# =============================================================================
# TRAINING LOOP
# =============================================================================

@dataclass
class TrainingState:
    epoch: int
    best_loss: float
    history: Dict[str, List[float]]

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


def train_epoch(
    model: INNATE3D_TGV,
    optimizer: optim.Optimizer,
    config: Dict,
    epoch: int,
    device: torch.device
) -> Dict[str, float]:
    """Tek epoch eğitimi."""
    model.train()

    # IC oluştur
    state = model.tgv_initial_condition(batch_size=config["batch_size"])
    state = FluidState3D(
        u=state.u.to(device), v=state.v.to(device), w=state.w.to(device),
        p=state.p.to(device),
        omega_x=state.omega_x.to(device), omega_y=state.omega_y.to(device),
        omega_z=state.omega_z.to(device),
        t=state.t.to(device)
    )

    W = get_curriculum_weights(epoch, config)

    total_loss = 0.0
    ns_loss = 0.0
    cont_loss = 0.0
    energy_loss = 0.0
    enstrophy_loss = 0.0

    optimizer.zero_grad()

    num_steps = config["num_steps"]
    dt = config["dt"]

    state_prev = state

    for step in range(num_steps):
        # Bir adım at - hem projeksiyonlu hem projeksiyonsuz state al
        state_curr_projected, state_curr_unprojected = model.step_with_unprojected(state_prev, dt)

        # NS Residual loss - UNPROJECTED state ile!
        # (Projeksiyon basınç gradyanını ekler, NS denklemi onu ayrı içerir)
        loss_ns = compute_ns_residual_loss(model, state_prev, state_curr_unprojected, dt)
        ns_loss += loss_ns.item()

        # Continuity loss - PROJECTED state ile (div-free olmalı)
        loss_cont = compute_continuity_loss(model, state_curr_projected)
        cont_loss += loss_cont.item()

        # Energy loss - PROJECTED state ile (fiziksel evrim)
        if W['energy'] > 0:
            loss_energy = compute_energy_loss(state_prev, state_curr_projected, config["nu"], dt)
            energy_loss += loss_energy.item()
        else:
            loss_energy = torch.tensor(0.0, device=device)

        # Enstrophy loss - PROJECTED state ile
        if W['enstrophy'] > 0:
            loss_enstrophy = compute_enstrophy_loss(state_prev, state_curr_projected)
            enstrophy_loss += loss_enstrophy.item()
        else:
            loss_enstrophy = torch.tensor(0.0, device=device)

        # Weighted total
        step_loss = (
            W['ns_residual'] * loss_ns +
            W['continuity'] * loss_cont +
            W['energy'] * loss_energy +
            W['enstrophy'] * loss_enstrophy
        )
        total_loss += step_loss

        # Sonraki adım için projeksiyonlu state kullan
        state_prev = state_curr_projected

    # Backward
    total_loss.backward()
    optimizer.step()

    return {
        'total_loss': total_loss.item() / num_steps,
        'ns_residual_loss': ns_loss / num_steps,
        'continuity_loss': cont_loss / num_steps,
        'energy_loss': energy_loss / num_steps,
        'enstrophy_loss': enstrophy_loss / num_steps,
    }


def evaluate(
    model: INNATE3D_TGV,
    config: Dict,
    device: torch.device
) -> Dict[str, float]:
    """Model değerlendirmesi."""
    model.eval()

    with torch.no_grad():
        state = model.tgv_initial_condition(batch_size=1)
        state = FluidState3D(
            u=state.u.to(device), v=state.v.to(device), w=state.w.to(device),
            p=state.p.to(device),
            omega_x=state.omega_x.to(device), omega_y=state.omega_y.to(device),
            omega_z=state.omega_z.to(device),
            t=state.t.to(device)
        )

        dt = config["dt"]
        T_final = config["T_final"]
        num_steps = int(T_final / dt)

        energies = [state.kinetic_energy().item()]
        enstrophies = [state.enstrophy().item()]
        divergences = [model.spectral_ops.divergence(state.u, state.v, state.w).abs().mean().item()]
        times = [0.0]

        for step in range(num_steps):
            state = model.step(state, dt)

            if (step + 1) % (num_steps // config["num_snapshots"]) == 0:
                energies.append(state.kinetic_energy().item())
                enstrophies.append(state.enstrophy().item())
                div = model.spectral_ops.divergence(state.u, state.v, state.w)
                divergences.append(div.abs().mean().item())
                times.append(state.t.item())

    return {
        'times': times,
        'energies': energies,
        'enstrophies': enstrophies,
        'divergences': divergences,
        'final_energy': energies[-1],
        'energy_decay': (energies[0] - energies[-1]) / energies[0] if energies[0] > 0 else 0,
        'max_divergence': max(divergences),
        'mean_divergence': sum(divergences) / len(divergences),
    }


def save_checkpoint(
    model: INNATE3D_TGV,
    optimizer: optim.Optimizer,
    training_state: TrainingState,
    config: Dict,
    filepath: str
):
    """Checkpoint kaydet."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'training_state': training_state.to_dict(),
        'config': config,
    }
    torch.save(checkpoint, filepath)
    print(f"  Checkpoint saved: {filepath}")


def load_checkpoint(filepath: str, model: INNATE3D_TGV, optimizer, device):
    """Checkpoint yükle."""
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    training_state = TrainingState.from_dict(checkpoint['training_state'])
    print(f"  Checkpoint loaded: {filepath}, epoch {training_state.epoch}")
    return training_state


def train(config: Dict, resume_from: Optional[str] = None):
    """Ana eğitim döngüsü."""
    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    os.makedirs(config["results_dir"], exist_ok=True)

    # Model
    model = create_model(
        resolution=config["resolution"],
        nu=config["nu"],
        num_layers=config["num_layers"],
        neurons_per_layer=config["neurons_per_layer"],
        target_params=config["target_params"]
    )
    model = model.to(device)

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"]
    )

    # Training state
    if resume_from and os.path.exists(resume_from):
        training_state = load_checkpoint(resume_from, model, optimizer, device)
        start_epoch = training_state.epoch + 1
    else:
        training_state = TrainingState(
            epoch=0,
            best_loss=float('inf'),
            history={
                'total_loss': [], 'ns_residual_loss': [], 'continuity_loss': [],
                'energy_loss': [], 'enstrophy_loss': [],
                'eval_energy_decay': [], 'eval_max_div': []
            }
        )
        start_epoch = 1

    print(f"\n{'='*60}")
    print("INNATE3D TGV Training")
    print(f"{'='*60}")
    print(f"Epochs: {start_epoch} -> {config['epochs']}")
    print(f"Resolution: {config['resolution']}^3")
    print(f"Parameters: {model.count_parameters()['total']:,}")
    print(f"{'='*60}\n")

    start_time = time.time()

    for epoch in range(start_epoch, config["epochs"] + 1):
        losses = train_epoch(model, optimizer, config, epoch, device)

        for key, value in losses.items():
            training_state.history[key].append(value)

        if losses['total_loss'] < training_state.best_loss:
            training_state.best_loss = losses['total_loss']

        training_state.epoch = epoch

        if epoch % config["print_every"] == 0 or epoch == 1:
            phase = "A" if epoch < config["phase_A_end"] else ("B" if epoch < config["phase_B_end"] else "C")
            elapsed = time.time() - start_time
            print(
                f"[Epoch {epoch:5d}] Phase {phase} | "
                f"Loss: {losses['total_loss']:.4e} | "
                f"NS: {losses['ns_residual_loss']:.4e} | "
                f"Div: {losses['continuity_loss']:.4e} | "
                f"Time: {elapsed:.1f}s"
            )

        if epoch % config["eval_every"] == 0:
            eval_metrics = evaluate(model, config, device)
            training_state.history['eval_energy_decay'].append(eval_metrics['energy_decay'])
            training_state.history['eval_max_div'].append(eval_metrics['max_divergence'])
            print(
                f"  Eval: E_decay={eval_metrics['energy_decay']:.4f} | "
                f"Max_div={eval_metrics['max_divergence']:.2e}"
            )

        if epoch % config["checkpoint_every"] == 0:
            filepath = os.path.join(
                config["results_dir"],
                f"{config['checkpoint_prefix']}epoch_{epoch}.pth"
            )
            save_checkpoint(model, optimizer, training_state, config, filepath)

    # Final save
    final_path = os.path.join(config["results_dir"], "final_model.pth")
    save_checkpoint(model, optimizer, training_state, config, final_path)

    history_path = os.path.join(config["results_dir"], "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(training_state.history, f, indent=2)

    print(f"\n{'='*60}")
    print("Training completed!")
    print(f"Best loss: {training_state.best_loss:.4e}")
    print(f"Total time: {time.time() - start_time:.1f}s")
    print(f"{'='*60}")

    return model, training_state


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train INNATE3D for TGV")
    parser.add_argument("--epochs", type=int, default=CONFIG["epochs"])
    parser.add_argument("--lr", type=float, default=CONFIG["lr"])
    parser.add_argument("--resolution", type=int, default=CONFIG["resolution"])
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    CONFIG["epochs"] = args.epochs
    CONFIG["lr"] = args.lr
    CONFIG["resolution"] = args.resolution

    model, state = train(CONFIG, resume_from=args.resume)
