"""
DeepINNATE: 1000+ Parametreli Derin Fizik-Gömülü Ağ

INNATE nöronlarını birleştirerek daha güçlü bir model oluşturur.
Yapay sinir ağları felsefesi: Basit nöronları birleştirerek karmaşık davranış.

Mimari:
- Multi-layer fizik nöronları (stacked Advection, Vorticity)
- Spatial modulation (8x8 learnable grid per layer)
- Residual connections
- Learnable fusion weights

Parametre Dağılımı (~1000):
- 8 Advection layer × 64 spatial weights = 512
- 4 Vorticity layer × 64 spatial weights = 256
- 2 Diffusion layer × 64 spatial weights = 128
- Fusion/gate weights = ~50
- Base neuron params = ~50
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Tuple, List, Optional
from dataclasses import dataclass

from innate import (
    Advection, Vorticity, Projection, TimeMarcher,
    SpectralOps, FluidState, DEVICE
)


class SpatialModulation(nn.Module):
    """
    Uzamsal modülasyon katmanı.

    8x8 öğrenilebilir grid → resolution×resolution interpolate
    Her fizik nöronu için farklı bölgelerde farklı davranış.

    Params: 64 (8×8)
    """
    def __init__(self, resolution: int, grid_size: int = 8):
        super().__init__()
        self.resolution = resolution
        self.grid_size = grid_size

        # 8x8 öğrenilebilir grid (1'e yakın başla)
        self.weights = nn.Parameter(torch.ones(1, 1, grid_size, grid_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x'i spatial weights ile modüle et."""
        # 8x8'den resolution×resolution'a interpolate
        weights = F.interpolate(
            self.weights,
            size=(self.resolution, self.resolution),
            mode='bilinear',
            align_corners=True
        )

        # Sigmoid ile [0.5, 1.5] aralığına kısıtla
        weights = torch.sigmoid(weights) + 0.5

        return x * weights


class AdvectionBlock(nn.Module):
    """
    Advection + Spatial Modulation + Residual

    Params: 1 (base) + 64 (spatial) = 65
    """
    def __init__(self, resolution: int, diff_ops: SpectralOps):
        super().__init__()
        self.advection = Advection(resolution, diff_ops=diff_ops)
        self.spatial_mod_u = SpatialModulation(resolution)
        self.spatial_mod_v = SpatialModulation(resolution)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, state: FluidState) -> Tuple[torch.Tensor, torch.Tensor]:
        """Advection with spatial modulation."""
        adv_u, adv_v = self.advection(state)

        # Spatial modulation
        adv_u = self.spatial_mod_u(adv_u)
        adv_v = self.spatial_mod_v(adv_v)

        return adv_u, adv_v


class VorticityBlock(nn.Module):
    """
    Vorticity + Spatial Modulation

    Params: 1 (base) + 64 (spatial) = 65
    """
    def __init__(self, resolution: int, diff_ops: SpectralOps):
        super().__init__()
        self.vorticity = Vorticity(resolution, diff_ops=diff_ops)
        self.spatial_mod = SpatialModulation(resolution)

    def forward(self, state: FluidState) -> torch.Tensor:
        """Vorticity advection with spatial modulation."""
        vort_adv = self.vorticity(state)
        return self.spatial_mod(vort_adv)


class DiffusionBlock(nn.Module):
    """
    Learnable diffusion with spatial modulation.

    Params: 64 (spatial) + 1 (scale) = 65
    """
    def __init__(self, resolution: int, diff_ops: SpectralOps):
        super().__init__()
        self.diff_ops = diff_ops
        self.spatial_mod = SpatialModulation(resolution)
        self.diffusion_scale = nn.Parameter(torch.ones(1))

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        """Laplacian with spatial modulation."""
        lap = self.diff_ops.laplacian(field)
        lap = self.spatial_mod(lap)
        return self.diffusion_scale * lap


class FusionLayer(nn.Module):
    """
    Birden fazla advection/vorticity çıktısını birleştir.

    Params: n_inputs (fusion weights)
    """
    def __init__(self, n_inputs: int):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(n_inputs) / n_inputs)

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        """Weighted sum of inputs."""
        weights = F.softmax(self.weights, dim=0)
        result = torch.zeros_like(inputs[0])
        for w, x in zip(weights, inputs):
            result = result + w * x
        return result


class DeepINNATE(nn.Module):
    """
    1000+ Parametreli Derin INNATE

    Yapı:
    - 8 Advection block (stacked, parallel branches)
    - 4 Vorticity block
    - 2 Diffusion block
    - Fusion layers
    - Projection (divergence-free)
    - Time marching

    Total params: ~1000
    """
    def __init__(
        self,
        resolution: int = 64,
        n_advection: int = 8,
        n_vorticity: int = 4,
        n_diffusion: int = 2,
        nu: float = 0.001,
    ):
        super().__init__()
        self.resolution = resolution
        self.nu = nn.Parameter(torch.tensor(nu))

        # Shared spectral ops
        self.diff_ops = SpectralOps(resolution)

        # Advection blocks (parallel branches)
        self.advection_blocks = nn.ModuleList([
            AdvectionBlock(resolution, self.diff_ops)
            for _ in range(n_advection)
        ])

        # Vorticity blocks
        self.vorticity_blocks = nn.ModuleList([
            VorticityBlock(resolution, self.diff_ops)
            for _ in range(n_vorticity)
        ])

        # Diffusion blocks
        self.diffusion_blocks = nn.ModuleList([
            DiffusionBlock(resolution, self.diff_ops)
            for _ in range(n_diffusion)
        ])

        # Fusion layers
        self.adv_fusion_u = FusionLayer(n_advection)
        self.adv_fusion_v = FusionLayer(n_advection)
        self.vort_fusion = FusionLayer(n_vorticity)
        self.diff_fusion = FusionLayer(n_diffusion)

        # Projection
        self.projector = Projection(resolution, diff_ops=self.diff_ops)

        # Time marcher
        self.time_marcher = TimeMarcher(resolution)

        # Learnable scales
        self.adv_scale = nn.Parameter(torch.ones(1))
        self.vort_scale = nn.Parameter(torch.ones(1))
        self.diff_scale = nn.Parameter(torch.ones(1))

        # dt için learnable factor
        self.dt_factor = nn.Parameter(torch.tensor(1.0))

    def count_parameters(self) -> int:
        """Toplam parametre sayısı."""
        return sum(p.numel() for p in self.parameters())

    def compute_rhs(self, state: FluidState) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Navier-Stokes sağ taraf - derin hesaplama.

        du/dt = -u·∇u + ν∇²u
        dω/dt = -u·∇ω + ν∇²ω
        """
        # === ADVECTION (8 parallel branches → fusion) ===
        adv_u_list = []
        adv_v_list = []
        for block in self.advection_blocks:
            au, av = block(state)
            adv_u_list.append(au)
            adv_v_list.append(av)

        adv_u = self.adv_fusion_u(adv_u_list)
        adv_v = self.adv_fusion_v(adv_v_list)

        # === VORTICITY (4 branches → fusion) ===
        vort_list = []
        for block in self.vorticity_blocks:
            vort_list.append(block(state))
        vort_adv = self.vort_fusion(vort_list)

        # === DIFFUSION (2 branches → fusion) ===
        diff_u_list = [block(state.u) for block in self.diffusion_blocks]
        diff_v_list = [block(state.v) for block in self.diffusion_blocks]
        diff_omega_list = [block(state.vorticity) for block in self.diffusion_blocks]

        diff_u = self.diff_fusion(diff_u_list)
        diff_v = self.diff_fusion(diff_v_list)
        diff_omega = self.diff_fusion(diff_omega_list)

        # === RHS ===
        nu = torch.abs(self.nu)  # Pozitif viskozite

        du_dt = -self.adv_scale * adv_u + nu * self.diff_scale * diff_u
        dv_dt = -self.adv_scale * adv_v + nu * self.diff_scale * diff_v
        domega_dt = -self.vort_scale * vort_adv + nu * self.diff_scale * diff_omega

        return du_dt, dv_dt, domega_dt

    def step(self, state: FluidState, dt: float = 0.01) -> FluidState:
        """Tek zaman adımı."""
        # RHS hesapla
        du_dt, dv_dt, domega_dt = self.compute_rhs(state)

        # Effective dt
        effective_dt = dt * torch.sigmoid(self.dt_factor)

        # Euler step
        u_new = state.u + effective_dt * du_dt
        v_new = state.v + effective_dt * dv_dt
        omega_new = state.vorticity + effective_dt * domega_dt

        # Projection (divergence-free)
        u_new, v_new, p_new = self.projector(u_new, v_new)

        return FluidState(
            u=u_new,
            v=v_new,
            p=p_new,
            vorticity=omega_new,
            t=state.t + dt
        )

    def forward(self, initial_state: FluidState, num_steps: int, dt: float = 0.01) -> List[FluidState]:
        """Forward pass - num_steps adım simülasyon."""
        states = [initial_state]
        state = initial_state

        for _ in range(num_steps):
            state = self.step(state, dt)
            states.append(state)

        return states

    def physics_loss(self, state: FluidState) -> dict:
        """Fizik kayıpları (regularizasyon için)."""
        # Divergence (sıfır olmalı)
        div = self.diff_ops.divergence(state.u, state.v)
        div_loss = (div ** 2).mean()

        # Energy stability (çok büyümemeli)
        energy = 0.5 * (state.u**2 + state.v**2).mean()

        return {
            'divergence': div_loss,
            'energy': energy,
        }


# =============================================================================
# LAMB-OSEEN TEST
# =============================================================================

def create_lamb_oseen_ic(resolution: int, Re: float, device=None) -> FluidState:
    """Lamb-Oseen vortex başlangıç koşulu."""
    if device is None:
        device = DEVICE

    nu = 1.0 / Re
    Gamma = 2 * np.pi
    r_c0 = 0.5
    x0, y0 = np.pi, np.pi

    # Grid
    L = 2 * np.pi
    x = np.linspace(0, L, resolution, endpoint=False)
    y = np.linspace(0, L, resolution, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # Analytical solution at t=0
    dx = X - x0
    dy = Y - y0
    r2 = dx**2 + dy**2
    r = np.sqrt(r2 + 1e-10)

    # Vorticity
    omega = (Gamma / (np.pi * r_c0**2)) * np.exp(-r2 / r_c0**2)

    # Velocity (from Biot-Savart)
    v_theta = (Gamma / (2 * np.pi * r)) * (1 - np.exp(-r2 / r_c0**2))
    theta = np.arctan2(dy, dx)
    u = -v_theta * np.sin(theta)
    v = v_theta * np.cos(theta)

    # To tensors
    u_t = torch.tensor(u, dtype=torch.float32, device=device).unsqueeze(0)
    v_t = torch.tensor(v, dtype=torch.float32, device=device).unsqueeze(0)
    omega_t = torch.tensor(omega, dtype=torch.float32, device=device).unsqueeze(0)
    p_t = torch.zeros_like(u_t)

    return FluidState(
        u=u_t, v=v_t, p=p_t, vorticity=omega_t,
        t=torch.tensor(0.0, device=device)
    )


def lamb_oseen_analytical(X, Y, t, Re, Gamma=2*np.pi, r_c0=0.5):
    """Lamb-Oseen analitik çözüm."""
    nu = 1.0 / Re
    x0, y0 = np.pi, np.pi

    r_c = np.sqrt(4 * nu * t + r_c0**2)

    dx = X - x0
    dy = Y - y0
    r2 = dx**2 + dy**2
    r = np.sqrt(r2 + 1e-10)

    omega = (Gamma / (np.pi * r_c**2)) * np.exp(-r2 / r_c**2)

    v_theta = (Gamma / (2 * np.pi * r)) * (1 - np.exp(-r2 / r_c**2))
    theta = np.arctan2(dy, dx)
    u = -v_theta * np.sin(theta)
    v = v_theta * np.cos(theta)

    return u, v, omega


def train_deep_innate(
    resolution: int = 64,
    Re: float = 1000,
    num_epochs: int = 500,
    num_steps: int = 100,
    dt: float = 0.01,
):
    """DeepINNATE'i Lamb-Oseen üzerinde eğit."""

    print("="*60)
    print("DeepINNATE - LAMB-OSEEN TRAINING")
    print("="*60)

    # Model
    model = DeepINNATE(
        resolution=resolution,
        n_advection=8,
        n_vorticity=4,
        n_diffusion=2,
        nu=1.0/Re,
    ).to(DEVICE)

    n_params = model.count_parameters()
    print(f"Model parametreleri: {n_params}")
    print(f"Device: {DEVICE}")
    print(f"Resolution: {resolution}")
    print(f"Re: {Re}")
    print()

    # Grid for analytical solution
    L = 2 * np.pi
    x = np.linspace(0, L, resolution, endpoint=False)
    y = np.linspace(0, L, resolution, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)

    # Training
    model.train()
    best_loss = float('inf')

    for epoch in range(num_epochs):
        # Curriculum: start with fewer steps
        current_steps = min(5 + epoch // 20, num_steps)

        # Initial condition
        state = create_lamb_oseen_ic(resolution, Re)

        # Forward
        states = model(state, num_steps=current_steps, dt=dt)

        # Loss: match analytical solution at each step
        total_loss = torch.tensor(0.0, device=DEVICE)

        for i, s in enumerate(states):
            t = i * dt
            if t > 0:
                # Analytical
                u_ref, v_ref, omega_ref = lamb_oseen_analytical(X, Y, t, Re)
                u_ref = torch.tensor(u_ref, dtype=torch.float32, device=DEVICE)
                v_ref = torch.tensor(v_ref, dtype=torch.float32, device=DEVICE)

                # MSE loss
                loss_u = F.mse_loss(s.u.squeeze(), u_ref)
                loss_v = F.mse_loss(s.v.squeeze(), v_ref)
                total_loss = total_loss + loss_u + loss_v

        # Physics loss
        phys = model.physics_loss(states[-1])
        total_loss = total_loss + 0.1 * phys['divergence']

        # Energy stability
        energies = [0.5 * (s.u**2 + s.v**2).mean() for s in states]
        for i in range(1, len(energies)):
            growth = energies[i] / (energies[i-1] + 1e-8)
            if growth > 1.1:  # Max 10% growth per step
                total_loss = total_loss + 0.1 * (growth - 1.1)**2

        # Backward
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Logging
        if epoch % 50 == 0:
            # Validation
            model.eval()
            with torch.no_grad():
                val_state = create_lamb_oseen_ic(resolution, Re)
                val_states = model(val_state, num_steps=num_steps, dt=dt)

                final_t = num_steps * dt
                u_ref, v_ref, _ = lamb_oseen_analytical(X, Y, final_t, Re)

                u_pred = val_states[-1].u.squeeze().cpu().numpy()
                v_pred = val_states[-1].v.squeeze().cpu().numpy()

                l2_error = np.sqrt(np.mean((u_pred - u_ref)**2 + (v_pred - v_ref)**2))

            model.train()

            print(f"Epoch {epoch:4d} | Loss: {total_loss.item():.4e} | "
                  f"L2 Error: {l2_error:.4e} | Steps: {current_steps}")

            if total_loss.item() < best_loss:
                best_loss = total_loss.item()

    # Final validation
    print("\n" + "="*60)
    print("FINAL VALIDATION")
    print("="*60)

    model.eval()
    with torch.no_grad():
        state = create_lamb_oseen_ic(resolution, Re)
        states = model(state, num_steps=num_steps, dt=dt)

        # Errors at different times
        times = [0.25, 0.5, 0.75, 1.0]
        for t_target in times:
            idx = int(t_target / dt)
            if idx < len(states):
                u_ref, v_ref, omega_ref = lamb_oseen_analytical(X, Y, t_target, Re)
                u_pred = states[idx].u.squeeze().cpu().numpy()
                v_pred = states[idx].v.squeeze().cpu().numpy()
                omega_pred = states[idx].vorticity.squeeze().cpu().numpy()

                l2_vel = np.sqrt(np.mean((u_pred - u_ref)**2 + (v_pred - v_ref)**2))
                l2_omega = np.sqrt(np.mean((omega_pred - omega_ref)**2))

                print(f"t={t_target:.2f} | Velocity L2: {l2_vel:.4e} | Vorticity L2: {l2_omega:.4e}")

    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--resolution', type=int, default=64)
    parser.add_argument('--Re', type=float, default=1000)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--dt', type=float, default=0.01)

    args = parser.parse_args()

    model = train_deep_innate(
        resolution=args.resolution,
        Re=args.Re,
        num_epochs=args.epochs,
        num_steps=args.steps,
        dt=args.dt,
    )

    print(f"\nFinal model parameters: {model.count_parameters()}")
