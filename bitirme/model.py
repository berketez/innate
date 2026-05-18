"""
INNATE3D Network for Taylor-Green Vortex 3D

DÜZELTILMIŞ VERSİYON - Bellek verimli tasarım.

Önceki hata: 832 ayrı nöron objesi → bellek patlaması
Düzeltme: Her layer'da TEK nöron + learnable modulator

Mimari:
    20 layer, her biri:
    - Tek Advection3D veya Diffusion nöronu
    - Physics-informed MLP modulator (E, Z, div, t → scale)

Hedef: ~10K parametre

TGV3D:
    Domain: [0, 2π]³
    IC: u = sin(x)cos(y)cos(z), v = -cos(x)sin(y)cos(z), w = 0
    BC: Periodic
    Re ≈ 1000 (ν = 0.001)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from dataclasses import dataclass
import sys
import os

# INNATE kütüphanesinden import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from innate import (
    FluidState3D, SpectralOps3D,
    Advection3D, Projection3D,
    DEVICE
)


class DiffusionOp(nn.Module):
    """
    Basit difüzyon operatörü: ν∇²u
    Learnable: viscosity_scale (1 param)
    """
    def __init__(self, nu: float, spectral_ops: SpectralOps3D):
        super().__init__()
        self.nu = nu
        self.ops = spectral_ops
        self.viscosity_scale = nn.Parameter(torch.ones(1))

    def forward(self, u: torch.Tensor, v: torch.Tensor, w: torch.Tensor
               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        nu_eff = self.nu * torch.clamp(self.viscosity_scale, 0.1, 5.0)
        return (
            nu_eff * self.ops.laplacian(u),
            nu_eff * self.ops.laplacian(v),
            nu_eff * self.ops.laplacian(w)
        )


class PhysicsModulator(nn.Module):
    """
    Physics-informed modulator MLP.
    Input: [E, Z, div, t] → Output: [scale_u, scale_v, scale_w]
    """
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 3)
        )
        # Initialize to output ~1.0
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, energy: torch.Tensor, enstrophy: torch.Tensor,
                divergence: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """Returns scale factors [3] for u, v, w."""
        # Normalize inputs
        features = torch.stack([
            energy / 0.125,  # Normalize by IC energy
            enstrophy / 1.0,
            divergence * 1000,  # Scale up small divergence
            time / 1.0
        ], dim=-1)

        # Output: base 1.0 + small learned correction
        return 1.0 + 0.1 * self.net(features)


class INNATELayer(nn.Module):
    """
    Tek INNATE Layer - BİR nöron + modulator.

    Bellek verimli: 832 ayrı obje yerine tek obje.
    """
    def __init__(
        self,
        neuron_type: str,  # 'advection' veya 'diffusion'
        resolution: int,
        nu: float,
        spectral_ops: SpectralOps3D,
        hidden_dim: int = 64
    ):
        super().__init__()
        self.neuron_type = neuron_type
        self.ops = spectral_ops

        # TEK nöron
        if neuron_type == 'advection':
            self.neuron = Advection3D(resolution, diff_ops=spectral_ops)
        else:
            self.neuron = DiffusionOp(nu, spectral_ops)

        # Physics-informed modulator
        self.modulator = PhysicsModulator(hidden_dim)

        # Layer scale
        self.layer_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, state: FluidState3D) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Layer forward pass."""
        # Compute physics diagnostics (detached for stability)
        with torch.no_grad():
            energy = state.kinetic_energy().mean()
            enstrophy = state.enstrophy().mean()
            div = self.ops.divergence(state.u, state.v, state.w).abs().mean()
            time = state.t.mean()

        # Get modulation scales
        scales = self.modulator(energy, enstrophy, div, time)  # [3]

        # Apply neuron
        if self.neuron_type == 'advection':
            rhs_u, rhs_v, rhs_w = self.neuron(state)
        else:
            rhs_u, rhs_v, rhs_w = self.neuron(state.u, state.v, state.w)

        # Apply scales
        layer_s = torch.sigmoid(self.layer_scale)
        rhs_u = layer_s * scales[0] * rhs_u
        rhs_v = layer_s * scales[1] * rhs_v
        rhs_w = layer_s * scales[2] * rhs_w

        return rhs_u, rhs_v, rhs_w


class INNATE3D_TGV(nn.Module):
    """
    INNATE 3D Network for Taylor-Green Vortex - DÜZELTILMIŞ VERSİYON

    20 layer, her biri tek nöron + modulator.
    Advection ve Diffusion dönüşümlü.
    """
    def __init__(
        self,
        resolution: int = 32,
        nu: float = 0.001,
        num_layers: int = 20,
        hidden_dim: int = 64
    ):
        super().__init__()
        self.resolution = resolution
        self.nu = nu
        self.num_layers = num_layers
        self.domain_size = 2 * math.pi

        # Paylaşılan spektral operatörler (TEK INSTANCE)
        self.spectral_ops = SpectralOps3D(resolution, domain_size=self.domain_size)

        # Sequential layers - her biri TEK nöron
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            neuron_type = 'advection' if i % 2 == 0 else 'diffusion'
            self.layers.append(INNATELayer(
                neuron_type=neuron_type,
                resolution=resolution,
                nu=nu,
                spectral_ops=self.spectral_ops,
                hidden_dim=hidden_dim
            ))

        # Final projeksiyon
        self.projection = Projection3D(resolution, diff_ops=self.spectral_ops)

        # Grid koordinatları (register_buffer - parametre değil)
        dx = self.domain_size / resolution
        coords = torch.linspace(0, self.domain_size - dx, resolution)
        X, Y, Z = torch.meshgrid(coords, coords, coords, indexing='ij')
        self.register_buffer('X', X)
        self.register_buffer('Y', Y)
        self.register_buffer('Z', Z)

    def tgv_initial_condition(self, batch_size: int = 1) -> FluidState3D:
        """Taylor-Green Vortex 3D IC"""
        X = self.X.unsqueeze(0).expand(batch_size, -1, -1, -1)
        Y = self.Y.unsqueeze(0).expand(batch_size, -1, -1, -1)
        Z = self.Z.unsqueeze(0).expand(batch_size, -1, -1, -1)

        u = torch.sin(X) * torch.cos(Y) * torch.cos(Z)
        v = -torch.cos(X) * torch.sin(Y) * torch.cos(Z)
        w = torch.zeros_like(X)
        p = (torch.cos(2*X) + torch.cos(2*Y)) * (torch.cos(2*Z) + 2) / 16

        omega_x, omega_y, omega_z = self.spectral_ops.curl(u, v, w)
        t = torch.zeros(batch_size, 1, device=X.device)

        return FluidState3D(
            u=u, v=v, w=w, p=p,
            omega_x=omega_x, omega_y=omega_y, omega_z=omega_z,
            t=t
        )

    def step(self, state: FluidState3D, dt: float) -> FluidState3D:
        """Tek zaman adımı."""
        u, v, w = state.u, state.v, state.w

        for layer in self.layers:
            # Layer RHS hesapla
            current_state = FluidState3D(
                u=u, v=v, w=w, p=state.p,
                omega_x=state.omega_x, omega_y=state.omega_y, omega_z=state.omega_z,
                t=state.t
            )
            rhs_u, rhs_v, rhs_w = layer(current_state)

            # Update (advection: -, diffusion: +)
            if layer.neuron_type == 'advection':
                u = u - dt * rhs_u
                v = v - dt * rhs_v
                w = w - dt * rhs_w
            else:
                u = u + dt * rhs_u
                v = v + dt * rhs_v
                w = w + dt * rhs_w

        # Projeksiyon
        u, v, w, p = self.projection(u, v, w, dt=dt)

        # Vortisite güncelle
        omega_x, omega_y, omega_z = self.spectral_ops.curl(u, v, w)
        t_new = state.t + dt

        return FluidState3D(
            u=u, v=v, w=w, p=p,
            omega_x=omega_x, omega_y=omega_y, omega_z=omega_z,
            t=t_new
        )

    def step_with_unprojected(self, state: FluidState3D, dt: float
                              ) -> Tuple[FluidState3D, FluidState3D]:
        """
        Tek zaman adımı - hem projeksiyonlu hem projeksiyonsuz state döndür.
        NS residual loss için gerekli.
        """
        u, v, w = state.u, state.v, state.w

        for layer in self.layers:
            current_state = FluidState3D(
                u=u, v=v, w=w, p=state.p,
                omega_x=state.omega_x, omega_y=state.omega_y, omega_z=state.omega_z,
                t=state.t
            )
            rhs_u, rhs_v, rhs_w = layer(current_state)

            if layer.neuron_type == 'advection':
                u = u - dt * rhs_u
                v = v - dt * rhs_v
                w = w - dt * rhs_w
            else:
                u = u + dt * rhs_u
                v = v + dt * rhs_v
                w = w + dt * rhs_w

        t_new = state.t + dt

        # Unprojected state
        omega_x_u, omega_y_u, omega_z_u = self.spectral_ops.curl(u, v, w)
        unprojected = FluidState3D(
            u=u, v=v, w=w, p=state.p,
            omega_x=omega_x_u, omega_y=omega_y_u, omega_z=omega_z_u,
            t=t_new
        )

        # Projected state
        u_p, v_p, w_p, p = self.projection(u, v, w, dt=dt)
        omega_x, omega_y, omega_z = self.spectral_ops.curl(u_p, v_p, w_p)
        projected = FluidState3D(
            u=u_p, v=v_p, w=w_p, p=p,
            omega_x=omega_x, omega_y=omega_y, omega_z=omega_z,
            t=t_new
        )

        return projected, unprojected

    def forward(self, state: FluidState3D, num_steps: int, dt: float) -> List[FluidState3D]:
        """Birden fazla zaman adımı."""
        states = [state]
        for _ in range(num_steps):
            state = self.step(state, dt)
            states.append(state)
        return states

    def get_diagnostics(self, state: FluidState3D, dt: float = 0.002) -> dict:
        """Fiziksel tanı değerleri."""
        with torch.no_grad():
            energy = state.kinetic_energy().mean().item()
            enstrophy = state.enstrophy().mean().item()
            helicity = state.helicity().mean().item()

            div = self.spectral_ops.divergence(state.u, state.v, state.w)
            div_mean = div.abs().mean().item()
            div_max = div.abs().max().item()

            max_vel = state.velocity_magnitude().max().item()
            dx = self.domain_size / self.resolution
            cfl = max_vel * dt / dx

        return {
            'kinetic_energy': energy,
            'enstrophy': enstrophy,
            'helicity': helicity,
            'divergence_mean': div_mean,
            'divergence_max': div_max,
            'max_velocity': max_vel,
            'cfl': cfl,
            'time': state.t.mean().item()
        }

    def count_parameters(self) -> dict:
        """Parametre sayıları."""
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)

        layer_params = sum(
            sum(p.numel() for p in layer.parameters() if p.requires_grad)
            for layer in self.layers
        )
        proj_params = sum(p.numel() for p in self.projection.parameters() if p.requires_grad)

        return {
            'total': total,
            'layers': layer_params,
            'projection': proj_params,
            'num_layers': self.num_layers
        }


def create_model(
    resolution: int = 32,
    nu: float = 0.001,
    num_layers: int = 20,
    neurons_per_layer: int = None,  # Artık kullanılmıyor, uyumluluk için
    target_params: int = 10000
) -> INNATE3D_TGV:
    """
    INNATE3D modeli oluştur.

    Yeni mimari:
        - num_layers layer (advection/diffusion dönüşümlü)
        - Her layer: tek nöron + MLP modulator

    Parametre hesabı:
        Her layer: ~450 param (MLP: 4*64 + 64*3 + biases = 451)
        20 layer: ~9000 param
        + projection: ~2 param
        Total: ~9K param
    """
    # hidden_dim'i target_params'a göre ayarla
    # Her layer: 4*h + h + h*3 + 3 + 2 (nöron params) ≈ 7h + 7
    # num_layers * (7h + 7) ≈ target_params
    # h ≈ (target_params / num_layers - 7) / 7

    hidden_dim = max(16, int((target_params / num_layers - 7) / 7))
    hidden_dim = min(hidden_dim, 128)  # Cap at 128

    model = INNATE3D_TGV(
        resolution=resolution,
        nu=nu,
        num_layers=num_layers,
        hidden_dim=hidden_dim
    )

    params = model.count_parameters()
    print(f"\n{'='*60}")
    print(f"INNATE3D Network (Memory-Efficient)")
    print(f"{'='*60}")
    print(f"Architecture: {num_layers} layers × 1 neuron + MLP(h={hidden_dim})")
    print(f"Layer pattern: Advection → Diffusion → ...")
    print(f"Resolution: {resolution}³")
    print(f"Total parameters: {params['total']:,}")
    print(f"{'='*60}\n")

    return model


if __name__ == "__main__":
    print("Testing INNATE3D_TGV (memory-efficient)...")
    print(f"Device: {DEVICE}")

    # Bellek temizle
    if DEVICE.type == 'mps':
        torch.mps.empty_cache()

    model = create_model(
        resolution=32,
        nu=0.001,
        num_layers=20,
        target_params=10000
    )
    model = model.to(DEVICE)

    state = model.tgv_initial_condition(batch_size=1)
    state = FluidState3D(
        u=state.u.to(DEVICE), v=state.v.to(DEVICE), w=state.w.to(DEVICE),
        p=state.p.to(DEVICE),
        omega_x=state.omega_x.to(DEVICE), omega_y=state.omega_y.to(DEVICE),
        omega_z=state.omega_z.to(DEVICE),
        t=state.t.to(DEVICE)
    )

    dt = 0.002
    print("\nInitial state:")
    for k, v in model.get_diagnostics(state, dt=dt).items():
        print(f"  {k}: {v:.6f}")

    print("\nRunning 5 steps...")
    import time
    start = time.time()
    for _ in range(5):
        state = model.step(state, dt)
    if DEVICE.type == 'mps':
        torch.mps.synchronize()
    elapsed = time.time() - start
    print(f"5 steps took: {elapsed:.3f} sec")

    print("\nFinal state:")
    for k, v in model.get_diagnostics(state, dt=dt).items():
        print(f"  {k}: {v:.6f}")

    print("\nTest completed!")
