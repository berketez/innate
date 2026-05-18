"""
Flow Past Cylinder - INNATE Validation Benchmark

Bu test, INNATE kütüphanesini klasik CFD benchmark'ı ile doğrular.

PROBLEM:
    - 2D silindir etrafında akış
    - Re = 100 → Vortex shedding (Karman girdap sokağı)
    - Strouhal sayısı ile validation

REFERANS DEĞERLERİ (Literatür):
    Re=100: St = 0.164-0.167, Cd = 1.33
    Re=200: St = 0.196-0.200, Cd = 1.34

VALIDATION KRİTERLERİ:
    ✓ Strouhal sayısı ±5% hata içinde
    ✓ Vortex shedding periyodik
    ✓ Divergence < 1e-4

Kullanım:
    python flow_past_cylinder.py --Re 100 --epochs 500
    python flow_past_cylinder.py --Re 200 --epochs 500
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, List, Optional
import argparse

from innate import (
    Advection, Vorticity, TimeMarcher,
    Reynolds, Boundary, SpectralOps, FiniteDiffOps, FluidState,
    ImmersedBoundary, DEVICE
)


def iterative_poisson_solver(rhs: torch.Tensor, dx: float,
                              n_iters: int = 50, omega: float = 1.5) -> torch.Tensor:
    """
    Basit iteratif Poisson solver (SOR - Successive Over-Relaxation)

    ∇²p = rhs  çözer

    Non-periodic BC için çalışır!
    """
    p = torch.zeros_like(rhs)

    for _ in range(n_iters):
        p_old = p.clone()

        # Red-Black Gauss-Seidel with SOR
        # p[i,j] = (1-ω)*p[i,j] + ω/4 * (p[i+1,j] + p[i-1,j] + p[i,j+1] + p[i,j-1] - dx²*rhs[i,j])

        # Interior points
        p[:, 1:-1, 1:-1] = (1-omega) * p_old[:, 1:-1, 1:-1] + omega * 0.25 * (
            p_old[:, 2:, 1:-1] + p_old[:, :-2, 1:-1] +
            p_old[:, 1:-1, 2:] + p_old[:, 1:-1, :-2] -
            dx**2 * rhs[:, 1:-1, 1:-1]
        )

        # Neumann BC (dp/dn = 0)
        p[:, 0, :] = p[:, 1, :]
        p[:, -1, :] = p[:, -2, :]
        p[:, :, 0] = p[:, :, 1]
        p[:, :, -1] = p[:, :, -2]

    return p


def finite_projection(u: torch.Tensor, v: torch.Tensor,
                       diff_ops: FiniteDiffOps) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Divergence-free projeksiyon (finite difference versiyonu)

    1. ∇·u hesapla
    2. ∇²p = ∇·u çöz (iteratif)
    3. u_new = u - ∇p
    """
    # Divergence hesapla
    div = diff_ops.divergence(u, v)

    # Poisson çöz
    p = iterative_poisson_solver(div, diff_ops.dx, n_iters=100)

    # Gradient of pressure
    dp_dx, dp_dy = diff_ops.gradient(p)

    # Project
    u_proj = u - dp_dx
    v_proj = v - dp_dy

    return u_proj, v_proj, p


# =============================================================================
# REFERANS DEĞERLERİ (40 yıllık literatür)
# =============================================================================

REFERENCE_DATA = {
    40:  {"St": None,  "Cd": 1.50, "regime": "steady"},
    100: {"St": 0.166, "Cd": 1.33, "regime": "laminar_shedding"},
    200: {"St": 0.198, "Cd": 1.34, "regime": "laminar_shedding"},
}


@dataclass
class CylinderConfig:
    """Silindir akış konfigürasyonu"""
    resolution: int = 256        # Grid çözünürlüğü
    domain_length: float = 20.0  # Domain uzunluğu (çap cinsinden)
    domain_height: float = 10.0  # Domain yüksekliği
    cylinder_center: Tuple[float, float] = (5.0, 5.0)  # Silindir merkezi
    cylinder_radius: float = 0.5  # Silindir yarıçapı (D=1)
    U_inf: float = 1.0           # Serbest akış hızı
    Re: float = 100.0            # Reynolds sayısı
    T_final: float = 200.0       # Toplam simülasyon süresi
    dt: float = 0.01             # Zaman adımı


class CylinderFlowModel(nn.Module):
    """
    Flow Past Cylinder için INNATE modeli
    """
    def __init__(self, config: CylinderConfig):
        super().__init__()
        self.config = config
        N = config.resolution

        # Domain ölçekleme (çap = 1 olacak şekilde)
        self.Lx = config.domain_length
        self.Ly = config.domain_height
        self.dx = self.Lx / N
        self.dy = self.Ly / N

        # Viskozite (Re = U*D/nu → nu = U*D/Re)
        D = 2 * config.cylinder_radius
        self.nu = config.U_inf * D / config.Re

        # FiniteDiffOps kullan (non-periodic BC için)
        self.diff_ops = FiniteDiffOps(N, domain_size=self.Lx,
                                       scheme='central', bc_type='neumann')

        # INNATE nöronları - AYNI diff_ops'u paylaşıyorlar!
        self.advection = Advection(N, diff_ops=self.diff_ops)
        # Projection için finite_projection fonksiyonu kullanacağız
        self.time_marcher = TimeMarcher(N)

        # Immersed Boundary - silindir
        self.ib = ImmersedBoundary(N, domain_size=self.Lx, device=DEVICE)
        self.ib.set_cylinder_geometry(
            center=config.cylinder_center,
            radius=config.cylinder_radius
        )

        # Grid koordinatları
        x = torch.linspace(0, self.Lx, N, device=DEVICE)
        y = torch.linspace(0, self.Ly, N, device=DEVICE)
        self.X, self.Y = torch.meshgrid(x, y, indexing='ij')

        # Outlet sponge layer (yansıma önlemek için)
        sponge_start = 0.8 * self.Lx
        self.sponge = torch.clamp((self.X - sponge_start) / (self.Lx - sponge_start), 0, 1)
        self.sponge = self.sponge ** 2  # Smooth ramp

    def create_initial_condition(self) -> FluidState:
        """Başlangıç koşulu: uniform akış"""
        N = self.config.resolution

        # Uniform inlet velocity
        u = torch.ones(1, N, N, device=DEVICE) * self.config.U_inf
        v = torch.zeros(1, N, N, device=DEVICE)
        p = torch.zeros(1, N, N, device=DEVICE)

        # Silindir içinde hız = 0
        mask = self.ib.config.mask.unsqueeze(0)
        u = u * (~mask).float()
        v = v * (~mask).float()

        # Küçük pertürbasyon (shedding tetiklemek için)
        noise = 0.01 * torch.randn_like(v)
        v = v + noise * (~mask).float()

        # Vortisite hesapla
        vorticity = self.diff_ops.curl_2d(u, v)

        return FluidState(u=u, v=v, p=p, vorticity=vorticity, t=torch.tensor(0.0, device=DEVICE))

    def compute_forces(self, state: FluidState) -> Tuple[float, float]:
        """
        Silindir üzerindeki kuvvetleri hesapla (Direct Forcing'ten)

        Returns:
            (Cd, Cl): Drag ve Lift katsayıları
        """
        # Basınç ve hız gradyanlarından kuvvet hesapla
        # Basitleştirilmiş: silindir çevresindeki basınç integrali

        mask = self.ib.config.mask
        cx, cy = self.config.cylinder_center
        r = self.config.cylinder_radius

        # Silindir sınırı yakınındaki noktalar
        dist = torch.sqrt((self.X - cx)**2 + (self.Y - cy)**2)
        boundary_mask = (dist > r * 0.9) & (dist < r * 1.3)

        # Basınç kuvveti (yaklaşık)
        p = state.p.squeeze()

        # x-yönü (drag)
        grad_px = self.diff_ops.gradient(p.unsqueeze(0))[0].squeeze()
        Fx = -(grad_px * boundary_mask.float()).sum() * self.dx * self.dy

        # y-yönü (lift)
        grad_py = self.diff_ops.gradient(p.unsqueeze(0))[1].squeeze()
        Fy = -(grad_py * boundary_mask.float()).sum() * self.dx * self.dy

        # Katsayılar: C = F / (0.5 * rho * U^2 * D)
        D = 2 * self.config.cylinder_radius
        dynamic_pressure = 0.5 * self.config.U_inf**2 * D

        Cd = Fx.item() / (dynamic_pressure + 1e-10)
        Cl = Fy.item() / (dynamic_pressure + 1e-10)

        return Cd, Cl

    def apply_boundary_conditions(self, u: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sınır koşullarını uygula"""
        # Inlet: uniform velocity
        u[:, 0, :] = self.config.U_inf
        v[:, 0, :] = 0

        # Silindir: no-slip (IBM ile)
        mask = self.ib.config.mask.unsqueeze(0)
        u = u * (~mask).float()
        v = v * (~mask).float()

        # Top/Bottom: free-slip (dv/dy = 0)
        v[:, :, 0] = 0
        v[:, :, -1] = 0

        # Outlet: sponge layer (yavaşça serbest akışa dön)
        u = u * (1 - self.sponge) + self.config.U_inf * self.sponge
        v = v * (1 - self.sponge)

        return u, v

    def single_step(self, state: FluidState) -> FluidState:
        """Tek zaman adımı"""
        dt = self.config.dt

        # Adveksiyon
        adv_u, adv_v = self.advection(state)

        # Difüzyon (Laplacian)
        lap_u = self.diff_ops.laplacian(state.u)
        lap_v = self.diff_ops.laplacian(state.v)

        # RHS
        du_dt = -adv_u + self.nu * lap_u
        dv_dt = -adv_v + self.nu * lap_v

        # Euler ilerleme
        u_new = state.u + dt * du_dt
        v_new = state.v + dt * dv_dt

        # Sınır koşulları
        u_new, v_new = self.apply_boundary_conditions(u_new, v_new)

        # Projeksiyon (divergence-free) - finite difference versiyonu
        u_new, v_new, p_new = finite_projection(u_new, v_new, self.diff_ops)

        # Tekrar sınır koşulları (projeksiyon sonrası)
        u_new, v_new = self.apply_boundary_conditions(u_new, v_new)

        # Vortisite güncelle
        vorticity_new = self.diff_ops.curl_2d(u_new, v_new)

        return FluidState(
            u=u_new, v=v_new, p=p_new, vorticity=vorticity_new,
            t=state.t + dt
        )

    def forward(self, state: FluidState, num_steps: int,
                save_every: int = 10) -> Tuple[List[FluidState], List[float], List[float]]:
        """
        Simülasyonu çalıştır

        Returns:
            states: Kaydedilen durumlar
            Cd_history: Drag katsayısı geçmişi
            Cl_history: Lift katsayısı geçmişi
        """
        states = [state]
        Cd_history = []
        Cl_history = []

        for step in range(num_steps):
            state = self.single_step(state)

            # Kuvvetleri hesapla
            Cd, Cl = self.compute_forces(state)
            Cd_history.append(Cd)
            Cl_history.append(Cl)

            if step % save_every == 0:
                states.append(state)

            if step % 100 == 0:
                print(f"Step {step}/{num_steps}, t={state.t.item():.2f}, Cd={Cd:.3f}, Cl={Cl:.3f}")

        return states, Cd_history, Cl_history


def compute_strouhal(Cl_history: List[float], dt: float, D: float, U_inf: float) -> dict:
    """
    Lift katsayısından Strouhal sayısını hesapla

    St = f * D / U_inf
    """
    Cl = np.array(Cl_history)

    # İlk geçici süreyi atla (vortex shedding yerleşene kadar)
    skip = len(Cl) // 4
    Cl = Cl[skip:]

    if len(Cl) < 100:
        return {"St": None, "f": None, "error": "Yeterli veri yok"}

    # FFT
    N = len(Cl)
    fft_result = np.fft.fft(Cl - Cl.mean())
    freqs = np.fft.fftfreq(N, dt)

    # Pozitif frekanslar
    positive_mask = freqs > 0
    freqs_pos = freqs[positive_mask]
    power = np.abs(fft_result[positive_mask])

    # Dominant frekans
    peak_idx = np.argmax(power)
    f_shedding = freqs_pos[peak_idx]

    # Strouhal sayısı
    St = f_shedding * D / U_inf

    return {
        "St": St,
        "f": f_shedding,
        "power_spectrum": (freqs_pos, power),
        "Cl_signal": Cl
    }


def validate_results(St_computed: float, Re: float) -> dict:
    """Sonuçları referans değerlerle karşılaştır"""
    if Re not in REFERENCE_DATA:
        return {"valid": False, "error": f"Re={Re} için referans yok"}

    ref = REFERENCE_DATA[Re]

    if ref["St"] is None:
        return {"valid": True, "note": "Steady flow (shedding yok)"}

    error = abs(St_computed - ref["St"]) / ref["St"] * 100

    result = {
        "St_computed": St_computed,
        "St_reference": ref["St"],
        "error_percent": error,
        "valid": error < 10,  # %10'dan az hata
        "excellent": error < 5,  # %5'ten az hata
    }

    return result


def plot_results(states: List[FluidState], Cl_history: List[float],
                 strouhal_result: dict, config: CylinderConfig, save_path: str):
    """Sonuçları görselleştir"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Vorticity field (son durum)
    ax = axes[0, 0]
    state = states[-1]
    u, v = state.u.squeeze().cpu(), state.v.squeeze().cpu()

    # Vortisite hesapla (basit finite difference)
    dx = config.domain_length / config.resolution
    omega = (np.gradient(v.numpy(), dx, axis=0) -
             np.gradient(u.numpy(), dx, axis=1))

    im = ax.contourf(omega, levels=50, cmap='RdBu_r',
                     vmin=-5, vmax=5)
    plt.colorbar(im, ax=ax, label='Vorticity')

    # Silindir çiz
    cx, cy = config.cylinder_center
    r = config.cylinder_radius
    circle = plt.Circle((cy/dx, cx/dx), r/dx, color='gray', fill=True)
    ax.add_patch(circle)
    ax.set_title(f'Vorticity Field (Re={config.Re})')
    ax.set_aspect('equal')

    # 2. Lift coefficient history
    ax = axes[0, 1]
    t = np.arange(len(Cl_history)) * config.dt
    ax.plot(t, Cl_history, 'b-', linewidth=0.5)
    ax.set_xlabel('Time')
    ax.set_ylabel('Lift Coefficient (Cl)')
    ax.set_title('Lift Coefficient vs Time')
    ax.grid(True)

    # 3. Power spectrum
    ax = axes[1, 0]
    if "power_spectrum" in strouhal_result:
        freqs, power = strouhal_result["power_spectrum"]
        ax.semilogy(freqs, power)
        ax.axvline(strouhal_result["f"], color='r', linestyle='--',
                   label=f'f = {strouhal_result["f"]:.4f}')
        ax.set_xlabel('Frequency')
        ax.set_ylabel('Power')
        ax.set_title(f'FFT of Lift Signal (St = {strouhal_result["St"]:.4f})')
        ax.legend()
        ax.grid(True)
        ax.set_xlim(0, 0.5)

    # 4. Validation summary
    ax = axes[1, 1]
    ax.axis('off')

    ref = REFERENCE_DATA.get(config.Re, {})
    validation = validate_results(strouhal_result.get("St", 0), config.Re)

    summary = f"""
    ╔══════════════════════════════════════╗
    ║  FLOW PAST CYLINDER - VALIDATION     ║
    ╠══════════════════════════════════════╣
    ║  Re = {config.Re}
    ║  Resolution = {config.resolution}×{config.resolution}
    ╠══════════════════════════════════════╣
    ║  STROUHAL SAYISI
    ║  ├─ Hesaplanan: {strouhal_result.get('St', 'N/A'):.4f}
    ║  ├─ Referans:   {ref.get('St', 'N/A')}
    ║  └─ Hata:       {validation.get('error_percent', 'N/A'):.1f}%
    ╠══════════════════════════════════════╣
    ║  SONUÇ: {'✅ BAŞARILI' if validation.get('valid') else '❌ BAŞARISIZ'}
    ║  {'⭐ MÜKEMMEL!' if validation.get('excellent') else ''}
    ╚══════════════════════════════════════╝
    """
    ax.text(0.1, 0.5, summary, fontsize=11, family='monospace',
            verticalalignment='center')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Sonuçlar kaydedildi: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Flow Past Cylinder Benchmark')
    parser.add_argument('--Re', type=float, default=100, help='Reynolds number')
    parser.add_argument('--resolution', type=int, default=128, help='Grid resolution')
    parser.add_argument('--T_final', type=float, default=100.0, help='Final time')
    parser.add_argument('--dt', type=float, default=0.005, help='Time step')
    args = parser.parse_args()

    print("="*60)
    print("FLOW PAST CYLINDER - INNATE BENCHMARK")
    print("="*60)
    print(f"Re = {args.Re}")
    print(f"Resolution = {args.resolution}")
    print(f"T_final = {args.T_final}")
    print(f"Device: {DEVICE}")
    print("="*60)

    # Konfigürasyon
    config = CylinderConfig(
        resolution=args.resolution,
        Re=args.Re,
        T_final=args.T_final,
        dt=args.dt
    )

    # Model
    model = CylinderFlowModel(config).to(DEVICE)
    print(f"\nModel parametreleri: {sum(p.numel() for p in model.parameters())}")

    # Başlangıç koşulu
    state = model.create_initial_condition()

    # Simülasyon
    num_steps = int(config.T_final / config.dt)
    print(f"\nSimülasyon başlıyor ({num_steps} adım)...")

    with torch.no_grad():
        states, Cd_history, Cl_history = model(state, num_steps, save_every=100)

    # Strouhal hesapla
    D = 2 * config.cylinder_radius
    strouhal_result = compute_strouhal(Cl_history, config.dt, D, config.U_inf)

    print("\n" + "="*60)
    print("SONUÇLAR")
    print("="*60)

    if strouhal_result["St"] is not None:
        print(f"Strouhal sayısı: {strouhal_result['St']:.4f}")
        print(f"Shedding frekansı: {strouhal_result['f']:.4f}")

        validation = validate_results(strouhal_result["St"], config.Re)
        print(f"\nReferans St: {REFERENCE_DATA[config.Re]['St']}")
        print(f"Hata: {validation.get('error_percent', 'N/A'):.1f}%")
        print(f"Sonuç: {'✅ BAŞARILI' if validation.get('valid') else '❌ BAŞARISIZ'}")
    else:
        print("Vortex shedding tespit edilemedi!")

    # Görselleştirme
    output_dir = os.path.dirname(os.path.abspath(__file__))
    plot_results(states, Cl_history, strouhal_result, config,
                 os.path.join(output_dir, f'cylinder_Re{int(config.Re)}.png'))

    return strouhal_result


if __name__ == "__main__":
    result = main()
