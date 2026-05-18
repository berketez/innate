"""
Lamb-Oseen Vortex Benchmark Testi

Analitik çözümü olan tek vortex decay problemi.
INNATE modelinin doğruluğunu test etmek için ideal başlangıç noktası.

Fizik:
    - Tek aksisimetrik vortex
    - Viskoz difüzyon ile decay
    - Core radius: r_c(t) = sqrt(4νt + r_c0²)
    - Peak vorticity: ω_max(t) ∝ 1/r_c(t)²

Analitik Çözüm:
    ω(r,t) = (Γ / π r_c²) * exp(-r²/r_c²)
    v_θ(r,t) = (Γ / 2πr) * (1 - exp(-r²/r_c²))

Kullanım:
    python -m tests.periodic_2d.lamb_oseen --resolution 64 --Re 1000 --epochs 500
"""

import sys
import os
import argparse
import numpy as np
import torch
import math
from pathlib import Path

# Parent directory'yi path'e ekle
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from innate import INNATE, FluidState, SpectralOps, DEVICE
from tests.common.device import get_device, print_device_info, setup_device_optimizations
from tests.common.trainer_base import BenchmarkTrainer, TrainerConfig, ValidationMetrics
from tests.common.visualizer import FlowVisualizer
import torch.nn as nn


# =============================================================================
# EXTENDED INNATE - Test için ek parametrelerle wrapper
# =============================================================================

class ExtendedINNATE(nn.Module):
    """
    INNATE modelini wrap eden genişletilmiş versiyon.

    Bu sınıf innate.py'yi DEĞİŞTİRMEDEN ek öğrenilebilir parametreler ekler.
    PyTorch modelini wrap etmek gibi - kütüphaneyi değiştirmiyoruz.

    Ek parametreler (~90):
    - spectral_correction: Spektral düzeltme katsayıları (21)
    - diffusion_scales: Multi-scale difüzyon (16)
    - advection_weights: Adveksiyon yön ağırlıkları (16)
    - spatial_bias: Uzamsal bias (2x4x4 = 32)
    - energy_correction: Enerji düzeltme (4)
    - vorticity_scales: Vortisite ölçekleme (4)

    Toplam: ~93 ek parametre + INNATE'in 9 = ~102 parametre
    """

    def __init__(self, resolution: int, re_range: tuple, bc_type: str = 'periodic'):
        super().__init__()

        # Base INNATE model
        self.base_model = INNATE(resolution, re_range, bc_type)
        self.resolution = resolution

        # ===== EK ÖĞRENİLEBİLİR PARAMETRELER =====

        # Spektral düzeltme katsayıları (radial bins)
        num_bins = resolution // 3  # ~21 for res=64
        self.spectral_correction = nn.Parameter(torch.ones(num_bins))

        # Multi-scale difüzyon katsayıları
        self.diffusion_scales = nn.Parameter(torch.ones(16))

        # Adveksiyon yön ağırlıkları [du_dx, du_dy, dv_dx, dv_dy] x 4 scales
        self.advection_weights = nn.Parameter(torch.ones(16))

        # Uzamsal bias (low-res grid interpolated to full res)
        self.spatial_bias_u = nn.Parameter(torch.zeros(1, 4, 4))
        self.spatial_bias_v = nn.Parameter(torch.zeros(1, 4, 4))

        # Enerji düzeltme katsayıları
        self.energy_correction = nn.Parameter(torch.ones(4))

        # Vortisite ölçekleme
        self.vorticity_scales = nn.Parameter(torch.ones(4))

        # Skew-symmetric ağırlık
        self.skew_weight = nn.Parameter(torch.tensor(0.5))

        # Interpolation için grid
        self._setup_interpolation_grid()

    def _setup_interpolation_grid(self):
        """Spatial bias interpolation için grid."""
        # 4x4'ten resolution x resolution'a interpolate edeceğiz
        pass  # torch.nn.functional.interpolate runtime'da kullanılacak

    def _apply_spectral_correction(self, field: torch.Tensor) -> torch.Tensor:
        """Spektral düzeltme uygula - radial bins ile."""
        from torch.fft import fft2, ifft2, fftfreq

        # Coefficients'ı [0.8, 1.2] aralığına kısıtla (daha dar)
        coeffs = torch.sigmoid(self.spectral_correction) * 0.4 + 0.8

        # FFT
        f_hat = fft2(field)

        # Radial distance hesapla
        n = self.resolution
        kx = torch.fft.fftfreq(n, device=field.device)
        ky = torch.fft.fftfreq(n, device=field.device)
        KX, KY = torch.meshgrid(kx, ky, indexing='ij')
        k_mag = torch.sqrt(KX**2 + KY**2)

        # Radial bins (0 to 0.5 arası, num_bins parçaya böl)
        num_bins = len(coeffs)
        bin_edges = torch.linspace(0, 0.5, num_bins + 1, device=field.device)

        # Her frekans için uygun coefficient'ı bul
        correction_map = torch.ones_like(k_mag)
        for i in range(num_bins):
            mask = (k_mag >= bin_edges[i]) & (k_mag < bin_edges[i + 1])
            correction_map = torch.where(mask, coeffs[i], correction_map)

        # Uygula
        corrected = f_hat * correction_map

        return ifft2(corrected).real

    def _apply_spatial_bias(self, u: torch.Tensor, v: torch.Tensor):
        """Spatial bias ekle (4x4'ten interpolate)."""
        import torch.nn.functional as F

        # 4x4'ten resolution x resolution'a interpolate
        bias_u = F.interpolate(
            self.spatial_bias_u.unsqueeze(0),
            size=(self.resolution, self.resolution),
            mode='bilinear',
            align_corners=True
        ).squeeze(0)

        bias_v = F.interpolate(
            self.spatial_bias_v.unsqueeze(0),
            size=(self.resolution, self.resolution),
            mode='bilinear',
            align_corners=True
        ).squeeze(0)

        # Bias'ı küçük tut
        scale = 0.01
        return u + scale * bias_u, v + scale * bias_v

    def _apply_energy_correction(self, u: torch.Tensor, v: torch.Tensor):
        """Enerji koruması için düzeltme."""
        # Mevcut enerji
        energy = 0.5 * (u**2 + v**2).mean()

        # Düzeltme faktörü (1'e yakın olmalı)
        correction = torch.sigmoid(self.energy_correction.mean()) * 0.2 + 0.9

        return u * correction, v * correction

    def step(self, state: FluidState, hidden=None, observed_data=None):
        """
        Genişletilmiş step - base model + ADDITIVE düzeltmeler.

        ÖNEMLİ: Multiplicative scaling yerine additive correction kullanıyoruz
        çünkü çarpımlar step'ler boyunca birikip explode ediyor.
        """
        # ============================================================
        # 1. BASE MODEL STEP (değiştirilmeden)
        # ============================================================
        new_state, new_hidden = self.base_model.step(state, hidden, observed_data)

        # ============================================================
        # 2. ADDITIVE CORRECTIONS (küçük, birikmeyen)
        # ============================================================

        # Spatial bias - çok küçük additive correction
        import torch.nn.functional as F
        bias_u = F.interpolate(
            self.spatial_bias_u.unsqueeze(0),
            size=(self.resolution, self.resolution),
            mode='bilinear', align_corners=True
        ).squeeze(0).squeeze(0)

        bias_v = F.interpolate(
            self.spatial_bias_v.unsqueeze(0),
            size=(self.resolution, self.resolution),
            mode='bilinear', align_corners=True
        ).squeeze(0).squeeze(0)

        # Çok küçük scale (0.001) - birikse bile sorun olmaz
        new_state.u = new_state.u + 0.001 * torch.tanh(bias_u)
        new_state.v = new_state.v + 0.001 * torch.tanh(bias_v)

        # ============================================================
        # 3. ADAPTIVE DAMPING (enerji patlamasını önle)
        # ============================================================
        # Mevcut enerjiyi hesapla
        energy = 0.5 * (new_state.u**2 + new_state.v**2).mean()

        # Hedef enerji (başlangıçtan azalmalı - fiziksel)
        target_energy = 0.5 * (state.u**2 + state.v**2).mean()

        # Enerji çok artarsa damping uygula
        if energy > target_energy * 1.1:  # %10'dan fazla artış
            damping = torch.sqrt(target_energy / (energy + 1e-8))
            damping = torch.clamp(damping, 0.9, 1.0)  # En fazla %10 damping
            new_state.u = new_state.u * damping
            new_state.v = new_state.v * damping

        return new_state, new_hidden

    def forward(self, initial_state: FluidState, num_steps: int, observed_data=None):
        """Forward pass - base model ile aynı interface."""
        states = [initial_state]
        state = initial_state
        hidden = None

        for step_idx in range(num_steps):
            step_data = None
            if observed_data is not None and step_idx in observed_data:
                step_data = observed_data[step_idx]

            state, hidden = self.step(state, hidden, step_data)
            states.append(state)

        return states

    def physics_loss(self, state: FluidState):
        """Base model physics loss + ek regularizasyon."""
        losses = self.base_model.physics_loss(state)

        # Ek regularizasyon: parametrelerin 1'e yakın kalması
        losses['param_reg'] = (
            (self.spectral_correction - 1).pow(2).mean() * 0.01 +
            (self.diffusion_scales - 1).pow(2).mean() * 0.01 +
            (self.advection_weights - 1).pow(2).mean() * 0.01 +
            self.spatial_bias_u.pow(2).mean() * 0.1 +
            self.spatial_bias_v.pow(2).mean() * 0.1
        )

        return losses

    # Delegate other methods to base model
    def __getattr__(self, name):
        if name in ['base_model', 'resolution', 'spectral_correction',
                    'diffusion_scales', 'advection_weights', 'spatial_bias_u',
                    'spatial_bias_v', 'energy_correction', 'vorticity_scales',
                    'skew_weight', '_setup_interpolation_grid',
                    '_apply_spectral_correction', '_apply_spatial_bias',
                    '_apply_energy_correction']:
            return super().__getattr__(name)
        return getattr(self.base_model, name)


# =============================================================================
# LAMB-OSEEN VORTEX ANALİTİK ÇÖZÜMÜ
# =============================================================================

class LambOseenAnalytical:
    """
    Lamb-Oseen vortex analitik çözümü.
    
    Args:
        Gamma: Sirkülasyon (varsayılan 2π)
        nu: Kinematik viskozite
        r_c0: Başlangıç core radius (varsayılan 0.5)
        center: Vortex merkezi (x0, y0)
    """
    
    def __init__(
        self,
        Gamma: float = 2 * np.pi,
        nu: float = 0.001,
        r_c0: float = 0.5,
        center: tuple = (np.pi, np.pi),
    ):
        self.Gamma = Gamma
        self.nu = nu
        self.r_c0 = r_c0
        self.x0, self.y0 = center
    
    def core_radius(self, t: float) -> float:
        """Core radius: r_c(t) = sqrt(4νt + r_c0²)"""
        return np.sqrt(4 * self.nu * t + self.r_c0**2)
    
    def vorticity(self, X: np.ndarray, Y: np.ndarray, t: float) -> np.ndarray:
        """
        Vortisite alanı.
        
        ω(r,t) = (Γ / π r_c²) * exp(-r²/r_c²)
        """
        r_c = self.core_radius(t)
        r2 = (X - self.x0)**2 + (Y - self.y0)**2
        
        omega = (self.Gamma / (np.pi * r_c**2)) * np.exp(-r2 / r_c**2)
        
        return omega
    
    def velocity(self, X: np.ndarray, Y: np.ndarray, t: float) -> tuple:
        """
        Hız alanı (Kartezyen).
        
        v_θ(r,t) = (Γ / 2πr) * (1 - exp(-r²/r_c²))
        u = -v_θ * sin(θ), v = v_θ * cos(θ)
        """
        r_c = self.core_radius(t)
        dx = X - self.x0
        dy = Y - self.y0
        r2 = dx**2 + dy**2
        r = np.sqrt(r2 + 1e-10)  # Singularite önle
        
        # Azimuthal velocity
        v_theta = (self.Gamma / (2 * np.pi * r)) * (1 - np.exp(-r2 / r_c**2))
        
        # Kartezyen
        theta = np.arctan2(dy, dx)
        u = -v_theta * np.sin(theta)
        v = v_theta * np.cos(theta)
        
        return u, v
    
    def energy(self, t: float) -> float:
        """
        Toplam kinetik enerji (yaklaşık).
        
        E(t) ∝ Γ² / r_c²
        """
        r_c = self.core_radius(t)
        # Normalize edilmiş enerji
        return 0.5 * (self.Gamma**2) / (4 * np.pi * r_c**2)
    
    def enstrophy(self, t: float) -> float:
        """
        Toplam enstrofi (yaklaşık).
        
        Ω(t) ∝ Γ² / r_c⁴
        """
        r_c = self.core_radius(t)
        return 0.5 * (self.Gamma**2) / (np.pi**2 * r_c**4)


# =============================================================================
# BAŞLANGIÇ KOŞULU OLUŞTURUCU
# =============================================================================

def create_lamb_oseen_initial_state(
    resolution: int,
    Re: float,
    Gamma: float = 2 * np.pi,
    r_c0: float = 0.5,
    center: tuple = (np.pi, np.pi),
    device: torch.device = None,
) -> FluidState:
    """
    Lamb-Oseen vortex başlangıç durumu oluştur.
    
    Args:
        resolution: Grid çözünürlüğü
        Re: Reynolds sayısı
        Gamma: Sirkülasyon
        r_c0: Başlangıç core radius
        center: Vortex merkezi
        device: Torch device
    
    Returns:
        FluidState: Başlangıç durumu
    """
    if device is None:
        device = DEVICE
    
    nu = 1.0 / Re
    
    # Grid
    L = 2 * np.pi
    x = np.linspace(0, L, resolution, endpoint=False)
    y = np.linspace(0, L, resolution, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Analitik çözüm t=0
    analytical = LambOseenAnalytical(Gamma=Gamma, nu=nu, r_c0=r_c0, center=center)
    u_np, v_np = analytical.velocity(X, Y, t=0)
    omega_np = analytical.vorticity(X, Y, t=0)
    
    # Torch tensors
    u = torch.tensor(u_np, dtype=torch.float32, device=device).unsqueeze(0)
    v = torch.tensor(v_np, dtype=torch.float32, device=device).unsqueeze(0)
    omega = torch.tensor(omega_np, dtype=torch.float32, device=device).unsqueeze(0)
    p = torch.zeros_like(u)
    
    return FluidState(
        u=u, v=v, p=p,
        vorticity=omega,
        t=torch.tensor(0.0, device=device)
    )


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

class LambOseenBenchmark:
    """
    Lamb-Oseen Vortex Benchmark Runner.
    
    Args:
        resolution: Grid çözünürlüğü
        Re: Reynolds sayısı
        Gamma: Sirkülasyon
        r_c0: Başlangıç core radius
        t_final: Bitiş zamanı
        results_dir: Sonuç klasörü
    """
    
    def __init__(
        self,
        resolution: int = 64,
        Re: float = 1000,
        Gamma: float = 2 * np.pi,
        r_c0: float = 0.5,
        t_final: float = 2.0,
        results_dir: str = 'results/lamb_oseen',
        extended: bool = False,  # Extended model (100 params) kullan
    ):
        self.resolution = resolution
        self.Re = Re
        self.Gamma = Gamma
        self.r_c0 = r_c0
        self.t_final = t_final
        self.nu = 1.0 / Re
        self.extended = extended

        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Analitik çözüm
        self.analytical = LambOseenAnalytical(
            Gamma=Gamma, nu=self.nu, r_c0=r_c0
        )

        # Grid
        L = 2 * np.pi
        x = np.linspace(0, L, resolution, endpoint=False)
        y = np.linspace(0, L, resolution, endpoint=False)
        self.X, self.Y = np.meshgrid(x, y, indexing='ij')

        # Visualizer
        self.viz = FlowVisualizer(results_dir=str(self.results_dir))

        # Model
        self.model = None

    def create_model(self):
        """INNATE veya ExtendedINNATE modeli oluştur."""
        if self.extended:
            # Extended model (~100 params)
            model = ExtendedINNATE(
                resolution=self.resolution,
                re_range=(self.Re * 0.5, self.Re * 2),
                bc_type='periodic'
            ).to(DEVICE)

            # Reynolds sayısını başlangıç değerine ayarla
            with torch.no_grad():
                model.base_model.reynolds_learner.reynolds.fill_(self.Re)

            print(f"ExtendedINNATE model oluşturuldu")
        else:
            # Standard INNATE (~9 params)
            model = INNATE(
                resolution=self.resolution,
                re_range=(self.Re * 0.5, self.Re * 2),
                bc_type='periodic'
            ).to(DEVICE)

            # Reynolds sayısını başlangıç değerine ayarla
            with torch.no_grad():
                model.reynolds_learner.reynolds.fill_(self.Re)

        self.model = model
        return model
    
    def get_initial_state(self, batch_size: int = 1, device=None) -> FluidState:
        """Başlangıç durumu."""
        return create_lamb_oseen_initial_state(
            resolution=self.resolution,
            Re=self.Re,
            Gamma=self.Gamma,
            r_c0=self.r_c0,
            device=device or DEVICE,
        )
    
    def get_analytical_solution(self, t: float) -> tuple:
        """Analitik çözüm."""
        u, v = self.analytical.velocity(self.X, self.Y, t)
        return u, v
    
    def run_simulation(self, num_steps: int = 200) -> dict:
        """
        Simülasyon çalıştır.

        Args:
            num_steps: Maksimum adım sayısı (t_final'e ulaşılmazsa)

        Returns:
            dict: Simülasyon sonuçları
        """
        if self.model is None:
            self.create_model()

        self.model.eval()

        with torch.no_grad():
            initial_state = self.get_initial_state()
            states = [initial_state]
            state = initial_state

            # t_final'e ulaşana veya num_steps tükenene kadar devam et
            hidden = None
            for step in range(num_steps):
                state, hidden = self.model.step(state, hidden)
                states.append(state)

                # t_final kontrolü
                current_t = state.t.item() if hasattr(state.t, 'item') else float(state.t)
                if current_t >= self.t_final:
                    break
        
        # Sonuçları topla
        results = {
            'u': [s.u.cpu().numpy() for s in states],
            'v': [s.v.cpu().numpy() for s in states],
            'vorticity': [s.vorticity.cpu().numpy() for s in states],
            't': [s.t.item() if hasattr(s.t, 'item') else i * 0.01 for i, s in enumerate(states)],
        }
        
        # Enerji ve enstrofi
        results['energy'] = [0.5 * np.mean(u**2 + v**2) 
                            for u, v in zip(results['u'], results['v'])]
        results['enstrophy'] = [0.5 * np.mean(w**2) for w in results['vorticity']]
        
        return results
    
    def compute_errors(self, results: dict) -> dict:
        """
        Analitik çözümle hata hesapla.
        
        Args:
            results: Simülasyon sonuçları
        
        Returns:
            dict: Hata metrikleri
        """
        times = results['t']
        l2_errors = []
        linf_errors = []
        
        for i, t in enumerate(times):
            u_pred = results['u'][i].squeeze()
            v_pred = results['v'][i].squeeze()
            
            u_ref, v_ref = self.analytical.velocity(self.X, self.Y, t)
            
            # L2 error
            l2_u = np.sqrt(np.mean((u_pred - u_ref)**2))
            l2_v = np.sqrt(np.mean((v_pred - v_ref)**2))
            l2_errors.append(np.sqrt(l2_u**2 + l2_v**2))
            
            # L∞ error
            linf_u = np.max(np.abs(u_pred - u_ref))
            linf_v = np.max(np.abs(v_pred - v_ref))
            linf_errors.append(max(linf_u, linf_v))
        
        return {
            'times': times,
            'l2_errors': l2_errors,
            'linf_errors': linf_errors,
            'final_l2': l2_errors[-1],
            'final_linf': linf_errors[-1],
            'max_l2': max(l2_errors),
        }
    
    def validate(self, num_steps: int = 200) -> ValidationMetrics:
        """
        Validasyon metrikleri hesapla.
        
        Args:
            num_steps: Adım sayısı
        
        Returns:
            ValidationMetrics: Sonuçlar
        """
        results = self.run_simulation(num_steps)
        errors = self.compute_errors(results)
        
        # Final state
        final_u = results['u'][-1].squeeze()
        final_v = results['v'][-1].squeeze()
        final_omega = results['vorticity'][-1].squeeze()
        final_t = results['t'][-1]

        # Referans - aynı normalizasyonla hesapla (domain ortalaması)
        ref_u, ref_v = self.analytical.velocity(self.X, self.Y, final_t)
        ref_omega = self.analytical.vorticity(self.X, self.Y, final_t)

        # Enerji ve enstrofi - AYNI FORMÜL ile karşılaştır
        ref_energy = 0.5 * np.mean(ref_u**2 + ref_v**2)
        ref_enstrophy = 0.5 * np.mean(ref_omega**2)

        pred_energy = 0.5 * np.mean(final_u**2 + final_v**2)
        pred_enstrophy = 0.5 * np.mean(final_omega**2)
        
        # Korelasyon
        pred_flat = np.concatenate([final_u.flatten(), final_v.flatten()])
        ref_flat = np.concatenate([ref_u.flatten(), ref_v.flatten()])
        correlation = np.corrcoef(pred_flat, ref_flat)[0, 1]
        
        # Divergence
        if self.model is not None:
            with torch.no_grad():
                final_state = self.get_initial_state()
                states = self.model(final_state, num_steps=num_steps)
                div = self.model.projector.divergence_error(
                    states[-1].u, states[-1].v
                ).item()
        else:
            div = 0.0
        
        return ValidationMetrics(
            l2_error=errors['final_l2'],
            linf_error=errors['final_linf'],
            energy_error=abs(pred_energy - ref_energy) / (ref_energy + 1e-10),
            enstrophy_error=abs(pred_enstrophy - ref_enstrophy) / (ref_enstrophy + 1e-10),
            divergence=div,
            correlation=correlation,
        )
    
    def visualize_results(self, results: dict, errors: dict) -> None:
        """Sonuçları görselleştir."""
        times = results['t']
        
        # Snapshot times
        snapshot_indices = [0, len(times)//4, len(times)//2, -1]
        
        for idx in snapshot_indices:
            t = times[idx]
            
            # INNATE
            omega_pred = results['vorticity'][idx].squeeze()
            
            # Analitik
            omega_ref = self.analytical.vorticity(self.X, self.Y, t)
            
            # Karşılaştırma
            self.viz.plot_comparison(
                omega_pred, omega_ref, t=t,
                innate_label='INNATE',
                reference_label='Analytical',
                save=f'comparison_t{t:.2f}.png'
            )
        
        # Enerji decay
        ref_energy = [self.analytical.energy(t) for t in times]
        self.viz.plot_energy_decay(
            np.array(times),
            np.array(results['energy']),
            reference_energies=np.array(ref_energy),
            title='Lamb-Oseen Vortex: Enerji Sönümü',
            save='energy_decay.png'
        )
        
        # Error evolution
        self.viz.plot_error_evolution(
            np.array(times),
            np.array(errors['l2_errors']),
            title='Lamb-Oseen Vortex: L2 Hata Evrimi',
            save='error_evolution.png'
        )
        
        # Animasyon
        ref_vorticity = [self.analytical.vorticity(self.X, self.Y, t) for t in times]
        self.viz.create_animation(
            results['vorticity'],
            times=np.array(times),
            reference_sequence=ref_vorticity,
            title='Lamb-Oseen Vortex',
            save='vorticity_animation.gif',
            fps=15
        )
    
    def run_full_benchmark(
        self,
        num_epochs: int = 500,
        num_steps: int = 200,
        train: bool = True,
    ) -> dict:
        """
        Tam benchmark çalıştır.
        
        Args:
            num_epochs: Eğitim epoch sayısı
            num_steps: Simülasyon adım sayısı
            train: Eğitim yap (False ise sadece validasyon)
        
        Returns:
            dict: Tüm sonuçlar
        """
        print("=" * 60)
        print("LAMB-OSEEN VORTEX BENCHMARK")
        print("=" * 60)
        print_device_info()
        print(f"\nParametreler:")
        print(f"  Resolution: {self.resolution}")
        print(f"  Reynolds:   {self.Re}")
        print(f"  Γ (Gamma):  {self.Gamma:.4f}")
        print(f"  r_c0:       {self.r_c0}")
        print(f"  t_final:    {self.t_final}")
        print()
        
        # Model
        self.create_model()
        print(f"Model parametreleri: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Eğitim
        if train:
            config = TrainerConfig(
                num_epochs=num_epochs,
                learning_rate=1e-3,
                curriculum_stages=[10, 25, 50, 100],
                save_every=100,
                results_dir=str(self.results_dir),
            )
            
            trainer = BenchmarkTrainer(
                model=self.model,
                config=config,
                initial_condition_fn=lambda b, d: self.get_initial_state(b, d),
                analytical_solution_fn=lambda t: self.get_analytical_solution(t),
            )
            
            print("\n--- Eğitim ---")
            history = trainer.train()
            
            # Eğitim grafiği
            self.viz.plot_training_history(
                history.__dict__,
                save='training_history.png'
            )
        
        # Simülasyon
        print("\n--- Simülasyon ---")
        results = self.run_simulation(num_steps)
        errors = self.compute_errors(results)
        
        print(f"Final L2 Error:  {errors['final_l2']:.4e}")
        print(f"Final L∞ Error:  {errors['final_linf']:.4e}")
        print(f"Max L2 Error:    {errors['max_l2']:.4e}")
        
        # Validasyon
        print("\n--- Validasyon ---")
        metrics = self.validate(num_steps)
        print(metrics)
        
        # Görselleştirme
        print("\n--- Görselleştirme ---")
        self.visualize_results(results, errors)
        
        print(f"\nSonuçlar kaydedildi: {self.results_dir}")
        
        return {
            'results': results,
            'errors': errors,
            'metrics': metrics,
        }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Lamb-Oseen Vortex Benchmark')
    parser.add_argument('--resolution', type=int, default=64, help='Grid çözünürlüğü')
    parser.add_argument('--Re', type=float, default=1000, help='Reynolds sayısı')
    parser.add_argument('--Gamma', type=float, default=2*np.pi, help='Sirkülasyon')
    parser.add_argument('--r_c0', type=float, default=0.5, help='Başlangıç core radius')
    parser.add_argument('--t_final', type=float, default=2.0, help='Bitiş zamanı')
    parser.add_argument('--epochs', type=int, default=500, help='Eğitim epoch sayısı')
    parser.add_argument('--steps', type=int, default=200, help='Simülasyon adım sayısı')
    parser.add_argument('--no-train', action='store_true', help='Eğitim atla')
    parser.add_argument('--results-dir', type=str, default='results/lamb_oseen', help='Sonuç klasörü')
    parser.add_argument('--extended', action='store_true', help='ExtendedINNATE kullan (~100 parametre)')

    args = parser.parse_args()

    # Device optimizasyonları
    setup_device_optimizations()

    # Benchmark
    benchmark = LambOseenBenchmark(
        resolution=args.resolution,
        Re=args.Re,
        Gamma=args.Gamma,
        r_c0=args.r_c0,
        t_final=args.t_final,
        results_dir=args.results_dir,
        extended=args.extended,
    )
    
    benchmark.run_full_benchmark(
        num_epochs=args.epochs,
        num_steps=args.steps,
        train=not args.no_train,
    )


if __name__ == "__main__":
    main()

