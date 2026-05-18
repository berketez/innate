"""
Vortex Dipole (Lamb-Chaplygin) Benchmark Testi

Zıt işaretli iki vortex'in birlikte hareket ettiği klasik test case.
DNS referans verisi ile karşılaştırma gerektirir.

Fizik:
    - İki zıt işaretli vortex
    - Net momentum = 0, net sirkülasyon = 0
    - Dipol düz çizgide yol alır
    - Self-induced velocity ile hareket

Kullanım:
    python -m tests.periodic_2d.vortex_dipole --resolution 64 --Re 1000
"""

import sys
import argparse
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from innate import INNATE, FluidState, DEVICE
from tests.common.device import get_device, print_device_info, setup_device_optimizations
from tests.common.trainer_base import BenchmarkTrainer, TrainerConfig, ValidationMetrics
from tests.common.visualizer import FlowVisualizer
from tests.common.dns_solver import PseudoSpectralDNS


class VortexDipoleBenchmark:
    """
    Vortex Dipole Benchmark.
    
    Lamb-Chaplygin tipi vortex dipole.
    """
    
    def __init__(
        self,
        resolution: int = 64,
        Re: float = 1000,
        separation: float = 1.0,
        Gamma: float = 2 * np.pi,
        r_c: float = 0.3,
        t_final: float = 5.0,
        dns_resolution: int = 256,
        results_dir: str = 'results/vortex_dipole',
    ):
        self.resolution = resolution
        self.Re = Re
        self.separation = separation
        self.Gamma = Gamma
        self.r_c = r_c
        self.t_final = t_final
        self.dns_resolution = dns_resolution
        self.nu = 1.0 / Re
        
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.viz = FlowVisualizer(results_dir=str(self.results_dir))
        self.model = None
        self.dns_result = None
    
    def create_initial_condition(self, N: int) -> tuple:
        """
        Dipole başlangıç koşulu oluştur.
        
        İki Gaussian vortex, zıt işaretli.
        """
        L = 2 * np.pi
        x = np.linspace(0, L, N, endpoint=False)
        y = np.linspace(0, L, N, endpoint=False)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Vortex merkezleri
        x0, y0 = np.pi, np.pi
        x1 = x0 - self.separation / 2
        x2 = x0 + self.separation / 2
        
        # İki Gaussian vortex
        r1_sq = (X - x1)**2 + (Y - y0)**2
        r2_sq = (X - x2)**2 + (Y - y0)**2
        
        omega1 = (self.Gamma / (np.pi * self.r_c**2)) * np.exp(-r1_sq / self.r_c**2)
        omega2 = -(self.Gamma / (np.pi * self.r_c**2)) * np.exp(-r2_sq / self.r_c**2)
        
        omega = omega1 + omega2
        
        # Hız stream function'dan
        # ∇²ψ = -ω, u = ∂ψ/∂y, v = -∂ψ/∂x
        k = np.fft.fftfreq(N, d=1/N) * 2 * np.pi / L
        kx, ky = np.meshgrid(k, k, indexing='ij')
        k_sq = kx**2 + ky**2
        k_sq[0, 0] = 1.0
        
        omega_hat = np.fft.fft2(omega)
        psi_hat = omega_hat / k_sq
        psi_hat[0, 0] = 0
        
        u = np.real(np.fft.ifft2(1j * ky * psi_hat))
        v = np.real(np.fft.ifft2(-1j * kx * psi_hat))
        
        return u, v, omega
    
    def generate_dns_reference(self) -> dict:
        """DNS referans verisi üret."""
        print("DNS referans üretiliyor...")
        
        dns = PseudoSpectralDNS(self.dns_resolution, self.Re)
        u0, v0, _ = self.create_initial_condition(self.dns_resolution)
        
        result = dns.solve(u0, v0, t_final=self.t_final, save_every=50)
        
        # INNATE çözünürlüğüne downsample
        self.dns_result = {
            'u': dns.downsample(result.u, self.resolution),
            'v': dns.downsample(result.v, self.resolution),
            't': result.t,
            'energy': result.energy,
            'enstrophy': result.enstrophy,
        }
        
        print(f"DNS tamamlandı: {len(result.t)} snapshot")
        return self.dns_result
    
    def get_initial_state(self, batch_size: int = 1, device=None) -> FluidState:
        """Başlangıç durumu."""
        if device is None:
            device = DEVICE
        
        u, v, omega = self.create_initial_condition(self.resolution)
        
        return FluidState(
            u=torch.tensor(u, dtype=torch.float32, device=device).unsqueeze(0),
            v=torch.tensor(v, dtype=torch.float32, device=device).unsqueeze(0),
            p=torch.zeros(1, self.resolution, self.resolution, device=device),
            vorticity=torch.tensor(omega, dtype=torch.float32, device=device).unsqueeze(0),
            t=torch.tensor(0.0, device=device)
        )
    
    def create_model(self) -> INNATE:
        """Model oluştur."""
        self.model = INNATE(
            resolution=self.resolution,
            re_range=(self.Re * 0.5, self.Re * 2),
            bc_type='periodic'
        ).to(DEVICE)
        
        with torch.no_grad():
            self.model.reynolds_learner.reynolds.fill_(self.Re)
        
        return self.model
    
    def run_full_benchmark(
        self,
        num_epochs: int = 500,
        num_steps: int = 200,
        train: bool = True,
    ) -> dict:
        """Tam benchmark."""
        print("=" * 60)
        print("VORTEX DIPOLE BENCHMARK")
        print("=" * 60)
        print_device_info()
        
        # DNS referans
        if self.dns_result is None:
            self.generate_dns_reference()
        
        # Model
        self.create_model()
        
        if train:
            config = TrainerConfig(
                num_epochs=num_epochs,
                results_dir=str(self.results_dir),
            )
            
            trainer = BenchmarkTrainer(
                model=self.model,
                dns_reference=self.dns_result,
                config=config,
                initial_condition_fn=lambda b, d: self.get_initial_state(b, d),
            )
            
            trainer.train()
        
        # Simülasyon ve validasyon
        self.model.eval()
        with torch.no_grad():
            initial = self.get_initial_state()
            states = self.model(initial, num_steps=num_steps)
        
        # Görselleştirme
        omega_pred = [s.vorticity.cpu().numpy().squeeze() for s in states]
        times = np.linspace(0, self.t_final, len(omega_pred))
        
        # DNS ile karşılaştırma
        for i in [0, len(omega_pred)//2, -1]:
            t = times[i]
            dns_idx = np.argmin(np.abs(self.dns_result['t'] - t))
            
            # Vortisite hesapla (DNS'ten)
            L = 2 * np.pi
            N = self.resolution
            k = np.fft.fftfreq(N, d=1/N) * 2 * np.pi / L
            kx, ky = np.meshgrid(k, k, indexing='ij')
            
            u_dns = self.dns_result['u'][dns_idx]
            v_dns = self.dns_result['v'][dns_idx]
            
            omega_dns = np.real(np.fft.ifft2(
                1j * kx * np.fft.fft2(v_dns) - 1j * ky * np.fft.fft2(u_dns)
            ))
            
            self.viz.plot_comparison(
                omega_pred[i], omega_dns, t=t,
                reference_label='DNS',
                save=f'comparison_t{t:.2f}.png'
            )
        
        # Animasyon
        self.viz.create_animation(
            omega_pred, times=times,
            title='Vortex Dipole',
            save='vorticity_animation.gif'
        )
        
        print(f"Sonuçlar: {self.results_dir}")
        return {'states': states}


def main():
    parser = argparse.ArgumentParser(description='Vortex Dipole Benchmark')
    parser.add_argument('--resolution', type=int, default=64)
    parser.add_argument('--Re', type=float, default=1000)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--no-train', action='store_true')
    
    args = parser.parse_args()
    setup_device_optimizations()
    
    benchmark = VortexDipoleBenchmark(resolution=args.resolution, Re=args.Re)
    benchmark.run_full_benchmark(num_epochs=args.epochs, train=not args.no_train)


if __name__ == "__main__":
    main()

