"""
Vortex Merger Benchmark Testi

Aynı işaretli iki vortex'in birleşme dinamikleri.
Klasik türbülans test case'i.

Fizik:
    - İki aynı işaretli (co-rotating) vortex
    - Başlangıçta ayrı, zamanla birleşir
    - Critical distance: birleşme için minimum mesafe
    - Merger time scales with Re

Kullanım:
    python -m tests.periodic_2d.vortex_merger --resolution 64 --Re 1000
"""

import sys
import argparse
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from innate import INNATE, FluidState, DEVICE
from tests.common.device import print_device_info, setup_device_optimizations
from tests.common.trainer_base import BenchmarkTrainer, TrainerConfig
from tests.common.visualizer import FlowVisualizer
from tests.common.dns_solver import PseudoSpectralDNS


class VortexMergerBenchmark:
    """Vortex Merger Benchmark."""
    
    def __init__(
        self,
        resolution: int = 64,
        Re: float = 1000,
        separation: float = 1.5,
        Gamma: float = 2 * np.pi,
        r_c: float = 0.3,
        t_final: float = 10.0,
        dns_resolution: int = 256,
        results_dir: str = 'results/vortex_merger',
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
        """Co-rotating vortex pair başlangıç koşulu."""
        L = 2 * np.pi
        x = np.linspace(0, L, N, endpoint=False)
        y = np.linspace(0, L, N, endpoint=False)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        x0, y0 = np.pi, np.pi
        x1 = x0 - self.separation / 2
        x2 = x0 + self.separation / 2
        
        # İki aynı işaretli vortex
        r1_sq = (X - x1)**2 + (Y - y0)**2
        r2_sq = (X - x2)**2 + (Y - y0)**2
        
        omega1 = (self.Gamma / (np.pi * self.r_c**2)) * np.exp(-r1_sq / self.r_c**2)
        omega2 = (self.Gamma / (np.pi * self.r_c**2)) * np.exp(-r2_sq / self.r_c**2)
        
        omega = omega1 + omega2
        
        # Hız
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
        """DNS referans."""
        print("DNS referans üretiliyor...")
        
        dns = PseudoSpectralDNS(self.dns_resolution, self.Re)
        u0, v0, _ = self.create_initial_condition(self.dns_resolution)
        
        result = dns.solve(u0, v0, t_final=self.t_final, save_every=50)
        
        self.dns_result = {
            'u': dns.downsample(result.u, self.resolution),
            'v': dns.downsample(result.v, self.resolution),
            't': result.t,
            'energy': result.energy,
            'enstrophy': result.enstrophy,
        }
        
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
    
    def detect_merger_time(self, omega_sequence: list) -> float:
        """Birleşme zamanını tespit et."""
        # Basit yaklaşım: max vorticity'nin pozisyonunu takip et
        # Birleşme olduğunda tek peak kalır
        
        for i, omega in enumerate(omega_sequence):
            omega_np = omega.squeeze() if hasattr(omega, 'squeeze') else omega
            
            # Peak sayısı (local maxima)
            from scipy.ndimage import maximum_filter
            local_max = maximum_filter(omega_np, size=5) == omega_np
            peaks = np.sum(local_max & (omega_np > 0.5 * np.max(omega_np)))
            
            if peaks <= 1:
                return i * self.t_final / len(omega_sequence)
        
        return self.t_final
    
    def run_full_benchmark(self, num_epochs: int = 500, train: bool = True) -> dict:
        """Tam benchmark."""
        print("=" * 60)
        print("VORTEX MERGER BENCHMARK")
        print("=" * 60)
        print_device_info()
        
        if self.dns_result is None:
            self.generate_dns_reference()
        
        self.create_model()
        
        if train:
            config = TrainerConfig(num_epochs=num_epochs, results_dir=str(self.results_dir))
            trainer = BenchmarkTrainer(
                model=self.model,
                dns_reference=self.dns_result,
                config=config,
                initial_condition_fn=lambda b, d: self.get_initial_state(b, d),
            )
            trainer.train()
        
        # Simülasyon
        self.model.eval()
        with torch.no_grad():
            states = self.model(self.get_initial_state(), num_steps=200)
        
        # Görselleştirme
        omega_pred = [s.vorticity.cpu().numpy().squeeze() for s in states]
        times = np.linspace(0, self.t_final, len(omega_pred))
        
        self.viz.create_animation(omega_pred, times=times, save='merger_animation.gif')
        
        # Birleşme zamanı
        merger_time = self.detect_merger_time(omega_pred)
        print(f"Tespit edilen birleşme zamanı: {merger_time:.2f}")
        
        return {'merger_time': merger_time}


def main():
    parser = argparse.ArgumentParser(description='Vortex Merger Benchmark')
    parser.add_argument('--resolution', type=int, default=64)
    parser.add_argument('--Re', type=float, default=1000)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--no-train', action='store_true')
    
    args = parser.parse_args()
    setup_device_optimizations()
    
    benchmark = VortexMergerBenchmark(resolution=args.resolution, Re=args.Re)
    benchmark.run_full_benchmark(num_epochs=args.epochs, train=not args.no_train)


if __name__ == "__main__":
    main()

