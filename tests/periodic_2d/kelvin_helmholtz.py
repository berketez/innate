"""
Kelvin-Helmholtz Instability (Double Shear Layer) Benchmark

Klasik hidrodinamik instabilite testi.
Shear layer'da küçük pertürbasyon → büyük vortex yapıları.

Fizik:
    - İki paralel akış, zıt yönlerde
    - Shear layer boyunca velocity gradient
    - Küçük pertürbasyon → instability
    - Vortex roll-up ve pairing

Kullanım:
    python -m tests.periodic_2d.kelvin_helmholtz --resolution 64 --Re 10000
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


class KelvinHelmholtzBenchmark:
    """Kelvin-Helmholtz Instability Benchmark."""
    
    def __init__(
        self,
        resolution: int = 64,
        Re: float = 10000,
        delta: float = 0.05,  # Shear layer thickness
        U0: float = 1.0,  # Mean flow velocity
        perturbation_amp: float = 0.01,
        t_final: float = 8.0,
        dns_resolution: int = 256,
        results_dir: str = 'results/kelvin_helmholtz',
    ):
        self.resolution = resolution
        self.Re = Re
        self.delta = delta
        self.U0 = U0
        self.perturbation_amp = perturbation_amp
        self.t_final = t_final
        self.dns_resolution = dns_resolution
        self.nu = 1.0 / Re
        
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.viz = FlowVisualizer(results_dir=str(self.results_dir))
        self.model = None
        self.dns_result = None
    
    def create_initial_condition(self, N: int) -> tuple:
        """Double shear layer başlangıç koşulu."""
        L = 2 * np.pi
        x = np.linspace(0, L, N, endpoint=False)
        y = np.linspace(0, L, N, endpoint=False)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Shear layer profili (tanh)
        y1, y2 = L/4, 3*L/4  # İki shear layer
        
        u = self.U0 * (
            np.tanh((Y - y1) / self.delta) - 
            np.tanh((Y - y2) / self.delta) - 1
        )
        
        # Pertürbasyon (en tehlikeli mod)
        k_pert = 2  # Wavenumber
        u += self.perturbation_amp * np.sin(k_pert * X) * (
            np.exp(-((Y - y1) / self.delta)**2) +
            np.exp(-((Y - y2) / self.delta)**2)
        )
        
        v = self.perturbation_amp * np.cos(k_pert * X) * (
            np.exp(-((Y - y1) / self.delta)**2) -
            np.exp(-((Y - y2) / self.delta)**2)
        )
        
        # Vortisite
        k = np.fft.fftfreq(N, d=1/N) * 2 * np.pi / L
        kx, ky = np.meshgrid(k, k, indexing='ij')
        
        omega = np.real(np.fft.ifft2(
            1j * kx * np.fft.fft2(v) - 1j * ky * np.fft.fft2(u)
        ))
        
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
    
    def compute_mixing_layer_thickness(self, omega: np.ndarray) -> float:
        """Mixing layer kalınlığını hesapla."""
        # Momentum thickness veya vorticity thickness
        N = omega.shape[0]
        y = np.linspace(0, 2*np.pi, N)
        
        # y-ortalama
        omega_mean = np.mean(np.abs(omega), axis=0)
        
        # Threshold-based thickness
        threshold = 0.1 * np.max(omega_mean)
        active = omega_mean > threshold
        
        if np.any(active):
            return np.sum(active) * (2*np.pi / N)
        return 0.0
    
    def run_full_benchmark(self, num_epochs: int = 500, train: bool = True) -> dict:
        """Tam benchmark."""
        print("=" * 60)
        print("KELVIN-HELMHOLTZ INSTABILITY BENCHMARK")
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
        
        omega_pred = [s.vorticity.cpu().numpy().squeeze() for s in states]
        times = np.linspace(0, self.t_final, len(omega_pred))
        
        # Animasyon
        self.viz.create_animation(omega_pred, times=times, save='kh_animation.gif')
        
        # Mixing layer kalınlığı
        thicknesses = [self.compute_mixing_layer_thickness(o) for o in omega_pred]
        
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(times, thicknesses)
        ax.set_xlabel('Time')
        ax.set_ylabel('Mixing Layer Thickness')
        ax.set_title('K-H Instability: Mixing Layer Growth')
        fig.savefig(self.results_dir / 'mixing_growth.png', dpi=150)
        plt.close()
        
        return {'thicknesses': thicknesses}


def main():
    parser = argparse.ArgumentParser(description='Kelvin-Helmholtz Benchmark')
    parser.add_argument('--resolution', type=int, default=64)
    parser.add_argument('--Re', type=float, default=10000)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--no-train', action='store_true')
    
    args = parser.parse_args()
    setup_device_optimizations()
    
    benchmark = KelvinHelmholtzBenchmark(resolution=args.resolution, Re=args.Re)
    benchmark.run_full_benchmark(num_epochs=args.epochs, train=not args.no_train)


if __name__ == "__main__":
    main()

