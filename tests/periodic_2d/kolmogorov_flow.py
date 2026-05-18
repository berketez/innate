"""
Kolmogorov Flow Benchmark

Sinusoidal forcing ile sürdürülen 2D akış.
Re arttıkça instability → turbulence transition.

Fizik:
    - External forcing: f_x = A * sin(k_f * y)
    - Steady-state çözüm: u = (A / ν k_f²) * sin(k_f * y)
    - Re_c ≈ √2 (kritik Reynolds)
    - Re > Re_c → instability, 2D turbulence

Kullanım:
    python -m tests.periodic_2d.kolmogorov_flow --resolution 64 --Re 40
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


class KolmogorovFlowBenchmark:
    """Kolmogorov Flow Benchmark."""
    
    def __init__(
        self,
        resolution: int = 64,
        Re: float = 40,  # Reynolds number
        k_f: int = 4,  # Forcing wavenumber
        A: float = 1.0,  # Forcing amplitude
        perturbation_amp: float = 0.1,
        t_final: float = 50.0,
        dns_resolution: int = 256,
        results_dir: str = 'results/kolmogorov_flow',
    ):
        self.resolution = resolution
        self.Re = Re
        self.k_f = k_f
        self.A = A
        self.perturbation_amp = perturbation_amp
        self.t_final = t_final
        self.dns_resolution = dns_resolution
        self.nu = 1.0 / Re
        
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.viz = FlowVisualizer(results_dir=str(self.results_dir))
        self.model = None
        self.dns_result = None
    
    def analytical_steady_state(self, Y: np.ndarray) -> np.ndarray:
        """
        Laminar steady-state çözümü.
        
        u = (A / ν k_f²) * sin(k_f * y)
        """
        return (self.A / (self.nu * self.k_f**2)) * np.sin(self.k_f * Y)
    
    def create_initial_condition(self, N: int) -> tuple:
        """
        Perturbed laminar state başlangıç koşulu.
        """
        L = 2 * np.pi
        x = np.linspace(0, L, N, endpoint=False)
        y = np.linspace(0, L, N, endpoint=False)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Laminar base flow
        u = self.analytical_steady_state(Y)
        v = np.zeros_like(u)
        
        # Pertürbasyon (en tehlikeli mod)
        k_pert = self.k_f
        u += self.perturbation_amp * np.cos(k_pert * X) * np.cos(self.k_f * Y)
        v += self.perturbation_amp * np.sin(k_pert * X) * np.sin(self.k_f * Y)
        
        # Vortisite
        k = np.fft.fftfreq(N, d=1/N) * 2 * np.pi / L
        kx, ky = np.meshgrid(k, k, indexing='ij')
        
        omega = np.real(np.fft.ifft2(
            1j * kx * np.fft.fft2(v) - 1j * ky * np.fft.fft2(u)
        ))
        
        return u, v, omega
    
    def get_forcing(self, N: int, device=None) -> tuple:
        """Kolmogorov forcing terimi."""
        if device is None:
            device = DEVICE
        
        L = 2 * np.pi
        y = np.linspace(0, L, N, endpoint=False)
        Y = np.tile(y, (N, 1))
        
        f_x = self.A * np.sin(self.k_f * Y)
        f_y = np.zeros_like(f_x)
        
        return (
            torch.tensor(f_x, dtype=torch.float32, device=device),
            torch.tensor(f_y, dtype=torch.float32, device=device)
        )
    
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
    
    def check_instability(self, omega_sequence: list) -> dict:
        """
        Instability kontrolü.
        
        Laminar'dan sapma ölç.
        """
        rms_values = []
        for omega in omega_sequence:
            omega_np = omega.squeeze() if hasattr(omega, 'squeeze') else omega
            rms = np.sqrt(np.mean(omega_np**2))
            rms_values.append(rms)
        
        # Growth rate
        if len(rms_values) > 10:
            early_rms = np.mean(rms_values[:10])
            late_rms = np.mean(rms_values[-10:])
            
            is_unstable = late_rms > 2 * early_rms
            growth_factor = late_rms / (early_rms + 1e-10)
        else:
            is_unstable = False
            growth_factor = 1.0
        
        return {
            'is_unstable': is_unstable,
            'growth_factor': growth_factor,
            'rms_values': rms_values,
        }
    
    def run_full_benchmark(self, num_epochs: int = 500, train: bool = True) -> dict:
        """Tam benchmark."""
        print("=" * 60)
        print("KOLMOGOROV FLOW BENCHMARK")
        print("=" * 60)
        print_device_info()
        print(f"Re = {self.Re}, k_f = {self.k_f}")
        print(f"Critical Re ≈ {np.sqrt(2) * self.k_f:.2f}")
        
        self.create_model()
        
        # NOTE: Kolmogorov flow için forcing gerekli
        # Şimdilik sadece decay simüle ediyoruz
        
        if train:
            config = TrainerConfig(num_epochs=num_epochs, results_dir=str(self.results_dir))
            trainer = BenchmarkTrainer(
                model=self.model,
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
        self.viz.create_animation(omega_pred, times=times, save='kolmogorov_animation.gif')
        
        # Instability analizi
        instability = self.check_instability(omega_pred)
        print(f"Unstable: {instability['is_unstable']}")
        print(f"Growth factor: {instability['growth_factor']:.2f}")
        
        # RMS vorticity plot
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(times, instability['rms_values'])
        ax.set_xlabel('Time')
        ax.set_ylabel('RMS Vorticity')
        ax.set_title(f'Kolmogorov Flow Instability (Re={self.Re})')
        fig.savefig(self.results_dir / 'rms_vorticity.png', dpi=150)
        plt.close()
        
        return instability


def main():
    parser = argparse.ArgumentParser(description='Kolmogorov Flow Benchmark')
    parser.add_argument('--resolution', type=int, default=64)
    parser.add_argument('--Re', type=float, default=40)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--no-train', action='store_true')
    
    args = parser.parse_args()
    setup_device_optimizations()
    
    benchmark = KolmogorovFlowBenchmark(resolution=args.resolution, Re=args.Re)
    benchmark.run_full_benchmark(num_epochs=args.epochs, train=not args.no_train)


if __name__ == "__main__":
    main()

