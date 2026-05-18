"""
2D Decaying Turbulence Benchmark

Klasik türbülans test case'i.
Random başlangıç → inverse cascade → büyük yapılar.

Fizik:
    - 2D türbülansta inverse energy cascade
    - Enstrofi forward cascade
    - Büyük scale vortex'ler oluşur
    - E(k) ~ k^(-3) (enstrophy inertial range)
    - E(k) ~ k^(-5/3) (energy inertial range)

Kullanım:
    python -m tests.periodic_2d.decaying_turbulence --resolution 64 --Re 10000
"""

import sys
import argparse
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from innate import INNATE, FluidState, DEVICE
from tests.common.device import print_device_info, setup_device_optimizations
from tests.common.trainer_base import BenchmarkTrainer, TrainerConfig, MetricsCalculator
from tests.common.visualizer import FlowVisualizer
from tests.common.dns_solver import PseudoSpectralDNS


class DecayingTurbulenceBenchmark:
    """2D Decaying Turbulence Benchmark."""
    
    def __init__(
        self,
        resolution: int = 64,
        Re: float = 10000,
        k_peak: int = 8,  # Peak wavenumber
        initial_energy: float = 0.5,
        t_final: float = 20.0,
        dns_resolution: int = 256,
        results_dir: str = 'results/decaying_turbulence',
        seed: int = 42,
    ):
        self.resolution = resolution
        self.Re = Re
        self.k_peak = k_peak
        self.initial_energy = initial_energy
        self.t_final = t_final
        self.dns_resolution = dns_resolution
        self.nu = 1.0 / Re
        self.seed = seed
        
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.viz = FlowVisualizer(results_dir=str(self.results_dir))
        self.model = None
        self.dns_result = None
    
    def create_initial_condition(self, N: int, seed: int = None) -> tuple:
        """
        Random vorticity field başlangıç koşulu.
        
        Enerji spektrumu: E(k) ~ k^2 * exp(-(k/k_peak)^2)
        """
        if seed is not None:
            np.random.seed(seed)
        
        L = 2 * np.pi
        k = np.fft.fftfreq(N, d=1/N) * 2 * np.pi / L
        kx, ky = np.meshgrid(k, k, indexing='ij')
        k_mag = np.sqrt(kx**2 + ky**2)
        k_mag[0, 0] = 1.0  # Avoid division by zero
        
        # Target energy spectrum
        E_k = k_mag**2 * np.exp(-(k_mag / self.k_peak)**2)
        
        # Random phases
        phase = 2 * np.pi * np.random.random((N, N))
        
        # Vortisite (spectral)
        omega_hat = np.sqrt(E_k) * np.exp(1j * phase)
        omega_hat[0, 0] = 0  # Zero mean
        
        # Hermitian symmetry for real field
        omega_hat = 0.5 * (omega_hat + np.conj(omega_hat[::-1, ::-1]))
        
        omega = np.real(np.fft.ifft2(omega_hat))
        
        # Normalize to target energy
        k_sq = kx**2 + ky**2
        k_sq[0, 0] = 1.0
        
        psi_hat = np.fft.fft2(omega) / k_sq
        psi_hat[0, 0] = 0
        
        u = np.real(np.fft.ifft2(1j * ky * psi_hat))
        v = np.real(np.fft.ifft2(-1j * kx * psi_hat))
        
        current_energy = 0.5 * np.mean(u**2 + v**2)
        scale = np.sqrt(self.initial_energy / (current_energy + 1e-10))
        
        u *= scale
        v *= scale
        omega *= scale
        
        return u, v, omega
    
    def generate_dns_reference(self) -> dict:
        """DNS referans."""
        print("DNS referans üretiliyor...")
        
        dns = PseudoSpectralDNS(self.dns_resolution, self.Re)
        u0, v0, _ = self.create_initial_condition(self.dns_resolution, seed=self.seed)
        
        result = dns.solve(u0, v0, t_final=self.t_final, save_every=100)
        
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
        
        u, v, omega = self.create_initial_condition(self.resolution, seed=self.seed)
        
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
    
    def run_full_benchmark(self, num_epochs: int = 500, train: bool = True) -> dict:
        """Tam benchmark."""
        print("=" * 60)
        print("2D DECAYING TURBULENCE BENCHMARK")
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
        self.viz.create_animation(omega_pred, times=times, save='turbulence_animation.gif')
        
        # Enerji spektrumu karşılaştırma
        import matplotlib.pyplot as plt
        
        for i in [0, len(states)//2, -1]:
            u = states[i].u
            v = states[i].v
            k, E_k = MetricsCalculator.energy_spectrum(u, v)
            
            # DNS spektrumu
            dns_idx = min(i, len(self.dns_result['u']) - 1)
            u_dns = torch.tensor(self.dns_result['u'][dns_idx])
            v_dns = torch.tensor(self.dns_result['v'][dns_idx])
            k_dns, E_k_dns = MetricsCalculator.energy_spectrum(u_dns, v_dns)
            
            self.viz.plot_energy_spectrum(
                k, E_k,
                reference_k=k_dns, reference_E_k=E_k_dns,
                title=f'Energy Spectrum @ t = {times[i]:.1f}',
                save=f'spectrum_t{times[i]:.1f}.png'
            )
        
        # Enerji decay
        energies = [0.5 * torch.mean(s.u**2 + s.v**2).item() for s in states]
        self.viz.plot_energy_decay(
            times, np.array(energies),
            reference_energies=self.dns_result['energy'],
            reference_times=self.dns_result['t'],
            title='2D Decaying Turbulence: Energy',
            save='energy_decay.png'
        )
        
        return {'energies': energies}


def main():
    parser = argparse.ArgumentParser(description='Decaying Turbulence Benchmark')
    parser.add_argument('--resolution', type=int, default=64)
    parser.add_argument('--Re', type=float, default=10000)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--no-train', action='store_true')
    
    args = parser.parse_args()
    setup_device_optimizations()
    
    benchmark = DecayingTurbulenceBenchmark(resolution=args.resolution, Re=args.Re)
    benchmark.run_full_benchmark(num_epochs=args.epochs, train=not args.no_train)


if __name__ == "__main__":
    main()

