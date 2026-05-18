"""
3D Isotropic Turbulence Benchmark

Klasik 3D türbülans test case'i.
Kolmogorov -5/3 enerji spektrumu.

Fizik:
    - 3D forward energy cascade
    - Kolmogorov theory: E(k) ~ ε^(2/3) k^(-5/3)
    - Viscous dissipation at small scales
    - Isotropic and homogeneous

Kullanım:
    python -m tests.3d.isotropic_3d --resolution 32 --Re 1000
"""

import sys
import argparse
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from innate import FluidState3D, DEVICE
from tests.common.device import print_device_info, setup_device_optimizations
from tests.common.visualizer import FlowVisualizer
from tests.common.dns_solver import PseudoSpectralDNS3D


class Isotropic3DBenchmark:
    """3D Isotropic Turbulence Benchmark."""
    
    def __init__(
        self,
        resolution: int = 32,
        Re: float = 1000,
        k_peak: int = 4,  # Initial peak wavenumber
        initial_energy: float = 0.5,
        t_final: float = 10.0,
        results_dir: str = 'results/isotropic_3d',
        seed: int = 42,
    ):
        self.resolution = resolution
        self.Re = Re
        self.k_peak = k_peak
        self.initial_energy = initial_energy
        self.t_final = t_final
        self.nu = 1.0 / Re
        self.seed = seed
        
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.viz = FlowVisualizer(results_dir=str(self.results_dir))
        self.dns_result = None
    
    def create_initial_condition(self, N: int, seed: int = None):
        """
        Random isotropic başlangıç koşulu.
        
        Divergence-free random field with prescribed energy spectrum.
        """
        if seed is not None:
            np.random.seed(seed)
        
        L = 2 * np.pi
        k = np.fft.fftfreq(N, d=1/N) * 2 * np.pi / L
        kx, ky, kz = np.meshgrid(k, k, k, indexing='ij')
        k_mag = np.sqrt(kx**2 + ky**2 + kz**2)
        k_mag[0, 0, 0] = 1.0  # Avoid division by zero
        
        # Target energy spectrum: E(k) ~ k^2 * exp(-(k/k_peak)^2)
        E_k = k_mag**2 * np.exp(-(k_mag / self.k_peak)**2)
        
        # Random phases (3 components)
        phase_u = 2 * np.pi * np.random.random((N, N, N))
        phase_v = 2 * np.pi * np.random.random((N, N, N))
        phase_w = 2 * np.pi * np.random.random((N, N, N))
        
        # Random velocity (spectral)
        u_hat = np.sqrt(E_k) * np.exp(1j * phase_u)
        v_hat = np.sqrt(E_k) * np.exp(1j * phase_v)
        w_hat = np.sqrt(E_k) * np.exp(1j * phase_w)
        
        # Make divergence-free (projection)
        k_sq = kx**2 + ky**2 + kz**2
        k_sq[0, 0, 0] = 1.0
        
        div_hat = kx * u_hat + ky * v_hat + kz * w_hat
        p_hat = div_hat / k_sq
        p_hat[0, 0, 0] = 0
        
        u_hat -= kx * p_hat
        v_hat -= ky * p_hat
        w_hat -= kz * p_hat
        
        # Zero DC
        u_hat[0, 0, 0] = 0
        v_hat[0, 0, 0] = 0
        w_hat[0, 0, 0] = 0
        
        # To physical space
        u = np.real(np.fft.ifftn(u_hat))
        v = np.real(np.fft.ifftn(v_hat))
        w = np.real(np.fft.ifftn(w_hat))
        
        # Normalize to target energy
        current_energy = 0.5 * np.mean(u**2 + v**2 + w**2)
        scale = np.sqrt(self.initial_energy / (current_energy + 1e-10))
        
        u *= scale
        v *= scale
        w *= scale
        
        return u, v, w
    
    def compute_energy_spectrum(self, u: np.ndarray, v: np.ndarray, 
                                 w: np.ndarray) -> tuple:
        """
        1D energy spectrum hesapla (shell averaging).
        """
        N = u.shape[0]
        L = 2 * np.pi
        
        k = np.fft.fftfreq(N, d=1/N) * 2 * np.pi / L
        kx, ky, kz = np.meshgrid(k, k, k, indexing='ij')
        k_mag = np.sqrt(kx**2 + ky**2 + kz**2)
        
        u_hat = np.fft.fftn(u)
        v_hat = np.fft.fftn(v)
        w_hat = np.fft.fftn(w)
        
        E_hat = 0.5 * (np.abs(u_hat)**2 + np.abs(v_hat)**2 + np.abs(w_hat)**2) / N**6
        
        # Shell averaging
        k_max = N // 2
        k_bins = np.arange(0.5, k_max + 0.5, 1)
        E_k = np.zeros(len(k_bins) - 1)
        
        for i in range(len(k_bins) - 1):
            mask = (k_mag >= k_bins[i]) & (k_mag < k_bins[i + 1])
            E_k[i] = np.sum(E_hat[mask])
        
        k_values = np.arange(1, k_max)
        
        return k_values, E_k
    
    def generate_dns_reference(self) -> dict:
        """DNS referans üret."""
        print("3D Isotropic Turbulence DNS üretiliyor...")
        
        dns = PseudoSpectralDNS3D(self.resolution, self.Re)
        u0, v0, w0 = self.create_initial_condition(self.resolution, seed=self.seed)
        
        result = dns.solve(u0, v0, w0, t_final=self.t_final, save_every=20)
        
        self.dns_result = result
        print(f"DNS tamamlandı: {len(result['t'])} snapshot")
        return result
    
    def run_full_benchmark(self) -> dict:
        """Tam benchmark."""
        print("=" * 60)
        print("3D ISOTROPIC TURBULENCE BENCHMARK")
        print("=" * 60)
        print_device_info()
        print(f"Resolution: {self.resolution}")
        print(f"Reynolds: {self.Re}")
        print(f"Initial k_peak: {self.k_peak}")
        
        self.generate_dns_reference()
        
        times = self.dns_result['t']
        energies = self.dns_result['energy']
        
        # Görselleştirme
        import matplotlib.pyplot as plt
        
        # Enerji decay
        fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
        ax.plot(times, energies / energies[0], 'b-', linewidth=2)
        ax.set_xlabel('Time')
        ax.set_ylabel('E(t) / E(0)')
        ax.set_title('3D Isotropic Turbulence: Energy Decay')
        ax.grid(True, alpha=0.3)
        fig.savefig(self.results_dir / 'energy_decay.png', bbox_inches='tight')
        plt.close()
        
        # Enerji spektrumu (başlangıç, orta, son)
        fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
        
        colors = ['blue', 'green', 'red']
        labels = ['t=0', f't={self.t_final/2:.1f}', f't={self.t_final:.1f}']
        indices = [0, len(times)//2, -1]
        
        for color, label, idx in zip(colors, labels, indices):
            k, E_k = self.compute_energy_spectrum(
                self.dns_result['u'][idx],
                self.dns_result['v'][idx],
                self.dns_result['w'][idx]
            )
            ax.loglog(k, E_k, color=color, linewidth=2, label=label)
        
        # -5/3 reference
        k_ref = k[2:len(k)//2]
        E_ref = E_k[2] * (k_ref / k_ref[0])**(-5/3)
        ax.loglog(k_ref, E_ref, 'k--', linewidth=1.5, label='k^{-5/3}')
        
        ax.set_xlabel('Wavenumber k')
        ax.set_ylabel('E(k)')
        ax.set_title('3D Isotropic Turbulence: Energy Spectrum')
        ax.legend()
        ax.grid(True, alpha=0.3, which='both')
        fig.savefig(self.results_dir / 'energy_spectrum.png', bbox_inches='tight')
        plt.close()
        
        # Mid-plane vorticity
        mid = self.resolution // 2
        for idx, t in [(0, 0), (-1, self.t_final)]:
            u = self.dns_result['u'][idx]
            v = self.dns_result['v'][idx]
            w = self.dns_result['w'][idx]
            
            # Vortisite magnitude
            N = self.resolution
            L = 2 * np.pi
            k = np.fft.fftfreq(N, d=1/N) * 2 * np.pi / L
            kx, ky, kz = np.meshgrid(k, k, k, indexing='ij')
            
            u_hat = np.fft.fftn(u)
            v_hat = np.fft.fftn(v)
            w_hat = np.fft.fftn(w)
            
            omega_x = np.real(np.fft.ifftn(1j * ky * w_hat - 1j * kz * v_hat))
            omega_y = np.real(np.fft.ifftn(1j * kz * u_hat - 1j * kx * w_hat))
            omega_z = np.real(np.fft.ifftn(1j * kx * v_hat - 1j * ky * u_hat))
            
            omega_mag = np.sqrt(omega_x**2 + omega_y**2 + omega_z**2)
            
            fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
            im = ax.imshow(omega_mag[:, :, mid].T, origin='lower', cmap='hot')
            ax.set_title(f'Vorticity Magnitude @ t={t:.1f}')
            plt.colorbar(im, ax=ax)
            fig.savefig(self.results_dir / f'vorticity_t{t:.1f}.png', bbox_inches='tight')
            plt.close()
        
        print(f"\nSonuçlar: {self.results_dir}")
        
        return {
            'times': times,
            'energy': energies,
        }


def main():
    parser = argparse.ArgumentParser(description='3D Isotropic Turbulence Benchmark')
    parser.add_argument('--resolution', type=int, default=32)
    parser.add_argument('--Re', type=float, default=1000)
    parser.add_argument('--t_final', type=float, default=10.0)
    
    args = parser.parse_args()
    setup_device_optimizations()
    
    benchmark = Isotropic3DBenchmark(resolution=args.resolution, Re=args.Re, t_final=args.t_final)
    benchmark.run_full_benchmark()


if __name__ == "__main__":
    main()

