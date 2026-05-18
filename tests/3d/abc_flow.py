"""
ABC (Arnold-Beltrami-Childress) Flow Benchmark

Beltrami flow: ω = λu (vortisite hız ile orantılı)
Euler denklemlerinin tam çözümü.
Chaotic behavior ve Lyapunov exponent analizi.

Fizik:
    - Beltrami property: ω × u = 0
    - Euler için steady-state (ν=0)
    - NS için decay (ν>0)
    - Lagrangian chaos

Analitik:
    u = A*sin(z) + C*cos(y)
    v = B*sin(x) + A*cos(z)
    w = C*sin(y) + B*cos(x)
    
    Klasik değerler: A = B = C = 1

Kullanım:
    python -m tests.3d.abc_flow --resolution 32 --Re 100
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


def abc_flow_initial(N: int, A: float = 1.0, B: float = 1.0, C: float = 1.0,
                     device=None):
    """
    ABC flow başlangıç koşulu.
    
    u = A*sin(z) + C*cos(y)
    v = B*sin(x) + A*cos(z)
    w = C*sin(y) + B*cos(x)
    """
    if device is None:
        device = DEVICE
    
    L = 2 * np.pi
    x = torch.linspace(0, L, N, device=device)
    y = torch.linspace(0, L, N, device=device)
    z = torch.linspace(0, L, N, device=device)
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
    
    u = A * torch.sin(Z) + C * torch.cos(Y)
    v = B * torch.sin(X) + A * torch.cos(Z)
    w = C * torch.sin(Y) + B * torch.cos(X)
    
    # Vortisite (Beltrami: ω = u çünkü k=1)
    omega_x = u.clone()
    omega_y = v.clone()
    omega_z = w.clone()
    
    return FluidState3D(
        u=u.unsqueeze(0),
        v=v.unsqueeze(0),
        w=w.unsqueeze(0),
        p=torch.zeros(1, N, N, N, device=device),
        omega_x=omega_x.unsqueeze(0),
        omega_y=omega_y.unsqueeze(0),
        omega_z=omega_z.unsqueeze(0),
        t=torch.tensor(0.0, device=device)
    )


class ABCFlowBenchmark:
    """ABC Flow Benchmark."""
    
    def __init__(
        self,
        resolution: int = 32,
        Re: float = 100,
        A: float = 1.0,
        B: float = 1.0,
        C: float = 1.0,
        t_final: float = 10.0,
        results_dir: str = 'results/abc_flow',
    ):
        self.resolution = resolution
        self.Re = Re
        self.A, self.B, self.C = A, B, C
        self.t_final = t_final
        self.nu = 1.0 / Re
        
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.viz = FlowVisualizer(results_dir=str(self.results_dir))
        self.dns_result = None
    
    def analytical_decay(self, t: float) -> float:
        """
        ABC flow analitik enerji decay'i.
        
        Beltrami flow için: E(t) = E(0) * exp(-2νk²t)
        k=1 için: E(t) = E(0) * exp(-2νt)
        """
        return np.exp(-2 * self.nu * t)
    
    def compute_helicity(self, u: np.ndarray, v: np.ndarray, w: np.ndarray) -> float:
        """
        Helicity hesapla: H = <u · ω>
        
        ABC flow için maksimal helicity beklenir.
        """
        N = u.shape[0]
        L = 2 * np.pi
        
        k = np.fft.fftfreq(N, d=1/N) * 2 * np.pi / L
        kx, ky, kz = np.meshgrid(k, k, k, indexing='ij')
        
        u_hat = np.fft.fftn(u)
        v_hat = np.fft.fftn(v)
        w_hat = np.fft.fftn(w)
        
        omega_x = np.real(np.fft.ifftn(1j * ky * w_hat - 1j * kz * v_hat))
        omega_y = np.real(np.fft.ifftn(1j * kz * u_hat - 1j * kx * w_hat))
        omega_z = np.real(np.fft.ifftn(1j * kx * v_hat - 1j * ky * u_hat))
        
        helicity = np.mean(u * omega_x + v * omega_y + w * omega_z)
        
        return helicity
    
    def generate_dns_reference(self) -> dict:
        """DNS referans üret."""
        print("ABC Flow DNS üretiliyor...")
        
        dns = PseudoSpectralDNS3D(self.resolution, self.Re)
        
        # Başlangıç koşulu
        L = 2 * np.pi
        N = self.resolution
        x = np.linspace(0, L, N, endpoint=False)
        y = np.linspace(0, L, N, endpoint=False)
        z = np.linspace(0, L, N, endpoint=False)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        u0 = self.A * np.sin(Z) + self.C * np.cos(Y)
        v0 = self.B * np.sin(X) + self.A * np.cos(Z)
        w0 = self.C * np.sin(Y) + self.B * np.cos(X)
        
        result = dns.solve(u0, v0, w0, t_final=self.t_final, save_every=20)
        
        self.dns_result = result
        print(f"DNS tamamlandı: {len(result['t'])} snapshot")
        return result
    
    def run_full_benchmark(self) -> dict:
        """Tam benchmark."""
        print("=" * 60)
        print("ABC FLOW BENCHMARK")
        print("=" * 60)
        print_device_info()
        print(f"Resolution: {self.resolution}")
        print(f"Reynolds: {self.Re}")
        print(f"A, B, C = {self.A}, {self.B}, {self.C}")
        
        self.generate_dns_reference()
        
        times = self.dns_result['t']
        energies = self.dns_result['energy']
        
        # Helicity hesapla
        helicities = []
        for i in range(len(times)):
            h = self.compute_helicity(
                self.dns_result['u'][i],
                self.dns_result['v'][i],
                self.dns_result['w'][i]
            )
            helicities.append(h)
        
        helicities = np.array(helicities)
        
        # Analitik ile karşılaştır
        analytical_energy = energies[0] * np.array([self.analytical_decay(t) for t in times])
        
        # Görselleştirme
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
        
        # Enerji
        axes[0].plot(times, energies / energies[0], 'b-', linewidth=2, label='DNS')
        axes[0].plot(times, analytical_energy / energies[0], 'r--', linewidth=2, label='Analytical')
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('E(t) / E(0)')
        axes[0].set_title('Energy Decay')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Helicity
        axes[1].plot(times, helicities / helicities[0], 'b-', linewidth=2)
        axes[1].set_xlabel('Time')
        axes[1].set_ylabel('H(t) / H(0)')
        axes[1].set_title('Helicity Decay')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig.savefig(self.results_dir / 'energy_helicity.png', bbox_inches='tight')
        plt.close()
        
        # Error
        relative_error = np.abs(energies - analytical_energy) / energies[0]
        print(f"\nMax relative energy error: {np.max(relative_error):.4e}")
        print(f"Final helicity ratio: {helicities[-1] / helicities[0]:.4f}")
        
        print(f"\nSonuçlar: {self.results_dir}")
        
        return {
            'times': times,
            'energy': energies,
            'helicity': helicities,
            'energy_error': relative_error,
        }


def main():
    parser = argparse.ArgumentParser(description='ABC Flow Benchmark')
    parser.add_argument('--resolution', type=int, default=32)
    parser.add_argument('--Re', type=float, default=100)
    parser.add_argument('--t_final', type=float, default=10.0)
    
    args = parser.parse_args()
    setup_device_optimizations()
    
    benchmark = ABCFlowBenchmark(resolution=args.resolution, Re=args.Re, t_final=args.t_final)
    benchmark.run_full_benchmark()


if __name__ == "__main__":
    main()

