"""
3D Taylor-Green Vortex Benchmark

Klasik 3D türbülans transition test case'i.
Vortex stretching ve enerji dissipasyon peak zamanı.

Fizik:
    - Başlangıçta laminar 3D vortex yapısı
    - Vortex stretching (2D'de yok!)
    - Enerji cascade ve dissipation
    - Peak dissipation time: t* ≈ 9 (Re=1600)

Analitik Başlangıç:
    u = sin(x)cos(y)cos(z)
    v = -cos(x)sin(y)cos(z)
    w = 0
    p = (cos(2x) + cos(2y))(cos(2z) + 2) / 16

Kullanım:
    python -m tests.3d.tgv_3d --resolution 32 --Re 400
"""

import sys
import argparse
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from innate import FluidState3D, SpectralOps3D, DEVICE
from tests.common.device import print_device_info, setup_device_optimizations
from tests.common.visualizer import FlowVisualizer
from tests.common.dns_solver import PseudoSpectralDNS3D


def taylor_green_3d_initial(N: int, device=None):
    """
    3D Taylor-Green vortex başlangıç koşulu.
    
    u = sin(x)cos(y)cos(z)
    v = -cos(x)sin(y)cos(z)
    w = 0
    """
    if device is None:
        device = DEVICE
    
    L = 2 * np.pi
    x = torch.linspace(0, L, N, device=device)
    y = torch.linspace(0, L, N, device=device)
    z = torch.linspace(0, L, N, device=device)
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
    
    u = torch.sin(X) * torch.cos(Y) * torch.cos(Z)
    v = -torch.cos(X) * torch.sin(Y) * torch.cos(Z)
    w = torch.zeros_like(u)
    
    # Vortisite (curl)
    omega_x = torch.sin(X) * torch.cos(Y) * torch.sin(Z)
    omega_y = torch.cos(X) * torch.sin(Y) * torch.sin(Z)
    omega_z = -2 * torch.sin(X) * torch.sin(Y) * torch.cos(Z)
    
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


class TGV3DBenchmark:
    """3D Taylor-Green Vortex Benchmark."""
    
    def __init__(
        self,
        resolution: int = 32,
        Re: float = 400,
        t_final: float = 20.0,
        dns_resolution: int = 64,
        results_dir: str = 'results/tgv_3d',
    ):
        self.resolution = resolution
        self.Re = Re
        self.t_final = t_final
        self.dns_resolution = dns_resolution
        self.nu = 1.0 / Re
        
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.viz = FlowVisualizer(results_dir=str(self.results_dir))
        self.dns_result = None
    
    def generate_dns_reference(self) -> dict:
        """DNS referans üret."""
        print("3D DNS referans üretiliyor (bu uzun sürebilir)...")
        
        dns = PseudoSpectralDNS3D(self.dns_resolution, self.Re)
        
        # Başlangıç koşulu
        L = 2 * np.pi
        N = self.dns_resolution
        x = np.linspace(0, L, N, endpoint=False)
        y = np.linspace(0, L, N, endpoint=False)
        z = np.linspace(0, L, N, endpoint=False)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        u0 = np.sin(X) * np.cos(Y) * np.cos(Z)
        v0 = -np.cos(X) * np.sin(Y) * np.cos(Z)
        w0 = np.zeros_like(u0)
        
        result = dns.solve(u0, v0, w0, t_final=self.t_final, save_every=50)
        
        self.dns_result = {
            'u': dns.downsample(result['u'], self.resolution),
            'v': dns.downsample(result['v'], self.resolution),
            'w': dns.downsample(result['w'], self.resolution),
            't': result['t'],
            'energy': result['energy'],
        }
        
        print(f"DNS tamamlandı: {len(result['t'])} snapshot")
        return self.dns_result
    
    def compute_dissipation(self, u: np.ndarray, v: np.ndarray, w: np.ndarray) -> float:
        """
        Enstrofi-bazlı dissipasyon hesabı.
        
        ε = 2ν * Ω = 2ν * <ω²>
        """
        N = u.shape[0]
        L = 2 * np.pi
        
        k = np.fft.fftfreq(N, d=1/N) * 2 * np.pi / L
        kx, ky, kz = np.meshgrid(k, k, k, indexing='ij')
        
        u_hat = np.fft.fftn(u)
        v_hat = np.fft.fftn(v)
        w_hat = np.fft.fftn(w)
        
        # Vortisite
        omega_x = np.real(np.fft.ifftn(1j * ky * w_hat - 1j * kz * v_hat))
        omega_y = np.real(np.fft.ifftn(1j * kz * u_hat - 1j * kx * w_hat))
        omega_z = np.real(np.fft.ifftn(1j * kx * v_hat - 1j * ky * u_hat))
        
        enstrophy = 0.5 * np.mean(omega_x**2 + omega_y**2 + omega_z**2)
        dissipation = 2 * self.nu * enstrophy
        
        return dissipation
    
    def find_peak_dissipation_time(self, times: np.ndarray, 
                                    dissipations: np.ndarray) -> float:
        """Peak dissipasyon zamanını bul."""
        peak_idx = np.argmax(dissipations)
        return times[peak_idx]
    
    def run_full_benchmark(self, generate_dns: bool = True) -> dict:
        """Tam benchmark."""
        print("=" * 60)
        print("3D TAYLOR-GREEN VORTEX BENCHMARK")
        print("=" * 60)
        print_device_info()
        print(f"Resolution: {self.resolution}")
        print(f"Reynolds: {self.Re}")
        
        if generate_dns and self.dns_result is None:
            self.generate_dns_reference()
        
        if self.dns_result is not None:
            # Dissipasyon hesapla
            dissipations = []
            for i in range(len(self.dns_result['t'])):
                u = self.dns_result['u'][i]
                v = self.dns_result['v'][i]
                w = self.dns_result['w'][i]
                diss = self.compute_dissipation(u, v, w)
                dissipations.append(diss)
            
            dissipations = np.array(dissipations)
            times = self.dns_result['t']
            
            peak_time = self.find_peak_dissipation_time(times, dissipations)
            print(f"\nPeak dissipation time: {peak_time:.2f}")
            print(f"(Referans Re=1600: t* ≈ 9)")
            
            # Görselleştirme
            import matplotlib.pyplot as plt
            
            # Enerji ve dissipasyon
            fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
            
            axes[0].plot(times, self.dns_result['energy'])
            axes[0].set_xlabel('Time')
            axes[0].set_ylabel('Kinetic Energy')
            axes[0].set_title('Energy Decay')
            axes[0].grid(True, alpha=0.3)
            
            axes[1].plot(times, dissipations)
            axes[1].axvline(peak_time, color='r', linestyle='--', label=f'Peak @ t={peak_time:.1f}')
            axes[1].set_xlabel('Time')
            axes[1].set_ylabel('Dissipation Rate')
            axes[1].set_title('Dissipation Rate')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            fig.savefig(self.results_dir / 'energy_dissipation.png', bbox_inches='tight')
            plt.close()
            
            # Vortisite magnitude (mid-plane)
            mid = self.resolution // 2
            for i, t in enumerate([0, peak_time, self.t_final]):
                idx = np.argmin(np.abs(times - t))
                
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
                ax.set_title(f'Vorticity Magnitude (z={mid}) @ t={t:.1f}')
                plt.colorbar(im, ax=ax)
                fig.savefig(self.results_dir / f'vorticity_t{t:.1f}.png', bbox_inches='tight')
                plt.close()
            
            print(f"\nSonuçlar: {self.results_dir}")
            
            return {
                'times': times,
                'energy': self.dns_result['energy'],
                'dissipation': dissipations,
                'peak_time': peak_time,
            }
        
        return {}


def main():
    parser = argparse.ArgumentParser(description='3D Taylor-Green Vortex Benchmark')
    parser.add_argument('--resolution', type=int, default=32)
    parser.add_argument('--Re', type=float, default=400)
    parser.add_argument('--t_final', type=float, default=20.0)
    parser.add_argument('--no-dns', action='store_true', help='DNS üretme')
    
    args = parser.parse_args()
    setup_device_optimizations()
    
    benchmark = TGV3DBenchmark(
        resolution=args.resolution, 
        Re=args.Re,
        t_final=args.t_final
    )
    benchmark.run_full_benchmark(generate_dns=not args.no_dns)


if __name__ == "__main__":
    main()

