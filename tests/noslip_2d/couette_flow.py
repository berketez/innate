"""
Couette Flow Benchmark

İki paralel plaka arası akış.
Analitik çözümü var - basit validasyon için ideal.

Fizik:
    - Alt plaka sabit (u=0)
    - Üst plaka hareket eder (u=U)
    - Steady-state: u(y) = U * y / H (lineer profil)
    - Startup: erf profili

Analitik Çözüm (transient):
    u(y,t) = U * y/H - (2U/π) * Σ (1/n) * sin(nπy/H) * exp(-n²π²νt/H²)

Kullanım:
    python -m tests.noslip_2d.couette_flow --resolution 64 --Re 100
"""

import sys
import argparse
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from innate import INNATE, FluidState, ImmersedBoundary, DEVICE
from tests.common.device import print_device_info, setup_device_optimizations
from tests.common.trainer_base import ValidationMetrics
from tests.common.visualizer import FlowVisualizer


class CouetteFlowBenchmark:
    """
    Couette Flow Benchmark.
    
    Analitik çözümle karşılaştırma.
    """
    
    def __init__(
        self,
        resolution: int = 64,
        Re: float = 100,
        U: float = 1.0,  # Üst plaka hızı
        H: float = 1.0,  # Kanal yüksekliği
        t_final: float = 5.0,
        results_dir: str = 'results/couette_flow',
    ):
        self.resolution = resolution
        self.Re = Re
        self.U = U
        self.H = H
        self.t_final = t_final
        self.nu = H * U / Re  # ν = UH/Re
        
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.viz = FlowVisualizer(results_dir=str(self.results_dir))
        self.model = None
        self.ib = None
    
    def analytical_steady_state(self, y: np.ndarray) -> np.ndarray:
        """
        Steady-state çözüm: u = U * y / H
        """
        return self.U * y / self.H
    
    def analytical_transient(self, y: np.ndarray, t: float, n_terms: int = 50) -> np.ndarray:
        """
        Transient çözüm (Fourier serisi).

        Doğru formül:
        u(y,t) = U*y/H + (2U/π) * Σ ((-1)^n / n) * sin(nπy/H) * exp(-n²π²νt/H²)

        t=0'da: u = 0 (akışkan durgun)
        t→∞'da: u = U*y/H (lineer steady-state)
        """
        u = self.analytical_steady_state(y)

        for n in range(1, n_terms + 1):
            decay = np.exp(-n**2 * np.pi**2 * self.nu * t / self.H**2)
            # (-1)^n faktörü kritik - alternan seri
            u += (2 * self.U / (n * np.pi)) * ((-1)**n) * np.sin(n * np.pi * y / self.H) * decay

        return u
    
    def setup_immersed_boundary(self) -> ImmersedBoundary:
        """Channel geometrisi ile IBM oluştur."""
        self.ib = ImmersedBoundary(self.resolution, domain_size=self.H, device=DEVICE)
        self.ib.set_channel_geometry(
            wall_velocity=self.U, 
            wall_thickness=1, 
            moving_wall='top'
        )
        return self.ib
    
    def get_initial_state(self, batch_size: int = 1, device=None) -> FluidState:
        """Başlangıç durumu (sıfır hız)."""
        if device is None:
            device = DEVICE
        
        N = self.resolution
        
        u = torch.zeros(1, N, N, device=device)
        v = torch.zeros(1, N, N, device=device)
        p = torch.zeros(1, N, N, device=device)
        omega = torch.zeros(1, N, N, device=device)
        
        return FluidState(
            u=u, v=v, p=p,
            vorticity=omega,
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
        
        self.setup_immersed_boundary()
        self.model.register_boundary(self.ib)
        
        return self.model
    
    def run_simulation(self, num_steps: int = 200) -> dict:
        """Simülasyon çalıştır."""
        if self.model is None:
            self.create_model()
        
        self.model.eval()
        
        with torch.no_grad():
            initial_state = self.get_initial_state()
            states = self.model(initial_state, num_steps=num_steps)
        
        return {'states': states}
    
    def compute_errors(self, states: list, times: np.ndarray) -> dict:
        """Analitik çözümle hata hesapla."""
        N = self.resolution
        y = np.linspace(0, self.H, N)
        
        l2_errors = []
        
        for i, (state, t) in enumerate(zip(states, times)):
            u_pred = state.u.cpu().numpy().squeeze()
            
            # y boyunca ortalama profil
            u_profile = np.mean(u_pred, axis=0)  # x boyunca ortalama
            
            # Analitik
            u_analytical = self.analytical_transient(y, t)
            
            # L2 error
            l2 = np.sqrt(np.mean((u_profile - u_analytical)**2))
            l2_errors.append(l2)
        
        return {
            'times': times,
            'l2_errors': l2_errors,
            'final_l2': l2_errors[-1],
        }
    
    def run_full_benchmark(self, num_steps: int = 200) -> dict:
        """Tam benchmark."""
        print("=" * 60)
        print("COUETTE FLOW BENCHMARK")
        print("=" * 60)
        print_device_info()
        print(f"Resolution: {self.resolution}")
        print(f"Reynolds: {self.Re}")
        print(f"U (wall velocity): {self.U}")
        
        self.create_model()
        
        print("\n--- Simülasyon ---")
        result = self.run_simulation(num_steps)
        
        # Adaptif state.t değerlerini kullan (linspace yerine)
        times = np.array([s.t.item() if hasattr(s.t, 'item') else float(s.t)
                          for s in result['states']])
        errors = self.compute_errors(result['states'], times)
        
        print(f"Final L2 Error: {errors['final_l2']:.4e}")
        
        # Görselleştirme
        import matplotlib.pyplot as plt
        
        N = self.resolution
        y = np.linspace(0, self.H, N)
        
        # Profil karşılaştırması
        fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=150)
        
        snapshot_times = [0.1, 0.5, 1.0, self.t_final]
        snapshot_indices = [int(t / self.t_final * (len(times) - 1)) for t in snapshot_times]
        
        for ax, idx in zip(axes.flat, snapshot_indices):
            t = times[idx]
            u_pred = result['states'][idx].u.cpu().numpy().squeeze()
            u_profile = np.mean(u_pred, axis=0)
            u_analytical = self.analytical_transient(y, t)
            
            ax.plot(u_profile, y, 'b-', linewidth=2, label='INNATE')
            ax.plot(u_analytical, y, 'r--', linewidth=2, label='Analytical')
            ax.set_xlabel('u')
            ax.set_ylabel('y')
            ax.set_title(f't = {t:.2f}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig.savefig(self.results_dir / 'profile_comparison.png', bbox_inches='tight')
        plt.close()
        
        # Error evolution
        self.viz.plot_error_evolution(
            times, np.array(errors['l2_errors']),
            title='Couette Flow: L2 Error',
            save='error_evolution.png'
        )
        
        print(f"\nSonuçlar: {self.results_dir}")
        
        return errors


def main():
    parser = argparse.ArgumentParser(description='Couette Flow Benchmark')
    parser.add_argument('--resolution', type=int, default=64)
    parser.add_argument('--Re', type=float, default=100)
    parser.add_argument('--steps', type=int, default=200)
    
    args = parser.parse_args()
    setup_device_optimizations()
    
    benchmark = CouetteFlowBenchmark(resolution=args.resolution, Re=args.Re)
    benchmark.run_full_benchmark(num_steps=args.steps)


if __name__ == "__main__":
    main()

