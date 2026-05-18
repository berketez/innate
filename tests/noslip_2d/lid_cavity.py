"""
Lid-Driven Cavity Benchmark

Klasik CFD benchmark problemi.
Ghia et al. (1982) referans verileri ile karşılaştırma.

Fizik:
    - Kare kavite (kapalı kutu)
    - Üst duvar sabit hızla hareket eder
    - Diğer duvarlar sabit (no-slip)
    - Re arttıkça: tek vortex → corner vortices

Referans:
    Ghia, U., Ghia, K. N., & Shin, C. T. (1982).
    High-Re solutions for incompressible flow using the
    Navier-Stokes equations and a multigrid method.
    Journal of computational physics, 48(3), 387-411.

Kullanım:
    python -m tests.noslip_2d.lid_cavity --resolution 64 --Re 100
"""

import sys
import argparse
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from innate import INNATE, FluidState, ImmersedBoundary, DEVICE
from tests.common.device import print_device_info, setup_device_optimizations
from tests.common.trainer_base import BenchmarkTrainer, TrainerConfig, ValidationMetrics
from tests.common.visualizer import FlowVisualizer


# =============================================================================
# GHIA ET AL. 1982 REFERANS VERİLERİ
# =============================================================================

GHIA_DATA = {
    100: {
        # y-koordinatları (merkez çizgi)
        'y': np.array([0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719,
                       0.2813, 0.4531, 0.5000, 0.6172, 0.7344, 0.8516,
                       0.9531, 0.9609, 0.9688, 0.9766, 1.0000]),
        # u-hızı (y boyunca, x=0.5)
        'u': np.array([0.00000, -0.03717, -0.04192, -0.04775, -0.06434, -0.10150,
                       -0.15662, -0.21090, -0.20581, -0.13641, 0.00332, 0.23151,
                       0.68717, 0.73722, 0.78871, 0.84123, 1.00000]),
        # x-koordinatları (merkez çizgi)
        'x': np.array([0.0000, 0.0625, 0.0703, 0.0781, 0.0938, 0.1563,
                       0.2266, 0.2344, 0.5000, 0.8047, 0.8594, 0.9063,
                       0.9453, 0.9531, 0.9609, 0.9688, 1.0000]),
        # v-hızı (x boyunca, y=0.5)
        'v': np.array([0.00000, 0.09233, 0.10091, 0.10890, 0.12317, 0.16077,
                       0.17507, 0.17527, 0.05454, -0.24533, -0.22445, -0.16914,
                       -0.10313, -0.08864, -0.07391, -0.05906, 0.00000]),
    },
    400: {
        'y': np.array([0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719,
                       0.2813, 0.4531, 0.5000, 0.6172, 0.7344, 0.8516,
                       0.9531, 0.9609, 0.9688, 0.9766, 1.0000]),
        'u': np.array([0.00000, -0.08186, -0.09266, -0.10338, -0.14612, -0.24299,
                       -0.32726, -0.17119, -0.11477, 0.02135, 0.16256, 0.29093,
                       0.55892, 0.61756, 0.68439, 0.75837, 1.00000]),
        'x': np.array([0.0000, 0.0625, 0.0703, 0.0781, 0.0938, 0.1563,
                       0.2266, 0.2344, 0.5000, 0.8047, 0.8594, 0.9063,
                       0.9453, 0.9531, 0.9609, 0.9688, 1.0000]),
        'v': np.array([0.00000, 0.18360, 0.19713, 0.20920, 0.22965, 0.28124,
                       0.30203, 0.30174, 0.05186, -0.38598, -0.44993, -0.23827,
                       -0.22847, -0.19254, -0.15663, -0.12146, 0.00000]),
    },
    1000: {
        'y': np.array([0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719,
                       0.2813, 0.4531, 0.5000, 0.6172, 0.7344, 0.8516,
                       0.9531, 0.9609, 0.9688, 0.9766, 1.0000]),
        'u': np.array([0.00000, -0.18109, -0.20196, -0.22220, -0.29730, -0.38289,
                       -0.27805, -0.10648, -0.06080, 0.05702, 0.18719, 0.33304,
                       0.46604, 0.51117, 0.57492, 0.65928, 1.00000]),
        'x': np.array([0.0000, 0.0625, 0.0703, 0.0781, 0.0938, 0.1563,
                       0.2266, 0.2344, 0.5000, 0.8047, 0.8594, 0.9063,
                       0.9453, 0.9531, 0.9609, 0.9688, 1.0000]),
        'v': np.array([0.00000, 0.27485, 0.29012, 0.30353, 0.32627, 0.37095,
                       0.33075, 0.32235, 0.02526, -0.31966, -0.42665, -0.51550,
                       -0.39188, -0.33714, -0.27669, -0.21388, 0.00000]),
    },
}


class LidDrivenCavityBenchmark:
    """
    Lid-Driven Cavity Benchmark.
    
    Immersed Boundary Method ile no-slip sınır koşulları.
    """
    
    def __init__(
        self,
        resolution: int = 64,
        Re: float = 100,
        lid_velocity: float = 1.0,
        t_final: float = 50.0,  # Steady-state'e ulaşmak için uzun süre
        results_dir: str = 'results/lid_cavity',
    ):
        self.resolution = resolution
        self.Re = Re
        self.lid_velocity = lid_velocity
        self.t_final = t_final
        self.nu = 1.0 / Re
        
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.viz = FlowVisualizer(results_dir=str(self.results_dir))
        self.model = None
        self.ib = None
        
        # Ghia referans
        if Re in GHIA_DATA:
            self.ghia = GHIA_DATA[Re]
        else:
            # En yakın Re
            available = list(GHIA_DATA.keys())
            closest = min(available, key=lambda x: abs(x - Re))
            self.ghia = GHIA_DATA[closest]
            print(f"Uyarı: Re={Re} için Ghia verisi yok, Re={closest} kullanılıyor")
    
    def setup_immersed_boundary(self) -> ImmersedBoundary:
        """Cavity geometrisi ile IBM oluştur."""
        self.ib = ImmersedBoundary(self.resolution, domain_size=1.0, device=DEVICE)
        self.ib.set_cavity_geometry(lid_velocity=self.lid_velocity, wall_thickness=1)
        return self.ib
    
    def get_initial_state(self, batch_size: int = 1, device=None) -> FluidState:
        """Başlangıç durumu (sıfır hız)."""
        if device is None:
            device = DEVICE
        
        N = self.resolution
        
        # Sıfır hız (quiescent)
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
        """Model oluştur ve IBM kaydet."""
        self.model = INNATE(
            resolution=self.resolution,
            re_range=(self.Re * 0.5, self.Re * 2),
            bc_type='periodic'  # IBM kullanacağız
        ).to(DEVICE)
        
        with torch.no_grad():
            self.model.reynolds_learner.reynolds.fill_(self.Re)
        
        # Immersed Boundary kaydet
        self.setup_immersed_boundary()
        self.model.register_boundary(self.ib)
        
        return self.model
    
    def extract_centerline_velocities(self, u: np.ndarray, v: np.ndarray) -> dict:
        """
        Merkez çizgi hızlarını çıkar.
        
        Args:
            u, v: Hız alanları [N, N]
        
        Returns:
            dict: Centerline velocities
        """
        N = u.shape[0]
        mid = N // 2
        
        # y boyunca (x = 0.5), u hızı
        u_centerline = u[mid, :]
        y_coords = np.linspace(0, 1, N)
        
        # x boyunca (y = 0.5), v hızı
        v_centerline = v[:, mid]
        x_coords = np.linspace(0, 1, N)
        
        return {
            'y': y_coords,
            'u': u_centerline,
            'x': x_coords,
            'v': v_centerline,
        }
    
    def compare_with_ghia(self, centerline: dict) -> dict:
        """Ghia verileri ile karşılaştır."""
        # Interpolate to Ghia points
        from scipy.interpolate import interp1d
        
        # u(y) at x=0.5
        u_interp = interp1d(centerline['y'], centerline['u'], kind='linear')
        u_at_ghia = u_interp(self.ghia['y'])
        u_error = np.sqrt(np.mean((u_at_ghia - self.ghia['u'])**2))
        
        # v(x) at y=0.5
        v_interp = interp1d(centerline['x'], centerline['v'], kind='linear')
        v_at_ghia = v_interp(self.ghia['x'])
        v_error = np.sqrt(np.mean((v_at_ghia - self.ghia['v'])**2))
        
        return {
            'u_rmse': u_error,
            'v_rmse': v_error,
            'u_at_ghia': u_at_ghia,
            'v_at_ghia': v_at_ghia,
        }
    
    def run_to_steady_state(self, num_steps: int = 500, 
                            convergence_threshold: float = 1e-6) -> dict:
        """
        Steady-state'e kadar simülasyon.
        
        Args:
            num_steps: Maksimum adım sayısı
            convergence_threshold: Yakınsama eşiği
        
        Returns:
            dict: Final durum ve metrikler
        """
        if self.model is None:
            self.create_model()
        
        self.model.eval()
        
        state = self.get_initial_state()
        prev_u = state.u.clone()
        
        states = [state]
        residuals = []
        
        with torch.no_grad():
            for step in range(num_steps):
                # Tek adım
                new_states = self.model(state, num_steps=1)
                state = new_states[-1]
                
                # Residual
                residual = torch.mean((state.u - prev_u)**2).item()
                residuals.append(residual)
                prev_u = state.u.clone()
                
                # Kaydet
                if step % 50 == 0:
                    states.append(state)
                    print(f"Step {step}: residual = {residual:.2e}")
                
                # Yakınsama kontrolü
                if residual < convergence_threshold:
                    print(f"Converged at step {step}")
                    break
        
        states.append(state)
        
        return {
            'final_state': state,
            'states': states,
            'residuals': residuals,
        }
    
    def run_full_benchmark(self, num_steps: int = 500, train: bool = False) -> dict:
        """Tam benchmark."""
        print("=" * 60)
        print("LID-DRIVEN CAVITY BENCHMARK")
        print("=" * 60)
        print_device_info()
        print(f"Resolution: {self.resolution}")
        print(f"Reynolds: {self.Re}")
        
        self.create_model()
        
        print("\n--- Steady-State Simülasyonu ---")
        result = self.run_to_steady_state(num_steps)
        
        # Final hız alanı
        final_u = result['final_state'].u.cpu().numpy().squeeze()
        final_v = result['final_state'].v.cpu().numpy().squeeze()
        
        # Centerline velocities
        centerline = self.extract_centerline_velocities(final_u, final_v)
        
        # Ghia karşılaştırması
        comparison = self.compare_with_ghia(centerline)
        
        print(f"\nGhia Karşılaştırması:")
        print(f"  u RMSE: {comparison['u_rmse']:.4f}")
        print(f"  v RMSE: {comparison['v_rmse']:.4f}")
        
        # Görselleştirme
        import matplotlib.pyplot as plt
        
        # Streamlines
        fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
        N = self.resolution
        x = np.linspace(0, 1, N)
        y = np.linspace(0, 1, N)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        ax.streamplot(X.T, Y.T, final_u.T, final_v.T, density=2, linewidth=0.5)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'Lid-Driven Cavity Streamlines (Re={self.Re})')
        ax.set_aspect('equal')
        fig.savefig(self.results_dir / 'streamlines.png', bbox_inches='tight')
        plt.close()
        
        # Ghia karşılaştırma grafiği
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
        
        # u(y) at x=0.5
        axes[0].plot(centerline['u'], centerline['y'], 'b-', label='INNATE')
        axes[0].plot(self.ghia['u'], self.ghia['y'], 'ro', markersize=6, label='Ghia (1982)')
        axes[0].set_xlabel('u')
        axes[0].set_ylabel('y')
        axes[0].set_title(f'u-velocity at x=0.5 (Re={self.Re})')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # v(x) at y=0.5
        axes[1].plot(centerline['x'], centerline['v'], 'b-', label='INNATE')
        axes[1].plot(self.ghia['x'], self.ghia['v'], 'ro', markersize=6, label='Ghia (1982)')
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('v')
        axes[1].set_title(f'v-velocity at y=0.5 (Re={self.Re})')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig.savefig(self.results_dir / 'ghia_comparison.png', bbox_inches='tight')
        plt.close()
        
        # Residual history
        fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
        ax.semilogy(result['residuals'])
        ax.set_xlabel('Step')
        ax.set_ylabel('Residual')
        ax.set_title('Convergence History')
        ax.grid(True, alpha=0.3)
        fig.savefig(self.results_dir / 'convergence.png', bbox_inches='tight')
        plt.close()
        
        print(f"\nSonuçlar: {self.results_dir}")
        
        return {
            'centerline': centerline,
            'comparison': comparison,
            'result': result,
        }


def main():
    parser = argparse.ArgumentParser(description='Lid-Driven Cavity Benchmark')
    parser.add_argument('--resolution', type=int, default=64)
    parser.add_argument('--Re', type=float, default=100)
    parser.add_argument('--steps', type=int, default=500)
    
    args = parser.parse_args()
    setup_device_optimizations()
    
    benchmark = LidDrivenCavityBenchmark(resolution=args.resolution, Re=args.Re)
    benchmark.run_full_benchmark(num_steps=args.steps)


if __name__ == "__main__":
    main()

