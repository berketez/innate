"""
Flow Görselleştirme Modülü

Akış alanları, karşılaştırma grafikleri ve animasyonlar için araçlar.

Kullanım:
    from tests.common.visualizer import FlowVisualizer
    
    viz = FlowVisualizer(results_dir='results')
    
    # Tek snapshot
    viz.plot_vorticity(omega, t=0.5, save='vorticity_t0.5.png')
    
    # Karşılaştırma
    viz.plot_comparison(innate_result, dns_result, t=1.0)
    
    # Animasyon
    viz.create_animation(states, fps=30)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize, TwoSlopeNorm
from pathlib import Path
from typing import List, Optional, Tuple, Union, Dict, Any
import torch


class FlowVisualizer:
    """
    Akış görselleştirme araçları.
    
    Args:
        results_dir: Sonuç klasörü
        figsize: Varsayılan figür boyutu
        dpi: Varsayılan DPI
        cmap: Varsayılan colormap
    """
    
    def __init__(
        self,
        results_dir: str = 'results',
        figsize: Tuple[int, int] = (12, 5),
        dpi: int = 150,
        cmap: str = 'RdBu_r',
    ):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.figsize = figsize
        self.dpi = dpi
        self.cmap = cmap
        
        # Style
        plt.style.use('seaborn-v0_8-whitegrid')
    
    def _to_numpy(self, tensor: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """Tensor'ü numpy'a çevir."""
        if isinstance(tensor, torch.Tensor):
            return tensor.squeeze().detach().cpu().numpy()
        return np.squeeze(tensor)
    
    def plot_vorticity(
        self,
        omega: Union[torch.Tensor, np.ndarray],
        t: float = 0.0,
        title: Optional[str] = None,
        save: Optional[str] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        ax: Optional[plt.Axes] = None,
    ) -> Optional[plt.Figure]:
        """
        Vortisite alanı çiz.
        
        Args:
            omega: Vortisite alanı [Nx, Ny]
            t: Zaman
            title: Başlık
            save: Kayıt dosya adı
            vmin, vmax: Renk skalası limitleri
            ax: Mevcut axes (None ise yeni oluştur)
        
        Returns:
            Figure (ax verilmediyse)
        """
        omega_np = self._to_numpy(omega)
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 5), dpi=self.dpi)
            own_fig = True
        else:
            fig = ax.figure
            own_fig = False
        
        # Simetrik colorbar
        if vmin is None or vmax is None:
            vmax_abs = np.max(np.abs(omega_np))
            vmin, vmax = -vmax_abs, vmax_abs
        
        im = ax.imshow(
            omega_np.T, origin='lower', cmap=self.cmap,
            vmin=vmin, vmax=vmax, extent=[0, 2*np.pi, 0, 2*np.pi]
        )
        
        plt.colorbar(im, ax=ax, label='ω (Vortisite)')
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(title or f'Vortisite @ t = {t:.2f}')
        ax.set_aspect('equal')
        
        if save and own_fig:
            fig.savefig(self.results_dir / save, bbox_inches='tight', dpi=self.dpi)
        
        if own_fig:
            return fig
    
    def plot_velocity_field(
        self,
        u: Union[torch.Tensor, np.ndarray],
        v: Union[torch.Tensor, np.ndarray],
        t: float = 0.0,
        skip: int = 4,
        title: Optional[str] = None,
        save: Optional[str] = None,
        show_magnitude: bool = True,
    ) -> plt.Figure:
        """
        Hız alanı (quiver plot) çiz.
        
        Args:
            u, v: Hız bileşenleri
            t: Zaman
            skip: Ok atlama sayısı
            title: Başlık
            save: Kayıt dosya adı
            show_magnitude: Arka planda hız büyüklüğü göster
        """
        u_np = self._to_numpy(u)
        v_np = self._to_numpy(v)
        
        N = u_np.shape[0]
        x = np.linspace(0, 2*np.pi, N)
        y = np.linspace(0, 2*np.pi, N)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        fig, ax = plt.subplots(figsize=(6, 5), dpi=self.dpi)
        
        if show_magnitude:
            mag = np.sqrt(u_np**2 + v_np**2)
            im = ax.imshow(
                mag.T, origin='lower', cmap='viridis',
                extent=[0, 2*np.pi, 0, 2*np.pi], alpha=0.7
            )
            plt.colorbar(im, ax=ax, label='|u|')
        
        ax.quiver(
            X[::skip, ::skip], Y[::skip, ::skip],
            u_np[::skip, ::skip], v_np[::skip, ::skip],
            color='black', alpha=0.8
        )
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(title or f'Hız Alanı @ t = {t:.2f}')
        ax.set_aspect('equal')
        
        if save:
            fig.savefig(self.results_dir / save, bbox_inches='tight', dpi=self.dpi)
        
        return fig
    
    def plot_comparison(
        self,
        innate_omega: Union[torch.Tensor, np.ndarray],
        reference_omega: Union[torch.Tensor, np.ndarray],
        t: float = 0.0,
        innate_label: str = 'INNATE',
        reference_label: str = 'Reference',
        title: Optional[str] = None,
        save: Optional[str] = None,
        metrics: Optional[Dict[str, float]] = None,
    ) -> plt.Figure:
        """
        INNATE vs Reference karşılaştırma grafiği.
        
        Args:
            innate_omega: INNATE vortisite tahmini
            reference_omega: Referans (DNS/analitik) vortisite
            t: Zaman
            innate_label, reference_label: Etiketler
            title: Başlık
            save: Kayıt dosya adı
            metrics: Gösterilecek metrikler
        """
        innate_np = self._to_numpy(innate_omega)
        ref_np = self._to_numpy(reference_omega)
        error_np = innate_np - ref_np
        
        # Ortak renk skalası
        vmax = max(np.max(np.abs(innate_np)), np.max(np.abs(ref_np)))
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=self.dpi)
        
        # INNATE
        im0 = axes[0].imshow(
            innate_np.T, origin='lower', cmap=self.cmap,
            vmin=-vmax, vmax=vmax, extent=[0, 2*np.pi, 0, 2*np.pi]
        )
        axes[0].set_title(f'{innate_label} @ t = {t:.2f}')
        plt.colorbar(im0, ax=axes[0])
        
        # Reference
        im1 = axes[1].imshow(
            ref_np.T, origin='lower', cmap=self.cmap,
            vmin=-vmax, vmax=vmax, extent=[0, 2*np.pi, 0, 2*np.pi]
        )
        axes[1].set_title(f'{reference_label} @ t = {t:.2f}')
        plt.colorbar(im1, ax=axes[1])
        
        # Error
        err_max = np.max(np.abs(error_np))
        im2 = axes[2].imshow(
            error_np.T, origin='lower', cmap='coolwarm',
            vmin=-err_max, vmax=err_max, extent=[0, 2*np.pi, 0, 2*np.pi]
        )
        axes[2].set_title(f'Hata (L2: {np.sqrt(np.mean(error_np**2)):.2e})')
        plt.colorbar(im2, ax=axes[2])
        
        for ax in axes:
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_aspect('equal')
        
        if title:
            fig.suptitle(title, fontsize=12, y=1.02)
        
        if metrics:
            metrics_str = ' | '.join([f'{k}: {v:.2e}' for k, v in metrics.items()])
            fig.text(0.5, -0.05, metrics_str, ha='center', fontsize=10)
        
        plt.tight_layout()
        
        if save:
            fig.savefig(self.results_dir / save, bbox_inches='tight', dpi=self.dpi)
        
        return fig
    
    def plot_energy_decay(
        self,
        times: np.ndarray,
        energies: np.ndarray,
        reference_energies: Optional[np.ndarray] = None,
        reference_times: Optional[np.ndarray] = None,
        analytical_fn: Optional[callable] = None,
        title: str = 'Enerji Sönümü',
        save: Optional[str] = None,
    ) -> plt.Figure:
        """
        Enerji sönüm grafiği.
        
        Args:
            times: Zaman dizisi
            energies: INNATE enerji değerleri
            reference_energies: Referans enerji değerleri
            reference_times: Referans zaman dizisi
            analytical_fn: Analitik enerji fonksiyonu f(t) -> E
            title: Başlık
            save: Kayıt dosya adı
        """
        fig, ax = plt.subplots(figsize=(8, 5), dpi=self.dpi)
        
        # Normalize
        E0 = energies[0]
        ax.plot(times, energies / E0, 'b-', linewidth=2, label='INNATE')
        
        if reference_energies is not None:
            ref_times = reference_times if reference_times is not None else times
            ref_E0 = reference_energies[0]
            ax.plot(ref_times, reference_energies / ref_E0, 'r--', 
                   linewidth=2, label='DNS Reference')
        
        if analytical_fn is not None:
            t_fine = np.linspace(times[0], times[-1], 100)
            E_analytical = np.array([analytical_fn(t) for t in t_fine])
            ax.plot(t_fine, E_analytical / E_analytical[0], 'g:', 
                   linewidth=2, label='Analytical')
        
        ax.set_xlabel('Zaman (t)')
        ax.set_ylabel('E(t) / E(0)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save:
            fig.savefig(self.results_dir / save, bbox_inches='tight', dpi=self.dpi)
        
        return fig
    
    def plot_error_evolution(
        self,
        times: np.ndarray,
        l2_errors: np.ndarray,
        title: str = 'Hata Evrimi',
        save: Optional[str] = None,
        log_scale: bool = True,
    ) -> plt.Figure:
        """
        L2 hata evrimi grafiği.
        """
        fig, ax = plt.subplots(figsize=(8, 5), dpi=self.dpi)
        
        ax.plot(times, l2_errors, 'b-', linewidth=2)
        
        if log_scale:
            ax.set_yscale('log')
        
        ax.set_xlabel('Zaman (t)')
        ax.set_ylabel('L2 Hata')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        if save:
            fig.savefig(self.results_dir / save, bbox_inches='tight', dpi=self.dpi)
        
        return fig
    
    def plot_training_history(
        self,
        history: Dict[str, List[float]],
        save: Optional[str] = None,
    ) -> plt.Figure:
        """
        Eğitim geçmişi grafiği.
        
        Args:
            history: {'train_loss': [...], 'physics_loss': [...], ...}
            save: Kayıt dosya adı
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=self.dpi)
        
        # Loss
        epochs = history.get('epochs', range(len(history['train_loss'])))
        axes[0].plot(epochs, history['train_loss'], 'b-', label='Total Loss')
        if 'physics_loss' in history:
            axes[0].plot(epochs, history['physics_loss'], 'r--', label='Physics Loss')
        if 'data_loss' in history:
            axes[0].plot(epochs, history['data_loss'], 'g:', label='Data Loss')
        
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_yscale('log')
        axes[0].legend()
        axes[0].set_title('Training Loss')
        axes[0].grid(True, alpha=0.3)
        
        # Learning rate
        if 'learning_rate' in history:
            axes[1].plot(epochs, history['learning_rate'], 'purple')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Learning Rate')
            axes[1].set_yscale('log')
            axes[1].set_title('Learning Rate Schedule')
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            fig.savefig(self.results_dir / save, bbox_inches='tight', dpi=self.dpi)
        
        return fig
    
    def plot_energy_spectrum(
        self,
        k: np.ndarray,
        E_k: np.ndarray,
        reference_k: Optional[np.ndarray] = None,
        reference_E_k: Optional[np.ndarray] = None,
        show_kolmogorov: bool = True,
        title: str = 'Enerji Spektrumu',
        save: Optional[str] = None,
    ) -> plt.Figure:
        """
        Enerji spektrumu grafiği.
        
        Args:
            k: Wavenumber dizisi
            E_k: Enerji spektrumu
            reference_k, reference_E_k: Referans spektrum
            show_kolmogorov: -5/3 eğimi göster
            title: Başlık
            save: Kayıt dosya adı
        """
        fig, ax = plt.subplots(figsize=(8, 6), dpi=self.dpi)
        
        ax.loglog(k, E_k, 'b-', linewidth=2, label='INNATE')
        
        if reference_E_k is not None:
            ref_k = reference_k if reference_k is not None else k
            ax.loglog(ref_k, reference_E_k, 'r--', linewidth=2, label='Reference')
        
        if show_kolmogorov:
            # -5/3 slope reference line
            k_ref = k[len(k)//4:len(k)//2]
            E_ref = E_k[len(k)//4] * (k_ref / k_ref[0])**(-5/3)
            ax.loglog(k_ref, E_ref, 'k:', linewidth=1.5, label='k^{-5/3}')
        
        ax.set_xlabel('Wavenumber k')
        ax.set_ylabel('E(k)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3, which='both')
        
        if save:
            fig.savefig(self.results_dir / save, bbox_inches='tight', dpi=self.dpi)
        
        return fig
    
    def create_animation(
        self,
        omega_sequence: Union[List, np.ndarray],
        times: Optional[np.ndarray] = None,
        reference_sequence: Optional[Union[List, np.ndarray]] = None,
        fps: int = 30,
        title: str = 'Vortisite Evrimi',
        save: str = 'vorticity_animation.gif',
    ) -> animation.FuncAnimation:
        """
        Vortisite animasyonu oluştur.
        
        Args:
            omega_sequence: Vortisite zaman serisi [Nt, Nx, Ny]
            times: Zaman dizisi
            reference_sequence: Referans zaman serisi (karşılaştırma için)
            fps: Frame rate
            title: Başlık
            save: Kayıt dosya adı
        """
        # Numpy'a çevir
        if isinstance(omega_sequence, list):
            omega_sequence = np.array([self._to_numpy(o) for o in omega_sequence])
        else:
            omega_sequence = self._to_numpy(omega_sequence)
        
        n_frames = len(omega_sequence)
        times = times if times is not None else np.arange(n_frames)
        
        # Ortak renk skalası
        vmax = np.max(np.abs(omega_sequence))
        
        if reference_sequence is not None:
            if isinstance(reference_sequence, list):
                reference_sequence = np.array([self._to_numpy(o) for o in reference_sequence])
            else:
                reference_sequence = self._to_numpy(reference_sequence)
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=100)
            vmax = max(vmax, np.max(np.abs(reference_sequence)))
        else:
            fig, ax = plt.subplots(figsize=(6, 5), dpi=100)
            axes = [ax]
        
        # İlk frame
        ims = []
        for i, ax in enumerate(axes if reference_sequence is not None else [axes[0]]):
            if i == 0:
                data = omega_sequence[0]
                label = 'INNATE'
            else:
                data = reference_sequence[0]
                label = 'Reference'
            
            im = ax.imshow(
                data.T, origin='lower', cmap=self.cmap,
                vmin=-vmax, vmax=vmax, extent=[0, 2*np.pi, 0, 2*np.pi],
                animated=True
            )
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title(f'{label} @ t = {times[0]:.2f}')
            ax.set_aspect('equal')
            plt.colorbar(im, ax=ax)
            ims.append(im)
        
        fig.suptitle(title)
        plt.tight_layout()
        
        def update(frame):
            for i, im in enumerate(ims):
                if i == 0:
                    data = omega_sequence[frame]
                else:
                    data = reference_sequence[frame]
                
                im.set_array(data.T)
                im.axes.set_title(
                    f'{"INNATE" if i == 0 else "Reference"} @ t = {times[frame]:.2f}'
                )
            return ims
        
        anim = animation.FuncAnimation(
            fig, update, frames=n_frames,
            interval=1000/fps, blit=True
        )
        
        # Kaydet
        if save:
            save_path = self.results_dir / save
            if save.endswith('.gif'):
                anim.save(save_path, writer='pillow', fps=fps)
            elif save.endswith('.mp4'):
                anim.save(save_path, writer='ffmpeg', fps=fps)
            print(f"Animasyon kaydedildi: {save_path}")
        
        return anim
    
    def create_summary_figure(
        self,
        innate_states: List,
        reference_states: Optional[List] = None,
        times: Optional[np.ndarray] = None,
        metrics: Optional[Dict] = None,
        save: str = 'summary.png',
    ) -> plt.Figure:
        """
        Özet figür oluştur (tüm sonuçlar tek sayfada).
        
        Args:
            innate_states: INNATE durumları listesi
            reference_states: Referans durumları
            times: Zaman dizisi
            metrics: Metrikler
            save: Kayıt dosya adı
        """
        n_snapshots = min(4, len(innate_states))
        indices = np.linspace(0, len(innate_states) - 1, n_snapshots, dtype=int)
        
        fig = plt.figure(figsize=(16, 12), dpi=self.dpi)
        
        # Layout: 3 rows
        # Row 1: INNATE snapshots
        # Row 2: Reference snapshots (varsa) veya velocity field
        # Row 3: Metrics (energy decay, error, spectrum)
        
        if reference_states is not None:
            n_rows = 3
        else:
            n_rows = 2
        
        # Row 1: INNATE vortisite
        for i, idx in enumerate(indices):
            ax = fig.add_subplot(n_rows, n_snapshots, i + 1)
            state = innate_states[idx]
            omega = self._to_numpy(state.vorticity if hasattr(state, 'vorticity') else state)
            t = times[idx] if times is not None else idx
            
            vmax = np.max(np.abs(omega))
            ax.imshow(
                omega.T, origin='lower', cmap=self.cmap,
                vmin=-vmax, vmax=vmax, extent=[0, 2*np.pi, 0, 2*np.pi]
            )
            ax.set_title(f'INNATE t={t:.2f}')
            ax.set_aspect('equal')
        
        # Row 2: Reference veya velocity
        if reference_states is not None:
            for i, idx in enumerate(indices):
                ax = fig.add_subplot(n_rows, n_snapshots, n_snapshots + i + 1)
                omega = self._to_numpy(reference_states[idx])
                t = times[idx] if times is not None else idx
                
                vmax = np.max(np.abs(omega))
                ax.imshow(
                    omega.T, origin='lower', cmap=self.cmap,
                    vmin=-vmax, vmax=vmax, extent=[0, 2*np.pi, 0, 2*np.pi]
                )
                ax.set_title(f'Reference t={t:.2f}')
                ax.set_aspect('equal')
        
        # Row 3: Metrics
        if metrics is not None:
            # Energy
            ax_energy = fig.add_subplot(n_rows, 3, n_rows * 3 - 2)
            if 'times' in metrics and 'energies' in metrics:
                ax_energy.plot(metrics['times'], metrics['energies'], 'b-')
                if 'reference_energies' in metrics:
                    ax_energy.plot(metrics['times'], metrics['reference_energies'], 'r--')
            ax_energy.set_xlabel('Time')
            ax_energy.set_ylabel('Energy')
            ax_energy.set_title('Energy Decay')
            
            # Error
            ax_error = fig.add_subplot(n_rows, 3, n_rows * 3 - 1)
            if 'times' in metrics and 'l2_errors' in metrics:
                ax_error.semilogy(metrics['times'], metrics['l2_errors'], 'b-')
            ax_error.set_xlabel('Time')
            ax_error.set_ylabel('L2 Error')
            ax_error.set_title('Error Evolution')
            
            # Summary text
            ax_text = fig.add_subplot(n_rows, 3, n_rows * 3)
            ax_text.axis('off')
            summary_text = '\n'.join([
                f'{k}: {v:.4e}' if isinstance(v, float) else f'{k}: {v}'
                for k, v in metrics.items()
                if not isinstance(v, (list, np.ndarray))
            ])
            ax_text.text(0.1, 0.5, summary_text, fontsize=10, 
                        verticalalignment='center', family='monospace')
        
        plt.tight_layout()
        
        if save:
            fig.savefig(self.results_dir / save, bbox_inches='tight', dpi=self.dpi)
        
        return fig


if __name__ == "__main__":
    print("Visualizer modülü yüklendi.")
    
    # Basit test
    viz = FlowVisualizer(results_dir='test_results')
    
    # Test veri
    N = 64
    x = np.linspace(0, 2*np.pi, N)
    y = np.linspace(0, 2*np.pi, N)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    omega = np.sin(X) * np.sin(Y)
    
    fig = viz.plot_vorticity(omega, t=0.0)
    plt.close(fig)
    
    print("Test başarılı!")

