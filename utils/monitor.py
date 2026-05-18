"""
Physics Monitor - Fizik İzleme ve Diagnostik Araçları

Bu modül eğitim sırasında fiziksel büyüklükleri izlemek için
hook-tabanlı bir monitor sağlar.

Kullanım:
    from utils import PhysicsMonitor

    model = INNATE(64)
    monitor = PhysicsMonitor(model)

    # Eğitim döngüsünde
    for epoch in range(epochs):
        states = model(initial_state, num_steps=100)
        monitor.log(states[-1], epoch)

    # Sonuçları görselleştir
    monitor.plot()
    monitor.save('physics_history.json')
"""

import torch
import json
import math
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field, asdict
from collections import defaultdict


@dataclass
class PhysicsSnapshot:
    """Tek bir zaman adımındaki fiziksel büyüklükler."""
    epoch: int
    step: int
    time: float
    energy: float
    enstrophy: float
    divergence: float
    vorticity_max: float
    vorticity_mean: float
    cfl: float
    reynolds: float

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class PhysicsHistory:
    """Tüm eğitim boyunca fiziksel büyüklüklerin geçmişi."""
    snapshots: List[PhysicsSnapshot] = field(default_factory=list)

    def add(self, snapshot: PhysicsSnapshot):
        self.snapshots.append(snapshot)

    def get_series(self, key: str) -> List[float]:
        """Belirli bir metriğin zaman serisi."""
        return [getattr(s, key) for s in self.snapshots]

    def to_dict(self) -> Dict:
        return {'snapshots': [s.to_dict() for s in self.snapshots]}

    def save(self, filepath: str):
        """JSON olarak kaydet."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> 'PhysicsHistory':
        """JSON'dan yükle."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        history = cls()
        for s in data['snapshots']:
            history.add(PhysicsSnapshot(**s))
        return history


class PhysicsMonitor:
    """
    Fizik izleme sınıfı.

    PyTorch hooks kullanarak model çıktılarını izler ve
    fiziksel büyüklükleri kaydeder.

    Args:
        model: INNATE modeli
        log_every: Her kaç epoch'ta bir log al
        warn_thresholds: Uyarı eşikleri

    Örnek:
        monitor = PhysicsMonitor(model, log_every=10)

        for epoch in range(1000):
            states = model(state, num_steps=100)
            warnings = monitor.log(states[-1], epoch)

            if warnings:
                print(f"Warnings: {warnings}")

        monitor.plot()
    """

    def __init__(
        self,
        model,
        log_every: int = 1,
        warn_thresholds: Optional[Dict[str, float]] = None
    ):
        self.model = model
        self.log_every = log_every
        self.history = PhysicsHistory()
        self.current_epoch = 0
        self.current_step = 0

        # Varsayılan uyarı eşikleri
        self.warn_thresholds = warn_thresholds or {
            'divergence': 1e-3,      # Divergence çok yüksek
            'energy_growth': 1.5,     # Enerji %50'den fazla arttı
            'cfl': 1.0,              # CFL > 1 instabil
            'vorticity_max': 100.0,  # Vortisite patladı
        }

        self._last_energy = None
        self._hooks = []

    def log(self, state, epoch: int, step: int = 0) -> List[str]:
        """
        Mevcut state'i logla ve uyarıları döndür.

        Args:
            state: FluidState
            epoch: Epoch numarası
            step: Simülasyon step numarası

        Returns:
            Uyarı mesajları listesi
        """
        self.current_epoch = epoch
        self.current_step = step

        if epoch % self.log_every != 0:
            return []

        # Fiziksel büyüklükleri hesapla
        with torch.no_grad():
            diagnostics = self.model.get_diagnostics(state)

        # Snapshot oluştur
        snapshot = PhysicsSnapshot(
            epoch=epoch,
            step=step,
            time=state.t.mean().item() if hasattr(state.t, 'mean') else float(state.t),
            energy=diagnostics['energy'],
            enstrophy=diagnostics['enstrophy'],
            divergence=diagnostics['divergence'],
            vorticity_max=state.vorticity.abs().max().item(),
            vorticity_mean=diagnostics['vorticity_mag'],
            cfl=diagnostics['cfl'],
            reynolds=diagnostics['reynolds']
        )

        self.history.add(snapshot)

        # Uyarıları kontrol et
        warnings = self._check_warnings(snapshot)

        self._last_energy = snapshot.energy
        return warnings

    def _check_warnings(self, snapshot: PhysicsSnapshot) -> List[str]:
        """Fiziksel uyarıları kontrol et."""
        warnings = []

        # Divergence kontrolü
        if snapshot.divergence > self.warn_thresholds['divergence']:
            warnings.append(
                f"HIGH DIVERGENCE: {snapshot.divergence:.2e} > {self.warn_thresholds['divergence']:.2e}"
            )

        # CFL kontrolü
        if snapshot.cfl > self.warn_thresholds['cfl']:
            warnings.append(
                f"CFL VIOLATION: {snapshot.cfl:.2f} > {self.warn_thresholds['cfl']:.1f}"
            )

        # Enerji büyümesi kontrolü
        if self._last_energy is not None and self._last_energy > 0:
            growth = snapshot.energy / self._last_energy
            if growth > self.warn_thresholds['energy_growth']:
                warnings.append(
                    f"ENERGY EXPLOSION: {growth:.2f}x growth in one epoch"
                )

        # Vortisite patlaması kontrolü
        if snapshot.vorticity_max > self.warn_thresholds['vorticity_max']:
            warnings.append(
                f"VORTICITY EXPLOSION: max={snapshot.vorticity_max:.2f}"
            )

        return warnings

    def get_summary(self) -> Dict[str, float]:
        """Özet istatistikler."""
        if not self.history.snapshots:
            return {}

        energies = self.history.get_series('energy')
        enstrophies = self.history.get_series('enstrophy')
        divergences = self.history.get_series('divergence')

        return {
            'energy_initial': energies[0] if energies else 0,
            'energy_final': energies[-1] if energies else 0,
            'energy_decay': (energies[0] - energies[-1]) / energies[0] if energies and energies[0] > 0 else 0,
            'enstrophy_peak': max(enstrophies) if enstrophies else 0,
            'divergence_mean': sum(divergences) / len(divergences) if divergences else 0,
            'divergence_max': max(divergences) if divergences else 0,
            'num_snapshots': len(self.history.snapshots),
        }

    def print_summary(self):
        """Özeti yazdır."""
        summary = self.get_summary()
        print("\n" + "=" * 50)
        print("PHYSICS MONITOR SUMMARY")
        print("=" * 50)
        for key, value in summary.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.6f}")
            else:
                print(f"  {key}: {value}")
        print("=" * 50)

    def plot(self, save_path: Optional[str] = None):
        """
        Fiziksel büyüklükleri görselleştir.

        Args:
            save_path: Kaydedilecek dosya yolu (None ise göster)
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not found. Install with: pip install matplotlib")
            return

        if not self.history.snapshots:
            print("No data to plot")
            return

        epochs = self.history.get_series('epoch')
        energy = self.history.get_series('energy')
        enstrophy = self.history.get_series('enstrophy')
        divergence = self.history.get_series('divergence')
        cfl = self.history.get_series('cfl')

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Energy
        axes[0, 0].semilogy(epochs, energy, 'b-', label='Energy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Kinetic Energy')
        axes[0, 0].set_title('Energy Evolution')
        axes[0, 0].grid(True, alpha=0.3)

        # Enstrophy
        axes[0, 1].semilogy(epochs, enstrophy, 'r-', label='Enstrophy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Enstrophy')
        axes[0, 1].set_title('Enstrophy Evolution')
        axes[0, 1].grid(True, alpha=0.3)

        # Divergence
        axes[1, 0].semilogy(epochs, divergence, 'g-', label='Divergence')
        axes[1, 0].axhline(y=self.warn_thresholds['divergence'], color='r',
                          linestyle='--', label='Threshold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('|∇·u|')
        axes[1, 0].set_title('Divergence (should be ~0)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # CFL
        axes[1, 1].plot(epochs, cfl, 'm-', label='CFL')
        axes[1, 1].axhline(y=1.0, color='r', linestyle='--', label='Stability limit')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('CFL Number')
        axes[1, 1].set_title('CFL Number (should be < 1)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Saved to {save_path}")
        else:
            plt.show()

    def save(self, filepath: str):
        """Geçmişi JSON olarak kaydet."""
        self.history.save(filepath)
        print(f"Physics history saved to {filepath}")

    def load(self, filepath: str):
        """Geçmişi JSON'dan yükle."""
        self.history = PhysicsHistory.load(filepath)
        print(f"Physics history loaded from {filepath}")

    def register_custom_metric(
        self,
        name: str,
        compute_fn: Callable,
        warn_threshold: Optional[float] = None
    ):
        """
        Özel metrik ekle.

        Args:
            name: Metrik adı
            compute_fn: state -> float hesaplama fonksiyonu
            warn_threshold: Uyarı eşiği

        Örnek:
            monitor.register_custom_metric(
                'helicity',
                lambda state: compute_helicity(state),
                warn_threshold=0.1
            )
        """
        # TODO: Implement custom metrics
        pass
