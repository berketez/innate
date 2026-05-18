"""
Benchmark Trainer Base Sınıfı

Tüm benchmark testleri için ortak eğitim altyapısı.

Kullanım:
    from tests.common.trainer_base import BenchmarkTrainer, ValidationMetrics
    
    trainer = BenchmarkTrainer(
        model=model,
        dns_reference=dns_result,
        config=TrainerConfig(...)
    )
    
    history = trainer.train(num_epochs=1000)
    metrics = trainer.validate()
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable, Any
from pathlib import Path
import json
import time

from .device import DEVICE


@dataclass
class TrainerConfig:
    """Eğitim konfigürasyonu."""
    # Temel ayarlar
    num_epochs: int = 1000
    batch_size: int = 1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    
    # Scheduler
    scheduler_type: str = 'cosine'  # 'cosine', 'plateau', 'none'
    scheduler_patience: int = 50
    
    # Curriculum learning
    use_curriculum: bool = True
    curriculum_stages: List[int] = field(default_factory=lambda: [10, 25, 50, 100])
    
    # Kayıt
    save_every: int = 100
    log_every: int = 10
    results_dir: str = 'results'
    
    # Erken durdurma
    early_stopping: bool = True
    early_stopping_patience: int = 200
    early_stopping_min_delta: float = 1e-6
    
    # Fiziksel kısıtlar
    physics_weight: float = 1.0
    data_weight: float = 1.0
    divergence_weight: float = 10.0


@dataclass
class ValidationMetrics:
    """Validasyon metrikleri."""
    l2_error: float
    linf_error: float
    energy_error: float
    enstrophy_error: float
    divergence: float
    correlation: float
    
    def is_success(self, 
                   l2_threshold: float = 0.05,
                   energy_threshold: float = 0.10,
                   divergence_threshold: float = 1e-4) -> bool:
        """Başarı kriterleri."""
        return (
            self.l2_error < l2_threshold and
            self.energy_error < energy_threshold and
            self.divergence < divergence_threshold
        )
    
    def to_dict(self) -> Dict[str, float]:
        """Dict'e çevir."""
        return {
            'l2_error': self.l2_error,
            'linf_error': self.linf_error,
            'energy_error': self.energy_error,
            'enstrophy_error': self.enstrophy_error,
            'divergence': self.divergence,
            'correlation': self.correlation,
        }
    
    def __str__(self) -> str:
        status = "✓" if self.is_success() else "✗"
        return (
            f"ValidationMetrics {status}:\n"
            f"  L2 Error:      {self.l2_error:.4e}\n"
            f"  L∞ Error:      {self.linf_error:.4e}\n"
            f"  Energy Error:  {self.energy_error:.2%}\n"
            f"  Enstrophy Err: {self.enstrophy_error:.2%}\n"
            f"  Divergence:    {self.divergence:.4e}\n"
            f"  Correlation:   {self.correlation:.4f}"
        )


@dataclass
class TrainingHistory:
    """Eğitim geçmişi."""
    epochs: List[int] = field(default_factory=list)
    train_loss: List[float] = field(default_factory=list)
    physics_loss: List[float] = field(default_factory=list)
    data_loss: List[float] = field(default_factory=list)
    divergence_loss: List[float] = field(default_factory=list)
    learning_rate: List[float] = field(default_factory=list)
    validation_metrics: List[Dict] = field(default_factory=list)
    
    def save(self, path: str) -> None:
        """JSON olarak kaydet."""
        with open(path, 'w') as f:
            json.dump({
                'epochs': self.epochs,
                'train_loss': self.train_loss,
                'physics_loss': self.physics_loss,
                'data_loss': self.data_loss,
                'divergence_loss': self.divergence_loss,
                'learning_rate': self.learning_rate,
                'validation_metrics': self.validation_metrics,
            }, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'TrainingHistory':
        """JSON'dan yükle."""
        with open(path, 'r') as f:
            data = json.load(f)
        history = cls()
        for key, value in data.items():
            setattr(history, key, value)
        return history


class BenchmarkTrainer:
    """
    Benchmark Trainer
    
    INNATE modeli için ortak eğitim altyapısı.
    DNS referans verisi ile karşılaştırmalı eğitim yapar.
    """
    
    def __init__(
        self,
        model: nn.Module,
        dns_reference: Optional[Dict] = None,
        config: Optional[TrainerConfig] = None,
        initial_condition_fn: Optional[Callable] = None,
        analytical_solution_fn: Optional[Callable] = None,
    ):
        """
        Args:
            model: INNATE modeli
            dns_reference: DNS referans verisi (opsiyonel)
            config: Eğitim konfigürasyonu
            initial_condition_fn: Başlangıç koşulu fonksiyonu
            analytical_solution_fn: Analitik çözüm fonksiyonu (varsa)
        """
        self.model = model.to(DEVICE)
        self.dns_reference = dns_reference
        self.config = config or TrainerConfig()
        self.initial_condition_fn = initial_condition_fn
        self.analytical_solution_fn = analytical_solution_fn
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Scheduler
        if self.config.scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer, T_max=self.config.num_epochs
            )
        elif self.config.scheduler_type == 'plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, mode='min', 
                patience=self.config.scheduler_patience,
                factor=0.5
            )
        else:
            self.scheduler = None
        
        # History
        self.history = TrainingHistory()
        
        # Results directory
        self.results_dir = Path(self.config.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Erken durdurma
        self.best_loss = float('inf')
        self.patience_counter = 0
    
    def create_initial_state(self, batch_size: int = 1):
        """Başlangıç durumu oluştur."""
        if self.initial_condition_fn is not None:
            return self.initial_condition_fn(batch_size, DEVICE)
        else:
            raise NotImplementedError("initial_condition_fn tanımlanmalı")
    
    def compute_data_loss(self, predicted, target) -> torch.Tensor:
        """Data loss hesapla (MSE)."""
        if target is None:
            return torch.tensor(0.0, device=DEVICE)
        
        loss = 0.0
        if hasattr(predicted, 'u') and 'u' in target:
            loss += torch.mean((predicted.u - target['u'])**2)
        if hasattr(predicted, 'v') and 'v' in target:
            loss += torch.mean((predicted.v - target['v'])**2)
        
        return loss
    
    def compute_physics_loss(self, state) -> torch.Tensor:
        """Fiziksel kısıt loss'u."""
        # Divergence loss
        if hasattr(self.model, 'projector'):
            div_loss = self.model.projector.divergence_error(state.u, state.v)
        else:
            div_loss = torch.tensor(0.0, device=DEVICE)
        
        return div_loss
    
    def train_step(self, num_steps: int, target_data: Optional[Dict] = None) -> Dict[str, float]:
        """Tek eğitim adımı."""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Başlangıç durumu
        initial_state = self.create_initial_state()
        
        # Forward pass
        states = self.model(initial_state, num_steps=num_steps)
        final_state = states[-1]
        
        # Loss hesapla
        data_loss = self.compute_data_loss(final_state, target_data)
        physics_loss = self.compute_physics_loss(final_state)
        
        total_loss = (
            self.config.data_weight * data_loss + 
            self.config.divergence_weight * physics_loss
        )
        
        # Backward
        total_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()

        # Energy ve enstrophy hesapla
        with torch.no_grad():
            energy = final_state.kinetic_energy().mean().item()
            enstrophy = final_state.enstrophy().mean().item()

        return {
            'total_loss': total_loss.item(),
            'data_loss': data_loss.item() if isinstance(data_loss, torch.Tensor) else data_loss,
            'physics_loss': physics_loss.item(),
            'energy': energy,
            'enstrophy': enstrophy,
        }
    
    def train(self, num_epochs: Optional[int] = None) -> TrainingHistory:
        """
        Model eğit.
        
        Args:
            num_epochs: Epoch sayısı (None ise config'den al)
        
        Returns:
            TrainingHistory: Eğitim geçmişi
        """
        num_epochs = num_epochs or self.config.num_epochs
        
        # Curriculum stages
        if self.config.use_curriculum:
            stages = self.config.curriculum_stages
            epochs_per_stage = num_epochs // len(stages)
        else:
            stages = [self.config.curriculum_stages[-1]]
            epochs_per_stage = num_epochs
        
        print(f"Eğitim başlıyor...")
        print(f"  Device: {DEVICE}")
        print(f"  Epochs: {num_epochs}")
        print(f"  Curriculum: {stages if self.config.use_curriculum else 'Disabled'}")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            # Curriculum: mevcut aşama
            stage_idx = min(epoch // epochs_per_stage, len(stages) - 1)
            num_steps = stages[stage_idx]
            
            # Eğitim adımı
            losses = self.train_step(num_steps)
            
            # Scheduler güncelle
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(losses['total_loss'])
                else:
                    self.scheduler.step()
            
            # History güncelle
            self.history.epochs.append(epoch)
            self.history.train_loss.append(losses['total_loss'])
            self.history.physics_loss.append(losses['physics_loss'])
            self.history.data_loss.append(losses['data_loss'])
            self.history.learning_rate.append(self.optimizer.param_groups[0]['lr'])
            
            # Log
            if epoch % self.config.log_every == 0:
                elapsed = time.time() - start_time
                print(f"Epoch {epoch:4d}/{num_epochs} | "
                      f"Loss: {losses['total_loss']:.4e} | "
                      f"E: {losses['energy']:.4e} | "
                      f"Z: {losses['enstrophy']:.4e} | "
                      f"Steps: {num_steps} | "
                      f"Time: {elapsed:.1f}s")
            
            # Kaydet
            if epoch % self.config.save_every == 0 and epoch > 0:
                self._save_checkpoint(epoch)
            
            # Erken durdurma
            if self.config.early_stopping:
                if losses['total_loss'] < self.best_loss - self.config.early_stopping_min_delta:
                    self.best_loss = losses['total_loss']
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                
                if self.patience_counter >= self.config.early_stopping_patience:
                    print(f"Erken durdurma: {epoch} epoch'ta")
                    break
        
        # Final kayıt
        self._save_checkpoint(epoch, final=True)
        self.history.save(self.results_dir / 'training_history.json')
        
        total_time = time.time() - start_time
        print(f"Eğitim tamamlandı: {total_time:.1f}s")
        
        return self.history
    
    def validate(self, num_steps: int = 100) -> ValidationMetrics:
        """
        Model validasyonu.
        
        Args:
            num_steps: Simülasyon adım sayısı
        
        Returns:
            ValidationMetrics: Validasyon sonuçları
        """
        self.model.eval()
        
        with torch.no_grad():
            # Başlangıç durumu
            initial_state = self.create_initial_state()
            
            # Forward
            states = self.model(initial_state, num_steps=num_steps)
            
            # Referans al
            if self.analytical_solution_fn is not None:
                # Analitik çözüm
                ref_u, ref_v = self.analytical_solution_fn(
                    states[-1].t.item() if hasattr(states[-1], 't') else num_steps * 0.01
                )
                ref_u = torch.tensor(ref_u, device=DEVICE).unsqueeze(0)
                ref_v = torch.tensor(ref_v, device=DEVICE).unsqueeze(0)
            elif self.dns_reference is not None:
                # DNS referans
                ref_u = torch.tensor(self.dns_reference['u'][-1], device=DEVICE).unsqueeze(0)
                ref_v = torch.tensor(self.dns_reference['v'][-1], device=DEVICE).unsqueeze(0)
            else:
                ref_u = None
                ref_v = None
            
            # Metrikler
            final_state = states[-1]
            
            if ref_u is not None:
                l2_error = torch.sqrt(
                    torch.mean((final_state.u - ref_u)**2 + (final_state.v - ref_v)**2)
                ).item()
                
                linf_error = torch.max(
                    torch.abs(final_state.u - ref_u).max(),
                    torch.abs(final_state.v - ref_v).max()
                ).item()
                
                # Korelasyon
                pred_flat = torch.cat([final_state.u.flatten(), final_state.v.flatten()])
                ref_flat = torch.cat([ref_u.flatten(), ref_v.flatten()])
                correlation = torch.corrcoef(
                    torch.stack([pred_flat, ref_flat])
                )[0, 1].item()
            else:
                l2_error = 0.0
                linf_error = 0.0
                correlation = 1.0
            
            # Enerji
            pred_energy = 0.5 * torch.mean(final_state.u**2 + final_state.v**2).item()
            init_energy = 0.5 * torch.mean(states[0].u**2 + states[0].v**2).item()
            
            if self.dns_reference is not None:
                ref_energy = self.dns_reference['energy'][-1]
                energy_error = abs(pred_energy - ref_energy) / (ref_energy + 1e-10)
            else:
                # Analitik decay ile karşılaştır
                energy_error = abs(pred_energy - init_energy) / (init_energy + 1e-10)
            
            # Enstrofi
            if hasattr(final_state, 'vorticity') and final_state.vorticity is not None:
                pred_enstrophy = 0.5 * torch.mean(final_state.vorticity**2).item()
                init_enstrophy = 0.5 * torch.mean(states[0].vorticity**2).item()
                enstrophy_error = abs(pred_enstrophy - init_enstrophy) / (init_enstrophy + 1e-10)
            else:
                enstrophy_error = 0.0
            
            # Divergence
            if hasattr(self.model, 'projector'):
                divergence = self.model.projector.divergence_error(
                    final_state.u, final_state.v
                ).item()
            else:
                divergence = 0.0
        
        return ValidationMetrics(
            l2_error=l2_error,
            linf_error=linf_error,
            energy_error=energy_error,
            enstrophy_error=enstrophy_error,
            divergence=divergence,
            correlation=correlation,
        )
    
    def _save_checkpoint(self, epoch: int, final: bool = False) -> None:
        """Checkpoint kaydet."""
        suffix = 'final' if final else f'epoch_{epoch}'
        path = self.results_dir / f'checkpoint_{suffix}.pt'
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
        }, path)


class MetricsCalculator:
    """
    Fiziksel metrikler hesaplayıcı.
    
    Spektral analiz, enerji bütçesi, korunum yasaları.
    """
    
    @staticmethod
    def kinetic_energy(u: torch.Tensor, v: torch.Tensor) -> float:
        """Kinetik enerji: E = 0.5 * <u² + v²>"""
        return 0.5 * torch.mean(u**2 + v**2).item()
    
    @staticmethod
    def enstrophy(omega: torch.Tensor) -> float:
        """Enstrofi: Ω = 0.5 * <ω²>"""
        return 0.5 * torch.mean(omega**2).item()
    
    @staticmethod
    def energy_spectrum(u: torch.Tensor, v: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        1D enerji spektrumu hesapla.
        
        Returns:
            (k, E_k): Wavenumber ve enerji spektrumu
        """
        u_np = u.squeeze().cpu().numpy()
        v_np = v.squeeze().cpu().numpy()
        
        N = u_np.shape[0]
        
        # FFT
        u_hat = np.fft.fft2(u_np)
        v_hat = np.fft.fft2(v_np)
        
        # Enerji yoğunluğu
        E_hat = 0.5 * (np.abs(u_hat)**2 + np.abs(v_hat)**2) / N**4
        
        # Radyal ortalamalama
        kx = np.fft.fftfreq(N, d=1/N)
        ky = np.fft.fftfreq(N, d=1/N)
        KX, KY = np.meshgrid(kx, ky, indexing='ij')
        K = np.sqrt(KX**2 + KY**2)
        
        # Binning
        k_max = N // 2
        k_bins = np.arange(0.5, k_max + 0.5, 1)
        E_k = np.zeros(len(k_bins) - 1)
        
        for i in range(len(k_bins) - 1):
            mask = (K >= k_bins[i]) & (K < k_bins[i + 1])
            E_k[i] = np.sum(E_hat[mask])
        
        k = np.arange(1, k_max)
        
        return k, E_k
    
    @staticmethod
    def l2_error(pred: torch.Tensor, ref: torch.Tensor) -> float:
        """L2 hata normu."""
        return torch.sqrt(torch.mean((pred - ref)**2)).item()
    
    @staticmethod
    def relative_l2_error(pred: torch.Tensor, ref: torch.Tensor) -> float:
        """Göreli L2 hata."""
        return torch.sqrt(torch.mean((pred - ref)**2) / torch.mean(ref**2)).item()


if __name__ == "__main__":
    print("Trainer Base modülü yüklendi.")
    print(f"Device: {DEVICE}")

