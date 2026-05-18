"""Bitirme2 konfigurasyonu -- nested dataclass yapisi."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import torch


# ── Domain ──────────────────────────────────────────────────────────────────

@dataclass
class DomainConfig:
    Lx: float = 6.0
    Ly: float = 10.0
    Lz: float = 4.0
    Nx: int = 96
    Ny: int = 160
    Nz: int = 64

    def __post_init__(self):
        for name in ("Nx", "Ny", "Nz"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0, got {getattr(self, name)}")

    @property
    def dx(self) -> float:
        return self.Lx / self.Nx

    @property
    def dy(self) -> float:
        return self.Ly / self.Ny

    @property
    def dz(self) -> float:
        return self.Lz / self.Nz

    @property
    def dx_min(self) -> float:
        return min(self.dx, self.dy, self.dz)

    @property
    def grid_spacings(self) -> tuple:
        return (self.dx, self.dy, self.dz)


# ── Physics ─────────────────────────────────────────────────────────────────

@dataclass
class PhysicsConfig:
    Re: float = 5000.0
    Ra: float = 1e5
    Pr: float = 0.71
    dt: float = 0.020
    # Sıcaklık konvansiyonu (KRİTİK — y-yön anlamı):
    #   T_hot       y = 0    (alt sınır, "taban")
    #   T_cold      y = Ly   (üst sınır, "tavan")
    # Yani sıcak alttan, soğuk yukarıdan → bu konfigürasyon
    # **stable stratification** (kararlı katmanlanma) DEĞİL,
    # **unstable stratification (RB-benzeri)** demek değildir;
    # forcing-driven shear flow + lineer baz profil var:
    #   T_base(y) = T_hot - (T_hot - T_cold) * y/Ly
    #   ∂T_base/∂y = -(T_hot - T_cold)/Ly = -ΔT/Ly  (negatif gradient)
    # Energy denkleminde `+v/Ly` source terimi bu konvansiyonla tutarlı:
    #   sıcak parçacık (θ>0) yukarı çıkarken (v>0) daha soğuk baz katmana
    #   girer → θ artar → +v·(ΔT/Ly)/ΔT = +v/Ly (boyutsuz) ✓
    # Buoyancy formu Ri·θ·ê_y ile birlikte: θ>0 (sıcak) yukarı kuvvet alır.
    # LES referans (les_solver.py) aynı konvansiyonu kullanır.
    T_hot: float = 20.0           # y = 0 (alt sınır)
    T_cold: float = 0.0           # y = Ly (üst sınır)
    forcing_mode: str = "kolmogorov"  # kolmogorov | uniform | stochastic
    k_f: int = 4
    non_boussinesq: bool = False      # True: Faz-2 yogunluk etkileri aktif
    T_ref_kelvin: float = 293.15      # 20 degC in Kelvin (referans sicaklik)

    def __post_init__(self):
        for name in ("Re", "Ra", "Pr"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0, got {getattr(self, name)}")
        if self.forcing_mode not in ("kolmogorov", "uniform", "stochastic"):
            raise ValueError(f"Invalid forcing_mode: {self.forcing_mode}")

    # -- computed (cache'lenmez, Re degisince otomatik guncellenir) --

    @property
    def nu(self) -> float:
        return 1.0 / self.Re

    @property
    def kappa(self) -> float:
        return 1.0 / (self.Re * self.Pr)

    @property
    def Ri(self) -> float:
        return self.Ra / (self.Re ** 2 * self.Pr)

    @property
    def dT(self) -> float:
        return self.T_hot - self.T_cold


# ── Model ───────────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    n_layers: int = 20
    use_eddy_viscosity: bool = True
    gradient_checkpointing: bool = False  # 96x160x64 grid 4090 16GB'ye sigar, checkpointing gereksiz
    # Tier 1 (Kritik)
    use_turbulent_prandtl: bool = True
    use_anisotropic_sgs: bool = True
    # Tier 2
    use_thermal_adv_modulator: bool = True
    use_per_layer_buoyancy: bool = True
    use_per_layer_dt: bool = True
    # Tier 3
    use_anisotropic_kappa: bool = True
    use_backscatter: bool = True        # init=0, ogrenilebilir
    use_forcing_harmonics: bool = True   # init=0, ogrenilebilir
    # v2: MLP SGS + IMEX + Scale-Similarity
    use_mlp_sgs: bool = True             # MLP: (|S|,|Omega|,R,Ri_g,Re,layer) -> (Cs,kappa)
    mlp_hidden_dim: int = 32             # MLP hidden layer boyutu
    use_scale_similarity: bool = True    # Mixed model: Smagorinsky + scale-similarity
    use_imex: bool = True                # IMEX: implicit mol. diffusion, explicit SGS
    # v3 (2026-05-09): Saf-INNATE Spectral-Cs (MLP YOK, Fourier mode-coefficient learnable)
    use_spectral_cs: bool = False        # True ise MLP-SGS bypass edilir, SpectralCsField devrede
    spectral_cs_kx_max: int = 5          # Düşük-k truncation x ekseni
    spectral_cs_ky_max: int = 8          # Düşük-k truncation y ekseni (büyük: dikey yapılar)
    spectral_cs_kz_max: int = 6          # Düşük-k truncation z ekseni


# ── Training ────────────────────────────────────────────────────────────────

@dataclass
class TrainingConfig:
    lr: float = 3e-4
    weight_decay: float = 0.0  # Fizik parametreleri weight decay'den zarar gorur (MLP yok)
    warmup_epochs: int = 100
    max_epochs: int = 1500
    num_steps: int = 1000
    loss_every: int = 20  # loss hesaplama sikligi (her N step'te)
    grad_clip: float = 5.0
    batch_size: int = 1
    curriculum_phase: str = "A"  # A | B | C | D
    checkpoint_dir: str = "results_v2/checkpoints"
    log_interval: int = 10
    use_gradient_routing: bool = True  # 3-group gradient routing (momentum/thermal/bridge)
    freeze_forcing: bool = False  # 2026-04-26: TUNING v2 - forcing.amplitude sabit (Goodhart fix)
    # Tier 1 — Param hijyeni (2026-04-29): denklem-değiştirici parametreleri sabitle
    # Buoyancy3D.s, Advection3D.mod, ThermalAdvection3D.mod, ThermalDiffusion3D.kappa_scale
    # → fixed=1.0 (Boussinesq/Newton/Fick kanonik). SGS Cs/cs_thermal trainable kalır.
    freeze_canonical_params: bool = True

    def __post_init__(self):
        if self.curriculum_phase not in ("A", "B", "C", "D"):
            raise ValueError(f"Invalid curriculum_phase: {self.curriculum_phase}")


# ── Ana Config ──────────────────────────────────────────────────────────────

@dataclass
class Config:
    domain: DomainConfig = field(default_factory=DomainConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    _device_override: Optional[str] = field(default=None, repr=False)

    # ── device ──

    @staticmethod
    def get_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @property
    def device(self) -> torch.device:
        if self._device_override is not None:
            return torch.device(self._device_override)
        return self.get_device()

    @device.setter
    def device(self, val: str | torch.device):
        self._device_override = str(val)

    # ── sweep helper ──

    def set_physics(self, Re: float, Ra: float, Pr: float = 0.71):
        """Re/Ra sweep icin fizik parametrelerini degistir."""
        self.physics.Re = Re
        self.physics.Ra = Ra
        self.physics.Pr = Pr

    # ── serialization ──

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("_device_override", None)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Config:
        return cls(
            domain=DomainConfig(**d.get("domain", {})),
            physics=PhysicsConfig(**d.get("physics", {})),
            model=ModelConfig(**d.get("model", {})),
            training=TrainingConfig(**d.get("training", {})),
        )

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> Config:
        return cls.from_dict(json.loads(Path(path).read_text()))

    # ── repr ──

    def __repr__(self) -> str:
        p = self.physics
        d = self.domain
        return (
            f"Config(\n"
            f"  device={self.device},\n"
            f"  domain=[{d.Nx}x{d.Ny}x{d.Nz}] L=({d.Lx},{d.Ly},{d.Lz}),\n"
            f"  physics=Re={p.Re:.0f} Ra={p.Ra:.0e} Pr={p.Pr} "
            f"nu={p.nu:.2e} Ri={p.Ri:.2e} "
            f"non_boussinesq={p.non_boussinesq},\n"
            f"  model=layers={self.model.n_layers} eddy={self.model.use_eddy_viscosity} "
            f"checkpoint={self.model.gradient_checkpointing},\n"
            f"  training=lr={self.training.lr} "
            f"epochs={self.training.max_epochs} "
            f"phase={self.training.curriculum_phase}\n"
            f")"
        )
