"""
INNATE: Intrinsic Navier-Stokes Neural Architecture for Temporal Evolution

Fizik-bilen nöron kütüphanesi - PINN'ler için torch.nn muadili.

Kullanım (ML kütüphanesi gibi):
    
    # Tek tek layer kullanımı (nn.Conv2d gibi)
    from innate import Advection, Vorticity, Projection
    
    adv = Advection(resolution=64)
    vort = Vorticity(resolution=64)
    proj = Projection(resolution=64)
    
    # PINN içinde kullanım
    class MyPINN(nn.Module):
        def __init__(self):
            super().__init__()
            self.advection = Advection(64)
            self.vorticity = Vorticity(64)
            self.projection = Projection(64)
    
    # Hazır container (opsiyonel)
    from innate import INNATE
    model = INNATE(resolution=64)

Fizik-Bilen Layerlar:
    - Advection      : u·∇u adveksiyon operatörü
    - Vorticity      : ∇×u vortisite evrimi  
    - Projection     : ∇·u = 0 divergence-free projeksiyon
    - TimeMarcher    : CFL-bazlı nedensel zaman ilerlemesi
    - DataInjector   : Seçici veri enjeksiyonu
    - Reynolds       : Öğrenilebilir Reynolds parametresi
    - Boundary       : Sınır koşulları (periodic, no-slip)

Yardımcı:
    - SpectralOps    : Spektral türev operatörleri
    - FluidState     : Akışkan durumu veri yapısı

Yazar: Berke Tezgöçen
Lisans: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fft import fft2, ifft2, fftn, ifftn, fftfreq
import math
import os
import warnings
from typing import Optional, Tuple, Dict
from dataclasses import dataclass
from enum import Enum


# =============================================================================
# CİHAZ SEÇİMİ VE OPTİMİZASYON
# =============================================================================

def get_device() -> torch.device:
    """
    Otomatik cihaz seçimi: CUDA → MPS → CPU
    
    Returns:
        En uygun torch.device
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def check_fft_support(device: torch.device) -> bool:
    """
    FFT desteğini kontrol et (özellikle MPS için).
    
    Args:
        device: Kontrol edilecek cihaz
    
    Returns:
        FFT destekleniyorsa True
    """
    try:
        test_tensor = torch.randn(4, 4, device=device)
        _ = torch.fft.fft2(test_tensor)
        return True
    except Exception:
        return False


def setup_device_optimizations(device: torch.device) -> None:
    """
    Cihaza özel optimizasyonları uygula.
    
    Args:
        device: Hedef cihaz
    """
    # Default dtype ayarla (MPS ve genel uyumluluk için float32)
    torch.set_default_dtype(torch.float32)
    
    if device.type == "cuda":
        # CUDA optimizasyonları
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        
    elif device.type == "mps":
        # MPS için FFT desteğini kontrol et
        if not check_fft_support(device):
            warnings.warn(
                "MPS'te FFT desteği bulunamadı. "
                "PYTORCH_ENABLE_MPS_FALLBACK=1 ayarlanabilir veya CPU kullanılabilir.",
                RuntimeWarning
            )
            # Fallback environment variable'ı kontrol et
            if not os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK'):
                os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'


def safe_fft2(x: torch.Tensor) -> torch.Tensor:
    """
    Güvenli FFT2 - MPS'te sorun olursa CPU'ya fallback.
    
    Args:
        x: Girdi tensörü
    
    Returns:
        FFT sonucu (orijinal device'ta)
    """
    original_device = x.device
    try:
        return fft2(x)
    except Exception:
        # MPS fallback: CPU'da hesapla, sonra geri taşı
        result = fft2(x.cpu())
        return result.to(original_device)


def safe_ifft2(x: torch.Tensor) -> torch.Tensor:
    """
    Güvenli IFFT2 - MPS'te sorun olursa CPU'ya fallback.
    
    Args:
        x: Girdi tensörü (kompleks)
    
    Returns:
        IFFT sonucu (orijinal device'ta)
    """
    original_device = x.device
    try:
        return ifft2(x)
    except Exception:
        # MPS fallback: CPU'da hesapla, sonra geri taşı
        result = ifft2(x.cpu())
        return result.to(original_device)


def safe_fftn(x: torch.Tensor, dim: Tuple[int, ...] = (-3, -2, -1)) -> torch.Tensor:
    """
    Güvenli 3D FFT - MPS'te sorun olursa CPU'ya fallback.
    
    Args:
        x: Girdi tensörü [B, Nx, Ny, Nz]
        dim: FFT boyutları
    
    Returns:
        3D FFT sonucu (orijinal device'ta)
    """
    original_device = x.device
    try:
        return fftn(x, dim=dim)
    except Exception:
        result = fftn(x.cpu(), dim=dim)
        return result.to(original_device)


def safe_ifftn(x: torch.Tensor, dim: Tuple[int, ...] = (-3, -2, -1)) -> torch.Tensor:
    """
    Güvenli 3D IFFT - MPS'te sorun olursa CPU'ya fallback.

    Args:
        x: Girdi tensörü (kompleks)
        dim: IFFT boyutları

    Returns:
        3D IFFT sonucu (orijinal device'ta)
    """
    original_device = x.device
    try:
        return ifftn(x, dim=dim)
    except Exception:
        result = ifftn(x.cpu(), dim=dim)
        return result.to(original_device)


def safe_rfftn(x: torch.Tensor, dim: Tuple[int, ...] = (-3, -2, -1)) -> torch.Tensor:
    """
    Real-to-complex 3D FFT - son boyut N//2+1 olur (~%30 hiz, ~%50 bellek).

    Tum alanlar (u, v, w, p, theta) real oldugu icin rfftn kullanmak
    fftn'e gore daha verimli: output boyutu [Nx, Ny, Nz//2+1] (complex).

    CUDA/MPS'te dogrudan cagir (torch.compile uyumlu, try/except yok).
    CPU fallback sadece bilinmeyen device'larda aktif.
    """
    if x.device.type in ("cuda", "mps", "cpu"):
        return torch.fft.rfftn(x, dim=dim)
    # Bilinmeyen device: CPU fallback
    original_device = x.device
    result = torch.fft.rfftn(x.cpu(), dim=dim)
    return result.to(original_device)


def safe_irfftn(x: torch.Tensor, s: Optional[Tuple[int, ...]] = None,
                dim: Tuple[int, ...] = (-3, -2, -1)) -> torch.Tensor:
    """
    Complex-to-real 3D IFFT - rfftn'in tersi. Output her zaman real.

    KRITIK: s parametresi orijinal spatial boyutlari (Nx, Ny, Nz) olmali.

    CUDA/MPS'te dogrudan cagir (torch.compile uyumlu, try/except yok).
    CPU fallback sadece bilinmeyen device'larda aktif.
    """
    if x.device.type in ("cuda", "mps", "cpu"):
        return torch.fft.irfftn(x, s=s, dim=dim)
    # Bilinmeyen device: CPU fallback
    original_device = x.device
    result = torch.fft.irfftn(x.cpu(), s=s, dim=dim)
    return result.to(original_device)


# Global device ve optimizasyonları ayarla
DEVICE = get_device()
setup_device_optimizations(DEVICE)


# =============================================================================
# PUBLIC API (ML kütüphanesi ergonomisi)
# =============================================================================

__all__ = [
    # =========================================
    # 2D Fizik-Bilen Nöronlar (nn.Conv2d gibi)
    # =========================================
    'Advection',
    'Vorticity', 
    'Projection',
    'TimeMarcher',
    'DataInjector',
    'Reynolds',
    'Boundary',
    
    # =========================================
    # 3D Fizik-Bilen Nöronlar (nn.Conv3d gibi)
    # =========================================
    'Advection3D',
    'Vorticity3D', 
    'Projection3D',
    'TimeMarcher3D',
    'DataInjector3D',
    'Boundary3D',
    'StrainRate3D',              # 3D spesifik
    'Helicity3D',                # 3D spesifik
    
    # =========================================
    # İleri Seviye 3D Nöronlar
    # =========================================
    'PressureCoupling3D',        # ω × u → pressure coupling
    'EnergyPreservingIntegrator3D',  # RK2/RK4/Symplectic
    'SpectralEnergyFlux3D',      # E(k), Π(k) diagnostic
    'EddyViscosity3D',           # SGS / Smagorinsky
    'MLPSGS',                    # MLP-based SGS closure (v2)
    'SpectralCsField',           # Saf-INNATE Cs field (Fourier mode-coefficient learnable, no MLP)

    # =========================================
    # Faz-2: Non-Boussinesq Nöronlar
    # =========================================
    'DensityUpdate3D',           # rho = rho_0 * T_0 / T (ideal gaz)
    'VariableDensityAdvection3D',# Yogunluk-moduleli adveksiyon
    'ContinuityNeuron3D',       # Kutle korunumu diagnostic
    'StateEquation3D',           # EOS dogrulama diagnostic

    # =========================================
    # Termal + Mixed Convection Nöronlar
    # =========================================
    'Forcing3D',                 # Dis forcing (kolmogorov/uniform/stoch)
    'Buoyancy3D',                # Termal kaldirim kuvveti
    'ThermalDiffusion3D',        # Termal difuzyon (kappa * nabla^2 T)
    'ThermalAdvection3D',        # Termal adveksiyon (u.nabla T)
    
    # =========================================
    # Diferansiyel Operatörler
    # =========================================
    'DiffOps',          # Base class
    'SpectralOps',      # 2D FFT (periodic BC)
    'FiniteDiffOps',    # 2D Finite Difference (genel BC)
    'SpectralOps3D',    # 3D FFT (periodic BC)
    'SpectralOps3DAniso',  # 3D FFT anisotropik (farkli L/N)
    'FiniteDiffOps3D',  # 3D Finite Difference (genel BC)
    
    # =========================================
    # Veri yapıları
    # =========================================
    'FluidState',     # 2D
    'FluidState3D',   # 3D
    'FlowRegime',
    
    # =========================================
    # Hazır container (opsiyonel)
    # =========================================
    'INNATE',
    
    # =========================================
    # Device ve yardımcılar
    # =========================================
    'DEVICE',
    'get_device',
    'setup_device_optimizations',
    'check_fft_support',
    'safe_fft2',
    'safe_ifft2',
    'safe_fftn',
    'safe_ifftn',
    'safe_rfftn',
    'safe_irfftn',
]


# =============================================================================
# BÖLÜM 1: TEMEL YAPILAR VE ARAÇLAR
# =============================================================================

@dataclass
class FluidState:
    """
    2D Akışkan durumu - skaler vortisite.
    
    Tensör boyutları: [B, H, W] (Batch, Height, Width)
    """
    u: torch.Tensor          # x-hız bileşeni [B, H, W]
    v: torch.Tensor          # y-hız bileşeni [B, H, W]
    p: torch.Tensor          # basınç alanı [B, H, W]
    vorticity: torch.Tensor  # vortisite ω = ∂v/∂x - ∂u/∂y [B, H, W] (skaler)
    t: torch.Tensor          # zaman [B, 1]
    
    def velocity_magnitude(self) -> torch.Tensor:
        return torch.sqrt(self.u**2 + self.v**2 + 1e-8)
    
    def kinetic_energy(self) -> torch.Tensor:
        return 0.5 * (self.u**2 + self.v**2).mean(dim=(-2, -1))
    
    def enstrophy(self) -> torch.Tensor:
        """Enstrofi: türbülans ölçüsü"""
        return 0.5 * (self.vorticity**2).mean(dim=(-2, -1))


@dataclass
class FluidState3D:
    """
    3D Akışkan durumu - vektörel vortisite.
    
    Tensör boyutları: [B, Nx, Ny, Nz] (Batch, X, Y, Z)
    
    Kullanım:
        state = FluidState3D(u, v, w, p, omega_x, omega_y, omega_z, t)
        speed = state.velocity_magnitude()
        energy = state.kinetic_energy()
        H = state.helicity()  # 3D spesifik
    """
    u: torch.Tensor        # x-hız bileşeni [B, Nx, Ny, Nz]
    v: torch.Tensor        # y-hız bileşeni [B, Nx, Ny, Nz]
    w: torch.Tensor        # z-hız bileşeni [B, Nx, Ny, Nz]
    p: torch.Tensor        # basınç alanı [B, Nx, Ny, Nz]
    omega_x: torch.Tensor  # vortisite x-bileşeni [B, Nx, Ny, Nz]
    omega_y: torch.Tensor  # vortisite y-bileşeni [B, Nx, Ny, Nz]
    omega_z: torch.Tensor  # vortisite z-bileşeni [B, Nx, Ny, Nz]
    t: torch.Tensor        # zaman [B, 1]
    
    def velocity_magnitude(self) -> torch.Tensor:
        """Hız büyüklüğü: |u| = sqrt(u² + v² + w²)"""
        return torch.sqrt(self.u**2 + self.v**2 + self.w**2 + 1e-8)
    
    def vorticity_magnitude(self) -> torch.Tensor:
        """Vortisite büyüklüğü: |ω| = sqrt(ωx² + ωy² + ωz²)"""
        return torch.sqrt(self.omega_x**2 + self.omega_y**2 + self.omega_z**2 + 1e-8)
    
    def kinetic_energy(self) -> torch.Tensor:
        """Kinetik enerji: E = 0.5 * (u² + v² + w²)"""
        return 0.5 * (self.u**2 + self.v**2 + self.w**2).mean(dim=(-3, -2, -1))
    
    def enstrophy(self) -> torch.Tensor:
        """Enstrofi: Z = 0.5 * |ω|² (türbülans ölçüsü)"""
        omega_sq = self.omega_x**2 + self.omega_y**2 + self.omega_z**2
        return 0.5 * omega_sq.mean(dim=(-3, -2, -1))
    
    def helicity(self) -> torch.Tensor:
        """
        Helicity: H = u · ω (3D spesifik)
        
        Hız ve vortisite arasındaki korelasyon.
        Türbülans enerji kaskadı analizinde önemli.
        """
        H = self.u * self.omega_x + self.v * self.omega_y + self.w * self.omega_z
        return H.mean(dim=(-3, -2, -1))


class FlowRegime(Enum):
    """Akış rejimi - nöronlar bunu öğrenecek"""
    LAMINAR = 1
    TRANSITIONAL = 2
    VORTEX_DOMINANT = 3


# =============================================================================
# BÖLÜM 2: DİFERANSİYEL OPERATÖRLER (NÖRONLARIN DNA'SI)
# =============================================================================

class DiffOps(nn.Module):
    """
    Diferansiyel Operatör Base Class

    Tüm türev operatörleri bu sınıftan türer:
    - SpectralOps: FFT tabanlı (periodic BC için ideal)
    - FiniteDiffOps: Finite difference (genel BC için)

    Bu abstraction sayesinde nöronlar operatör-agnostik çalışır.
    """
    def __init__(self, resolution: int, domain_size: float = 2*math.pi):
        super().__init__()
        self.resolution = resolution
        self.domain_size = domain_size
        self.dx = domain_size / resolution

    def gradient(self, f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """∇f = (∂f/∂x, ∂f/∂y) - Alt sınıflar implement etmeli"""
        raise NotImplementedError

    def laplacian(self, f: torch.Tensor) -> torch.Tensor:
        """∇²f = ∂²f/∂x² + ∂²f/∂y² - Alt sınıflar implement etmeli"""
        raise NotImplementedError

    def curl_2d(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """ω = ∇×u = ∂v/∂x - ∂u/∂y"""
        dv_dx, _ = self.gradient(v)
        _, du_dy = self.gradient(u)
        return dv_dx - du_dy

    def divergence(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """∇·u = ∂u/∂x + ∂v/∂y"""
        du_dx, _ = self.gradient(u)
        _, dv_dy = self.gradient(v)
        return du_dx + dv_dy

    def solve_poisson(self, rhs: torch.Tensor) -> torch.Tensor:
        """
        Poisson denklemi çöz: ∇²p = rhs

        Alt sınıflar implement etmeli:
        - SpectralOps: FFT ile doğrudan çözüm
        - FiniteDiffOps: Iteratif çözüm (SOR, Jacobi, vb.)

        Returns:
            p: Basınç alanı (ortalama = 0)
        """
        raise NotImplementedError

    def dealias(self, f: torch.Tensor) -> torch.Tensor:
        """Aliasing önleme - alt sınıflar override edebilir"""
        return f

    def apply_filter(self, f: torch.Tensor) -> torch.Tensor:
        """Spektral/spatial filtre - alt sınıflar override edebilir"""
        return f


class SpectralOps(DiffOps):
    """
    Spektral türev operatörleri - Fourier uzayında tam doğruluk.
    Finite difference değil, fiziksel olarak doğru.

    Bu sınıf ÖĞRENILMEZ - saf matematiksel operatörler.

    Kullanım:
        ops = SpectralOps(64)
        df_dx, df_dy = ops.gradient(f)
        lap_f = ops.laplacian(f)

    NOT: Sadece PERIODIC boundary conditions için uygundur!
    Non-periodic için FiniteDiffOps kullanın.
    """
    def __init__(self, resolution: int, domain_size: float = 2*math.pi):
        super().__init__(resolution, domain_size)

        # Dalga sayıları - sabit, öğrenilmez
        k = fftfreq(resolution, d=domain_size/resolution) * 2 * math.pi
        kx, ky = torch.meshgrid(k, k, indexing='ij')

        self.register_buffer('kx', kx)
        self.register_buffer('ky', ky)
        self.register_buffer('k_squared', kx**2 + ky**2)

        # Dealiasing filtresi (2/3 kuralı)
        k_max = resolution // 3
        dealias_mask = (torch.abs(kx) < k_max) & (torch.abs(ky) < k_max)
        self.register_buffer('dealias_mask', dealias_mask.float())

        # Exponential cutoff filter
        k_mag = torch.sqrt(kx**2 + ky**2)
        k_cutoff = resolution // 3
        alpha = 10.0
        p = 8
        exp_filter = torch.exp(-alpha * (k_mag / k_cutoff) ** p)
        exp_filter = torch.where(k_mag <= k_cutoff, torch.ones_like(exp_filter), exp_filter)
        self.register_buffer('exp_filter', exp_filter)

    def gradient(self, f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """∇f = (∂f/∂x, ∂f/∂y)"""
        f_hat = safe_fft2(f)
        df_dx = safe_ifft2(1j * self.kx * f_hat).real
        df_dy = safe_ifft2(1j * self.ky * f_hat).real
        return df_dx, df_dy
    
    def laplacian(self, f: torch.Tensor) -> torch.Tensor:
        """∇²f = ∂²f/∂x² + ∂²f/∂y²"""
        f_hat = safe_fft2(f)
        lap_f = safe_ifft2(-self.k_squared * f_hat).real
        return lap_f
    
    def curl_2d(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """ω = ∇×u = ∂v/∂x - ∂u/∂y (2D vortisite)"""
        dv_dx, _ = self.gradient(v)
        _, du_dy = self.gradient(u)
        return dv_dx - du_dy
    
    def divergence(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """∇·u = ∂u/∂x + ∂v/∂y"""
        du_dx, _ = self.gradient(u)
        _, dv_dy = self.gradient(v)
        return du_dx + dv_dy
    
    def dealias(self, f: torch.Tensor) -> torch.Tensor:
        """Aliasing önleme - nonlinear terimler için kritik"""
        f_hat = safe_fft2(f)
        return safe_ifft2(f_hat * self.dealias_mask).real

    def apply_filter(self, f: torch.Tensor) -> torch.Tensor:
        """
        Spektral filtre uygula - ÖĞRENİLMEZ.

        Bu filter stabilite için kritik:
        - Nyquist frekansına yakın modları sönümler
        - Checkerboard instabilitesini önler
        - Enerji birikimini kontrol eder
        """
        f_hat = safe_fft2(f)
        return safe_ifft2(f_hat * self.exp_filter).real

    def solve_poisson(self, rhs: torch.Tensor) -> torch.Tensor:
        """
        Poisson denklemi çöz: ∇²p = rhs (Spektral yöntem - doğrudan)

        FFT ile O(N log N) karmaşıklıkta tam çözüm.
        Sadece periodic BC için geçerli!
        """
        rhs_hat = safe_fft2(rhs)

        k_sq = self.k_squared.clone()
        k_sq[0, 0] = 1.0  # Divide by zero önle

        p_hat = rhs_hat / (-k_sq + 1e-10)
        p_hat[..., 0, 0] = 0  # Ortalama basınç = 0

        return safe_ifft2(p_hat).real


class FiniteDiffOps(DiffOps):
    """
    Finite Difference türev operatörleri - Genel boundary conditions için.

    SpectralOps'un aksine HER TÜRLÜ sınır koşuluyla çalışır:
    - Periodic
    - Dirichlet (fixed value)
    - Neumann (fixed gradient)
    - Open boundaries (inlet/outlet)

    Şemalar:
    - 'central': 2. derece merkezi fark (varsayılan, doğru)
    - 'upwind': 1. derece yukarı akış (stabil, difüzif)
    - 'central4': 4. derece merkezi fark (daha doğru)

    Kullanım:
        ops = FiniteDiffOps(64, domain_size=10.0)
        df_dx, df_dy = ops.gradient(f)
        lap_f = ops.laplacian(f)

    NOT: SpectralOps'tan daha az doğru ama daha genel.
    """
    def __init__(self, resolution: int, domain_size: float = 2*math.pi,
                 scheme: str = 'central', bc_type: str = 'periodic'):
        super().__init__(resolution, domain_size)
        self.scheme = scheme
        self.bc_type = bc_type

        # Convolution kernels for finite differences
        # Central difference: (f[i+1] - f[i-1]) / (2*dx)
        if scheme == 'central':
            # 2nd order central difference
            self.register_buffer('kernel_dx', torch.tensor([[0, 0, 0],
                                                            [-0.5, 0, 0.5],
                                                            [0, 0, 0]]).float().view(1, 1, 3, 3) / self.dx)
            self.register_buffer('kernel_dy', torch.tensor([[0, -0.5, 0],
                                                            [0, 0, 0],
                                                            [0, 0.5, 0]]).float().view(1, 1, 3, 3) / self.dx)
            # Laplacian: (f[i+1] - 2*f[i] + f[i-1]) / dx^2
            self.register_buffer('kernel_lap', torch.tensor([[0, 1, 0],
                                                              [1, -4, 1],
                                                              [0, 1, 0]]).float().view(1, 1, 3, 3) / (self.dx**2))
        elif scheme == 'central4':
            # 4th order central difference
            self.register_buffer('kernel_dx', torch.tensor([
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [1/12, -2/3, 0, 2/3, -1/12],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]
            ]).float().view(1, 1, 5, 5) / self.dx)
            self.register_buffer('kernel_dy', torch.tensor([
                [0, 0, 1/12, 0, 0],
                [0, 0, -2/3, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 2/3, 0, 0],
                [0, 0, -1/12, 0, 0]
            ]).float().view(1, 1, 5, 5) / self.dx)
            # 4th order Laplacian
            self.register_buffer('kernel_lap', torch.tensor([
                [0, 0, -1/12, 0, 0],
                [0, 0, 4/3, 0, 0],
                [-1/12, 4/3, -5, 4/3, -1/12],
                [0, 0, 4/3, 0, 0],
                [0, 0, -1/12, 0, 0]
            ]).float().view(1, 1, 5, 5) / (self.dx**2))
        elif scheme == 'upwind':
            # 1st order upwind (for advection stability)
            # Will be applied based on velocity direction in advection
            self.register_buffer('kernel_dx_forward', torch.tensor([[0, 0, 0],
                                                                     [0, -1, 1],
                                                                     [0, 0, 0]]).float().view(1, 1, 3, 3) / self.dx)
            self.register_buffer('kernel_dx_backward', torch.tensor([[0, 0, 0],
                                                                      [-1, 1, 0],
                                                                      [0, 0, 0]]).float().view(1, 1, 3, 3) / self.dx)
            self.register_buffer('kernel_dy_forward', torch.tensor([[0, 0, 0],
                                                                     [0, -1, 0],
                                                                     [0, 1, 0]]).float().view(1, 1, 3, 3) / self.dx)
            self.register_buffer('kernel_dy_backward', torch.tensor([[0, 1, 0],
                                                                      [0, -1, 0],
                                                                      [0, 0, 0]]).float().view(1, 1, 3, 3) / self.dx)
            # For laplacian, still use central
            self.register_buffer('kernel_lap', torch.tensor([[0, 1, 0],
                                                              [1, -4, 1],
                                                              [0, 1, 0]]).float().view(1, 1, 3, 3) / (self.dx**2))
            # Central for gradient (default)
            self.register_buffer('kernel_dx', torch.tensor([[0, 0, 0],
                                                            [-0.5, 0, 0.5],
                                                            [0, 0, 0]]).float().view(1, 1, 3, 3) / self.dx)
            self.register_buffer('kernel_dy', torch.tensor([[0, -0.5, 0],
                                                            [0, 0, 0],
                                                            [0, 0.5, 0]]).float().view(1, 1, 3, 3) / self.dx)

    def _apply_bc(self, f: torch.Tensor) -> torch.Tensor:
        """Apply boundary conditions via padding"""
        if self.bc_type == 'periodic':
            pad_size = self.kernel_dx.shape[-1] // 2
            return F.pad(f, (pad_size, pad_size, pad_size, pad_size), mode='circular')
        elif self.bc_type == 'neumann':
            # Zero gradient at boundaries (replicate)
            pad_size = self.kernel_dx.shape[-1] // 2
            return F.pad(f, (pad_size, pad_size, pad_size, pad_size), mode='replicate')
        elif self.bc_type == 'dirichlet':
            # Zero value at boundaries
            pad_size = self.kernel_dx.shape[-1] // 2
            return F.pad(f, (pad_size, pad_size, pad_size, pad_size), mode='constant', value=0)
        else:
            # Default: replicate (most stable for open boundaries)
            pad_size = self.kernel_dx.shape[-1] // 2
            return F.pad(f, (pad_size, pad_size, pad_size, pad_size), mode='replicate')

    def gradient(self, f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """∇f = (∂f/∂x, ∂f/∂y) using finite differences"""
        # Ensure 4D: [B, C, H, W]
        squeeze_batch = False
        if f.dim() == 3:
            f = f.unsqueeze(1)  # [B, 1, H, W]
            squeeze_batch = True
        elif f.dim() == 2:
            f = f.unsqueeze(0).unsqueeze(0)
            squeeze_batch = True

        # Apply boundary conditions
        f_padded = self._apply_bc(f)

        # Convolve
        df_dx = F.conv2d(f_padded, self.kernel_dx.to(f.device))
        df_dy = F.conv2d(f_padded, self.kernel_dy.to(f.device))

        if squeeze_batch:
            df_dx = df_dx.squeeze(1)
            df_dy = df_dy.squeeze(1)

        return df_dx, df_dy

    def laplacian(self, f: torch.Tensor) -> torch.Tensor:
        """∇²f = ∂²f/∂x² + ∂²f/∂y² using finite differences"""
        squeeze_batch = False
        if f.dim() == 3:
            f = f.unsqueeze(1)
            squeeze_batch = True
        elif f.dim() == 2:
            f = f.unsqueeze(0).unsqueeze(0)
            squeeze_batch = True

        # Apply boundary conditions
        f_padded = self._apply_bc(f)

        # Convolve
        lap_f = F.conv2d(f_padded, self.kernel_lap.to(f.device))

        if squeeze_batch:
            lap_f = lap_f.squeeze(1)

        return lap_f

    def gradient_upwind(self, f: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Upwind gradient for advection stability.

        Uses forward difference where velocity is positive,
        backward difference where velocity is negative.
        """
        if self.scheme != 'upwind':
            return self.gradient(f)

        squeeze_batch = False
        if f.dim() == 3:
            f = f.unsqueeze(1)
            u = u.unsqueeze(1)
            v = v.unsqueeze(1)
            squeeze_batch = True

        f_padded = self._apply_bc(f)

        # Forward and backward differences
        df_dx_fwd = F.conv2d(f_padded, self.kernel_dx_forward.to(f.device))
        df_dx_bwd = F.conv2d(f_padded, self.kernel_dx_backward.to(f.device))
        df_dy_fwd = F.conv2d(f_padded, self.kernel_dy_forward.to(f.device))
        df_dy_bwd = F.conv2d(f_padded, self.kernel_dy_backward.to(f.device))

        # Select based on velocity direction
        df_dx = torch.where(u > 0, df_dx_bwd, df_dx_fwd)
        df_dy = torch.where(v > 0, df_dy_bwd, df_dy_fwd)

        if squeeze_batch:
            df_dx = df_dx.squeeze(1)
            df_dy = df_dy.squeeze(1)

        return df_dx, df_dy

    def dealias(self, f: torch.Tensor) -> torch.Tensor:
        """No dealiasing needed for finite differences"""
        return f

    def apply_filter(self, f: torch.Tensor) -> torch.Tensor:
        """Simple smoothing filter for stability"""
        # 3x3 Gaussian-like smoothing
        kernel = torch.tensor([[1, 2, 1],
                               [2, 4, 2],
                               [1, 2, 1]]).float() / 16.0
        kernel = kernel.view(1, 1, 3, 3).to(f.device)

        squeeze_batch = False
        if f.dim() == 3:
            f = f.unsqueeze(1)
            squeeze_batch = True

        f_padded = self._apply_bc(f)
        filtered = F.conv2d(f_padded, kernel)

        if squeeze_batch:
            filtered = filtered.squeeze(1)

        return filtered

    def solve_poisson(self, rhs: torch.Tensor, max_iter: int = 500, tol: float = 1e-6) -> torch.Tensor:
        """
        Poisson denklemi çöz: ∇²p = rhs (Iteratif yöntem - Jacobi/SOR)

        Args:
            rhs: Sağ taraf (divergence)
            max_iter: Maksimum iterasyon sayısı
            tol: Yakınsama toleransı

        Returns:
            p: Basınç alanı (ortalama = 0)
        """
        # Batch boyutunu koru
        squeeze_batch = False
        if rhs.dim() == 2:
            rhs = rhs.unsqueeze(0)
            squeeze_batch = True

        B, H, W = rhs.shape

        # Neumann BC için compatibility condition: ortalama RHS = 0 olmalı
        rhs = rhs - rhs.mean(dim=(-2, -1), keepdim=True)

        # Başlangıç tahmini: sıfır
        p = torch.zeros_like(rhs)

        # Pure Jacobi (omega=1) - SOR (omega>1) diverge edebilir
        omega = 1.0

        # dx^2 faktörü
        dx2 = self.dx ** 2

        # Iteratif çözüm
        for iteration in range(max_iter):
            p_old = p.clone()

            # Padding ile komşuları al
            if self.bc_type == 'periodic':
                p_pad = F.pad(p, (1, 1, 1, 1), mode='circular')
            elif self.bc_type == 'neumann':
                p_pad = F.pad(p, (1, 1, 1, 1), mode='replicate')
            else:  # dirichlet
                p_pad = F.pad(p, (1, 1, 1, 1), mode='constant', value=0)

            # Komşu değerler
            p_left = p_pad[:, 1:-1, :-2]
            p_right = p_pad[:, 1:-1, 2:]
            p_up = p_pad[:, :-2, 1:-1]
            p_down = p_pad[:, 2:, 1:-1]

            # Jacobi güncellemesi: p = (neighbors - dx2*rhs) / 4
            p_jacobi = (p_left + p_right + p_up + p_down - dx2 * rhs) / 4.0

            # SOR güncellemesi
            p = (1 - omega) * p_old + omega * p_jacobi

            # Ortalamayı sıfırla (basınç gauge - Neumann BC için gerekli)
            p = p - p.mean(dim=(-2, -1), keepdim=True)

            # Yakınsama kontrolü (residual bazlı)
            if iteration % 10 == 0:
                residual = torch.abs(p - p_old).max()
                if residual < tol:
                    break

        if squeeze_batch:
            p = p.squeeze(0)

        return p

    def extra_repr(self) -> str:
        return f"resolution={self.resolution}, scheme={self.scheme}, bc={self.bc_type}"


class SpectralOps3D(nn.Module):
    """
    3D Spektral türev operatörleri - Fourier uzayında tam doğruluk.
    
    2D SpectralOps'un 3 boyutlu versiyonu.
    fftn/ifftn kullanarak x, y, z yönlerinde spektral türevler.
    
    Kullanım:
        ops = SpectralOps3D(resolution=64)
        df_dx, df_dy, df_dz = ops.gradient(f)
        lap_f = ops.laplacian(f)
        div = ops.divergence(u, v, w)
        omega_x, omega_y, omega_z = ops.curl(u, v, w)
    """
    def __init__(self, resolution: int, domain_size: float = 2*math.pi):
        super().__init__()
        self.resolution = resolution
        self.domain_size = domain_size
        
        # 3D dalga sayıları - sabit, öğrenilmez
        k = fftfreq(resolution, d=domain_size/resolution) * 2 * math.pi
        kx, ky, kz = torch.meshgrid(k, k, k, indexing='ij')
        
        self.register_buffer('kx', kx)
        self.register_buffer('ky', ky)
        self.register_buffer('kz', kz)
        self.register_buffer('k_squared', kx**2 + ky**2 + kz**2)
        
        # 3D Dealiasing filtresi (2/3 kuralı)
        k_max = resolution // 3
        dealias_mask = (torch.abs(kx) < k_max) & (torch.abs(ky) < k_max) & (torch.abs(kz) < k_max)
        self.register_buffer('dealias_mask', dealias_mask.float())
    
    def gradient(self, f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        3D Gradient: ∇f = (∂f/∂x, ∂f/∂y, ∂f/∂z)
        
        Args:
            f: Skaler alan [B, Nx, Ny, Nz]
        
        Returns:
            (df_dx, df_dy, df_dz): Gradient bileşenleri
        """
        f_hat = safe_fftn(f)
        df_dx = safe_ifftn(1j * self.kx * f_hat).real
        df_dy = safe_ifftn(1j * self.ky * f_hat).real
        df_dz = safe_ifftn(1j * self.kz * f_hat).real
        return df_dx, df_dy, df_dz
    
    def laplacian(self, f: torch.Tensor) -> torch.Tensor:
        """
        3D Laplacian: ∇²f = ∂²f/∂x² + ∂²f/∂y² + ∂²f/∂z²
        
        Args:
            f: Skaler alan [B, Nx, Ny, Nz]
        
        Returns:
            lap_f: Laplacian [B, Nx, Ny, Nz]
        """
        f_hat = safe_fftn(f)
        lap_f = safe_ifftn(-self.k_squared * f_hat).real
        return lap_f
    
    def curl(self, u: torch.Tensor, v: torch.Tensor, w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        3D Curl (Vortisite): ω = ∇×u
        
        ω_x = ∂w/∂y - ∂v/∂z
        ω_y = ∂u/∂z - ∂w/∂x
        ω_z = ∂v/∂x - ∂u/∂y
        
        Args:
            u, v, w: Hız bileşenleri [B, Nx, Ny, Nz]
        
        Returns:
            (omega_x, omega_y, omega_z): Vortisite vektörü
        """
        # Tüm gradyanları hesapla
        du_dx, du_dy, du_dz = self.gradient(u)
        dv_dx, dv_dy, dv_dz = self.gradient(v)
        dw_dx, dw_dy, dw_dz = self.gradient(w)
        
        # Curl bileşenleri
        omega_x = dw_dy - dv_dz
        omega_y = du_dz - dw_dx
        omega_z = dv_dx - du_dy
        
        return omega_x, omega_y, omega_z
    
    def divergence(self, u: torch.Tensor, v: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        3D Divergence: ∇·u = ∂u/∂x + ∂v/∂y + ∂w/∂z
        
        Args:
            u, v, w: Hız bileşenleri [B, Nx, Ny, Nz]
        
        Returns:
            div: Divergence skaler alanı [B, Nx, Ny, Nz]
        """
        du_dx, _, _ = self.gradient(u)
        _, dv_dy, _ = self.gradient(v)
        _, _, dw_dz = self.gradient(w)
        return du_dx + dv_dy + dw_dz
    
    def dealias(self, f: torch.Tensor) -> torch.Tensor:
        """
        3D Aliasing önleme - nonlinear terimler için kritik (2/3 kuralı)

        Args:
            f: Alan [B, Nx, Ny, Nz]

        Returns:
            Dealiased alan
        """
        f_hat = safe_fftn(f)
        return safe_ifftn(f_hat * self.dealias_mask).real

    def solve_poisson(self, rhs: torch.Tensor) -> torch.Tensor:
        """
        3D Poisson denklemi çöz: ∇²p = rhs (Spektral yöntem)
        """
        rhs_hat = safe_fftn(rhs)

        k_sq = self.k_squared.clone()
        k_sq[0, 0, 0] = 1.0  # Divide by zero önle

        p_hat = rhs_hat / (-k_sq + 1e-10)
        p_hat[..., 0, 0, 0] = 0  # Ortalama basınç = 0

        return safe_ifftn(p_hat).real


class SpectralOps3DAniso(nn.Module):
    """
    Anisotropik 3D spektral türev operatörleri.

    İsotropik SpectralOps3D ile aynı API, farklı resolution/domain boyutu
    desteği. Periyodik BC varsayar (Fourier bazlı).

    Her yön için farklı grid spacing ve domain uzunluğu destekler:
    Domain: Lx × Ly × Lz,  Grid: Nx × Ny × Nz

    Hiçbir learnable parameter içermez — tamamı register_buffer.

    Kullanım:
        ops = SpectralOps3DAniso(Nx=96, Ny=160, Nz=64, Lx=6.0, Ly=10.0, Lz=4.0)
        df_dx, df_dy, df_dz = ops.gradient(f)
        lap = ops.laplacian(f)
        ox, oy, oz = ops.curl(u, v, w)
        div = ops.divergence(u, v, w)
        f_clean = ops.dealias(f)
        p = ops.solve_poisson(rhs)
    """

    def __init__(self, Nx: int, Ny: int, Nz: int,
                 Lx: float, Ly: float, Lz: float):
        super().__init__()
        self.Nx, self.Ny, self.Nz = Nx, Ny, Nz
        self.Lx, self.Ly, self.Lz = Lx, Ly, Lz

        # Orijinal spatial boyutlar — irfftn icin ZORUNLU
        self.spatial_shape = (Nx, Ny, Nz)

        # --- Dalga sayilari (her yon icin farkli dx = L/N) ---
        # kx, ky: fftfreq (tam kompleks, Nx ve Ny elemanli)
        # kz: rfftfreq (yari-kompleks, Nz//2+1 elemanli — rfftn optimizasyonu)
        kx_1d = torch.fft.fftfreq(Nx, d=Lx / Nx) * 2 * math.pi
        ky_1d = torch.fft.fftfreq(Ny, d=Ly / Ny) * 2 * math.pi
        kz_1d = torch.fft.rfftfreq(Nz, d=Lz / Nz) * 2 * math.pi  # Nz//2+1

        # Nyquist modu sifirla (spectral ambiguity onleme)
        if Nx % 2 == 0:
            kx_1d[Nx // 2] = 0.0
        if Ny % 2 == 0:
            ky_1d[Ny // 2] = 0.0
        # kz: rfftfreq'da son eleman zaten Nyquist frekansi
        if Nz % 2 == 0:
            kz_1d[-1] = 0.0

        kx, ky, kz = torch.meshgrid(kx_1d, ky_1d, kz_1d, indexing='ij')

        self.register_buffer('kx', kx)   # [Nx, Ny, Nz//2+1]
        self.register_buffer('ky', ky)   # [Nx, Ny, Nz//2+1]
        self.register_buffer('kz', kz)   # [Nx, Ny, Nz//2+1]

        # k-kare — laplacian icin (origin = 0)
        k_squared = kx ** 2 + ky ** 2 + kz ** 2
        self.register_buffer('k_squared', k_squared)  # [Nx, Ny, Nz//2+1]

        # Tam boyutlu k-kare — spectrum analizi icin (train.py safe_fftn ile kullanir)
        kz_full_1d = torch.fft.fftfreq(Nz, d=Lz / Nz) * 2 * math.pi
        if Nz % 2 == 0:
            kz_full_1d[Nz // 2] = 0.0
        kx_f, ky_f, kz_f = torch.meshgrid(kx_1d, ky_1d, kz_full_1d, indexing='ij')
        k_squared_full = kx_f ** 2 + ky_f ** 2 + kz_f ** 2
        self.register_buffer('k_squared_full', k_squared_full)  # [Nx, Ny, Nz]

        # Poisson icin ayri buffer (tum k-kare=0 noktalari 1.0)
        k_sq_poisson = k_squared.clone()
        k_sq_poisson[k_sq_poisson == 0.0] = 1.0
        self.register_buffer('k_squared_poisson', k_sq_poisson)

        # --- Dealias mask: MODE INDEX uzerinden 2/3 kurali ---
        # mx, my: fftfreq (tam), mz: rfftfreq (yari) — buffer boyutlariyla tutarli
        mx = torch.fft.fftfreq(Nx, d=1.0) * Nx
        my = torch.fft.fftfreq(Ny, d=1.0) * Ny
        mz = torch.fft.rfftfreq(Nz, d=1.0) * Nz     # Nz//2+1 elemanli
        Mx, My, Mz = torch.meshgrid(mx, my, mz, indexing='ij')

        dealias_mask = (
            (torch.abs(Mx) < Nx // 3) &
            (torch.abs(My) < Ny // 3) &
            (torch.abs(Mz) < Nz // 3)      # rfftfreq >= 0, abs gereksiz ama zarar vermez
        )
        self.register_buffer('dealias_mask', dealias_mask.float())

        # --- Frekans band maskeleri (SGS frekans-bagimli Cs icin) ---
        k_mag = torch.sqrt(k_squared)
        k_max_val = k_mag.max().item()
        if k_max_val > 0:
            band_low = (k_mag < k_max_val / 3.0).float()
            band_high = (k_mag >= 2.0 * k_max_val / 3.0).float()
            band_mid = (1.0 - band_low - band_high).clamp(min=0.0)
        else:
            band_low = torch.ones_like(k_mag)
            band_mid = torch.zeros_like(k_mag)
            band_high = torch.zeros_like(k_mag)
        self.register_buffer('band_low', band_low)
        self.register_buffer('band_mid', band_mid)
        self.register_buffer('band_high', band_high)

    # ---- Helper: polimorfik FFT (Advection3D vb. siniflar icin) ----

    def to_hat(self, x: torch.Tensor) -> torch.Tensor:
        """Real field -> spectral domain (rfftn). Output: [B, Nx, Ny, Nz//2+1]."""
        return safe_rfftn(x)

    def from_hat(self, x_hat: torch.Tensor) -> torch.Tensor:
        """Spectral domain -> real field (irfftn). Output: [B, Nx, Ny, Nz]."""
        return safe_irfftn(x_hat, s=self.spatial_shape)

    # ---- Public API (gradient, laplacian, curl, vb.) ----

    def gradient(self, f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gradient: grad(f) = (df/dx, df/dy, df/dz).  f: [B, Nx, Ny, Nz]"""
        f_hat = safe_rfftn(f)
        s = self.spatial_shape
        df_dx = safe_irfftn(1j * self.kx * f_hat, s=s)
        df_dy = safe_irfftn(1j * self.ky * f_hat, s=s)
        df_dz = safe_irfftn(1j * self.kz * f_hat, s=s)
        return df_dx, df_dy, df_dz

    def laplacian(self, f: torch.Tensor) -> torch.Tensor:
        """Laplacian: nabla^2 f.  f: [B, Nx, Ny, Nz]"""
        f_hat = safe_rfftn(f)
        return safe_irfftn(-self.k_squared * f_hat, s=self.spatial_shape)

    def directional_laplacian(self, f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Yonsel Laplacian: (d2f/dx2, d2f/dy2, d2f/dz2) ayri ayri.
        Anisotropik diffusion icin gerekli.
        1 rFFT + 3 irFFT.
        """
        f_hat = safe_rfftn(f)
        s = self.spatial_shape
        d2f_dx2 = safe_irfftn(-(self.kx ** 2) * f_hat, s=s)
        d2f_dy2 = safe_irfftn(-(self.ky ** 2) * f_hat, s=s)
        d2f_dz2 = safe_irfftn(-(self.kz ** 2) * f_hat, s=s)
        return d2f_dx2, d2f_dy2, d2f_dz2

    # ---- _from_hat variants: pre-computed rFFT ile calisir (caching icin) ----
    # NOT: f_hat artik rfftn sonucu: [B, Nx, Ny, Nz//2+1] (complex)

    def gradient_from_hat(self, f_hat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gradient, f_hat = rfftn(f) onceden hesaplanmis. 0 FFT + 3 irFFT."""
        s = self.spatial_shape
        df_dx = safe_irfftn(1j * self.kx * f_hat, s=s)
        df_dy = safe_irfftn(1j * self.ky * f_hat, s=s)
        df_dz = safe_irfftn(1j * self.kz * f_hat, s=s)
        return df_dx, df_dy, df_dz

    def laplacian_from_hat(self, f_hat: torch.Tensor) -> torch.Tensor:
        """Laplacian, f_hat onceden hesaplanmis. 0 FFT + 1 irFFT."""
        return safe_irfftn(-self.k_squared * f_hat, s=self.spatial_shape)

    def directional_laplacian_from_hat(self, f_hat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Yonsel Laplacian, f_hat onceden hesaplanmis. 0 FFT + 3 irFFT."""
        s = self.spatial_shape
        d2f_dx2 = safe_irfftn(-(self.kx ** 2) * f_hat, s=s)
        d2f_dy2 = safe_irfftn(-(self.ky ** 2) * f_hat, s=s)
        d2f_dz2 = safe_irfftn(-(self.kz ** 2) * f_hat, s=s)
        return d2f_dx2, d2f_dy2, d2f_dz2

    def partial_x(self, f_hat: torch.Tensor) -> torch.Tensor:
        """df/dx, f_hat onceden hesaplanmis. 0 FFT + 1 irFFT."""
        return safe_irfftn(1j * self.kx * f_hat, s=self.spatial_shape)

    def partial_y(self, f_hat: torch.Tensor) -> torch.Tensor:
        """df/dy, f_hat onceden hesaplanmis. 0 FFT + 1 irFFT."""
        return safe_irfftn(1j * self.ky * f_hat, s=self.spatial_shape)

    def partial_z(self, f_hat: torch.Tensor) -> torch.Tensor:
        """df/dz, f_hat onceden hesaplanmis. 0 FFT + 1 irFFT."""
        return safe_irfftn(1j * self.kz * f_hat, s=self.spatial_shape)

    def divergence_from_hat(self, u_hat: torch.Tensor, v_hat: torch.Tensor, w_hat: torch.Tensor) -> torch.Tensor:
        """Divergence, u/v/w_hat onceden hesaplanmis. 0 FFT + 1 irFFT."""
        return safe_irfftn(
            1j * (self.kx * u_hat + self.ky * v_hat + self.kz * w_hat),
            s=self.spatial_shape
        )

    def curl_from_hat(self, u_hat: torch.Tensor, v_hat: torch.Tensor, w_hat: torch.Tensor
                      ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Curl, u/v/w_hat onceden hesaplanmis. 0 FFT + 3 irFFT."""
        s = self.spatial_shape
        ox = safe_irfftn(1j * (self.ky * w_hat - self.kz * v_hat), s=s)
        oy = safe_irfftn(1j * (self.kz * u_hat - self.kx * w_hat), s=s)
        oz = safe_irfftn(1j * (self.kx * v_hat - self.ky * u_hat), s=s)
        return ox, oy, oz

    def solve_poisson_from_hat(self, rhs_hat: torch.Tensor) -> torch.Tensor:
        """Poisson cozumu, rhs_hat onceden hesaplanmis. 0 FFT + 1 irFFT."""
        p_hat = rhs_hat / (-self.k_squared_poisson)
        p_hat[..., 0, 0, 0] = 0.0
        return safe_irfftn(p_hat, s=self.spatial_shape)

    def dealias_from_hat(self, f_hat: torch.Tensor) -> torch.Tensor:
        """Dealias, f_hat onceden hesaplanmis. 0 FFT + 1 irFFT."""
        return safe_irfftn(f_hat * self.dealias_mask, s=self.spatial_shape)

    def band_filter(self, f: torch.Tensor, band: str) -> torch.Tensor:
        """
        Frekans bandi filtreleme: f'nin belirli banddaki bilesenini dondur.
        band: 'low', 'mid', 'high'
        """
        f_hat = safe_rfftn(f)
        mask = getattr(self, f'band_{band}')
        return safe_irfftn(f_hat * mask, s=self.spatial_shape)

    def curl(self, u: torch.Tensor, v: torch.Tensor, w: torch.Tensor
             ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Curl: omega = curl(u,v,w).  u,v,w: [B, Nx, Ny, Nz]"""
        u_hat = safe_rfftn(u)
        v_hat = safe_rfftn(v)
        w_hat = safe_rfftn(w)
        s = self.spatial_shape
        ox = safe_irfftn(1j * (self.ky * w_hat - self.kz * v_hat), s=s)
        oy = safe_irfftn(1j * (self.kz * u_hat - self.kx * w_hat), s=s)
        oz = safe_irfftn(1j * (self.kx * v_hat - self.ky * u_hat), s=s)
        return ox, oy, oz

    def divergence(self, u: torch.Tensor, v: torch.Tensor, w: torch.Tensor
                   ) -> torch.Tensor:
        """Divergence: div(u,v,w).  u,v,w: [B, Nx, Ny, Nz]"""
        u_hat = safe_rfftn(u)
        v_hat = safe_rfftn(v)
        w_hat = safe_rfftn(w)
        return safe_irfftn(
            1j * (self.kx * u_hat + self.ky * v_hat + self.kz * w_hat),
            s=self.spatial_shape
        )

    def dealias(self, f: torch.Tensor) -> torch.Tensor:
        """2/3 kurali ile dealiasing.  f: [B, Nx, Ny, Nz]"""
        f_hat = safe_rfftn(f)
        return safe_irfftn(f_hat * self.dealias_mask, s=self.spatial_shape)

    def solve_poisson(self, rhs: torch.Tensor) -> torch.Tensor:
        """Poisson denklemi: nabla^2 p = rhs.  rhs: [B, Nx, Ny, Nz]"""
        rhs_hat = safe_rfftn(rhs)
        p_hat = rhs_hat / (-self.k_squared_poisson)
        p_hat[..., 0, 0, 0] = 0.0  # mean pressure = 0
        return safe_irfftn(p_hat, s=self.spatial_shape)


class FiniteDiffOps3D(nn.Module):
    """
    3D Finite Difference türev operatörleri - Genel boundary conditions için.

    SpectralOps3D'un aksine HER TÜRLÜ sınır koşuluyla çalışır:
    - Periodic
    - Dirichlet (fixed value)
    - Neumann (fixed gradient)

    Kullanım:
        ops = FiniteDiffOps3D(64, domain_size=2*pi)
        df_dx, df_dy, df_dz = ops.gradient(f)
        lap_f = ops.laplacian(f)
    """
    def __init__(self, resolution: int, domain_size: float = 2*math.pi,
                 scheme: str = 'central', bc_type: str = 'periodic'):
        super().__init__()
        self.resolution = resolution
        self.domain_size = domain_size
        self.dx = domain_size / resolution
        self.scheme = scheme
        self.bc_type = bc_type

    def _get_padding_mode(self) -> str:
        """BC tipine göre padding modu"""
        if self.bc_type == 'periodic':
            return 'circular'
        elif self.bc_type == 'neumann':
            return 'replicate'
        else:  # dirichlet
            return 'constant'

    def gradient(self, f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """∇f = (∂f/∂x, ∂f/∂y, ∂f/∂z) using central differences"""
        # Ensure 5D: [B, C, D, H, W]
        squeeze_dims = []
        if f.dim() == 3:
            f = f.unsqueeze(0).unsqueeze(0)
            squeeze_dims = [0, 1]
        elif f.dim() == 4:
            f = f.unsqueeze(1)
            squeeze_dims = [1]

        pad_mode = self._get_padding_mode()
        pad_value = 0 if self.bc_type == 'dirichlet' else None

        # Pad: (W_left, W_right, H_left, H_right, D_left, D_right)
        if pad_mode == 'constant':
            f_pad = F.pad(f, (1, 1, 1, 1, 1, 1), mode='constant', value=0)
        else:
            f_pad = F.pad(f, (1, 1, 1, 1, 1, 1), mode=pad_mode)

        # Central differences: (f[i+1] - f[i-1]) / (2*dx)
        df_dx = (f_pad[:, :, 2:, 1:-1, 1:-1] - f_pad[:, :, :-2, 1:-1, 1:-1]) / (2 * self.dx)
        df_dy = (f_pad[:, :, 1:-1, 2:, 1:-1] - f_pad[:, :, 1:-1, :-2, 1:-1]) / (2 * self.dx)
        df_dz = (f_pad[:, :, 1:-1, 1:-1, 2:] - f_pad[:, :, 1:-1, 1:-1, :-2]) / (2 * self.dx)

        for dim in reversed(squeeze_dims):
            df_dx = df_dx.squeeze(dim)
            df_dy = df_dy.squeeze(dim)
            df_dz = df_dz.squeeze(dim)

        return df_dx, df_dy, df_dz

    def laplacian(self, f: torch.Tensor) -> torch.Tensor:
        """∇²f = ∂²f/∂x² + ∂²f/∂y² + ∂²f/∂z²"""
        squeeze_dims = []
        if f.dim() == 3:
            f = f.unsqueeze(0).unsqueeze(0)
            squeeze_dims = [0, 1]
        elif f.dim() == 4:
            f = f.unsqueeze(1)
            squeeze_dims = [1]

        pad_mode = self._get_padding_mode()
        if pad_mode == 'constant':
            f_pad = F.pad(f, (1, 1, 1, 1, 1, 1), mode='constant', value=0)
        else:
            f_pad = F.pad(f, (1, 1, 1, 1, 1, 1), mode=pad_mode)

        # 7-point stencil Laplacian
        center = f_pad[:, :, 1:-1, 1:-1, 1:-1]
        lap = (f_pad[:, :, 2:, 1:-1, 1:-1] + f_pad[:, :, :-2, 1:-1, 1:-1] +
               f_pad[:, :, 1:-1, 2:, 1:-1] + f_pad[:, :, 1:-1, :-2, 1:-1] +
               f_pad[:, :, 1:-1, 1:-1, 2:] + f_pad[:, :, 1:-1, 1:-1, :-2] -
               6 * center) / (self.dx ** 2)

        for dim in reversed(squeeze_dims):
            lap = lap.squeeze(dim)

        return lap

    def curl(self, u: torch.Tensor, v: torch.Tensor, w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """3D Curl: ω = ∇×u"""
        du_dx, du_dy, du_dz = self.gradient(u)
        dv_dx, dv_dy, dv_dz = self.gradient(v)
        dw_dx, dw_dy, dw_dz = self.gradient(w)

        omega_x = dw_dy - dv_dz
        omega_y = du_dz - dw_dx
        omega_z = dv_dx - du_dy

        return omega_x, omega_y, omega_z

    def divergence(self, u: torch.Tensor, v: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """3D Divergence: ∇·u = ∂u/∂x + ∂v/∂y + ∂w/∂z"""
        du_dx, _, _ = self.gradient(u)
        _, dv_dy, _ = self.gradient(v)
        _, _, dw_dz = self.gradient(w)
        return du_dx + dv_dy + dw_dz

    def dealias(self, f: torch.Tensor) -> torch.Tensor:
        """No dealiasing needed for finite differences"""
        return f

    def solve_poisson(self, rhs: torch.Tensor, max_iter: int = 100, tol: float = 1e-6) -> torch.Tensor:
        """
        3D Poisson denklemi çöz: ∇²p = rhs (Iteratif - SOR)
        """
        squeeze_dims = []
        if rhs.dim() == 3:
            rhs = rhs.unsqueeze(0)
            squeeze_dims = [0]

        B, D, H, W = rhs.shape

        # Compatibility condition
        rhs = rhs - rhs.mean(dim=(-3, -2, -1), keepdim=True)

        p = torch.zeros_like(rhs)
        omega = 1.0  # Pure Jacobi - more stable
        dx2 = self.dx ** 2

        for iteration in range(max_iter):
            p_old = p.clone()

            # Padding
            pad_mode = self._get_padding_mode()
            if pad_mode == 'constant':
                p_pad = F.pad(p.unsqueeze(1), (1, 1, 1, 1, 1, 1), mode='constant', value=0).squeeze(1)
            else:
                p_pad = F.pad(p.unsqueeze(1), (1, 1, 1, 1, 1, 1), mode=pad_mode).squeeze(1)

            # 6 neighbors
            neighbors_sum = (p_pad[:, 2:, 1:-1, 1:-1] + p_pad[:, :-2, 1:-1, 1:-1] +
                           p_pad[:, 1:-1, 2:, 1:-1] + p_pad[:, 1:-1, :-2, 1:-1] +
                           p_pad[:, 1:-1, 1:-1, 2:] + p_pad[:, 1:-1, 1:-1, :-2])

            p_jacobi = (neighbors_sum - dx2 * rhs) / 6.0
            p = (1 - omega) * p_old + omega * p_jacobi
            p = p - p.mean(dim=(-3, -2, -1), keepdim=True)

            if torch.abs(p - p_old).max() < tol:
                break

        for dim in reversed(squeeze_dims):
            p = p.squeeze(dim)

        return p

    def extra_repr(self) -> str:
        return f"resolution={self.resolution}, scheme={self.scheme}, bc={self.bc_type}"


# =============================================================================
# BÖLÜM 3: FİZİK-BİLEN NÖRONLAR (2D)
# =============================================================================

class Advection(nn.Module):
    """
    (i) Adveksiyon Nöronu: u·∇ yapısını doğrudan temsil eder

    TEK NÖRON = TEK FİZİKSEL OPERATÖR
    Parametresiz fiziksel hesaplama + tek modülasyon katsayısı

    Öğrenilen: advection_modulator (1 parametre)

    Immersed Boundary Desteği:
        ImmersedBoundary.register_neuron() ile sınır bilgisi aktarılır.
        Sınır noktalarında adveksiyon sıfırlanır.

    Args:
        resolution: Grid çözünürlüğü
        diff_ops: Paylaşılan SpectralOps (opsiyonel, bellek tasarrufu için)
    """
    def __init__(self, resolution: int, diff_ops: Optional['SpectralOps'] = None):
        super().__init__()
        self.diff_ops = diff_ops if diff_ops is not None else SpectralOps(resolution)

        # TEK ÖĞRENİLEBİLİR PARAMETRE
        self.advection_modulator = nn.Parameter(torch.ones(1))

        # Immersed Boundary desteği (register_neuron ile set edilir)
        self.boundary_mask: Optional[torch.Tensor] = None
        self.wall_velocity: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    def forward(self, state: FluidState) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        u·∇u ve u·∇v hesapla - skew-symmetric form (enerji korunumu için).

        Sınır noktalarında adveksiyon = 0 (katı cisimde akış yok)
        """
        u, v = state.u, state.v

        # Spektral gradyanlar
        du_dx, du_dy = self.diff_ops.gradient(u)
        dv_dx, dv_dy = self.diff_ops.gradient(v)

        # ================================================================
        # SKEW-SYMMETRIC ADVECTION (Enerji Korunumu İçin)
        # 0.5 * (convective + divergence form)
        # ================================================================
        # Convective form: (u·∇)u
        conv_u = u * du_dx + v * du_dy
        conv_v = u * dv_dx + v * dv_dy

        # Divergence form: ∇·(u⊗u)
        d_uu_dx, _ = self.diff_ops.gradient(u * u)
        _, d_vu_dy = self.diff_ops.gradient(v * u)
        div_form_u = d_uu_dx + d_vu_dy

        d_uv_dx, _ = self.diff_ops.gradient(u * v)
        _, d_vv_dy = self.diff_ops.gradient(v * v)
        div_form_v = d_uv_dx + d_vv_dy

        # Skew-symmetric: 0.5 * (convective + divergence)
        adv_u = 0.5 * (conv_u + div_form_u)
        adv_v = 0.5 * (conv_v + div_form_v)

        # Dealiasing - 2/3 kuralı
        adv_u = self.diff_ops.dealias(adv_u)
        adv_v = self.diff_ops.dealias(adv_v)

        # Immersed Boundary: sınırda adveksiyon = 0
        if self.boundary_mask is not None:
            mask = self.boundary_mask
            if mask.dim() == 2 and adv_u.dim() == 3:
                mask = mask.unsqueeze(0)
            adv_u = torch.where(mask, torch.zeros_like(adv_u), adv_u)
            adv_v = torch.where(mask, torch.zeros_like(adv_v), adv_v)

        # Tek modülasyon parametresi
        return self.advection_modulator * adv_u, self.advection_modulator * adv_v

    def extra_repr(self) -> str:
        return f"modulator={self.advection_modulator.item():.4f}, has_boundary={self.boundary_mask is not None}"


class Vorticity(nn.Module):
    """
    (ii) Vortisite/Curl Nöronu: ∇×u üzerinden dönel yapılar
    
    TEK NÖRON = TEK FİZİKSEL OPERATÖR
    Kelvin teoremi: sirkülasyon korunumu yapısal olarak built-in
    
    Öğrenilen: circulation_preservation (1 parametre)
    
    ⚠️ HIZ-VORTİSİTE COUPLING HAKKINDA NOT:
    Bu modelde velocity-vorticity formülasyonu "loosely coupled":
    - u, v → ω hesaplanır (curl)
    - ω bağımsız olarak evrilir (adveksiyon + difüzyon)
    - Ama ω → u, v doğrudan geri besleme YOKTUR
    
    Tam coupled formülasyon için (DNS seviyesi):
    - Stream function: ∇²ψ = -ω, u = ∂ψ/∂y, v = -∂ψ/∂x
    - Biot-Savart integrali
    
    Bu tasarım tercihi bilinçlidir:
    - Avantaj: Modüler, her nöron bağımsız kullanılabilir
    - Dezavantaj: Uzun simülasyonlarda ω-u tutarsızlığı birikebilir
    - Çözüm: Periyodik olarak ω = curl(u,v) ile senkronize et
    
    Args:
        resolution: Grid çözünürlüğü
        diff_ops: Paylaşılan SpectralOps (opsiyonel)
    """
    def __init__(self, resolution: int, diff_ops: Optional['SpectralOps'] = None):
        super().__init__()
        self.diff_ops = diff_ops if diff_ops is not None else SpectralOps(resolution)
        
        # TEK ÖĞRENİLEBİLİR PARAMETRE: sirkülasyon koruma katsayısı
        self.circulation_preservation = nn.Parameter(torch.tensor(0.99))
    
    def forward(self, state: FluidState) -> torch.Tensor:
        """
        Vortisite evrimini hesapla: ∂ω/∂t - SAF FİZİK
        
        2D N-S vortisite formu:
        ∂ω/∂t = -u·∇ω (adveksiyon, difüzyon ayrı eklenir)
        """
        omega = state.vorticity
        u, v = state.u, state.v
        
        # Vortisite gradyanları (parametresiz)
        domega_dx, domega_dy = self.diff_ops.gradient(omega)
        
        # Vortisite adveksiyonu: u·∇ω - SAF FİZİK
        vort_advection = u * domega_dx + v * domega_dy
        vort_advection = self.diff_ops.dealias(vort_advection)
        
        # Kelvin teoremi: sirkülasyon korunmalı
        preservation = torch.clamp(self.circulation_preservation, 0.9, 1.0)

        return -vort_advection * preservation

    def extra_repr(self) -> str:
        return f"circulation_preservation={self.circulation_preservation.item():.4f}"


class Projection(nn.Module):
    """
    (iii) Divergence-Free Projeksiyon Nöronu (Helmholtz Ayrışımı)
    
    TEK NÖRON = SIKIŞTIRILAMAZLIK GARANTİSİ
    ∇·u = 0 yapısal olarak sağlanır, loss'a bırakılmaz!
    
    Öğrenilen: pressure_scale (1 parametre)
    
    ═══════════════════════════════════════════════════════════════════
    İKİ MOD DESTEKLENİR:
    ═══════════════════════════════════════════════════════════════════
    
    1. Helmholtz Projeksiyon (dt=None):
       ∇²p = ∇·u
       u_new = u - ∇p
       → Sonuç: ∇·u_new = ∇·u - ∇²p = ∇·u - ∇·u = 0 ✓
    
    2. Fractional-Step / Chorin Projeksiyon (dt verilirse):
       ∇²p = ∇·u / dt    ← dt ile bölünür!
       u_new = u - dt×∇p  ← dt ile çarpılır!
       → Sonuç: ∇·u_new = ∇·u - dt×∇²p = ∇·u - dt×(∇·u/dt) = 0 ✓
    
    Her iki mod da matematiksel olarak tam divergence-free garanti eder.
    
    ⚠️ NOT: Eski versiyonda Poisson'da /dt yoktu ama güncellemede ×dt
       vardı. Bu durumda ∇·u_new = (1-dt)×∇·u ≠ 0 oluyordu!
       Bu hata düzeltildi.
    
    Args:
        resolution: Grid çözünürlüğü
        diff_ops: Paylaşılan SpectralOps (opsiyonel)
    """
    def __init__(self, resolution: int, diff_ops: Optional['SpectralOps'] = None):
        super().__init__()
        self.resolution = resolution
        self.diff_ops = diff_ops if diff_ops is not None else SpectralOps(resolution)
        
        # TEK ÖĞRENİLEBİLİR PARAMETRE: basınç ölçek faktörü
        self.pressure_scale = nn.Parameter(torch.ones(1))
        
        # Immersed Boundary desteği (register_neuron ile set edilir)
        self.boundary_mask: Optional[torch.Tensor] = None
        self.wall_velocity: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    
    def forward(self, u: torch.Tensor, v: torch.Tensor, dt: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Hız alanını divergence-free uzaya projekte et - SAF FİZİK
        
        İki mod desteklenir:
        
        1. Helmholtz Projeksiyon (dt=None):
           ∇²p = ∇·u
           u_new = u - ∇p
           → Tam divergence-free: ∇·u_new = 0
        
        2. Fractional-Step / Chorin Projeksiyon (dt verilirse):
           ∇²p = ∇·u / dt
           u_new = u - dt × ∇p
           → Tam divergence-free: ∇·u_new = ∇·u - dt × (∇·u/dt) = 0
        
        Immersed Boundary:
           Projeksiyon sonrası sınırda wall_velocity zorlanır.
        
        ⚠️ ÖNCEKİ HATA: Poisson'da dt yoktu ama güncellemede dt vardı.
           Bu durumda ∇·u_new = (1-dt)×∇·u ≠ 0 oluyordu!
           Şimdi her iki mod da matematiksel olarak tutarlı.
        """
        # Mevcut divergence (parametresiz)
        div_u = self.diff_ops.divergence(u, v)
        
        # pressure_scale'i [0.5, 2.0] aralığında tut (fiziksel sınırlar)
        p_scale = torch.clamp(self.pressure_scale, 0.5, 2.0)
        
        # Basınç Poisson denklemi
        # NOT: pressure_scale Poisson RHS'ına gömüldü!
        # Bu sayede projeksiyon her zaman tam divergence-free olur.
        # Eskiden p'yi çarpıyordu → (1 - pressure_scale)×∇·u ≠ 0 hatası!
        if dt is not None:
            # Fractional-step: ∇²p = ∇·u / (dt × pressure_scale)
            div_for_poisson = div_u / (dt * p_scale + 1e-10)
        else:
            # Helmholtz: ∇²p = ∇·u / pressure_scale
            div_for_poisson = div_u / (p_scale + 1e-10)

        # Operatör-agnostik Poisson çözümü
        # SpectralOps: FFT ile doğrudan
        # FiniteDiffOps: SOR ile iteratif
        p = self.diff_ops.solve_poisson(div_for_poisson)
        
        # NOT: p artık pressure_scale ile çarpılmıyor!
        # Ölçek Poisson'a gömüldü, projeksiyon tam divergence-free.
        
        # Projeksiyon
        dp_dx, dp_dy = self.diff_ops.gradient(p)
        
        if dt is not None:
            # Fractional-step: u - dt × pressure_scale × ∇p
            # (Poisson'da /pressure_scale var, burada ×pressure_scale → birbirini götürür)
            u_proj = u - dt * p_scale * dp_dx
            v_proj = v - dt * p_scale * dp_dy
        else:
            # Helmholtz: u - pressure_scale × ∇p
            u_proj = u - p_scale * dp_dx
            v_proj = v - p_scale * dp_dy
        
        # Immersed Boundary: sınırda wall velocity zorla
        if self.boundary_mask is not None and self.wall_velocity is not None:
            mask = self.boundary_mask
            u_wall, v_wall = self.wall_velocity
            
            # Batch boyutu ekle
            if mask.dim() == 2 and u_proj.dim() == 3:
                mask = mask.unsqueeze(0)
                u_wall = u_wall.unsqueeze(0)
                v_wall = v_wall.unsqueeze(0)
            
            u_proj = torch.where(mask, u_wall.expand_as(u_proj), u_proj)
            v_proj = torch.where(mask, v_wall.expand_as(v_proj), v_proj)
        
        return u_proj, v_proj, p

    def divergence_error(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Divergence hatasını ölç - sıfır olmalı"""
        div = self.diff_ops.divergence(u, v)
        return div.abs().mean()

    def extra_repr(self) -> str:
        return f"pressure_scale={self.pressure_scale.item():.4f}, has_boundary={self.boundary_mask is not None}"


class TimeMarcher(nn.Module):
    """
    (iv) Zamana-Zorlamalı Nöron (Causal Time-Marching)

    TEK NÖRON = ZAMAN İLERLEMESİ
    Nedensellik yapısal: t₁ sadece t₀'dan türeyebilir

    Özellikler:
    - RK4 integrasyon (enerji korumalı, stabil)
    - Advective + Diffusive CFL koşulu
    - Adaptif zaman adımı

    Öğrenilen: dt_scale, stability_factor (2 parametre)
    """
    def __init__(self, resolution: int, dt_range: Tuple[float, float] = (0.0001, 0.05),
                 method: str = 'rk4'):
        super().__init__()
        self.resolution = resolution
        self.dt_min, self.dt_max = dt_range
        self.method = method  # 'euler', 'rk2', 'rk4'

        # TEK ÖĞRENİLEBİLİR PARAMETRELER
        self.dt_scale = nn.Parameter(torch.tensor(0.3))  # Daha konservatif başlangıç
        self.stability_factor = nn.Parameter(torch.tensor(0.99))

        # Grid spacing
        self.dx = 2 * math.pi / resolution

    def compute_adaptive_dt(self, state: FluidState, nu: float = 0.001) -> torch.Tensor:
        """
        Advective + Diffusive CFL koşulu.

        dt = min(dt_adv, dt_diff) * safety_factor

        Advective CFL: dt ≤ C * dx / max|u|
        Diffusive CFL: dt ≤ dx² / (2 * d * ν)  (d=2 for 2D)
        """
        # Maksimum hız büyüklüğü
        max_velocity = torch.amax(state.velocity_magnitude(), dim=(-2, -1))

        # Advective CFL
        cfl_adv = self.dx / (max_velocity + 1e-8)

        # Diffusive CFL (dx² / 4ν for 2D)
        # nu Tensor olabilir, scalar'a çevir
        if isinstance(nu, torch.Tensor):
            nu_val = nu.item() if nu.numel() == 1 else nu.mean().item()
        else:
            nu_val = float(nu)
        cfl_diff = (self.dx ** 2) / (4 * nu_val + 1e-10)

        # Minimum al (her iki kısıtı da sağla)
        cfl_dt = torch.minimum(cfl_adv, torch.full_like(cfl_adv, cfl_diff))

        # Öğrenilen ölçek faktörü ile modüle et
        dt_scale_clamped = torch.clamp(self.dt_scale, 0.05, 0.5)
        dt = dt_scale_clamped * cfl_dt

        # dt aralığına kısıtla
        dt = torch.clamp(dt, self.dt_min, self.dt_max)

        return dt

    def euler_step(self, u: torch.Tensor, v: torch.Tensor,
                   du_dt: torch.Tensor, dv_dt: torch.Tensor,
                   dt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward Euler: u_{n+1} = u_n + dt * f(u_n)"""
        return u + dt * du_dt, v + dt * dv_dt

    def rk2_step(self, u: torch.Tensor, v: torch.Tensor,
                 du_dt: torch.Tensor, dv_dt: torch.Tensor,
                 dt: torch.Tensor, rhs_func) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        RK2 (Heun's method):
        k1 = f(u_n)
        k2 = f(u_n + dt * k1)
        u_{n+1} = u_n + 0.5 * dt * (k1 + k2)
        """
        # k1 already computed as du_dt, dv_dt
        k1_u, k1_v = du_dt, dv_dt

        # Predictor step
        u_pred = u + dt * k1_u
        v_pred = v + dt * k1_v

        # k2 = f(u_pred, v_pred)
        k2_u, k2_v = rhs_func(u_pred, v_pred)

        # Final update
        u_new = u + 0.5 * dt * (k1_u + k2_u)
        v_new = v + 0.5 * dt * (k1_v + k2_v)

        return u_new, v_new

    def rk4_step(self, u: torch.Tensor, v: torch.Tensor,
                 du_dt: torch.Tensor, dv_dt: torch.Tensor,
                 dt: torch.Tensor, rhs_func) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Classical RK4:
        k1 = f(t_n, u_n)
        k2 = f(t_n + dt/2, u_n + dt/2 * k1)
        k3 = f(t_n + dt/2, u_n + dt/2 * k2)
        k4 = f(t_n + dt, u_n + dt * k3)
        u_{n+1} = u_n + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        """
        # k1 already computed
        k1_u, k1_v = du_dt, dv_dt

        # k2
        u_k2 = u + 0.5 * dt * k1_u
        v_k2 = v + 0.5 * dt * k1_v
        k2_u, k2_v = rhs_func(u_k2, v_k2)

        # k3
        u_k3 = u + 0.5 * dt * k2_u
        v_k3 = v + 0.5 * dt * k2_v
        k3_u, k3_v = rhs_func(u_k3, v_k3)

        # k4
        u_k4 = u + dt * k3_u
        v_k4 = v + dt * k3_v
        k4_u, k4_v = rhs_func(u_k4, v_k4)

        # Final update
        u_new = u + (dt / 6.0) * (k1_u + 2*k2_u + 2*k3_u + k4_u)
        v_new = v + (dt / 6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)

        return u_new, v_new

    def forward(
        self,
        state: FluidState,
        du_dt: torch.Tensor,
        dv_dt: torch.Tensor,
        dt: torch.Tensor,
        rhs_func=None
    ) -> FluidState:
        """
        Zaman ilerlemesi - RK4/RK2/Euler seçenekleriyle.

        Args:
            state: Mevcut durum
            du_dt: u için RHS (k1)
            dv_dt: v için RHS (k1)
            dt: Zaman adımı (dışarıdan geçirilir, tutarlılık için)
            rhs_func: RK2/RK4 için RHS fonksiyonu (u, v) -> (du_dt, dv_dt)

        Returns:
            Yeni FluidState
        """
        dt = dt.view(-1, 1, 1)  # Broadcasting için

        # Integrasyon metodu seç
        if self.method == 'rk4' and rhs_func is not None:
            u_new, v_new = self.rk4_step(state.u, state.v, du_dt, dv_dt, dt, rhs_func)
        elif self.method == 'rk2' and rhs_func is not None:
            u_new, v_new = self.rk2_step(state.u, state.v, du_dt, dv_dt, dt, rhs_func)
        else:
            # Euler fallback
            u_new, v_new = self.euler_step(state.u, state.v, du_dt, dv_dt, dt)

        # Yeni zaman
        t_new = state.t + dt.squeeze(-1).squeeze(-1)

        # Yeni durum oluştur
        new_state = FluidState(
            u=u_new,
            v=v_new,
            p=state.p,
            vorticity=state.vorticity,
            t=t_new
        )

        return new_state

    def extra_repr(self) -> str:
        return f"method={self.method}, dt_scale={self.dt_scale.item():.4f}, stability={self.stability_factor.item():.4f}"


class DataInjector(nn.Module):
    """
    (v) Seçici Veri-Enjeksiyon Kapısı
    
    TEK NÖRON = VERİ FÜZYONU
    Kritik bölgelerde gözlem verisini fiziksel olarak enjekte eder
    
    Öğrenilen: fusion_weight (1 parametre)
    
    Args:
        resolution: Grid çözünürlüğü
        diff_ops: Paylaşılan SpectralOps (opsiyonel)
    """
    def __init__(self, resolution: int, diff_ops: Optional['SpectralOps'] = None):
        super().__init__()
        self.resolution = resolution
        self.diff_ops = diff_ops if diff_ops is not None else SpectralOps(resolution)
        
        # TEK ÖĞRENİLEBİLİR PARAMETRE: veri füzyon ağırlığı
        self.fusion_weight = nn.Parameter(torch.tensor(0.1))
    
    def forward(
        self, 
        predicted_state: FluidState, 
        observed_data: Optional[Dict[str, torch.Tensor]] = None,
        injection_mask: Optional[torch.Tensor] = None
    ) -> FluidState:
        """
        Kritik bölgelerde gözlem verisini enjekte et - SAF FİZİK
        """
        if observed_data is None:
            return predicted_state
        
        # Kritiklik: vortisite gradyanı büyüklüğü (parametresiz)
        omega = predicted_state.vorticity
        domega_dx, domega_dy = self.diff_ops.gradient(omega)
        grad_omega = torch.sqrt(domega_dx**2 + domega_dy**2 + 1e-8)
        
        # Kritiklik haritası: normalize edilmiş gradyan
        criticality = grad_omega / (torch.amax(grad_omega, dim=(-2,-1), keepdim=True) + 1e-8)
        
        # Gate: kritiklik tabanlı (parametresiz)
        gate = torch.sigmoid(5 * (criticality - 0.5))  # Keskin geçiş
        
        # Lokasyon maskesi (varsa)
        if 'locations' in observed_data:
            locations = observed_data['locations']
            location_mask = torch.zeros_like(gate)
            if isinstance(locations, torch.Tensor):
                for b in range(locations.shape[0]):
                    for n in range(locations.shape[1]):
                        i, j = locations[b, n, 0].long(), locations[b, n, 1].long()
                        if 0 <= i < self.resolution and 0 <= j < self.resolution:
                            location_mask[b, i, j] = 1.0
            else:
                for loc in locations:
                    i, j = int(loc[0]), int(loc[1])
                    if 0 <= i < self.resolution and 0 <= j < self.resolution:
                        location_mask[:, i, j] = 1.0
            gate = gate * location_mask
        
        if injection_mask is not None:
            gate = gate * injection_mask.float()
        
        # Veri enjeksiyonu
        if 'u' in observed_data and 'v' in observed_data:
            w = torch.clamp(self.fusion_weight, 0, 0.5)
            blend = gate * w
            
            u_new = predicted_state.u * (1 - blend) + observed_data['u'] * blend
            v_new = predicted_state.v * (1 - blend) + observed_data['v'] * blend
            
            return FluidState(
                u=u_new, v=v_new, p=predicted_state.p,
                vorticity=predicted_state.vorticity, t=predicted_state.t
            )
        
        return predicted_state
    
    def get_criticality_map(self, state: FluidState) -> torch.Tensor:
        """Kritik bölge haritası (parametresiz)"""
        domega_dx, domega_dy = self.diff_ops.gradient(state.vorticity)
        grad_omega = torch.sqrt(domega_dx**2 + domega_dy**2 + 1e-8)
        return grad_omega / (torch.amax(grad_omega, dim=(-2,-1), keepdim=True) + 1e-8)

    def extra_repr(self) -> str:
        return f"fusion_weight={self.fusion_weight.item():.4f}"


class Reynolds(nn.Module):
    """
    (vi) Re-Öğrenen Parametrik Nöron
    
    TEK NÖRON = REYNOLDS SAYISI
    Re = UL/ν fiziksel olarak kısıtlı öğrenilebilir parametre
    
    Öğrenilen: log_re_raw (1 parametre)
    """
    def __init__(self, re_range: Tuple[float, float] = (50, 2000)):
        super().__init__()
        self.re_min, self.re_max = re_range
        self.log_re_min = math.log(re_range[0])
        self.log_re_max = math.log(re_range[1])
        
        # TEK ÖĞRENİLEBİLİR PARAMETRE: log Reynolds
        self.log_re_raw = nn.Parameter(torch.tensor(0.0))
    
    @property
    def reynolds(self) -> torch.Tensor:
        """Fiziksel olarak geçerli Reynolds sayısı"""
        normalized = torch.sigmoid(self.log_re_raw)
        log_re = self.log_re_min + (self.log_re_max - self.log_re_min) * normalized
        return torch.exp(log_re)
    
    @property
    def viscosity(self) -> torch.Tensor:
        """ν = 1/Re (boyutsuz formda)"""
        return 1.0 / self.reynolds
    
    def get_regime(self) -> FlowRegime:
        """Mevcut akış rejimini belirle"""
        re = self.reynolds.item()
        if re < 300:
            return FlowRegime.LAMINAR
        elif re < 1000:
            return FlowRegime.TRANSITIONAL
        else:
            return FlowRegime.VORTEX_DOMINANT
    
    def forward(self) -> Dict[str, torch.Tensor]:
        """
        Re'ye bağlı fiziksel parametreleri döndür - SAF FİZİK
        """
        re = self.reynolds
        
        # Fiziksel ölçekleme (parametresiz formüller)
        # Difüzyon ölçeği: 1/Re'ye orantılı
        diffusion_scale = torch.ones(1, device=re.device)
        # Adveksiyon ölçeği: sabit
        advection_scale = torch.ones(1, device=re.device)
        
        return {
            'reynolds': re,
            'viscosity': self.viscosity,
            'diffusion_scale': diffusion_scale,
            'advection_scale': advection_scale,
            'regime': self.get_regime()
        }
    
    def regularization_loss(self) -> torch.Tensor:
        """Re için fiziksel regularizasyon"""
        re = self.reynolds
        lower_penalty = F.relu(self.re_min - re)
        upper_penalty = F.relu(re - self.re_max)
        return lower_penalty + upper_penalty

    def extra_repr(self) -> str:
        return f"Re={self.reynolds.item():.1f}, nu={self.viscosity.item():.6f}, regime={self.get_regime().name}"


class Boundary(nn.Module):
    """
    (vii) BC/IC Öğrenen Sınır Nöronu
    
    TEK NÖRON = SINIR KOŞULLARI
    Periyodik: FFT zaten sağlar (parametresiz)
    No-slip: Sert projeksiyon (parametresiz)
    
    Öğrenilen: violation_penalty_weight (1 parametre)
    """
    def __init__(self, resolution: int, bc_type: str = 'periodic'):
        super().__init__()
        self.resolution = resolution
        self.bc_type = bc_type
        
        # TEK ÖĞRENİLEBİLİR PARAMETRE: ihlal ceza ağırlığı
        self.violation_penalty_weight = nn.Parameter(torch.tensor(10.0))
    
    def forward(self, state: FluidState, boundary_info: Optional[Dict] = None) -> FluidState:
        """
        Sınır koşullarını uygula - SAF FİZİK
        """
        if self.bc_type == 'periodic':
            # FFT zaten periyodik - parametresiz
            return state
        
        elif self.bc_type == 'no_slip':
            # Duvar sınırlarında u=v=0 zorla (hard constraint)
            u, v = state.u.clone(), state.v.clone()
            
            # Kenar maskeleme (parametresiz)
            u[:, 0, :] = 0; u[:, -1, :] = 0
            u[:, :, 0] = 0; u[:, :, -1] = 0
            v[:, 0, :] = 0; v[:, -1, :] = 0
            v[:, :, 0] = 0; v[:, :, -1] = 0
            
            return FluidState(
                u=u, v=v, p=state.p,
                vorticity=state.vorticity, t=state.t
            )
        
        return state
    
    def boundary_violation_loss(self, state: FluidState) -> torch.Tensor:
        """Sınır ihlal cezası"""
        if self.bc_type == 'no_slip':
            u, v = state.u, state.v
            boundary_u = torch.cat([u[:, 0, :], u[:, -1, :], u[:, :, 0], u[:, :, -1]], dim=-1)
            boundary_v = torch.cat([v[:, 0, :], v[:, -1, :], v[:, :, 0], v[:, :, -1]], dim=-1)
            violation = (boundary_u**2 + boundary_v**2).mean()
            return self.violation_penalty_weight * violation

        return torch.tensor(0.0, device=state.u.device)

    def extra_repr(self) -> str:
        return f"bc_type={self.bc_type}, penalty_weight={self.violation_penalty_weight.item():.4f}"


# =============================================================================
# BÖLÜM 4: ANA MODEL - INNATE
# =============================================================================

class INNATE(nn.Module):
    """
    INNATE: Intrinsic Navier-Stokes Neural Architecture for Temporal Evolution
    
    Tüm fizik-bilen nöronları birleştiren ana model.
    
    Bu bir PINN DEĞİL. Bu bir Physics-Native Neural Operator.
    
    Loss minimal. Kısıt nöronun DNA'sında.
    """
    def __init__(
        self, 
        resolution: int = 64,
        re_range: Tuple[float, float] = (50, 2000),
        bc_type: str = 'periodic'
    ):
        super().__init__()
        self.resolution = resolution
        
        # PAYLAŞILAN SpectralOps (bellek tasarrufu)
        self.diff_ops = SpectralOps(resolution)
        
        # Fizik-bilen layerlar - PAYLAŞILAN diff_ops ile (nn.Conv2d gibi)
        self.advection = Advection(resolution, diff_ops=self.diff_ops)
        self.vorticity = Vorticity(resolution, diff_ops=self.diff_ops)
        self.projector = Projection(resolution, diff_ops=self.diff_ops)
        self.time_marcher = TimeMarcher(resolution)  # diff_ops kullanmıyor
        self.data_injector = DataInjector(resolution, diff_ops=self.diff_ops)
        self.reynolds_learner = Reynolds(re_range)  # diff_ops kullanmıyor
        self.boundary = Boundary(resolution, bc_type)  # diff_ops kullanmıyor
        
        # Difüzyon modülatörü
        self.diffusion_modulator = nn.Parameter(torch.ones(1))
        
        # Immersed Boundary (opsiyonel, register_boundary ile set edilir)
        self.immersed_boundary: Optional['ImmersedBoundary'] = None
    
    def register_boundary(self, ib: 'ImmersedBoundary') -> None:
        """
        Immersed Boundary modülünü kaydet ve tüm nöronlara sınır bilgisi aktar.
        
        Bu metod çağrıldığında:
        - Advection nöronu sınır noktalarında adveksiyon = 0 yapar
        - Projection nöronu sınır noktalarında wall velocity zorlar
        - step() sonunda IBM forcing uygulanır
        
        Args:
            ib: ImmersedBoundary modülü (geometri tanımlanmış olmalı)
        
        Kullanım:
            model = INNATE(resolution=64)
            ib = ImmersedBoundary(resolution=64)
            ib.set_cavity_geometry(lid_velocity=1.0)
            model.register_boundary(ib)
        """
        self.immersed_boundary = ib
        
        # Tüm ilgili nöronlara sınır bilgisi kaydet
        ib.register_neuron(self.advection)
        ib.register_neuron(self.projector)
    
    def unregister_boundary(self) -> None:
        """Immersed Boundary modülünü kaldır."""
        if self.immersed_boundary is not None:
            self.immersed_boundary.unregister_neuron(self.advection)
            self.immersed_boundary.unregister_neuron(self.projector)
            self.immersed_boundary = None
    
    def compute_rhs(self, state: FluidState) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Navier-Stokes sağ taraf hesabı:

        ∂u/∂t = -u·∇u - ∇p + ν∇²u
        ∂ω/∂t = -u·∇ω + ν∇²ω (vortisite formülasyonu)
        """
        # Reynolds parametreleri
        re_params = self.reynolds_learner()
        nu = re_params['viscosity']

        # Adveksiyon terimi: -u·∇u (skew-symmetric form)
        adv_u, adv_v = self.advection(state)

        # Difüzyon terimi: ν∇²u
        diff_u = self.diff_ops.laplacian(state.u)
        diff_v = self.diff_ops.laplacian(state.v)

        # Vortisite dinamiği (Vorticity layer)
        # ∂ω/∂t = -u·∇ω + ν∇²ω (TAM 2D NS vortisite denklemi)
        vort_advection = self.vorticity(state)  # -u·∇ω
        diff_omega = self.diff_ops.laplacian(state.vorticity)  # ∇²ω
        domega_dt = vort_advection + nu * diff_omega  # TAM DENKLEM

        # Modülasyon
        diff_scale = re_params['diffusion_scale'] * self.diffusion_modulator
        adv_scale = re_params['advection_scale']

        # Toplam RHS (basınç projeksiyon sonrası eklenecek)
        du_dt = -adv_scale * adv_u + nu * diff_scale * diff_u
        dv_dt = -adv_scale * adv_v + nu * diff_scale * diff_v

        return du_dt, dv_dt, domega_dt

    def _velocity_rhs(self, u: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        RK4 için velocity-only RHS fonksiyonu.

        Args:
            u, v: Hız alanları

        Returns:
            (du_dt, dv_dt): Velocity RHS
        """
        # Geçici state oluştur
        temp_state = FluidState(
            u=u, v=v,
            p=torch.zeros_like(u),
            vorticity=self.diff_ops.curl_2d(u, v),  # Her zaman senkron
            t=torch.tensor(0.0, device=u.device)
        )

        # Reynolds parametreleri
        re_params = self.reynolds_learner()
        nu = re_params['viscosity']

        # Adveksiyon (skew-symmetric)
        adv_u, adv_v = self.advection(temp_state)

        # Difüzyon
        diff_u = self.diff_ops.laplacian(u)
        diff_v = self.diff_ops.laplacian(v)

        # Modülasyon
        diff_scale = re_params['diffusion_scale'] * self.diffusion_modulator
        adv_scale = re_params['advection_scale']

        du_dt = -adv_scale * adv_u + nu * diff_scale * diff_u
        dv_dt = -adv_scale * adv_v + nu * diff_scale * diff_v

        return du_dt, dv_dt

    def step(
        self,
        state: FluidState,
        hidden: Optional[torch.Tensor] = None,
        observed_data: Optional[Dict] = None
    ) -> Tuple[FluidState, torch.Tensor]:
        """
        Tek zaman adımı - RK4 integrasyon ile.

        Yeni özellikler:
        - RK4 zaman integrasyonu (enerji korumalı)
        - Tutarlı dt kullanımı (tek hesaplama)
        - Vorticity senkronizasyonu (her adımda)
        - Spektral filtreleme (yüksek-k sönümleme)
        """
        # 1. Reynolds parametrelerini al (nu için)
        re_params = self.reynolds_learner()
        nu = re_params['viscosity']

        # 2. Adaptif dt hesapla - TEK SEFER (tutarlılık için)
        dt = self.time_marcher.compute_adaptive_dt(state, nu)

        # 3. İlk RHS hesapla (k1)
        du_dt, dv_dt, domega_dt = self.compute_rhs(state)

        # 4. RK4 ile zaman ilerlet
        new_state = self.time_marcher(
            state, du_dt, dv_dt, dt,
            rhs_func=self._velocity_rhs
        )

        # 5. Vortisite evrimini uygula (aynı dt ile)
        dt_3d = dt.view(-1, 1, 1)
        new_state.vorticity = state.vorticity + dt_3d * domega_dt

        # 6. Divergence-free projeksiyon
        new_state.u, new_state.v, new_state.p = self.projector(
            new_state.u, new_state.v, dt_3d
        )

        # 7. Spektral filtreleme - DEVRE DIŞI
        # NOT: Her adımda filtre uygulamak aşırı sönüme neden olur.
        # RK4 + skew-symmetric advection + dealiasing stabilite için yeterli.
        # Gerekirse sadece instabilite durumunda aktifleştir:
        # new_state.u = self.diff_ops.apply_filter(new_state.u)
        # new_state.v = self.diff_ops.apply_filter(new_state.v)

        # 8. Vorticity senkronizasyonu (drift önleme)
        # Her adımda curl(u,v) ile senkronize et
        new_state.vorticity = self.diff_ops.curl_2d(new_state.u, new_state.v)

        # 9. Sınır koşulları
        new_state = self.boundary(new_state)

        # 10. Immersed Boundary forcing (varsa)
        if self.immersed_boundary is not None:
            new_state.u, new_state.v = self.immersed_boundary.apply_forcing(
                new_state.u, new_state.v, dt_3d
            )

        # 11. Veri enjeksiyonu (varsa)
        if observed_data is not None:
            new_state = self.data_injector(new_state, observed_data)

        return new_state, None
    
    def forward(
        self, 
        initial_state: FluidState, 
        num_steps: int,
        observed_data: Optional[Dict] = None
    ) -> list:
        """
        Zaman serisini simüle et.
        """
        states = [initial_state]
        state = initial_state
        hidden = None
        
        for step_idx in range(num_steps):
            # Adım-spesifik veri
            step_data = None
            if observed_data is not None and step_idx in observed_data:
                step_data = observed_data[step_idx]
            
            state, hidden = self.step(state, hidden, step_data)
            states.append(state)
        
        return states
    
    def physics_loss(self, state: FluidState) -> Dict[str, torch.Tensor]:
        """
        Minimal loss - çoğu kısıt nöron DNA'sında.
        Burada sadece soft regularizasyonlar.
        """
        losses = {}
        
        # Divergence (teorik olarak 0 olmalı, kontrol amaçlı)
        div_error = self.projector.divergence_error(state.u, state.v)
        losses['divergence'] = div_error
        
        # Reynolds regularizasyonu
        losses['reynolds_reg'] = self.reynolds_learner.regularization_loss()
        
        # Sınır ihlali
        losses['boundary'] = self.boundary.boundary_violation_loss(state)
        
        # NOT: energy_reg KALDIRILDI
        # Türbülans için enerji düşmesini teşvik etmek YANLIŞ
        # Viskoz sönüm zaten fizik denklemlerinde var (ν∇²u)

        return losses

    # =========================================================================
    # DIAGNOSTIC METHODS - Fiziksel Metrikleri Public API Olarak Aç
    # =========================================================================

    def get_energy(self, state: FluidState) -> float:
        """Kinetik enerji: E = 0.5 * mean(u² + v²)"""
        return state.kinetic_energy().mean().item()

    def get_enstrophy(self, state: FluidState) -> float:
        """Enstrofi: Z = 0.5 * mean(ω²)"""
        return state.enstrophy().mean().item()

    def get_divergence(self, state: FluidState) -> float:
        """Divergence hatası: |∇·u| (sıfır olmalı)"""
        div = self.diff_ops.divergence(state.u, state.v)
        return div.abs().mean().item()

    def get_vorticity_magnitude(self, state: FluidState) -> float:
        """Vortisite büyüklüğü: |ω|"""
        return state.vorticity.abs().mean().item()

    def get_cfl_number(self, state: FluidState) -> float:
        """CFL sayısı: max(|u|) * dt / dx"""
        u_max = state.velocity_magnitude().max().item()
        # Gerçek viscosity değerini reynolds_learner'dan al
        nu = self.reynolds_learner()['viscosity']
        dt_tensor = self.time_marcher.compute_adaptive_dt(state, nu=nu)
        # Batch>1 için max dt kullan (en kötü durum CFL)
        dt = dt_tensor.max().item() if dt_tensor.numel() > 1 else dt_tensor.item()
        dx = 2 * math.pi / self.resolution
        return u_max * dt / dx

    def get_reynolds(self) -> float:
        """Öğrenilmiş Reynolds sayısı"""
        return self.reynolds_learner.reynolds.item()

    def get_diagnostics(self, state: FluidState) -> Dict[str, float]:
        """Tüm diagnostikleri tek seferde döndür"""
        return {
            'energy': self.get_energy(state),
            'enstrophy': self.get_enstrophy(state),
            'divergence': self.get_divergence(state),
            'vorticity_mag': self.get_vorticity_magnitude(state),
            'cfl': self.get_cfl_number(state),
            'reynolds': self.get_reynolds(),
        }

    def extra_repr(self) -> str:
        return f"resolution={self.resolution}, Re={self.get_reynolds():.1f}"


# =============================================================================
# BÖLÜM 5: EĞİTİM VE TEST
# =============================================================================

class INNATETrainer:
    """
    INNATE modeli için eğitim sınıfı.
    
    SOAP + Time-Marching Curriculum ile eğitim.
    """
    def __init__(self, model: INNATE, device: str = 'cuda'):
        self.model = model.to(device)
        self.device = device
        
        # SOAP optimizer (Adam + momentum scheduling)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=100, T_mult=2
        )
    
    def create_initial_state(self, batch_size: int) -> FluidState:
        """Test için başlangıç durumu oluştur"""
        resolution = self.model.resolution
        
        # Taylor-Green vortex initial condition
        x = torch.linspace(0, 2*math.pi, resolution, device=self.device)
        y = torch.linspace(0, 2*math.pi, resolution, device=self.device)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        
        u = torch.sin(X) * torch.cos(Y)
        v = -torch.cos(X) * torch.sin(Y)
        p = 0.25 * (torch.cos(2*X) + torch.cos(2*Y))
        
        # Batch'e genişlet
        u = u.unsqueeze(0).expand(batch_size, -1, -1)
        v = v.unsqueeze(0).expand(batch_size, -1, -1)
        p = p.unsqueeze(0).expand(batch_size, -1, -1)
        
        omega = self.model.diff_ops.curl_2d(u, v)
        
        return FluidState(
            u=u, v=v, p=p, vorticity=omega,
            t=torch.zeros(batch_size, 1, device=self.device)
        )
    
    def curriculum_train(self, num_epochs: int = 1000, max_steps: int = 100):
        """
        Curriculum learning: kısa simülasyonlardan uzun simülasyonlara.
        """
        self.model.train()
        
        for epoch in range(num_epochs):
            # Curriculum: epoch ilerledikçe daha uzun simülasyon
            current_steps = min(5 + epoch // 50, max_steps)
            
            # Başlangıç durumu
            initial_state = self.create_initial_state(batch_size=4)
            
            # Forward pass
            states = self.model(initial_state, num_steps=current_steps)
            
            # Loss hesapla
            total_loss = torch.tensor(0.0, device=self.device)
            for state in states:
                losses = self.model.physics_loss(state)
                for name, loss in losses.items():
                    total_loss = total_loss + loss
            
            # Stability loss: enerji patlamamalı
            energies = [s.kinetic_energy().mean() for s in states]
            energy_growth = torch.stack(energies[1:]) / (torch.stack(energies[:-1]) + 1e-8)
            stability_loss = F.relu(energy_growth - 1.5).mean()  # Max %50 büyüme
            total_loss = total_loss + 10 * stability_loss
            
            # Backward
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            if epoch % 100 == 0:
                re = self.model.reynolds_learner.reynolds.item()
                regime = self.model.reynolds_learner.get_regime().name
                final_energy = states[-1].kinetic_energy().mean().item()
                final_enstrophy = states[-1].enstrophy().mean().item()
                
                print(f"Epoch {epoch}: Loss={total_loss.item():.4f}, "
                      f"Re={re:.1f} ({regime}), "
                      f"Steps={current_steps}, "
                      f"Energy={final_energy:.4f}, "
                      f"Enstrophy={final_enstrophy:.4f}")
    
    def test_stability(self, num_steps: int = 500) -> Dict:
        """
        Stabilite testi: zorlama olmadan çözüm stabil kalmalı.
        """
        self.model.eval()
        
        with torch.no_grad():
            initial_state = self.create_initial_state(batch_size=1)
            states = self.model(initial_state, num_steps=num_steps)
            
            energies = [s.kinetic_energy().item() for s in states]
            enstrophies = [s.enstrophy().item() for s in states]
            
            # Türbülans sönmemeli (Re yüksekse)
            re = self.model.reynolds_learner.reynolds.item()
            final_enstrophy = enstrophies[-1]
            initial_enstrophy = enstrophies[0]
            
            results = {
                'reynolds': re,
                'regime': self.model.reynolds_learner.get_regime().name,
                'energy_decay': energies[-1] / energies[0],
                'enstrophy_ratio': final_enstrophy / initial_enstrophy,
                'stable': energies[-1] < energies[0] * 2,  # Patlamamış
                'turbulence_sustained': final_enstrophy > 0.1 * initial_enstrophy if re > 500 else True
            }
            
            return results


# =============================================================================
# BÖLÜM 6: 3D FİZİK-BİLEN NÖRONLAR
# =============================================================================
# PyTorch'ta nn.Conv2d ve nn.Conv3d ayrı olduğu gibi,
# burada da 2D ve 3D nöronlar ayrı sınıflar olarak tanımlanır.
# Her nöron = TEK fiziksel operatör + öğrenilebilir parametre(ler)
# =============================================================================

class Advection3D(nn.Module):
    """
    3D Adveksiyon Nöronu: u·∇ operatörü (3 bileşenli)
    
    TEK NÖRON = TEK FİZİKSEL OPERATÖR
    
    Hesaplar:
        adv_u = u·∂u/∂x + v·∂u/∂y + w·∂u/∂z
        adv_v = u·∂v/∂x + v·∂v/∂y + w·∂v/∂z
        adv_w = u·∂w/∂x + v·∂w/∂y + w·∂w/∂z
    
    Öğrenilen: advection_modulator (1 parametre)
    
    Immersed Boundary Desteği:
        ImmersedBoundary3D.register_neuron() ile sınır bilgisi aktarılır.
    
    Kullanım:
        adv = Advection3D(resolution=64)
        adv_u, adv_v, adv_w = adv(state)
    """
    def __init__(self, resolution: int, diff_ops: Optional['SpectralOps3D'] = None,
                 use_lamb: bool = True):
        super().__init__()
        self.diff_ops = diff_ops if diff_ops is not None else SpectralOps3D(resolution)
        self.use_lamb = use_lamb

        # TEK ÖĞRENİLEBİLİR PARAMETRE: adveksiyon şiddeti
        self.advection_modulator = nn.Parameter(torch.ones(1))

        # Immersed Boundary desteği
        self.boundary_mask: Optional[torch.Tensor] = None
        self.wall_velocity: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None

    def forward(self, state: FluidState3D,
                u_hat=None, v_hat=None, w_hat=None,
                vel_grads=None,
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        3D adveksiyon hesapla: u·∇u, u·∇v, u·∇w

        Iki mod desteklenir:
          use_lamb=True  (default): Rotational/Lamb form
            u·∇u = ω×u + ∇(|u|²/2)
            ∇(|u|²/2) Leray projector tarafindan sifirlanir → sadece ω×u hesaplanir.
            Maliyet: 6 FFT (3 rFFT + 3 irFFT dealiasing)
          use_lamb=False: Skew-symmetric form
            0.5*(convective + divergence)
            Maliyet: 21 FFT (6 rFFT product + 9 irFFT partial + 6 dealias)

        Args:
            state: FluidState3D
            u_hat, v_hat, w_hat: Onceden hesaplanmis FFT'ler (opsiyonel, caching icin)
            vel_grads: 9-tuple (du_dx,..,dw_dz) onceden hesaplanmis gradyanlar
                       (gradient sharing icin — EddyViscosity ile paylasim)

        Returns:
            (adv_u, adv_v, adv_w): Adveksiyon bileşenleri
        """
        u, v, w = state.u, state.v, state.w
        ops = self.diff_ops

        # Polimorfik FFT: SpectralOps3DAniso -> rfftn, SpectralOps3D -> fftn
        _fft = ops.to_hat if hasattr(ops, 'to_hat') else safe_fftn

        # Cache'den veya yeni hesapla
        if u_hat is None:
            u_hat = _fft(u)
        if v_hat is None:
            v_hat = _fft(v)
        if w_hat is None:
            w_hat = _fft(w)

        # Gradient sharing: disaridan verilmisse yeniden hesaplama (9 irFFT tasarruf)
        if vel_grads is not None:
            du_dx, du_dy, du_dz, dv_dx, dv_dy, dv_dz, dw_dx, dw_dy, dw_dz = vel_grads
        else:
            du_dx, du_dy, du_dz = ops.gradient_from_hat(u_hat)
            dv_dx, dv_dy, dv_dz = ops.gradient_from_hat(v_hat)
            dw_dx, dw_dy, dw_dz = ops.gradient_from_hat(w_hat)

        if self.use_lamb:
            # ==============================================================
            # LAMB (ROTATIONAL) FORM: u·∇u = ω×u + ∇(|u|²/2)
            # ∇(|u|²/2) curl-free → Leray projector sifirlar → hesaplamiyoruz
            # Maliyet: 0 FFT (vorticity + cross product) + 6 FFT (dealiasing)
            # ==============================================================

            # Vorticity: ω = ∇×u — vel_grads'tan, 0 FFT
            omega_x = dw_dy - dv_dz
            omega_y = du_dz - dw_dx
            omega_z = dv_dx - du_dy

            # Lamb vector: ω×u — fiziksel uzayda carpma, 0 FFT
            adv_u = omega_y * w - omega_z * v
            adv_v = omega_z * u - omega_x * w
            adv_w = omega_x * v - omega_y * u

        else:
            # ==============================================================
            # SKEW-SYMMETRIC FORM: 0.5 * (convective + divergence)
            # Maliyet: 6 rFFT (products) + 9 irFFT (partials) + 6 FFT (dealias)
            # ==============================================================

            # Convective form: (u . nabla) phi
            conv_u = u * du_dx + v * du_dy + w * du_dz
            conv_v = u * dv_dx + v * dv_dy + w * dv_dz
            conv_w = u * dw_dx + v * dw_dy + w * dw_dz

            # Divergence form: symmetric product reuse (6 rFFT + 9 irFFT)
            uu_hat = _fft(u * u)
            uv_hat = _fft(u * v)
            uw_hat = _fft(u * w)
            vv_hat = _fft(v * v)
            vw_hat = _fft(v * w)
            ww_hat = _fft(w * w)

            div_form_u = ops.partial_x(uu_hat) + ops.partial_y(uv_hat) + ops.partial_z(uw_hat)
            div_form_v = ops.partial_x(uv_hat) + ops.partial_y(vv_hat) + ops.partial_z(vw_hat)
            div_form_w = ops.partial_x(uw_hat) + ops.partial_y(vw_hat) + ops.partial_z(ww_hat)

            adv_u = 0.5 * (conv_u + div_form_u)
            adv_v = 0.5 * (conv_v + div_form_v)
            adv_w = 0.5 * (conv_w + div_form_w)

        # Dealiasing (her iki mod icin ZORUNLU)
        adv_u = self.diff_ops.dealias(adv_u)
        adv_v = self.diff_ops.dealias(adv_v)
        adv_w = self.diff_ops.dealias(adv_w)
        
        # Immersed Boundary: sınırda adveksiyon = 0
        if self.boundary_mask is not None:
            mask = self.boundary_mask
            if mask.dim() == 3 and adv_u.dim() == 4:
                mask = mask.unsqueeze(0)
            adv_u = torch.where(mask, torch.zeros_like(adv_u), adv_u)
            adv_v = torch.where(mask, torch.zeros_like(adv_v), adv_v)
            adv_w = torch.where(mask, torch.zeros_like(adv_w), adv_w)
        
        # Tek modülasyon parametresi (clamp: uzun unrolling'de ustel buyumeyi onler)
        mod = torch.clamp(self.advection_modulator, 0.5, 1.5)
        return mod * adv_u, mod * adv_v, mod * adv_w

    def extra_repr(self) -> str:
        form = "lamb" if self.use_lamb else "skew-symmetric"
        return f"form={form}, modulator={self.advection_modulator.item():.4f}, has_boundary={self.boundary_mask is not None}"


class Vorticity3D(nn.Module):
    """
    3D Vortisite Nöronu: ∇×u (vektörel curl)
    
    TEK NÖRON = TEK FİZİKSEL OPERATÖR
    
    Hesaplar (curl):
        ω_x = ∂w/∂y - ∂v/∂z
        ω_y = ∂u/∂z - ∂w/∂x
        ω_z = ∂v/∂x - ∂u/∂y
    
    Ve vortisite adveksiyonu: u·∇ω (her bileşen için)
    
    Öğrenilen: circulation_preservation (1 parametre)
    
    Kullanım:
        vort = Vorticity3D(resolution=64)
        domega_x_dt, domega_y_dt, domega_z_dt = vort(state)
    """
    def __init__(self, resolution: int, diff_ops: Optional['SpectralOps3D'] = None):
        super().__init__()
        self.diff_ops = diff_ops if diff_ops is not None else SpectralOps3D(resolution)
        
        # TEK ÖĞRENİLEBİLİR PARAMETRE: sirkülasyon koruma
        self.circulation_preservation = nn.Parameter(torch.tensor(0.99))
    
    def forward(self, state: FluidState3D) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        3D vortisite evrimini hesapla: ∂ω/∂t
        
        Vortisite denklemi (inviscid kısım):
        ∂ω/∂t = -u·∇ω + ω·∇u (vortex stretching 3D'de!)
        
        Args:
            state: FluidState3D
        
        Returns:
            (domega_x_dt, domega_y_dt, domega_z_dt): Vortisite türevleri
        """
        u, v, w = state.u, state.v, state.w
        omega_x, omega_y, omega_z = state.omega_x, state.omega_y, state.omega_z
        
        # Hız gradyanları (vortex stretching için)
        du_dx, du_dy, du_dz = self.diff_ops.gradient(u)
        dv_dx, dv_dy, dv_dz = self.diff_ops.gradient(v)
        dw_dx, dw_dy, dw_dz = self.diff_ops.gradient(w)
        
        # Vortisite gradyanları (adveksiyon için)
        domega_x_dx, domega_x_dy, domega_x_dz = self.diff_ops.gradient(omega_x)
        domega_y_dx, domega_y_dy, domega_y_dz = self.diff_ops.gradient(omega_y)
        domega_z_dx, domega_z_dy, domega_z_dz = self.diff_ops.gradient(omega_z)
        
        # Vortisite adveksiyonu: -u·∇ω
        adv_omega_x = -(u * domega_x_dx + v * domega_x_dy + w * domega_x_dz)
        adv_omega_y = -(u * domega_y_dx + v * domega_y_dy + w * domega_y_dz)
        adv_omega_z = -(u * domega_z_dx + v * domega_z_dy + w * domega_z_dz)
        
        # Vortex stretching: ω·∇u (3D'ye özgü, 2D'de YOK!)
        # Bu terim türbülans enerji kaskadının temelidir
        stretch_x = omega_x * du_dx + omega_y * du_dy + omega_z * du_dz
        stretch_y = omega_x * dv_dx + omega_y * dv_dy + omega_z * dv_dz
        stretch_z = omega_x * dw_dx + omega_y * dw_dy + omega_z * dw_dz
        
        # Toplam: adveksiyon + stretching
        preservation = torch.clamp(self.circulation_preservation, 0.9, 1.0)
        
        domega_x_dt = self.diff_ops.dealias(adv_omega_x + stretch_x) * preservation
        domega_y_dt = self.diff_ops.dealias(adv_omega_y + stretch_y) * preservation
        domega_z_dt = self.diff_ops.dealias(adv_omega_z + stretch_z) * preservation
        
        return domega_x_dt, domega_y_dt, domega_z_dt
    
    def compute_curl(self, u: torch.Tensor, v: torch.Tensor, w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Hız alanından vortisite hesapla (yardımcı fonksiyon)"""
        return self.diff_ops.curl(u, v, w)

    def extra_repr(self) -> str:
        return f"circulation_preservation={self.circulation_preservation.item():.4f}"


class Projection3D(nn.Module):
    """
    3D Divergence-Free Projeksiyon Nöronu (Helmholtz Ayrışımı)
    
    TEK NÖRON = SIKIŞTIRILAMAZLIK GARANTİSİ
    ∇·u = 0 yapısal olarak sağlanır!
    
    Poisson denklemi: ∇²p = ∇·u
    Projeksiyon: u_new = u - ∇p
    
    Öğrenilen: YOK (0 parametre — tam div-free garanti)

    Immersed Boundary Desteği:
        ImmersedBoundary3D.register_neuron() ile sınır bilgisi aktarılır.

    Kullanım:
        proj = Projection3D(resolution=64)
        u_proj, v_proj, w_proj, p = proj(u, v, w, dt)
    """
    def __init__(self, resolution: int, diff_ops: Optional['SpectralOps3D'] = None):
        super().__init__()
        self.resolution = resolution
        self.diff_ops = diff_ops if diff_ops is not None else SpectralOps3D(resolution)

        # Immersed Boundary desteği
        self.boundary_mask: Optional[torch.Tensor] = None
        self.wall_velocity: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
    
    def forward(self, u: torch.Tensor, v: torch.Tensor, w: torch.Tensor, 
                dt: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Hız alanını divergence-free uzaya projekte et.
        
        İki mod desteklenir:
        
        1. Helmholtz Projeksiyon (dt=None):
           ∇²p = ∇·u
           u_new = u - ∇p
           → Tam divergence-free: ∇·u_new = 0
        
        2. Fractional-Step / Chorin Projeksiyon (dt verilirse):
           ∇²p = ∇·u / dt
           u_new = u - dt × ∇p
           → Tam divergence-free: ∇·u_new = ∇·u - dt × (∇·u/dt) = 0
        
        Args:
            u, v, w: Hız bileşenleri [B, Nx, Ny, Nz]
            dt: Zaman adımı (opsiyonel)
        
        Returns:
            (u_proj, v_proj, w_proj, p): Projeksiyon sonucu + basınç
        """
        # Mevcut divergence
        div_u = self.diff_ops.divergence(u, v, w)

        # 3D Poisson denklemi (pressure_scale kaldırıldı — matematiksel olarak
        # RHS'da bölen ve correction'da çarpan birbirini iptal ediyordu)
        if dt is not None:
            # Fractional-step: ∇²p = ∇·u / dt
            div_for_poisson = div_u / (dt + 1e-10)
        else:
            # Helmholtz: ∇²p = ∇·u
            div_for_poisson = div_u

        # Operatör-agnostik Poisson çözümü
        p = self.diff_ops.solve_poisson(div_for_poisson)

        # Projeksiyon: u_new = u - ∇p (veya u - dt×∇p)
        dp_dx, dp_dy, dp_dz = self.diff_ops.gradient(p)

        if dt is not None:
            # Fractional-step: u - dt × ∇p
            u_proj = u - dt * dp_dx
            v_proj = v - dt * dp_dy
            w_proj = w - dt * dp_dz
        else:
            # Helmholtz: u - ∇p
            u_proj = u - dp_dx
            v_proj = v - dp_dy
            w_proj = w - dp_dz
        
        # Immersed Boundary: sınırda wall velocity zorla
        if self.boundary_mask is not None and self.wall_velocity is not None:
            mask = self.boundary_mask
            u_wall, v_wall, w_wall = self.wall_velocity
            
            # Batch boyutu ekle
            if mask.dim() == 3 and u_proj.dim() == 4:
                mask = mask.unsqueeze(0)
                u_wall = u_wall.unsqueeze(0)
                v_wall = v_wall.unsqueeze(0)
                w_wall = w_wall.unsqueeze(0)
            
            u_proj = torch.where(mask, u_wall.expand_as(u_proj), u_proj)
            v_proj = torch.where(mask, v_wall.expand_as(v_proj), v_proj)
            w_proj = torch.where(mask, w_wall.expand_as(w_proj), w_proj)
        
        return u_proj, v_proj, w_proj, p
    
    def forward_leray(self, u: torch.Tensor, v: torch.Tensor, w: torch.Tensor,
                      dt: Optional[torch.Tensor] = None
                      ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Leray projector: tamamen Fourier uzayinda divergence-free projeksiyon.

        Matematik:
            k_dot_u = kx*u_hat + ky*v_hat + kz*w_hat
            u_proj_hat = u_hat - kx * k_dot_u / |k|^2
            (ayni: Helmholtz/Chorin ile matematiksel olarak AYNI)

        Maliyet: 3 rFFT + 4 irFFT = 7 FFT ops (eski forward: 10 ops)
        Tasarruf: 3 FFT/katman

        Kisit: Sadece periodic BC (spectral ops) ile calisir.
        Immersed boundary DESTEKLENMEZ (IB varsa forward() kullanilmali).
        """
        ops = self.diff_ops
        s = ops.spatial_shape

        # 3 rFFT
        u_hat = safe_rfftn(u)
        v_hat = safe_rfftn(v)
        w_hat = safe_rfftn(w)

        # Leray: u_proj = u - k(k.u)/|k|^2 — tamamen Fourier'de (0 FFT)
        k_dot_u = ops.kx * u_hat + ops.ky * v_hat + ops.kz * w_hat
        factor = k_dot_u / ops.k_squared_poisson  # k_sq_poisson[0,0,0]=1.0

        u_hat_proj = u_hat - ops.kx * factor
        v_hat_proj = v_hat - ops.ky * factor
        w_hat_proj = w_hat - ops.kz * factor

        # 3 irFFT
        u_proj = safe_irfftn(u_hat_proj, s=s)
        v_proj = safe_irfftn(v_hat_proj, s=s)
        w_proj = safe_irfftn(w_hat_proj, s=s)

        # Pressure: p_hat = -i*(k.u) / (|k|^2 * dt)  [1 irFFT]
        if dt is not None:
            p_hat = -1j * k_dot_u / (ops.k_squared_poisson * (dt + 1e-10))
        else:
            p_hat = -1j * k_dot_u / ops.k_squared_poisson
        p_hat[..., 0, 0, 0] = 0.0  # mean pressure = 0
        p = safe_irfftn(p_hat, s=s)

        return u_proj, v_proj, w_proj, p

    def divergence_error(self, u: torch.Tensor, v: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Divergence hatasını ölç - sıfır olmalı"""
        div = self.diff_ops.divergence(u, v, w)
        return div.abs().mean()

    def extra_repr(self) -> str:
        return f"resolution={self.resolution}, has_boundary={self.boundary_mask is not None}"


class TimeMarcher3D(nn.Module):
    """
    3D Zamana-Zorlamalı Nöron (Causal Time-Marching)

    TEK NÖRON = ZAMAN İLERLEMESİ
    Nedensellik yapısal: t₁ sadece t₀'dan türeyebilir
    3D CFL koşulu fiziksel olarak uygulanır (advective + diffusive)

    Özellikler:
    - RK4 integrasyon (4. derece doğruluk, enerji korumalı)
    - RK2 (Heun's method, 2. derece)
    - Euler fallback (rhs_func yoksa)
    - Advective + Diffusive CFL koşulu
    - Adaptif zaman adımı

    Öğrenilen: dt_scale (1 parametre)

    Kullanım:
        tm = TimeMarcher3D(resolution=64, method='rk4')
        new_state = tm(state, du_dt, dv_dt, dw_dt, rhs_func=compute_rhs)
    """
    def __init__(self, resolution: int, dt_range: Tuple[float, float] = (0.001, 0.1),
                 method: str = 'rk4', dx_min: Optional[float] = None):
        super().__init__()
        self.resolution = resolution
        self.dt_min, self.dt_max = dt_range
        self.method = method  # 'euler', 'rk2', 'rk4'

        # ÖĞRENİLEBİLİR PARAMETRE: sadece dt_scale
        self.dt_scale = nn.Parameter(torch.tensor(0.3))  # Konservatif başlangıç

        # Grid spacing (anizotropik grid'de en küçük dx kullanılır)
        self.dx = dx_min if dx_min is not None else 2 * math.pi / resolution

    def compute_adaptive_dt(self, state: FluidState3D, nu: float = 0.001) -> torch.Tensor:
        """
        Advective + Diffusive CFL koşulu (3D).

        Advective CFL: dt ≤ C * dx / max(|u|, |v|, |w|)
        Diffusive CFL: dt ≤ dx² / (6 * ν)  (6 = 2*d for 3D)
        """
        max_velocity = torch.amax(state.velocity_magnitude(), dim=(-3, -2, -1))

        # Advective CFL
        cfl_adv = self.dx / (max_velocity + 1e-8)

        # Diffusive CFL (dx² / 6ν for 3D)
        if isinstance(nu, torch.Tensor):
            nu_val = nu.item() if nu.numel() == 1 else nu.mean().item()
        else:
            nu_val = float(nu)
        cfl_diff = (self.dx ** 2) / (6 * nu_val + 1e-10)

        # Minimum al (her iki kısıtı da sağla)
        cfl_dt = torch.minimum(cfl_adv, torch.full_like(cfl_adv, cfl_diff))

        dt_scale_clamped = torch.clamp(self.dt_scale, 0.05, 0.5)
        dt = dt_scale_clamped * cfl_dt
        dt = torch.clamp(dt, self.dt_min, self.dt_max)

        return dt

    def _make_temp_state(self, state: FluidState3D,
                         du: torch.Tensor, dv: torch.Tensor, dw: torch.Tensor,
                         dox: torch.Tensor, doy: torch.Tensor, doz: torch.Tensor,
                         dt: torch.Tensor) -> FluidState3D:
        """Geçici state oluştur (RK ara adımları için)"""
        return FluidState3D(
            u=state.u + dt * du,
            v=state.v + dt * dv,
            w=state.w + dt * dw,
            p=state.p,
            omega_x=state.omega_x + dt * dox,
            omega_y=state.omega_y + dt * doy,
            omega_z=state.omega_z + dt * doz,
            t=state.t
        )

    def _euler_step(self, state: FluidState3D,
                    du_dt: torch.Tensor, dv_dt: torch.Tensor, dw_dt: torch.Tensor,
                    domega_x_dt: Optional[torch.Tensor],
                    domega_y_dt: Optional[torch.Tensor],
                    domega_z_dt: Optional[torch.Tensor],
                    dt: torch.Tensor) -> FluidState3D:
        """Forward Euler: u_{n+1} = u_n + dt * f(u_n)"""
        u_new = state.u + dt * du_dt
        v_new = state.v + dt * dv_dt
        w_new = state.w + dt * dw_dt

        if domega_x_dt is not None:
            ox_new = state.omega_x + dt * domega_x_dt
            oy_new = state.omega_y + dt * domega_y_dt
            oz_new = state.omega_z + dt * domega_z_dt
        else:
            ox_new, oy_new, oz_new = state.omega_x, state.omega_y, state.omega_z

        return FluidState3D(
            u=u_new, v=v_new, w=w_new, p=state.p,
            omega_x=ox_new, omega_y=oy_new, omega_z=oz_new,
            t=state.t + dt.squeeze(-1).squeeze(-1)
        )

    def _rk2_step(self, state: FluidState3D,
                  du_dt: torch.Tensor, dv_dt: torch.Tensor, dw_dt: torch.Tensor,
                  domega_x_dt: Optional[torch.Tensor],
                  domega_y_dt: Optional[torch.Tensor],
                  domega_z_dt: Optional[torch.Tensor],
                  dt: torch.Tensor, rhs_func) -> FluidState3D:
        """RK2 (Heun's method) - 2. derece doğruluk"""
        # k1 zaten hesaplanmış
        k1_u, k1_v, k1_w = du_dt, dv_dt, dw_dt
        k1_ox = domega_x_dt if domega_x_dt is not None else torch.zeros_like(state.omega_x)
        k1_oy = domega_y_dt if domega_y_dt is not None else torch.zeros_like(state.omega_y)
        k1_oz = domega_z_dt if domega_z_dt is not None else torch.zeros_like(state.omega_z)

        # k2 = f(state + dt * k1)
        s2 = self._make_temp_state(state, k1_u, k1_v, k1_w, k1_ox, k1_oy, k1_oz, dt)
        k2_u, k2_v, k2_w, k2_ox, k2_oy, k2_oz = rhs_func(s2)

        return FluidState3D(
            u=state.u + 0.5 * dt * (k1_u + k2_u),
            v=state.v + 0.5 * dt * (k1_v + k2_v),
            w=state.w + 0.5 * dt * (k1_w + k2_w),
            p=state.p,
            omega_x=state.omega_x + 0.5 * dt * (k1_ox + k2_ox),
            omega_y=state.omega_y + 0.5 * dt * (k1_oy + k2_oy),
            omega_z=state.omega_z + 0.5 * dt * (k1_oz + k2_oz),
            t=state.t + dt.squeeze(-1).squeeze(-1)
        )

    def _rk4_step(self, state: FluidState3D,
                  du_dt: torch.Tensor, dv_dt: torch.Tensor, dw_dt: torch.Tensor,
                  domega_x_dt: Optional[torch.Tensor],
                  domega_y_dt: Optional[torch.Tensor],
                  domega_z_dt: Optional[torch.Tensor],
                  dt: torch.Tensor, rhs_func) -> FluidState3D:
        """
        Classical RK4 - 4. derece doğruluk.

        k1 = f(t_n, y_n)          → zaten hesaplanmış (du_dt, dv_dt, dw_dt)
        k2 = f(t_n + dt/2, y_n + dt/2 * k1)
        k3 = f(t_n + dt/2, y_n + dt/2 * k2)
        k4 = f(t_n + dt, y_n + dt * k3)
        y_{n+1} = y_n + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        """
        # k1 zaten hesaplanmış
        k1_u, k1_v, k1_w = du_dt, dv_dt, dw_dt
        k1_ox = domega_x_dt if domega_x_dt is not None else torch.zeros_like(state.omega_x)
        k1_oy = domega_y_dt if domega_y_dt is not None else torch.zeros_like(state.omega_y)
        k1_oz = domega_z_dt if domega_z_dt is not None else torch.zeros_like(state.omega_z)

        # k2 = f(state + 0.5*dt * k1)
        s2 = self._make_temp_state(state, k1_u, k1_v, k1_w, k1_ox, k1_oy, k1_oz, 0.5 * dt)
        k2_u, k2_v, k2_w, k2_ox, k2_oy, k2_oz = rhs_func(s2)

        # k3 = f(state + 0.5*dt * k2)
        s3 = self._make_temp_state(state, k2_u, k2_v, k2_w, k2_ox, k2_oy, k2_oz, 0.5 * dt)
        k3_u, k3_v, k3_w, k3_ox, k3_oy, k3_oz = rhs_func(s3)

        # k4 = f(state + dt * k3)
        s4 = self._make_temp_state(state, k3_u, k3_v, k3_w, k3_ox, k3_oy, k3_oz, dt)
        k4_u, k4_v, k4_w, k4_ox, k4_oy, k4_oz = rhs_func(s4)

        # Combine: y_new = y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        factor = 1.0 / 6.0

        return FluidState3D(
            u=state.u + dt * factor * (k1_u + 2*k2_u + 2*k3_u + k4_u),
            v=state.v + dt * factor * (k1_v + 2*k2_v + 2*k3_v + k4_v),
            w=state.w + dt * factor * (k1_w + 2*k2_w + 2*k3_w + k4_w),
            p=state.p,
            omega_x=state.omega_x + dt * factor * (k1_ox + 2*k2_ox + 2*k3_ox + k4_ox),
            omega_y=state.omega_y + dt * factor * (k1_oy + 2*k2_oy + 2*k3_oy + k4_oy),
            omega_z=state.omega_z + dt * factor * (k1_oz + 2*k2_oz + 2*k3_oz + k4_oz),
            t=state.t + dt.squeeze(-1).squeeze(-1)
        )

    def forward(self, state: FluidState3D,
                du_dt: torch.Tensor, dv_dt: torch.Tensor, dw_dt: torch.Tensor,
                domega_x_dt: Optional[torch.Tensor] = None,
                domega_y_dt: Optional[torch.Tensor] = None,
                domega_z_dt: Optional[torch.Tensor] = None,
                rhs_func=None) -> FluidState3D:
        """
        Nedensel 3D zaman ilerlemesi - RK4/RK2/Euler seçenekleriyle.

        Args:
            state: Mevcut FluidState3D
            du_dt, dv_dt, dw_dt: Hız türevleri (k1 olarak kullanılır)
            domega_x_dt, domega_y_dt, domega_z_dt: Vortisite türevleri (opsiyonel)
            rhs_func: RK2/RK4 için RHS fonksiyonu
                      rhs_func(state) -> (du_dt, dv_dt, dw_dt, domega_x_dt, domega_y_dt, domega_z_dt)

        Returns:
            Yeni FluidState3D
        """
        dt = self.compute_adaptive_dt(state)
        dt = dt.view(-1, 1, 1, 1)  # 3D broadcasting için

        if self.method == 'rk4' and rhs_func is not None:
            return self._rk4_step(state, du_dt, dv_dt, dw_dt,
                                  domega_x_dt, domega_y_dt, domega_z_dt, dt, rhs_func)
        elif self.method == 'rk2' and rhs_func is not None:
            return self._rk2_step(state, du_dt, dv_dt, dw_dt,
                                  domega_x_dt, domega_y_dt, domega_z_dt, dt, rhs_func)
        else:
            # Euler fallback (rhs_func yoksa veya method='euler')
            return self._euler_step(state, du_dt, dv_dt, dw_dt,
                                    domega_x_dt, domega_y_dt, domega_z_dt, dt)

    def extra_repr(self) -> str:
        return f"method={self.method}, resolution={self.resolution}, dt_range=({self.dt_min}, {self.dt_max}), dt_scale={self.dt_scale.item():.4f}"


class Boundary3D(nn.Module):
    """
    3D Sınır Koşulları Nöronu
    
    TEK NÖRON = SINIR FİZİĞİ
    
    Desteklenen tipler:
    - 'periodic': Periyodik BC (FFT ile otomatik)
    - 'no_slip': Duvar BC (u=v=w=0 tüm bileşenler)
    - 'free_slip': Serbest kayma (normal bileşen=0, tanjant gradyan=0)
    
    Öğrenilen: bc_strength (1 parametre)
    
    Kullanım:
        bc = Boundary3D(resolution=64, bc_type='periodic')
        state = bc(state)
    """
    def __init__(self, resolution: int, bc_type: str = 'periodic'):
        super().__init__()
        self.resolution = resolution
        self.bc_type = bc_type
        
        # TEK ÖĞRENİLEBİLİR PARAMETRE
        self.bc_strength = nn.Parameter(torch.tensor(1.0))
        
        # İç alan maskesi (sınırlar hariç)
        mask = torch.ones(resolution, resolution, resolution)
        if bc_type in ['no_slip', 'free_slip']:
            mask[0, :, :] = 0
            mask[-1, :, :] = 0
            mask[:, 0, :] = 0
            mask[:, -1, :] = 0
            mask[:, :, 0] = 0
            mask[:, :, -1] = 0
        self.register_buffer('boundary_mask', mask)
        
        # Free-slip için yüz maskeleri (hangi yüzde hangi bileşen normal)
        # x-faces (i=0, i=-1): u normal, v,w tangential
        # y-faces (j=0, j=-1): v normal, u,w tangential
        # z-faces (k=0, k=-1): w normal, u,v tangential
        if bc_type == 'free_slip':
            # x-yüzleri maskesi
            x_face = torch.zeros(resolution, resolution, resolution)
            x_face[0, :, :] = 1
            x_face[-1, :, :] = 1
            self.register_buffer('x_face_mask', x_face)
            
            # y-yüzleri maskesi
            y_face = torch.zeros(resolution, resolution, resolution)
            y_face[:, 0, :] = 1
            y_face[:, -1, :] = 1
            self.register_buffer('y_face_mask', y_face)
            
            # z-yüzleri maskesi
            z_face = torch.zeros(resolution, resolution, resolution)
            z_face[:, :, 0] = 1
            z_face[:, :, -1] = 1
            self.register_buffer('z_face_mask', z_face)
    
    def forward(self, state: FluidState3D) -> FluidState3D:
        """Sınır koşullarını uygula"""
        if self.bc_type == 'periodic':
            return state  # FFT zaten periyodik
        
        strength = torch.clamp(self.bc_strength, 0.5, 1.0)
        mask = self.boundary_mask.unsqueeze(0)
        
        if self.bc_type == 'no_slip':
            # No-slip: Duvarda u=v=w=0 (tüm bileşenler)
            u = state.u * mask * strength + state.u * (1 - mask) * (1 - strength)
            v = state.v * mask * strength + state.v * (1 - mask) * (1 - strength)
            w = state.w * mask * strength + state.w * (1 - mask) * (1 - strength)
        else:  # free_slip
            # Free-slip: Normal bileşen=0, tanjant bileşenler korunur
            # x-yüzlerinde: u=0 (normal), v,w serbest (Neumann: ∂v/∂x=0, ∂w/∂x=0 → iç değerden kopyala)
            # y-yüzlerinde: v=0 (normal), u,w serbest
            # z-yüzlerinde: w=0 (normal), u,v serbest
            
            x_face = self.x_face_mask.unsqueeze(0)
            y_face = self.y_face_mask.unsqueeze(0)
            z_face = self.z_face_mask.unsqueeze(0)
            
            # Normal bileşenleri sıfırla
            u = state.u * (1 - x_face * strength)  # u: x-yüzlerinde 0
            v = state.v * (1 - y_face * strength)  # v: y-yüzlerinde 0
            w = state.w * (1 - z_face * strength)  # w: z-yüzlerinde 0
            
            # Tanjant bileşenler için Neumann BC: sınır değerini iç değerden al
            # x-yüzleri için v, w (∂v/∂x=0 → v[0] = v[1], v[-1] = v[-2])
            v = self._apply_neumann_x(v)
            w = self._apply_neumann_x(w)
            
            # y-yüzleri için u, w
            u = self._apply_neumann_y(u)
            w = self._apply_neumann_y(w)
            
            # z-yüzleri için u, v
            u = self._apply_neumann_z(u)
            v = self._apply_neumann_z(v)
        
        return FluidState3D(
            u=u, v=v, w=w, p=state.p,
            omega_x=state.omega_x, omega_y=state.omega_y, omega_z=state.omega_z,
            t=state.t
        )
    
    def _apply_neumann_x(self, f: torch.Tensor) -> torch.Tensor:
        """x-yönünde Neumann BC: ∂f/∂x = 0 sınırlarda"""
        f = f.clone()
        f[:, 0, :, :] = f[:, 1, :, :]
        f[:, -1, :, :] = f[:, -2, :, :]
        return f
    
    def _apply_neumann_y(self, f: torch.Tensor) -> torch.Tensor:
        """y-yönünde Neumann BC: ∂f/∂y = 0 sınırlarda"""
        f = f.clone()
        f[:, :, 0, :] = f[:, :, 1, :]
        f[:, :, -1, :] = f[:, :, -2, :]
        return f
    
    def _apply_neumann_z(self, f: torch.Tensor) -> torch.Tensor:
        """z-yönünde Neumann BC: ∂f/∂z = 0 sınırlarda"""
        f = f.clone()
        f[:, :, :, 0] = f[:, :, :, 1]
        f[:, :, :, -1] = f[:, :, :, -2]
        return f
    
    def boundary_violation_loss(self, state: FluidState3D) -> torch.Tensor:
        """Sınır koşulu ihlal kaybı"""
        if self.bc_type == 'periodic':
            return torch.tensor(0.0, device=state.u.device)
        
        if self.bc_type == 'no_slip':
            inv_mask = 1 - self.boundary_mask.unsqueeze(0)
            violation = (state.u.abs() * inv_mask).mean()
            violation = violation + (state.v.abs() * inv_mask).mean()
            violation = violation + (state.w.abs() * inv_mask).mean()
        else:  # free_slip
            # Normal bileşenler sınırda 0 olmalı
            x_face = self.x_face_mask.unsqueeze(0)
            y_face = self.y_face_mask.unsqueeze(0)
            z_face = self.z_face_mask.unsqueeze(0)
            violation = (state.u.abs() * x_face).mean()
            violation = violation + (state.v.abs() * y_face).mean()
            violation = violation + (state.w.abs() * z_face).mean()

        return violation

    def extra_repr(self) -> str:
        return f"resolution={self.resolution}, bc_type={self.bc_type}, bc_strength={self.bc_strength.item():.4f}"


class DataInjector3D(nn.Module):
    """
    3D Veri-Enjeksiyon Kapısı Nöronu
    
    TEK NÖRON = VERİ FÜZYONU
    Kritik bölgelerde gözlem verisini fiziksel olarak enjekte eder.
    
    Öğrenilen: fusion_weight (1 parametre)
    
    Kullanım:
        injector = DataInjector3D(resolution=64)
        fused_state = injector(predicted_state, observed_data)
    """
    def __init__(self, resolution: int, diff_ops: Optional['SpectralOps3D'] = None):
        super().__init__()
        self.resolution = resolution
        self.diff_ops = diff_ops if diff_ops is not None else SpectralOps3D(resolution)
        
        # TEK ÖĞRENİLEBİLİR PARAMETRE
        self.fusion_weight = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, predicted_state: FluidState3D, 
                observed_data: Optional[Dict[str, torch.Tensor]] = None) -> FluidState3D:
        """Gözlem verisini enjekte et"""
        if observed_data is None:
            return predicted_state
        
        # Vortisite büyüklüğü tabanlı kritiklik
        omega_mag = predicted_state.vorticity_magnitude()
        criticality = omega_mag / (torch.amax(omega_mag, dim=(-3, -2, -1), keepdim=True) + 1e-8)
        
        gate = torch.sigmoid(5 * (criticality - 0.5))
        
        if 'u' in observed_data and 'v' in observed_data and 'w' in observed_data:
            w = torch.clamp(self.fusion_weight, 0, 0.5)
            blend = gate * w
            
            u_new = predicted_state.u * (1 - blend) + observed_data['u'] * blend
            v_new = predicted_state.v * (1 - blend) + observed_data['v'] * blend
            w_new = predicted_state.w * (1 - blend) + observed_data['w'] * blend
            
            return FluidState3D(
                u=u_new, v=v_new, w=w_new, p=predicted_state.p,
                omega_x=predicted_state.omega_x, omega_y=predicted_state.omega_y,
                omega_z=predicted_state.omega_z, t=predicted_state.t
            )

        return predicted_state

    def extra_repr(self) -> str:
        return f"fusion_weight={self.fusion_weight.item():.4f}"


class StrainRate3D(nn.Module):
    """
    3D Strain Rate Tensor Nöronu (3D SPESİFİK)
    
    TEK NÖRON = STRAIN RATE
    
    Strain rate tensor: S_ij = 0.5 * (∂u_i/∂x_j + ∂u_j/∂x_i)
    
    LES (Large Eddy Simulation) ve türbülans modellerinde kritik.
    Smagorinsky SGS modeli bu tensöre dayanır.
    
    Öğrenilen: strain_modulator (1 parametre)
    
    Kullanım:
        strain = StrainRate3D(resolution=64)
        strain_mag = strain(state)  # |S| = sqrt(2 * S_ij * S_ij)
    """
    def __init__(self, resolution: int, diff_ops: Optional['SpectralOps3D'] = None):
        super().__init__()
        self.diff_ops = diff_ops if diff_ops is not None else SpectralOps3D(resolution)
        
        # TEK ÖĞRENİLEBİLİR PARAMETRE
        self.strain_modulator = nn.Parameter(torch.ones(1))
    
    def forward(self, state: FluidState3D) -> torch.Tensor:
        """
        Strain rate büyüklüğü hesapla: |S| = sqrt(2 * S_ij * S_ij)
        
        Args:
            state: FluidState3D
        
        Returns:
            strain_magnitude: [B, Nx, Ny, Nz]
        """
        u, v, w = state.u, state.v, state.w
        
        # Gradyanlar
        du_dx, du_dy, du_dz = self.diff_ops.gradient(u)
        dv_dx, dv_dy, dv_dz = self.diff_ops.gradient(v)
        dw_dx, dw_dy, dw_dz = self.diff_ops.gradient(w)
        
        # Strain rate tensor bileşenleri (simetrik, 6 bağımsız)
        S_xx = du_dx
        S_yy = dv_dy
        S_zz = dw_dz
        S_xy = 0.5 * (du_dy + dv_dx)
        S_xz = 0.5 * (du_dz + dw_dx)
        S_yz = 0.5 * (dv_dz + dw_dy)
        
        # |S|² = 2 * S_ij * S_ij (tekrar eden indeksler üzerine toplam)
        S_sq = (S_xx**2 + S_yy**2 + S_zz**2 + 
                2 * (S_xy**2 + S_xz**2 + S_yz**2))
        
        strain_mag = torch.sqrt(2 * S_sq + 1e-8)
        
        return self.strain_modulator * strain_mag
    
    def compute_tensor(self, state: FluidState3D) -> Dict[str, torch.Tensor]:
        """Tam strain rate tensor'ü döndür (6 bileşen)"""
        u, v, w = state.u, state.v, state.w
        
        du_dx, du_dy, du_dz = self.diff_ops.gradient(u)
        dv_dx, dv_dy, dv_dz = self.diff_ops.gradient(v)
        dw_dx, dw_dy, dw_dz = self.diff_ops.gradient(w)
        
        return {
            'S_xx': du_dx,
            'S_yy': dv_dy,
            'S_zz': dw_dz,
            'S_xy': 0.5 * (du_dy + dv_dx),
            'S_xz': 0.5 * (du_dz + dw_dx),
            'S_yz': 0.5 * (dv_dz + dw_dy),
        }

    def extra_repr(self) -> str:
        return f"strain_modulator={self.strain_modulator.item():.4f}"


class Helicity3D(nn.Module):
    """
    3D Helicity Nöronu (3D SPESİFİK)
    
    TEK NÖRON = HELİCİTY
    
    Helicity: H = u · ω = u*ω_x + v*ω_y + w*ω_z
    
    Hız ve vortisite vektörlerinin iç çarpımı.
    - 3D türbülans diagnostiği için kritik
    - Enerji kaskadı analizi
    - Topological flow features
    
    2D'de vortisite skalerdir, helicity tanımsızdır!
    
    Öğrenilen: helicity_scale (1 parametre)
    
    Kullanım:
        hel = Helicity3D()
        H = hel(state)  # Helicity alanı
    """
    def __init__(self):
        super().__init__()
        
        # TEK ÖĞRENİLEBİLİR PARAMETRE
        self.helicity_scale = nn.Parameter(torch.ones(1))
    
    def forward(self, state: FluidState3D) -> torch.Tensor:
        """
        Helicity alanı hesapla: H = u · ω
        
        Args:
            state: FluidState3D
        
        Returns:
            helicity: [B, Nx, Ny, Nz]
        """
        H = (state.u * state.omega_x + 
             state.v * state.omega_y + 
             state.w * state.omega_z)
        
        return self.helicity_scale * H
    
    def relative_helicity(self, state: FluidState3D) -> torch.Tensor:
        """
        Göreceli helicity: H / (|u| * |ω|)
        
        [-1, 1] aralığında normalize edilmiş.
        ±1: Hız ve vortisite paralel/antiparalel
        0: Hız ve vortisite dik
        """
        H = self.forward(state) / self.helicity_scale  # Raw helicity
        u_mag = state.velocity_magnitude()
        omega_mag = state.vorticity_magnitude()

        return H / (u_mag * omega_mag + 1e-8)

    def extra_repr(self) -> str:
        return f"helicity_scale={self.helicity_scale.item():.4f}"


class PressureCoupling3D(nn.Module):
    """
    3D Basınç-Vortisite Coupling Nöronu (İLERİ SEVİYE)
    
    TEK NÖRON = BASINÇ-VORTİSİTE BAĞLANTISI
    
    3D'de basınç ve vortisite coupling'i:
    ω × u → pressure correction feedback
    
    Bu terim momentum denkleminde:
    ∂u/∂t = -∇p + ν∇²u - (u·∇)u
    
    Vortisite formunda implicit ama bu nöron explicit yapar.
    
    Öğrenilen: coupling_strength (1 parametre)
    
    Kullanım:
        coupling = PressureCoupling3D(resolution=64)
        dp_correction = coupling(state)
    """
    def __init__(self, resolution: int, diff_ops: Optional['SpectralOps3D'] = None):
        super().__init__()
        self.resolution = resolution
        self.diff_ops = diff_ops if diff_ops is not None else SpectralOps3D(resolution)
        
        # TEK ÖĞRENİLEBİLİR PARAMETRE
        self.coupling_strength = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, state: FluidState3D) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        ω × u hesapla (Lamb vector) ve basınç düzeltmesi döndür.
        
        Lamb vector: L = ω × u
        Bu, Bernoulli basıncı ile ilişkili: ∇(p + 0.5|u|²) = -L (steady inviscid)
        
        Args:
            state: FluidState3D
        
        Returns:
            (L_x, L_y, L_z): Lamb vector bileşenleri (basınç düzeltme yönü)
        """
        u, v, w = state.u, state.v, state.w
        omega_x, omega_y, omega_z = state.omega_x, state.omega_y, state.omega_z
        
        # Lamb vector: L = ω × u
        L_x = omega_y * w - omega_z * v
        L_y = omega_z * u - omega_x * w
        L_z = omega_x * v - omega_y * u
        
        # Modülasyon
        strength = torch.clamp(self.coupling_strength, 0.0, 0.5)
        
        return strength * L_x, strength * L_y, strength * L_z
    
    def compute_pressure_correction(self, state: FluidState3D) -> torch.Tensor:
        """
        Lamb vector'dan basınç düzeltmesi hesapla.

        ∇²(dp) = ∇·L (Poisson denklemi)
        """
        L_x, L_y, L_z = self.forward(state)

        # Lamb vector divergence
        div_L = self.diff_ops.divergence(L_x, L_y, L_z)

        # Operatör-agnostik Poisson çözümü
        return self.diff_ops.solve_poisson(div_L)

    def extra_repr(self) -> str:
        return f"coupling_strength={self.coupling_strength.item():.4f}"


class EnergyPreservingIntegrator3D(nn.Module):
    """
    Enerji Koruyan Zaman İntegratörü (İLERİ SEVİYE - OPSİYONEL)
    
    TEK NÖRON = ENERJİ KORUYAN ZAMAN ADIMI
    
    ╔══════════════════════════════════════════════════════════════════╗
    ║  ⚠️ ÖNEMLİ: BU NÖRON VARSAYILAN DEĞİLDİR!                        ║
    ║                                                                  ║
    ║  Varsayılan eğitim için: TimeMarcher3D (Euler + CFL)             ║
    ║  Bu nöron sadece şu durumlarda kullanın:                         ║
    ║  - Çok uzun simülasyonlar (enerji drift kritik)                  ║
    ║  - Yüksek doğruluk gereken validasyon testleri                   ║
    ║  - Enerji korunumunun şart olduğu fizik analizleri               ║
    ║                                                                  ║
    ║  ML + Euler + CFL + Projection = çoğu durumda YETERLİ!           ║
    ╚══════════════════════════════════════════════════════════════════╝
    
    Forward Euler enerji drift yapar. Bu nöron alternatifleri sunar:
    - 'euler': Forward Euler (basit, hızlı, drift yapar)
    - 'rk2': Runge-Kutta 2 (Heun's method, 2x maliyet, daha az drift)
    - 'rk4': Runge-Kutta 4 (klasik, 4x maliyet, çok az drift)
    - 'implicit_midpoint': Symplectic (enerji koruyan, Newton iter. gerek)
    
    Öğrenilen: dt_scale, stability_factor (2 parametre)
    
    Kullanım:
        # SADECE GERÇEKTEN GEREKTİĞİNDE KULLAN!
        integrator = EnergyPreservingIntegrator3D(resolution=64, method='rk2')
        new_state = integrator(state, rhs_func)
    """
    def __init__(self, resolution: int, method: str = 'rk2', 
                 dt_range: Tuple[float, float] = (0.001, 0.1)):
        super().__init__()
        self.resolution = resolution
        self.method = method
        self.dt_min, self.dt_max = dt_range
        
        # ÖĞRENİLEBİLİR PARAMETRELER
        self.dt_scale = nn.Parameter(torch.tensor(0.5))
        self.stability_factor = nn.Parameter(torch.tensor(0.99))
    
    def compute_adaptive_dt(self, state: FluidState3D) -> torch.Tensor:
        """3D CFL-bazlı adaptif dt"""
        max_velocity = torch.amax(state.velocity_magnitude(), dim=(-3, -2, -1))
        dx = 2 * math.pi / self.resolution
        cfl_dt = dx / (max_velocity + 1e-8)
        
        dt_scale_clamped = torch.clamp(self.dt_scale, 0.1, 0.9)
        dt = dt_scale_clamped * cfl_dt
        return torch.clamp(dt, self.dt_min, self.dt_max)
    
    def forward(self, state: FluidState3D, 
                rhs_func) -> FluidState3D:
        """
        Zaman adımı al.
        
        Args:
            state: Mevcut FluidState3D
            rhs_func: RHS hesaplayan callable
                      rhs_func(state) -> (du_dt, dv_dt, dw_dt, domega_x_dt, domega_y_dt, domega_z_dt)
        
        Returns:
            Yeni FluidState3D
        """
        dt = self.compute_adaptive_dt(state)
        dt = dt.view(-1, 1, 1, 1)
        
        if self.method == 'euler':
            return self._euler_step(state, rhs_func, dt)
        elif self.method == 'rk2':
            return self._rk2_step(state, rhs_func, dt)
        elif self.method == 'rk4':
            return self._rk4_step(state, rhs_func, dt)
        elif self.method == 'implicit_midpoint':
            return self._implicit_midpoint_step(state, rhs_func, dt)
        else:
            raise ValueError(f"Bilinmeyen method: {self.method}")
    
    def _euler_step(self, state: FluidState3D, rhs_func, dt: torch.Tensor) -> FluidState3D:
        """Forward Euler"""
        du, dv, dw, domega_x, domega_y, domega_z = rhs_func(state)
        stability = torch.clamp(self.stability_factor, 0.9, 1.0)
        
        return FluidState3D(
            u=state.u + dt * du * stability,
            v=state.v + dt * dv * stability,
            w=state.w + dt * dw * stability,
            p=state.p,
            omega_x=state.omega_x + dt * domega_x * stability,
            omega_y=state.omega_y + dt * domega_y * stability,
            omega_z=state.omega_z + dt * domega_z * stability,
            t=state.t + dt.squeeze(-1).squeeze(-1)
        )
    
    def _rk2_step(self, state: FluidState3D, rhs_func, dt: torch.Tensor) -> FluidState3D:
        """Runge-Kutta 2 (Heun's method) - 2. derece doğruluk"""
        # k1 = f(t, y)
        k1_u, k1_v, k1_w, k1_ox, k1_oy, k1_oz = rhs_func(state)
        
        # y_mid = y + dt * k1
        mid_state = FluidState3D(
            u=state.u + dt * k1_u,
            v=state.v + dt * k1_v,
            w=state.w + dt * k1_w,
            p=state.p,
            omega_x=state.omega_x + dt * k1_ox,
            omega_y=state.omega_y + dt * k1_oy,
            omega_z=state.omega_z + dt * k1_oz,
            t=state.t + dt.squeeze(-1).squeeze(-1)
        )
        
        # k2 = f(t + dt, y_mid)
        k2_u, k2_v, k2_w, k2_ox, k2_oy, k2_oz = rhs_func(mid_state)
        
        # y_new = y + 0.5 * dt * (k1 + k2)
        stability = torch.clamp(self.stability_factor, 0.9, 1.0)
        
        return FluidState3D(
            u=state.u + 0.5 * dt * (k1_u + k2_u) * stability,
            v=state.v + 0.5 * dt * (k1_v + k2_v) * stability,
            w=state.w + 0.5 * dt * (k1_w + k2_w) * stability,
            p=state.p,
            omega_x=state.omega_x + 0.5 * dt * (k1_ox + k2_ox) * stability,
            omega_y=state.omega_y + 0.5 * dt * (k1_oy + k2_oy) * stability,
            omega_z=state.omega_z + 0.5 * dt * (k1_oz + k2_oz) * stability,
            t=state.t + dt.squeeze(-1).squeeze(-1)
        )
    
    def _rk4_step(self, state: FluidState3D, rhs_func, dt: torch.Tensor) -> FluidState3D:
        """Runge-Kutta 4 - 4. derece doğruluk, enerji drift düşük"""
        # k1
        k1_u, k1_v, k1_w, k1_ox, k1_oy, k1_oz = rhs_func(state)
        
        # k2
        s2 = self._make_temp_state(state, k1_u, k1_v, k1_w, k1_ox, k1_oy, k1_oz, 0.5 * dt)
        k2_u, k2_v, k2_w, k2_ox, k2_oy, k2_oz = rhs_func(s2)
        
        # k3
        s3 = self._make_temp_state(state, k2_u, k2_v, k2_w, k2_ox, k2_oy, k2_oz, 0.5 * dt)
        k3_u, k3_v, k3_w, k3_ox, k3_oy, k3_oz = rhs_func(s3)
        
        # k4
        s4 = self._make_temp_state(state, k3_u, k3_v, k3_w, k3_ox, k3_oy, k3_oz, dt)
        k4_u, k4_v, k4_w, k4_ox, k4_oy, k4_oz = rhs_func(s4)
        
        # Combine
        stability = torch.clamp(self.stability_factor, 0.9, 1.0)
        factor = stability / 6.0
        
        return FluidState3D(
            u=state.u + dt * factor * (k1_u + 2*k2_u + 2*k3_u + k4_u),
            v=state.v + dt * factor * (k1_v + 2*k2_v + 2*k3_v + k4_v),
            w=state.w + dt * factor * (k1_w + 2*k2_w + 2*k3_w + k4_w),
            p=state.p,
            omega_x=state.omega_x + dt * factor * (k1_ox + 2*k2_ox + 2*k3_ox + k4_ox),
            omega_y=state.omega_y + dt * factor * (k1_oy + 2*k2_oy + 2*k3_oy + k4_oy),
            omega_z=state.omega_z + dt * factor * (k1_oz + 2*k2_oz + 2*k3_oz + k4_oz),
            t=state.t + dt.squeeze(-1).squeeze(-1)
        )
    
    def _implicit_midpoint_step(self, state: FluidState3D, rhs_func, dt: torch.Tensor) -> FluidState3D:
        """Implicit Midpoint - Symplectic, enerji koruyan (1 iterasyon approx)"""
        # Midpoint yaklaşık: y_{n+1} ≈ y_n + dt * f((y_n + y_{n+1})/2)
        # İterasyon olmadan: f(y_n) ile başla, midpoint tahmin et
        
        k1_u, k1_v, k1_w, k1_ox, k1_oy, k1_oz = rhs_func(state)
        
        # Midpoint tahmini
        mid_state = self._make_temp_state(state, k1_u, k1_v, k1_w, k1_ox, k1_oy, k1_oz, 0.5 * dt)
        
        # Midpoint'te RHS
        km_u, km_v, km_w, km_ox, km_oy, km_oz = rhs_func(mid_state)
        
        stability = torch.clamp(self.stability_factor, 0.9, 1.0)
        
        return FluidState3D(
            u=state.u + dt * km_u * stability,
            v=state.v + dt * km_v * stability,
            w=state.w + dt * km_w * stability,
            p=state.p,
            omega_x=state.omega_x + dt * km_ox * stability,
            omega_y=state.omega_y + dt * km_oy * stability,
            omega_z=state.omega_z + dt * km_oz * stability,
            t=state.t + dt.squeeze(-1).squeeze(-1)
        )
    
    def _make_temp_state(self, state, du, dv, dw, dox, doy, doz, dt):
        """Geçici state oluştur (RK adımları için)"""
        return FluidState3D(
            u=state.u + dt * du,
            v=state.v + dt * dv,
            w=state.w + dt * dw,
            p=state.p,
            omega_x=state.omega_x + dt * dox,
            omega_y=state.omega_y + dt * doy,
            omega_z=state.omega_z + dt * doz,
            t=state.t
        )

    def extra_repr(self) -> str:
        return f"method={self.method}, energy_scale={self.energy_scale.item():.4f}, stability={self.stability_factor.item():.4f}"


class SpectralEnergyFlux3D(nn.Module):
    """
    3D Spektral Enerji Akısı Nöronu (DİAGNOSTİK)
    
    TEK NÖRON = SPEKTRAL ANALİZ
    
    3D türbülansta:
    - E(k): Enerji spektrumu
    - Π(k): Enerji akısı (cascade rate)
    
    Kolmogorov 5/3 yasası: E(k) ∝ k^(-5/3) (inertial range)
    
    NOT: Bu bir diagnostic nöron, loss için değil!
         "Bu gerçekten türbülans mı?" testinde altın standart.
    
    Öğrenilen: YOK (saf ölçüm)
    
    Kullanım:
        flux = SpectralEnergyFlux3D(resolution=64)
        E_k, Pi_k, k_bins = flux(state)
    """
    def __init__(self, resolution: int, num_bins: int = 32):
        super().__init__()
        self.resolution = resolution
        self.num_bins = num_bins
        
        # Dalga sayıları
        k = fftfreq(resolution, d=(2*math.pi)/resolution) * 2 * math.pi
        kx, ky, kz = torch.meshgrid(k, k, k, indexing='ij')
        k_mag = torch.sqrt(kx**2 + ky**2 + kz**2)
        
        self.register_buffer('kx', kx)
        self.register_buffer('ky', ky)
        self.register_buffer('kz', kz)
        self.register_buffer('k_magnitude', k_mag)
        
        # k binleri
        k_max = resolution // 2
        k_bins = torch.linspace(0, k_max, num_bins + 1)
        self.register_buffer('k_bins', k_bins)
    
    def forward(self, state: FluidState3D) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Spektral enerji ve akı hesapla.
        
        Args:
            state: FluidState3D
        
        Returns:
            E_k: Enerji spektrumu [num_bins]
            Pi_k: Enerji akısı [num_bins]
            k_centers: Bin merkezleri [num_bins]
        """
        # Hız Fourier dönüşümü
        u_hat = safe_fftn(state.u)
        v_hat = safe_fftn(state.v)
        w_hat = safe_fftn(state.w)
        
        # Spektral enerji yoğunluğu: 0.5 * |û|²
        E_hat = 0.5 * (u_hat.abs()**2 + v_hat.abs()**2 + w_hat.abs()**2)
        
        # Shell averaging: E(k) = ∫ E(k') δ(|k'| - k) dk'
        E_k = torch.zeros(self.num_bins, device=state.u.device)
        
        for i in range(self.num_bins):
            k_low = self.k_bins[i]
            k_high = self.k_bins[i + 1]
            shell_mask = (self.k_magnitude >= k_low) & (self.k_magnitude < k_high)
            if shell_mask.sum() > 0:
                E_k[i] = E_hat[0][shell_mask].mean()  # Batch 0
        
        # Enerji akısı Π(k) = -dE(<k)/dt ≈ cumsum(transfer)
        # Basitleştirilmiş: Π(k) ≈ k * E(k) (boyutsal analiz)
        k_centers = 0.5 * (self.k_bins[:-1] + self.k_bins[1:])
        Pi_k = k_centers * E_k  # Crude estimate
        
        return E_k, Pi_k, k_centers
    
    def check_kolmogorov(self, state: FluidState3D) -> Dict[str, float]:
        """
        Kolmogorov 5/3 yasasını kontrol et.
        
        Inertial range'de: E(k) ∝ k^(-5/3)
        Log-log fit yaparak exponent'ı bul.
        """
        E_k, _, k_centers = self.forward(state)
        
        # Sıfırdan büyük değerleri filtrele
        valid = (E_k > 1e-10) & (k_centers > 0)
        if valid.sum() < 3:
            return {'exponent': 0.0, 'is_turbulent': False}
        
        log_k = torch.log(k_centers[valid])
        log_E = torch.log(E_k[valid])
        
        # Linear fit: log(E) = exponent * log(k) + const
        # Least squares
        n = log_k.shape[0]
        sum_x = log_k.sum()
        sum_y = log_E.sum()
        sum_xy = (log_k * log_E).sum()
        sum_xx = (log_k * log_k).sum()
        
        exponent = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x**2 + 1e-8)
        
        # Kolmogorov: exponent ≈ -5/3 ≈ -1.67
        is_turbulent = -2.0 < exponent.item() < -1.3
        
        return {
            'exponent': exponent.item(),
            'is_turbulent': is_turbulent,
            'kolmogorov_deviation': abs(exponent.item() + 5/3)
        }

    def extra_repr(self) -> str:
        return f"resolution={self.resolution}, num_bins={self.num_bins}"


class EddyViscosity3D(nn.Module):
    """
    3D Eddy Viscosity / SGS Noronu - Gelismis Versiyon

    LES (Large Eddy Simulation) icin:
      nu_eff = nu + nu_t

    Desteklenen ozellikler:
      - Frekans-band bazli Cs (use_frequency_bands): low/mid/high frekans bandlari
        icin ayri Smagorinsky katsayilari → spectrum slope kontrolu
      - Turbulent Prandtl (use_turbulent_prandtl): kappa_t = nu_t / Pr_t
        → Nusselt dogrulugu icin kritik
      - Anisotropik SGS (use_anisotropic): y ve z yonlerinde farkli viskozite
        → buoyancy/shear anisotropisi yakalama
      - Backscatter (use_backscatter): negatif nu_t bileseni
        → enerji geri aktarimi (deneysel, default OFF)

    Eski uyumluluk: Tum flag'ler False ise mevcut tek-Cs davranisi korunur.
    """

    def __init__(self, resolution: int, diff_ops=None,
                 grid_spacings: Optional[Tuple[float, float, float]] = None,
                 use_frequency_bands: bool = False,
                 use_turbulent_prandtl: bool = False,
                 use_anisotropic: bool = False,
                 use_backscatter: bool = False,
                 use_local_cs: bool = False,
                 use_local_thermal: bool = False,
                 mlp_sgs: Optional['MLPSGS'] = None,
                 spectral_cs: Optional['SpectralCsField'] = None,
                 use_scale_similarity: bool = False):
        super().__init__()
        self.resolution = resolution
        self.diff_ops = diff_ops if diff_ops is not None else SpectralOps3D(resolution)
        self.use_anisotropic = use_anisotropic
        self.use_backscatter = use_backscatter
        self.mlp_sgs = mlp_sgs
        self.spectral_cs = spectral_cs
        self.use_scale_similarity = use_scale_similarity

        # Spectral-Cs varsa MLP ve diğerlerini bypass et (Saf-INNATE pathway)
        if spectral_cs is not None:
            self.use_frequency_bands = False
            self.use_turbulent_prandtl = False
            self.use_local_cs = False
            self.use_local_thermal = False
        # MLP varsa: cs_low/mid/high, cs_thermal, local_* OLUSTURMA
        elif mlp_sgs is not None:
            self.use_frequency_bands = False
            self.use_turbulent_prandtl = False
            self.use_local_cs = False
            self.use_local_thermal = False
        else:
            self.use_frequency_bands = use_frequency_bands
            self.use_turbulent_prandtl = use_turbulent_prandtl
            self.use_local_cs = use_local_cs
            self.use_local_thermal = use_local_thermal

        # Grid spacing: anizotropik grid'de (dx*dy*dz)^(1/3) = LES filter width
        if grid_spacings is not None:
            dx, dy, dz = grid_spacings
            self.delta = (dx * dy * dz) ** (1.0 / 3.0)
        else:
            self.delta = 2 * math.pi / resolution

        # Frekans-band Cs VEYA tekil Cs (MLP yoksa)
        if mlp_sgs is None:
            if use_frequency_bands:
                self.cs_low = nn.Parameter(torch.tensor(0.08))
                self.cs_mid = nn.Parameter(torch.tensor(0.15))
                self.cs_high = nn.Parameter(torch.tensor(0.22))
            else:
                self.smagorinsky_coeff = nn.Parameter(torch.tensor(0.15))

            # Thermal SGS: bagimsiz cs_thermal
            if use_turbulent_prandtl:
                self.cs_thermal = nn.Parameter(torch.tensor(0.10))

            # Lokal Cs modülasyonu
            if use_local_cs:
                self.local_alpha = nn.Parameter(torch.tensor(1.0))
                self.local_R_crit = nn.Parameter(torch.tensor(0.0))

            # Lokal termal modülasyon
            if use_local_thermal:
                self.thermal_beta = nn.Parameter(torch.tensor(1.0))
                self.thermal_Ri_crit = nn.Parameter(torch.tensor(0.0))

        # Anisotropik oranlar (MLP'den bagimsiz — her zaman mevcut)
        if use_anisotropic:
            self.aniso_ratio_y = nn.Parameter(torch.tensor(1.0))
            self.aniso_ratio_z = nn.Parameter(torch.tensor(1.0))

        # Backscatter (MLP'den bagimsiz)
        if use_backscatter:
            self.backscatter_coeff = nn.Parameter(torch.tensor(0.0))

        # Scale-similarity: C_ss parametresi
        if use_scale_similarity:
            self.C_ss = nn.Parameter(torch.tensor(0.1))

    def _get_strain_mag(self, state: FluidState3D) -> torch.Tensor:
        """Strain magnitude hesabi (cache-free, checkpoint-safe)."""
        return self._compute_strain_magnitude(state)

    def _compute_nu_t(self, state: FluidState3D, strain_mag: torch.Tensor,
                      vel_grads=None) -> torch.Tensor:
        """
        Band-bazli veya tekil Cs ile nu_t hesapla.
        use_local_cs=True ise: Cs lokal akis durumuna gore adapte olur.

        Lokal Cs mekanizmasi (Dinamik Smagorinsky ilhamli):
          R = |Omega| / (|S| + eps)   (vorticity-to-strain ratio)
          Cs_mod = sigmoid(alpha * (R - R_crit))
          R > R_crit: vortex-dominant bolge → Cs dusuk (az dissipasyon)
          R < R_crit: shear-dominant bolge → Cs yuksek (cok dissipasyon)
        """
        # -- Lokal Cs modülasyonu --
        local_mod = None
        if self.use_local_cs and vel_grads is not None:
            du_dx, du_dy, du_dz, dv_dx, dv_dy, dv_dz, dw_dx, dw_dy, dw_dz = vel_grads
            # Vorticity from existing gradients (0 extra FFT)
            omega_x = dw_dy - dv_dz
            omega_y = du_dz - dw_dx
            omega_z = dv_dx - du_dy
            omega_mag = torch.sqrt(omega_x**2 + omega_y**2 + omega_z**2 + 1e-8)
            # R = |Omega| / |S| (vorticity-to-strain ratio)
            R = omega_mag / (strain_mag + 1e-8)
            # Lokal modülasyon: sigmoid(alpha * (R_crit - R))
            # R < R_crit → mod ~1 (shear → yüksek Cs)
            # R > R_crit → mod ~0 (vortex → düşük Cs)
            alpha = F.softplus(self.local_alpha) + 0.1  # [0.1, inf), default ~1.1
            R_crit = torch.sigmoid(self.local_R_crit) * 2.0 + 0.5  # [0.5, 2.5]
            local_mod = torch.sigmoid(alpha * (R_crit - R))
            # Clamp: [0.2, 1.5] -- tamamen kapatma veya asiri artirma
            local_mod = 0.2 + 1.3 * local_mod

        if self.use_frequency_bands and hasattr(self.diff_ops, 'band_low'):
            # Fiziksel siralama: cs_l <= cs_m <= cs_h (softplus zinciri)
            cs_l = torch.clamp(self.cs_low, 0.05, 0.15)
            cs_m = (cs_l + F.softplus(self.cs_mid - self.cs_low)).clamp(0.08, 0.20)
            cs_h = (cs_m + F.softplus(self.cs_high - self.cs_mid)).clamp(0.10, 0.25)

            if local_mod is not None:
                cs_l = cs_l * local_mod
                cs_m = cs_m * local_mod
                cs_h = cs_h * local_mod

            nu_t_low = (cs_l * self.delta) ** 2 * self.diff_ops.band_filter(strain_mag, 'low')
            nu_t_mid = (cs_m * self.delta) ** 2 * self.diff_ops.band_filter(strain_mag, 'mid')
            nu_t_high = (cs_h * self.delta) ** 2 * self.diff_ops.band_filter(strain_mag, 'high')
            nu_t = nu_t_low + nu_t_mid + nu_t_high
        else:
            C_s = torch.clamp(self.smagorinsky_coeff, 0.05, 0.18)
            if local_mod is not None:
                C_s = C_s * local_mod
            nu_t = (C_s * self.delta) ** 2 * strain_mag
        return nu_t

    def forward(self, state: FluidState3D, nu_molecular: float = 0.0) -> torch.Tensor:
        """
        Izotropik efektif viskozite: nu_eff = nu + nu_t
        (Eski API uyumlulugu icin korunur)
        """
        strain_mag = self._get_strain_mag(state)
        nu_t = self._compute_nu_t(state, strain_mag)

        # Bug #7 fix: Backscatter — asymmetric clamp [-0.02, 0] + final
        # clamp(>=0) killed the mechanism (ablation diff=0.000, param dead).
        # New: symmetric clamp [-0.05, 0.02] AND allow mild anti-diffusion
        # (clamp floor = -0.3*nu_mol) so small-scale → large-scale energy flux
        # (backscatter) can actually contribute. CFL stability preserved.
        if self.use_backscatter:
            bs = torch.clamp(self.backscatter_coeff, -0.05, 0.02)
            nu_t = nu_t + bs * self.delta ** 2 * strain_mag
            nu_t = torch.clamp(nu_t, min=-0.3 * max(nu_molecular, 1e-6))

        return nu_molecular + nu_t

    def compute_anisotropic_nu(self, state: FluidState3D, nu_mol: float
                               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Anisotropik efektif viskozite: (nu_x, nu_y, nu_z).

        nu_x = nu_mol + nu_t
        nu_y = nu_mol + nu_t * aniso_ratio_y
        nu_z = nu_mol + nu_t * aniso_ratio_z

        Anisotropik mod OFF ise: (nu_eff, nu_eff, nu_eff) doner.
        """
        nu_eff = self.forward(state, nu_mol)
        if not self.use_anisotropic:
            return nu_eff, nu_eff, nu_eff

        nu_t = nu_eff - nu_mol
        ry = torch.clamp(self.aniso_ratio_y, 0.3, 3.0)
        rz = torch.clamp(self.aniso_ratio_z, 0.3, 3.0)
        nu_x = nu_mol + nu_t
        nu_y = nu_mol + nu_t * ry
        nu_z = nu_mol + nu_t * rz
        return nu_x, nu_y, nu_z

    def compute_thermal_eddy_diffusivity(self, state: FluidState3D) -> torch.Tensor:
        """
        Termal eddy difuzivitesi: kappa_t = (cs_thermal * delta)^2 * |S|.

        Bagimsiz cs_thermal parametresi ile (Cs'ye bagimli degil).
        Turbulent Prandtl OFF ise 0 doner (mevcut davranisi bozmaz).
        """
        if not self.use_turbulent_prandtl:
            return torch.tensor(0.0, device=state.u.device)

        strain_mag = self._get_strain_mag(state)
        cs_t = torch.clamp(self.cs_thermal, 0.03, 0.25)
        return (cs_t * self.delta) ** 2 * strain_mag

    def compute_all(self, state: FluidState3D, nu_mol: float,
                    u_hat=None, v_hat=None, w_hat=None,
                    vel_grads=None,
                    theta_hat=None, Ri: float = 0.0,
                    Re_normalized: float = 0.5,
                    layer_idx: int = 0,
                    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Strain'i 1 kez hesapla, kappa_t + anisotropik (nu_x, nu_y, nu_z) dondur.

        MLP SGS varsa: MLP'den (Cs, Pr_t) al, nu_t = (Cs·Δ)²|S|, kappa_t = nu_t/Pr_t.
        MLP yoksa: eski davranis (band-bazli Cs, cs_thermal, lokal modulasyon).

        Args:
            state: FluidState3D
            nu_mol: Molekuler viskozite (skaler)
            u_hat, v_hat, w_hat: Onceden hesaplanmis FFT'ler (optional)
            vel_grads: 9-tuple (du_dx,..,dw_dz) onceden hesaplanmis gradyanlar
            theta_hat: Onceden hesaplanmis sicaklik FFT'si
            Ri: Richardson sayisi = Ra/(Re^2*Pr)
            Re_normalized: Re / 20000 (MLP input icin)
            layer_idx: Katman indeksi (MLP input icin)

        Returns:
            (kappa_t, nu_x, nu_y, nu_z)
        """
        # -- Strain magnitude: TEK hesaplama --
        strain_mag = self._compute_strain_magnitude(state, u_hat, v_hat, w_hat, vel_grads)

        # -- Saf-INNATE Spectral-Cs pathway (2026-05-09, MLP yok) --
        # Cs(x,y,z) field Fourier mod katsayilarindan insa edilir.
        # Klasik Smagorinsky: nu_t = (Cs*Δ)^2 * |S|, |S| ile flow-dependent.
        # Cs field zamana bagli degil, sadece spatial structure ogrenilir.
        if self.spectral_cs is not None:
            Cs = self.spectral_cs()  # shape [Nx, Ny, Nz]
            # Broadcast batch dim
            Cs_b = Cs.unsqueeze(0)
            # nu_t = (Cs * delta)^2 * |S|
            nu_t = (Cs_b * self.delta) ** 2 * strain_mag
            # Reynolds analogy: kappa_t = nu_t / Pr_t (per-layer learnable)
            Pr_t = self.spectral_cs.get_Pr_t()
            kappa_t = nu_t / Pr_t

            # Skip MLP/legacy pathways. nu_mol caller'dan gelir.
            if self.use_anisotropic:
                aniso_y = self.spectral_cs.get_aniso_y()
                aniso_z = self.spectral_cs.get_aniso_z()
                nu_x = nu_mol + nu_t
                nu_y = nu_mol + nu_t * aniso_y
                nu_z = nu_mol + nu_t * aniso_z
            else:
                nu_eff = nu_mol + nu_t
                nu_x = nu_y = nu_z = nu_eff

            return kappa_t, nu_x, nu_y, nu_z

        # -- MLP SGS pathway --
        if self.mlp_sgs is not None:
            # Vorticity magnitude (from vel_grads, 0 extra FFT)
            if vel_grads is not None:
                du_dx, du_dy, du_dz, dv_dx, dv_dy, dv_dz, dw_dx, dw_dy, dw_dz = vel_grads
            elif u_hat is not None:
                ops = self.diff_ops
                du_dx, du_dy, du_dz = ops.gradient_from_hat(u_hat)
                dv_dx, dv_dy, dv_dz = ops.gradient_from_hat(v_hat)
                dw_dx, dw_dy, dw_dz = ops.gradient_from_hat(w_hat)
            else:
                ops = self.diff_ops
                du_dx, du_dy, du_dz = ops.gradient(state.u)
                dv_dx, dv_dy, dv_dz = ops.gradient(state.v)
                dw_dx, dw_dy, dw_dz = ops.gradient(state.w)

            omega_x = dw_dy - dv_dz
            omega_y = du_dz - dw_dx
            omega_z = dv_dx - du_dy
            omega_mag = torch.sqrt(omega_x**2 + omega_y**2 + omega_z**2 + 1e-8)

            # Gradient Richardson number
            if theta_hat is not None and Ri > 0:
                ops = self.diff_ops
                dtheta_dy = ops.partial_y_from_hat(theta_hat) if hasattr(ops, 'partial_y_from_hat') else \
                            ops.gradient_from_hat(theta_hat)[1]
                Ri_g = -Ri * dtheta_dy / (strain_mag ** 2 + 1e-8)
            else:
                Ri_g = torch.zeros_like(strain_mag)

            # MLP forward: Cs field + scalar Pr_t
            Cs, Pr_t = self.mlp_sgs(
                strain_mag, omega_mag, Ri_g,
                Re_normalized, layer_idx, self.delta
            )

            # nu_t = (Cs * delta)^2 * |S|
            nu_t = (Cs * self.delta) ** 2 * strain_mag

            # Reynolds analogy: kappa_t structurally bound to nu_t via Pr_t.
            # Momentum and thermal SGS cannot decouple — kappa_t collapse
            # to zero requires nu_t collapse too, which momentum loss prevents.
            kappa_t = nu_t / Pr_t

        else:
            # -- Legacy pathway: band-based Cs / cs_thermal / local modulation --
            nu_t = self._compute_nu_t(state, strain_mag, vel_grads=vel_grads)

            if self.use_turbulent_prandtl:
                cs_t = torch.clamp(self.cs_thermal, 0.03, 0.25)
                kappa_t = (cs_t * self.delta) ** 2 * strain_mag
            else:
                kappa_t = torch.tensor(0.0, device=state.u.device)

            # Lokal termal modulasyon (legacy)
            if self.use_local_thermal and theta_hat is not None and Ri > 0:
                ops = self.diff_ops
                dtheta_dy = ops.partial_y_from_hat(theta_hat) if hasattr(ops, 'partial_y_from_hat') else \
                            ops.gradient_from_hat(theta_hat)[1]
                Ri_g = -Ri * dtheta_dy / (strain_mag ** 2 + 1e-8)
                beta = F.softplus(self.thermal_beta) + 0.1
                Ri_crit = torch.sigmoid(self.thermal_Ri_crit) * 0.5 + 0.1
                thermal_mod = torch.sigmoid(beta * (Ri_crit - Ri_g))
                thermal_mod = 0.1 + 1.9 * thermal_mod
                kappa_t = kappa_t * thermal_mod

        # -- Backscatter (MLP'den bagimsiz) --
        # Bug #7 fix: symmetric clamp + allow mild anti-diffusion floor
        if self.use_backscatter:
            bs = torch.clamp(self.backscatter_coeff, -0.05, 0.02)
            nu_t = nu_t + bs * self.delta ** 2 * strain_mag
            # Floor: nu_mol accessible via state (compute_all path); use small negative.
            nu_t = torch.clamp(nu_t, min=-1e-5)

        # -- Scale-similarity mixed model --
        if self.use_scale_similarity and hasattr(self.diff_ops, 'to_hat'):
            ops = self.diff_ops
            C_ss_val = torch.clamp(self.C_ss, 0.0, 0.5)
            # Test filter: low-pass at k_max/2
            u_bar = self._test_filter(state.u, ops)
            v_bar = self._test_filter(state.v, ops)
            w_bar = self._test_filter(state.w, ops)
            # Leonard stress trace: L_ii = filter(u_i^2) - filter(u_i)^2
            L_trace = (self._test_filter(state.u**2, ops) - u_bar**2 +
                       self._test_filter(state.v**2, ops) - v_bar**2 +
                       self._test_filter(state.w**2, ops) - w_bar**2)
            # Scale-similarity contribution to nu_t
            scale_sim = C_ss_val * L_trace / (strain_mag + 1e-8)
            nu_t = nu_t + scale_sim
            # Scale-similarity adds forward scatter (L_trace >= 0 by Jensen).
            # Clamp prevents anti-diffusion from other sources (e.g. backscatter_coeff).
            nu_t = torch.clamp(nu_t, min=-0.3 * nu_mol)

        # -- Anisotropik nu --
        nu_eff = nu_mol + nu_t
        if self.use_anisotropic:
            ry = torch.clamp(self.aniso_ratio_y, 0.3, 3.0)
            rz = torch.clamp(self.aniso_ratio_z, 0.3, 3.0)
            nu_x = nu_eff
            nu_y = nu_mol + nu_t * ry
            nu_z = nu_mol + nu_t * rz
        else:
            nu_x = nu_y = nu_z = nu_eff

        return kappa_t, nu_x, nu_y, nu_z

    @staticmethod
    def _test_filter(f: torch.Tensor, ops) -> torch.Tensor:
        """Test filter: low-pass at k_max/2 for scale-similarity model."""
        f_hat = ops.to_hat(f)
        # Create test filter mask if not cached
        if not hasattr(ops, '_test_filter_mask'):
            k_mag = torch.sqrt(ops.k_squared)
            k_max = k_mag.max()
            ops._test_filter_mask = (k_mag < k_max / 2.0).float()
        return ops.from_hat(f_hat * ops._test_filter_mask)

    def _compute_strain_magnitude(self, state: FluidState3D,
                                    u_hat=None, v_hat=None, w_hat=None,
                                    vel_grads=None) -> torch.Tensor:
        """Strain rate magnitude: |S| = sqrt(2 * S_ij * S_ij)

        vel_grads verilirse dogrudan kullanir (gradient sharing, 9 irFFT tasarruf).
        u_hat/v_hat/w_hat verilirse gradient_from_hat kullanir (3 FFT tasarrufu).
        """
        ops = self.diff_ops
        if vel_grads is not None:
            du_dx, du_dy, du_dz, dv_dx, dv_dy, dv_dz, dw_dx, dw_dy, dw_dz = vel_grads
        elif u_hat is not None and v_hat is not None and w_hat is not None:
            du_dx, du_dy, du_dz = ops.gradient_from_hat(u_hat)
            dv_dx, dv_dy, dv_dz = ops.gradient_from_hat(v_hat)
            dw_dx, dw_dy, dw_dz = ops.gradient_from_hat(w_hat)
        else:
            u, v, w = state.u, state.v, state.w
            du_dx, du_dy, du_dz = ops.gradient(u)
            dv_dx, dv_dy, dv_dz = ops.gradient(v)
            dw_dx, dw_dy, dw_dz = ops.gradient(w)

        S_xx = du_dx
        S_yy = dv_dy
        S_zz = dw_dz
        S_xy = 0.5 * (du_dy + dv_dx)
        S_xz = 0.5 * (du_dz + dw_dx)
        S_yz = 0.5 * (dv_dz + dw_dy)

        S_sq = S_xx**2 + S_yy**2 + S_zz**2 + 2*(S_xy**2 + S_xz**2 + S_yz**2)

        return torch.sqrt(2 * S_sq + 1e-8)

    def sgs_dissipation(self, state: FluidState3D, nu_molecular: float = 0.0) -> torch.Tensor:
        """SGS enerji dissipasyonu: eps_sgs = 2 * nu_t * |S|^2"""
        nu_eff = self.forward(state, nu_molecular)
        nu_t = nu_eff - nu_molecular
        strain_mag = self._get_strain_mag(state)
        return 2 * nu_t * strain_mag**2

    def extra_repr(self) -> str:
        parts = [f"resolution={self.resolution}"]
        if self.use_frequency_bands:
            parts.append("freq_bands=True")
        if self.use_turbulent_prandtl:
            parts.append("cs_thermal=True")
        if self.use_anisotropic:
            parts.append("aniso=True")
        if self.use_backscatter:
            parts.append("backscatter=True")
        if hasattr(self, 'mlp_sgs') and self.mlp_sgs is not None:
            parts.append("mlp_sgs=True")
        if hasattr(self, 'use_scale_similarity') and self.use_scale_similarity:
            parts.append("scale_sim=True")
        return ", ".join(parts)


class MLPSGS(nn.Module):
    """
    MLP-based SGS closure: akis ozelliklerinden Cs tahmin eder, kappa_t
    fiziksel Reynolds analojisi ile turetilir (kappa_t = nu_t / Pr_t).

    Input (6D, per grid point):
      0: |S| * Delta          — strain rate magnitude (scaled by filter width)
      1: |Omega| * Delta      — vorticity magnitude (scaled by filter width)
      2: R = |Omega|/(|S|+e)  — vorticity-to-strain ratio
      3: 100*Ri_g (clip [-5,5]) — scaled gradient Richardson number
      4: Re / 20000           — normalized Reynolds
      5: layer_idx / n_layers — position in unrolling [0, 1]

    Output:
      Cs in [0.05, 0.25]      — Smagorinsky coefficient (spatial field)
      Pr_t in [0.3, 1.5]      — turbulent Prandtl number (single scalar)

    Shared across all layers. Per-layer bias allows layer-specific tuning of Cs.

    [Reynolds analogy reparam, 2026-04-17]: Previous Bug #1 fix split fc2 into
    fc2_cs and fc2_kappa (decoupled momentum/thermal SGS), but this broke the
    physical Reynolds analogy — model pushed kappa_coeff to 0.01 floor, killing
    thermal SGS. New structure: single fc2_cs head, kappa_t = nu_t / Pr_t
    with Pr_t as a learnable scalar (single param). Turbulent thermal transport
    is now structurally bound to momentum SGS, cannot collapse independently.

    [Bug #9 fix]: GELU activation instead of ReLU (smooth gradient, fewer dead
    neurons in 32-hidden small MLP).

    [Bug #8 fix]: Ri_g scale 100 → 10000 so the clamp [-5, 5] is actually
    reachable for realistic Ri_g ~ 2.9e-5 in forced Kolmogorov mixed convection.
    """

    def __init__(self, hidden_dim: int = 32, n_layers: int = 20):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        # Shared fc1: 6 -> hidden
        self.fc1 = nn.Linear(6, hidden_dim)
        # Single output head for Cs (kappa_t derived via Pr_t)
        self.fc2_cs = nn.Linear(hidden_dim, 1)

        # Per-layer output bias for Cs
        self.layer_bias_cs = nn.Parameter(torch.zeros(n_layers, 1))

        # Learnable scalar turbulent Prandtl number (Reynolds analogy)
        # Initialized at Pr_t = 0.85 (standard shear flow value).
        # log-parametrized for positivity; clamped to [0.3, 1.5] in forward.
        self.log_Pr_t = nn.Parameter(torch.log(torch.tensor(0.85)))

        # Init: Xavier uniform, output bias for Cs~0.155 (Lilly's value)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2_cs.weight, gain=0.1)
        # sigmoid(0.1) ~ 0.525 -> 0.05 + 0.20*0.525 ~ 0.155
        with torch.no_grad():
            self.fc2_cs.bias.copy_(torch.tensor([0.1]))

    @staticmethod
    def migrate_state_dict(state_dict: dict, prefix: str = "") -> dict:
        """
        Migrate old checkpoints to Pr_t reparametrization.

        Supported source formats:
          (a) Pre-Bug#1-fix: fc2.weight [2, 32], fc2.bias [2], layer_bias [20, 2]
          (b) Post-Bug#1-fix (2026-04-17 split): fc2_cs + fc2_kappa + layer_bias_{cs,kappa}

        In both cases, fc2_kappa / layer_bias_kappa data is DISCARDED (Reynolds
        analogy rebuilt structurally via log_Pr_t). fc2_cs is preserved from the
        split checkpoint or extracted (row 0) from the legacy joint fc2. log_Pr_t
        is initialized fresh at log(0.85) unless already present.

        Returns modified state_dict (in place).
        """
        def _migrate_one(sd: dict, pfx: str) -> None:
            # Legacy joint fc2 (pre-split)
            fc2_w = sd.pop(pfx + "fc2.weight", None)
            fc2_b = sd.pop(pfx + "fc2.bias", None)
            lb = sd.pop(pfx + "layer_bias", None)
            if fc2_w is not None:
                sd[pfx + "fc2_cs.weight"] = fc2_w[0:1].clone()
            if fc2_b is not None:
                sd[pfx + "fc2_cs.bias"] = fc2_b[0:1].clone()
            if lb is not None:
                sd[pfx + "layer_bias_cs"] = lb[:, 0:1].clone()
            # Discard thermal head from split checkpoints (Reynolds analogy reparam)
            sd.pop(pfx + "fc2_kappa.weight", None)
            sd.pop(pfx + "fc2_kappa.bias", None)
            sd.pop(pfx + "layer_bias_kappa", None)

        if prefix:
            _migrate_one(state_dict, prefix)
        else:
            # Auto-discover all mlp_sgs prefixes in state_dict
            prefixes = set()
            for k in list(state_dict.keys()):
                for suffix in ("fc2.weight", "fc2.bias", "layer_bias",
                               "fc2_kappa.weight", "fc2_kappa.bias",
                               "layer_bias_kappa"):
                    if k.endswith("." + suffix) or k == suffix:
                        pfx = k[: -len(suffix)]
                        if pfx.endswith("mlp_sgs.") or pfx == "":
                            prefixes.add(pfx)
            for pfx in prefixes:
                _migrate_one(state_dict, pfx)
        return state_dict

    def forward(self, strain_mag: torch.Tensor, omega_mag: torch.Tensor,
                Ri_g: torch.Tensor, Re_normalized: float,
                layer_idx: int, delta: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            strain_mag: |S| [B, Nx, Ny, Nz]
            omega_mag: |Omega| [B, Nx, Ny, Nz]
            Ri_g: gradient Richardson number [B, Nx, Ny, Nz]
            Re_normalized: Re / 20000 (scalar)
            layer_idx: current layer index (0-indexed)
            delta: LES filter width

        Returns:
            Cs: Smagorinsky coefficient field [B, Nx, Ny, Nz], range [0.05, 0.25]
            Pr_t: turbulent Prandtl number (scalar tensor), range [0.3, 1.5]
        """
        shape = strain_mag.shape

        # Normalize inputs
        # 2026-05-08 Codex consultation: s_norm/o_norm bounded değildi, türbülans
        # patladığında MLP'ye garbage input gidiyor (OOD), Cs decisions bozuluyor.
        # Clamp [0, 10] ekledi — normal range ~0.05, blow-up'ı dampleyecek.
        # nan_to_num: upstream NaN propagasyonu MLP'yi de NaN yapmasin.
        s_norm = torch.nan_to_num(strain_mag * delta, nan=0.0, posinf=10.0, neginf=0.0)
        o_norm = torch.nan_to_num(omega_mag * delta, nan=0.0, posinf=10.0, neginf=0.0)
        s_norm = torch.clamp(s_norm, 0.0, 10.0)
        o_norm = torch.clamp(o_norm, 0.0, 10.0)
        R = omega_mag / (strain_mag + 1e-8)      # vorticity/strain ratio
        R = torch.nan_to_num(R, nan=1.0, posinf=10.0, neginf=0.0)
        R = torch.clamp(R, 0.0, 10.0)
        # Bug #8 fix: Ri_g ~ 2.9e-5 in forced Kolmogorov Re=10K, Ra=1e5.
        # Scale 100 → clip[-5,5] never activates; scale 10000 brings physical
        # Ri_g to ~0.3 and clip becomes meaningful in strongly stratified pockets.
        Ri_g_scaled = Ri_g * 10000.0
        Ri_g_clip = torch.clamp(Ri_g_scaled, -5.0, 5.0)
        Re_field = torch.full_like(strain_mag, Re_normalized)
        layer_field = torch.full_like(strain_mag, layer_idx / self.n_layers)

        # Stack: [B, Nx, Ny, Nz, 6]
        x = torch.stack([s_norm, o_norm, R, Ri_g_clip, Re_field, layer_field], dim=-1)

        # Flatten spatial dims: [B*Nx*Ny*Nz, 6]
        x_flat = x.reshape(-1, 6)

        # MLP forward (Bug #9 fix: GELU instead of ReLU → smooth, fewer dead neurons)
        h = F.gelu(self.fc1(x_flat))

        # Single Cs head + per-layer bias
        raw_cs = self.fc2_cs(h).squeeze(-1) + self.layer_bias_cs[layer_idx, 0]

        # Output activation: bounded sigmoid
        Cs = 0.05 + 0.20 * torch.sigmoid(raw_cs)            # [0.05, 0.25]
        Cs = Cs.reshape(shape)

        # Turbulent Prandtl number (Reynolds analogy): kappa_t = nu_t / Pr_t
        # Clamp ensures physical range; log-param keeps it positive by construction.
        Pr_t = torch.clamp(torch.exp(self.log_Pr_t), 0.3, 1.5)

        return Cs, Pr_t

    def extra_repr(self) -> str:
        n_params = sum(p.numel() for p in self.parameters())
        return f"hidden={self.hidden_dim}, n_layers={self.n_layers}, params={n_params}"


# =============================================================================
# BÖLÜM 6a-2: SAF-INNATE SPECTRAL Cs FIELD (2026-05-09)
# =============================================================================
# MLP YOK — Cs(x,y,z) field'i Fourier mod katsayıları ile parametrize edilir.
# Berke'nin 2026-02-18 orijinal vizyonu: "MLP modulatorleri kaldırılacak,
# saf INNATE mimarisi". 2026-05-09'da gerçekleştirildi.
#
# Klasik Smagorinsky korunur: nu_t = (Cs*Δ)² × |S|. |S| flow-dependent
# adaptasyon sağlar, Cs(x,y,z) ise spatial structure ogrenilen kısım.
# =============================================================================

class SpectralCsField(nn.Module):
    """
    Saf-INNATE Cs spatial field — Fourier mode-coefficient learnable.

    Cs(x,y,z) = base + Real(IRFFT(c_hat_truncated))

    Düşük-k modlarına trunke edilmiş kompleks Fourier katsayıları öğrenilen
    parametreler. Real ve Imag ayrı tensörlerde tutulur (Adam optimizer
    real-valued grad ile çalışsın).

    Per-layer instance: her katman kendi Cs field'ini öğrenir (paylaşılmaz).

    Param count = 2 × kx_max × ky_max × kz_max + 2 + 2
                  (real + imag + base + log_Pr_t + aniso_y + aniso_z)

    Default 5×8×6 = 240 modes → 480 + 4 = 484 param/layer × 20 = 9680.
    """

    def __init__(self, Nx: int, Ny: int, Nz: int,
                 kx_max: int = 5, ky_max: int = 8, kz_max: int = 6,
                 base: float = 0.155, Pr_t_init: float = 0.85,
                 use_anisotropic: bool = True,
                 init_scale: float = 0.005):
        super().__init__()
        self.Nx, self.Ny, self.Nz = Nx, Ny, Nz
        # rfftn shape on real input: [Nx, Ny, Nz//2 + 1]
        self.kx_max = min(kx_max, Nx)
        self.ky_max = min(ky_max, Ny)
        self.kz_max = min(kz_max, Nz // 2 + 1)
        self._rfft_shape = (Nx, Ny, Nz // 2 + 1)

        # Trainable: real + imag parts of complex Fourier coefficients
        # Init: küçük random perturbation (mode 0,0,0 = base ile çakışmasın)
        self.coeffs_real = nn.Parameter(
            init_scale * torch.randn(self.kx_max, self.ky_max, self.kz_max)
        )
        self.coeffs_imag = nn.Parameter(
            init_scale * torch.randn(self.kx_max, self.ky_max, self.kz_max)
        )
        # k=0 component zero (DC handled by base scalar)
        with torch.no_grad():
            self.coeffs_real[0, 0, 0] = 0.0
            self.coeffs_imag[0, 0, 0] = 0.0

        # Base Smagorinsky scalar (Lilly default 0.155 ~ Cs²=0.024)
        self.base = nn.Parameter(torch.tensor(base))
        # Per-layer turbulent Prandtl (log-parametrized for positivity)
        self.log_Pr_t = nn.Parameter(torch.log(torch.tensor(Pr_t_init)))
        # Per-layer anisotropy ratios (y/z)
        self.use_anisotropic = use_anisotropic
        if use_anisotropic:
            self.log_aniso_y = nn.Parameter(torch.tensor(0.0))  # exp(0)=1
            self.log_aniso_z = nn.Parameter(torch.tensor(0.0))

    def forward(self) -> torch.Tensor:
        """Build Cs(x,y,z) field via inverse FFT."""
        device = self.coeffs_real.device
        dtype = self.coeffs_real.dtype
        # Allocate full rfft tensor (low-k truncation)
        spectrum = torch.zeros(self._rfft_shape, dtype=torch.complex64, device=device)
        spectrum[:self.kx_max, :self.ky_max, :self.kz_max] = (
            self.coeffs_real.to(torch.complex64)
            + 1j * self.coeffs_imag.to(torch.complex64)
        )
        # Inverse rFFT → real-valued spatial field
        Cs_perturbation = torch.fft.irfftn(
            spectrum, s=(self.Nx, self.Ny, self.Nz)
        ).to(dtype)
        # Add base + clamp [0.05, 0.30] (numerical safety + positivity)
        Cs = self.base + Cs_perturbation
        Cs = torch.clamp(Cs, 0.05, 0.30)
        return Cs

    def get_Pr_t(self) -> torch.Tensor:
        """Turbulent Prandtl number, clamped [0.3, 1.5]."""
        return torch.clamp(torch.exp(self.log_Pr_t), 0.3, 1.5)

    def get_aniso_y(self) -> torch.Tensor:
        if self.use_anisotropic:
            return torch.clamp(torch.exp(self.log_aniso_y), 0.5, 2.0)
        return torch.tensor(1.0, device=self.coeffs_real.device)

    def get_aniso_z(self) -> torch.Tensor:
        if self.use_anisotropic:
            return torch.clamp(torch.exp(self.log_aniso_z), 0.5, 2.0)
        return torch.tensor(1.0, device=self.coeffs_real.device)

    def extra_repr(self) -> str:
        n_params = sum(p.numel() for p in self.parameters())
        return (f"grid=({self.Nx}x{self.Ny}x{self.Nz}), "
                f"k_trunc=({self.kx_max}x{self.ky_max}x{self.kz_max}), "
                f"params={n_params}")


# =============================================================================
# BÖLÜM 6b: MIXED CONVECTION THERMAL NEURONS
# =============================================================================
# Bitirme2 projesi icin termal akis noronlari.
# Boussinesq yaklasimi, sicaklik perturbasyonu T' = T - T_base(y).

class Forcing3D(nn.Module):
    """
    Momentum denklemine dis kuvvet terimi (ruzgar).

    3 mod destekler:
      kolmogorov : F_x = A * sin(k_f * 2*pi*y/Ly + phi), F_y=0, F_z=0
      uniform    : F_x = A, F_y=0, F_z=0
      stochastic : F_x = A + sigma*eta(t), F_y=0, F_z=0 (Ornstein-Uhlenbeck)

    Harmonik uzanti (use_harmonics=True):
      Fx += A_k2 * sin(2*k_f*y + phase_k2) + A_k3 * sin(3*k_f*y + phase_k3)
      Init=0 → baslangicta mevcut davranisla ayni (4 ek param).

    Learnable: amplitude (hepsinde), sigma + tau_corr (sadece stochastic),
               harmonik amplitud ve fazlar (use_harmonics ise).
    Output shape: (Fx, Fy, Fz) her biri [1, 1, Ny, 1] - broadcasting icin.
    """

    def __init__(self, Ny: int, Ly: float, k_f: int = 1,
                 mode: str = "kolmogorov", use_harmonics: bool = False):
        super().__init__()
        assert mode in ("kolmogorov", "uniform", "stochastic")
        self.mode = mode
        self.Ly = Ly
        self.k_f = k_f
        self.use_harmonics = use_harmonics

        # -- learnable --
        # Forcing amplitude: steady-state'te F = nu * k_f^2 * U_target.
        # Re=5000, nu=2e-4, k_f=2*pi/Ly ile denge A ~ 1e-4.
        # Init=0.001, ogrenilebilir. Cok buyuk baslangic (1.0) enerji biriktirir.
        self.amplitude = nn.Parameter(torch.tensor(0.005))

        if mode == "stochastic":
            self.sigma = nn.Parameter(torch.tensor(0.1))
            self.tau_corr = nn.Parameter(torch.ones(1))

        # Harmonik parametreler (init=0 → baslangicta etkisiz)
        if use_harmonics:
            self.amplitude_k2 = nn.Parameter(torch.tensor(0.0))
            self.amplitude_k3 = nn.Parameter(torch.tensor(0.0))
            self.phase_k2 = nn.Parameter(torch.tensor(0.0))
            self.phase_k3 = nn.Parameter(torch.tensor(0.0))

        # -- non-learnable state --
        self.phi: float = 0.0          # random phase (kolmogorov)
        self.eta: float = 0.0          # OU process state (stochastic)

        # -- y grid buffer: [1, 1, Ny, 1] --
        y = torch.linspace(0, Ly, Ny + 1)[:-1]  # periodic, Ny noktali
        self.register_buffer("y_grid", y.view(1, 1, Ny, 1))

    def forward(self) -> tuple:
        """(Fx, Fy, Fz) dondurur, her biri [1, 1, Ny, 1]."""
        A = self.amplitude.clamp(0.001, 0.02)
        zero = torch.zeros_like(self.y_grid)

        if self.mode == "kolmogorov":
            arg = self.k_f * 2.0 * math.pi * self.y_grid / self.Ly + self.phi
            Fx = A * torch.sin(arg)
        elif self.mode == "uniform":
            Fx = A * torch.ones_like(self.y_grid)
        else:  # stochastic
            Fx = (A + self.eta) * torch.ones_like(self.y_grid)

        # Harmonikler (kolmogorov modunda anlamli)
        # Bug #10 fix: clamp harmonics to prevent unbounded growth + phase
        # identity-degeneracy during optimization. Symmetric ±0.01 (5× base amp).
        if self.use_harmonics and self.mode == "kolmogorov":
            y_norm = 2.0 * math.pi * self.y_grid / self.Ly
            A2 = self.amplitude_k2.clamp(-0.01, 0.01)
            A3 = self.amplitude_k3.clamp(-0.01, 0.01)
            phi2 = self.phase_k2.clamp(-math.pi, math.pi)
            phi3 = self.phase_k3.clamp(-math.pi, math.pi)
            Fx = Fx + A2 * torch.sin(2 * self.k_f * y_norm + phi2)
            Fx = Fx + A3 * torch.sin(3 * self.k_f * y_norm + phi3)

        return Fx, zero, zero

    def step_ou(self, dt: float):
        """Ornstein-Uhlenbeck process'i bir adim ilerlet."""
        if self.mode != "stochastic":
            return
        tau = self.tau_corr.detach().clamp(0.1, 10.0).item()
        sig = self.sigma.detach().clamp(0.0, 2.0).item()
        noise = torch.randn(1, device=self.y_grid.device).item()
        self.eta = self.eta * (1.0 - dt / tau) + sig * math.sqrt(2.0 * dt / tau) * noise

    def reset_phase(self):
        """Kolmogorov phase'i randomize et (generalization icin)."""
        self.phi = torch.rand(1).item() * 2.0 * math.pi
        self.eta = 0.0

    def extra_repr(self) -> str:
        p = f"mode={self.mode}, k_f={self.k_f}, Ly={self.Ly}"
        if self.mode == "stochastic":
            p += f", sigma={self.sigma.item():.3f}, tau={self.tau_corr.item():.3f}"
        if self.use_harmonics:
            p += ", harmonics=True"
        return p


class Buoyancy3D(nn.Module):
    """
    Termal buoyancy kuvveti: sicak parcacik yukselir, soguk duser.

    Boussinesq yaklasimi:
        F_buoy = Ri * buoyancy_strength * T' * e_y

    T' = perturbation sicaklik (mean=0 enforce edilir).
    Kuvvet sadece y-yonunde (dikey), x ve z bilesenleri sifir.

    Learnable: buoyancy_strength (1 parametre, init=1.0, clamp=[0, 50])
    """

    def __init__(self, Ri: float = 0.35):
        super().__init__()
        self.Ri = Ri
        self.buoyancy_strength = nn.Parameter(torch.tensor(0.5))

    def forward(self, theta: torch.Tensor) -> tuple:
        """
        Args:
            theta: T' perturbation sicaklik [B, Nx, Ny, Nz]
        Returns:
            (Fx, Fy, Fz) — Fx=0, Fz=0, Fy = Ri * strength * T'
        """
        strength = torch.clamp(self.buoyancy_strength, 0.0, 50.0)
        Fy = self.Ri * strength * theta
        zeros = torch.zeros_like(Fy)
        return zeros, Fy, zeros

    def set_Ri(self, Ri: float):
        """Richardson number'i guncelle (parameter sweep icin)."""
        self.Ri = Ri


class ThermalDiffusion3D(nn.Module):
    """
    Termal difuzyon: kappa * kappa_scale * nabla^2(T').

    Izotropik mod: kappa * kappa_scale * nabla^2(T')
    Anisotropik mod: (kappa*sx + kappa_t)*d2T/dx2 + (kappa*sy + kappa_t)*d2T/dy2
                     + (kappa*sz + kappa_t)*d2T/dz2

    Learnable: kappa_scale (izotropik) veya kappa_scale_x/y/z (anisotropik)
    SpectralOps: EVET (laplacian / directional_laplacian)
    """

    def __init__(self, kappa: float, spectral_ops, use_anisotropic: bool = False):
        """
        Args:
            kappa: Termal difuzivite = 1/(Re*Pr)
            spectral_ops: SpectralOps3DAniso instance (paylasimli)
            use_anisotropic: True ise x/y/z ayri kappa_scale
        """
        super().__init__()
        self.kappa = kappa
        self.ops = spectral_ops
        self.use_anisotropic = use_anisotropic

        if use_anisotropic:
            self.kappa_scale_x = nn.Parameter(torch.ones(1))
            self.kappa_scale_y = nn.Parameter(torch.ones(1))
            self.kappa_scale_z = nn.Parameter(torch.ones(1))
        else:
            self.kappa_scale = nn.Parameter(torch.ones(1))

    def forward(self, theta: torch.Tensor, kappa_t: Optional[torch.Tensor] = None,
                theta_hat: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            theta: T' perturbation [B, Nx, Ny, Nz]
            kappa_t: Turbulent eddy termal difuzivite (EddyViscosity3D'den)
                     None veya skaler 0.0 ise yok sayilir.
            theta_hat: Onceden hesaplanmis fftn(theta) (opsiyonel, FFT caching icin).
                       Verilirse directional_laplacian_from_hat / laplacian_from_hat kullanilir.
        Returns:
            diffusion: termal difuzyon terimi [B, Nx, Ny, Nz]
        """
        # kappa_t varsa ve sifir degilse ekle
        has_kappa_t = kappa_t is not None and not (isinstance(kappa_t, torch.Tensor) and kappa_t.dim() == 0 and kappa_t.item() == 0.0)

        if self.use_anisotropic:
            # Bug #4 fix: clamp [0.1, 20.0] → [0.3, 3.0]. Old range let the model
            # push κ_scale to 20× and effectively wipe out θ perturbations.
            # New range mirrors momentum aniso_ratio and keeps diffusion physical.
            sx = torch.clamp(self.kappa_scale_x, 0.3, 3.0)
            sy = torch.clamp(self.kappa_scale_y, 0.3, 3.0)
            sz = torch.clamp(self.kappa_scale_z, 0.3, 3.0)
            if theta_hat is not None:
                d2T_dx2, d2T_dy2, d2T_dz2 = self.ops.directional_laplacian_from_hat(theta_hat)
            else:
                d2T_dx2, d2T_dy2, d2T_dz2 = self.ops.directional_laplacian(theta)
            if has_kappa_t:
                diff = (self.kappa * sx + kappa_t) * d2T_dx2 \
                     + (self.kappa * sy + kappa_t) * d2T_dy2 \
                     + (self.kappa * sz + kappa_t) * d2T_dz2
            else:
                diff = self.kappa * (sx * d2T_dx2 + sy * d2T_dy2 + sz * d2T_dz2)
            return diff
        else:
            scale = torch.clamp(self.kappa_scale, 0.3, 3.0)  # Bug #4 fix
            if theta_hat is not None:
                lap = self.ops.laplacian_from_hat(theta_hat)
            else:
                lap = self.ops.laplacian(theta)
            if has_kappa_t:
                return (self.kappa * scale + kappa_t) * lap
            else:
                return self.kappa * scale * lap

    def set_kappa(self, kappa: float):
        """Sweep sirasinda kappa'yi guncelle."""
        self.kappa = kappa


class ThermalAdvection3D(nn.Module):
    """
    Termal adveksiyon: sicaklik perturbasyonunu hiz alani ile tasir.

    u . nabla(T') = mod * (u*dT'/dx + v*dT'/dy + w*dT'/dz)

    Convective form (Advection3D ile tutarli).
    Nonlinear terim icerdiginden dealiasing (2/3 kurali) ZORUNLU.

    Learnable parametre: thermal_adv_modulator (use_modulator=True ise, 1 param)
    SpectralOps: EVET (gradient + dealias)
    """

    def __init__(self, spectral_ops, use_modulator: bool = False):
        """
        Args:
            spectral_ops: SpectralOps3DAniso instance (paylasimli)
            use_modulator: True ise ogr. modulator ekle
        """
        super().__init__()
        self.ops = spectral_ops
        self.use_modulator = use_modulator
        if use_modulator:
            self.thermal_adv_modulator = nn.Parameter(torch.tensor(1.0))

    def forward(self, u: torch.Tensor, v: torch.Tensor, w: torch.Tensor,
                theta: torch.Tensor, theta_hat: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Termal adveksiyon: mod * u . nabla(T')

        Args:
            u, v, w: Hiz bilesenleri [B, Nx, Ny, Nz]
            theta: T' perturbation sicaklik [B, Nx, Ny, Nz]
            theta_hat: Onceden hesaplanmis fftn(theta) (opsiyonel, FFT caching icin).
                       Verilirse gradient_from_hat kullanilir (1 FFT tasarrufu).
        Returns:
            advection: mod * u.nabla(T') [B, Nx, Ny, Nz] (dealiased)
        """
        if theta_hat is not None:
            dT_dx, dT_dy, dT_dz = self.ops.gradient_from_hat(theta_hat)
        else:
            dT_dx, dT_dy, dT_dz = self.ops.gradient(theta)
        adv = u * dT_dx + v * dT_dy + w * dT_dz
        adv = self.ops.dealias(adv)
        if self.use_modulator:
            mod = torch.clamp(self.thermal_adv_modulator, 0.5, 2.0)
            adv = mod * adv
        return adv


# -----------------------------------------------------------------------------
# Faz 2: Non-Boussinesq Yogunluk Noronlari
# -----------------------------------------------------------------------------
# Boussinesq yaklasiminin otesinde, degisken yogunluk etkileri.
# Ideal gaz EOS: p = rho * R * T,  sabit referans basinc p_0.
# Tum degerler nondimensional (rho_0=1, T_0=1, R=1, p_0=1).

class DensityUpdate3D(nn.Module):
    """
    Non-Boussinesq yogunluk guncelleme: ideal gaz, sabit p_0.

    rho = rho_0 * T_0 / T_total
    T_total = T_base(y) + T' (nondim mutlak sicaklik, Kelvin/T_0)

    Guvenlik: T_safe = clamp(T_total, min=0.01), rho = clamp(rho, 0.5*rho_0, 2.0*rho_0)
    Learnable: YOK (fiziksel yasa)
    SpectralOps: HAYIR
    """

    def __init__(self, rho_0: float = 1.0, T_0: float = 1.0):
        """
        Args:
            rho_0: Referans yogunluk (nondim, genelde 1.0)
            T_0: Referans sicaklik (nondim, genelde 1.0)
        """
        super().__init__()
        self.register_buffer('rho_0', torch.tensor(rho_0))
        self.register_buffer('T_0', torch.tensor(T_0))

    def forward(self, T_total: torch.Tensor) -> torch.Tensor:
        """
        Ideal gaz yogunluk hesabi: rho = rho_0 * T_0 / T_total

        Args:
            T_total: Toplam nondimensional sicaklik [B, Nx, Ny, Nz]
                     (T_base + T', her zaman > 0 olmali)
        Returns:
            rho: Yogunluk alani [B, Nx, Ny, Nz]
        """
        T_safe = torch.clamp(T_total, min=0.01)
        rho = self.rho_0 * self.T_0 / T_safe
        return torch.clamp(rho, 0.5 * self.rho_0, 2.0 * self.rho_0)


class VariableDensityAdvection3D(nn.Module):
    """
    Yogunluk-moduleli adveksiyon: primitive form + rho kuplaji.

    density_factor = 1 + coupling * (rho / rho_mean - 1)
    adv_modified = adv_standard * density_factor

    Boussinesq limiti: coupling=0 -> density_factor=1 -> standart adveksiyon.
    Learnable: density_coupling (1 param, init=1.0, clamp=[0.5, 2.0])
    SpectralOps: EVET (gradient + dealias)
    """

    def __init__(self, spectral_ops):
        """
        Args:
            spectral_ops: SpectralOps3DAniso instance (paylasimli)
        """
        super().__init__()
        self.ops = spectral_ops
        self.density_coupling = nn.Parameter(torch.ones(1))

    def forward(self, u: torch.Tensor, v: torch.Tensor, w: torch.Tensor,
                rho: torch.Tensor) -> tuple:
        """
        Yogunluk-moduleli adveksiyon: standart (u.nabla)u * density_factor

        Args:
            u, v, w: Hiz bilesenleri [B, Nx, Ny, Nz]
            rho: Yogunluk alani [B, Nx, Ny, Nz]
        Returns:
            (adv_u, adv_v, adv_w): Yogunluk-moduleli adveksiyon terimleri
        """
        # 1. Standart adveksiyon: (u.nabla)u, (u.nabla)v, (u.nabla)w
        du_dx, du_dy, du_dz = self.ops.gradient(u)
        dv_dx, dv_dy, dv_dz = self.ops.gradient(v)
        dw_dx, dw_dy, dw_dz = self.ops.gradient(w)

        adv_u = self.ops.dealias(u * du_dx + v * du_dy + w * du_dz)
        adv_v = self.ops.dealias(u * dv_dx + v * dv_dy + w * dv_dz)
        adv_w = self.ops.dealias(u * dw_dx + v * dw_dy + w * dw_dz)

        # 2. Yogunluk modulasyonu
        coupling = torch.clamp(self.density_coupling, 0.5, 2.0)
        rho_mean = rho.mean(dim=(-3, -2, -1), keepdim=True)
        # rho_mean=0 korunmasi: domain-ortalama yogunluk sifir olamaz
        rho_mean = torch.clamp(rho_mean, min=1e-8)
        density_factor = 1.0 + coupling * (rho / rho_mean - 1.0)

        return adv_u * density_factor, adv_v * density_factor, adv_w * density_factor


class ContinuityNeuron3D(nn.Module):
    """
    DIAGNOSTIC: Kutle korunumu residuali.

    d(rho)/dt + nabla . (rho * u) = 0

    Loss terimi olarak kullanilir, prognostik degil.
    Learnable: YOK
    SpectralOps: EVET (divergence hesabi)
    """

    def __init__(self, spectral_ops, dt: float = 0.005):
        """
        Args:
            spectral_ops: SpectralOps3DAniso instance (paylasimli)
            dt: Zaman adimi (nondim)
        """
        super().__init__()
        self.ops = spectral_ops
        self.register_buffer('dt', torch.tensor(dt))

    def forward(self, rho_old: torch.Tensor, rho_new: torch.Tensor,
                u: torch.Tensor, v: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        Continuity residual: d(rho)/dt + nabla.(rho*u)

        Args:
            rho_old: Onceki zaman adimi yogunlugu [B, Nx, Ny, Nz]
            rho_new: Yeni zaman adimi yogunlugu [B, Nx, Ny, Nz]
            u, v, w: Hiz bilesenleri (rho_new zamaninda) [B, Nx, Ny, Nz]
        Returns:
            residual: Kutle korunumu residuali [B, Nx, Ny, Nz]
        """
        drho_dt = (rho_new - rho_old) / self.dt

        # nabla . (rho * u) = d(rho*u)/dx + d(rho*v)/dy + d(rho*w)/dz
        flux_div = self.ops.divergence(rho_new * u, rho_new * v, rho_new * w)

        return drho_dt + flux_div


class StateEquation3D(nn.Module):
    """
    Durum denklemi dogrulama (diagnostic).

    Ideal gaz: p_total = rho * R_specific * T_total
    EOS residual: |p_total - rho * R * T| / p_0

    Loss terimi olarak kullanilir. Egitim sirasinda EOS tutarliligi zorlanir.
    Learnable: YOK
    SpectralOps: HAYIR
    """

    def __init__(self, R_specific: float = 1.0, p_0: float = 1.0):
        """
        Args:
            R_specific: Nondimensional spesifik gaz sabiti (genelde 1.0)
            p_0: Referans basinc (normalizasyon icin, genelde 1.0)
        """
        super().__init__()
        self.register_buffer('R_specific', torch.tensor(R_specific))
        self.register_buffer('p_0', torch.tensor(p_0))

    def forward(self, rho: torch.Tensor, T_total: torch.Tensor,
                p: torch.Tensor) -> torch.Tensor:
        """
        EOS residuali: |p - rho * R * T| / p_0

        Args:
            rho: Yogunluk [B, Nx, Ny, Nz]
            T_total: Toplam sicaklik [B, Nx, Ny, Nz]
            p: Basinc [B, Nx, Ny, Nz]
        Returns:
            residual: Normalize EOS residuali [B, Nx, Ny, Nz]
        """
        p_eos = rho * self.R_specific * T_total
        return (p - p_eos).abs() / self.p_0


# =============================================================================
# BÖLÜM 7: IMMERSED BOUNDARY METHOD
# =============================================================================

@dataclass
class BoundaryConfig:
    """
    Sınır koşulu konfigürasyonu
    
    Attributes:
        mask: [H, W] veya [Nx, Ny, Nz] - True = katı cisim içi
        wall_velocity: Duvar hızı (u, v) veya (u, v, w)
        boundary_points: Sınır noktaları (Lagrangian markers) - opsiyonel
        geometry_type: 'cavity', 'channel', 'cylinder', 'custom'
    """
    mask: torch.Tensor
    wall_velocity: Tuple[float, ...]
    geometry_type: str = 'custom'
    boundary_points: Optional[torch.Tensor] = None


class ImmersedBoundary(nn.Module):
    """
    Hibrit Immersed Boundary Method - 2D
    
    No-slip sınır koşulları için Direct Forcing yaklaşımı.
    Hem ayrı modül olarak kullanılabilir, hem de nöronlara entegre edilebilir.
    
    Özellikler:
        - Geometri tanımları (cavity, channel, cylinder)
        - Sınır maskesi yönetimi
        - Direct Forcing IBM
        - Nöron entegrasyonu için register/unregister
    
    Kullanım:
        ib = ImmersedBoundary(resolution=64)
        ib.set_cavity_geometry(lid_velocity=1.0)
        
        # Nöronlara kaydet
        model.register_boundary(ib)
        
        # Veya ayrı kullan
        u, v = ib.apply_forcing(u, v, dt)
    """
    
    def __init__(self, resolution: int, domain_size: float = 2 * math.pi,
                 device: Optional[torch.device] = None):
        super().__init__()
        self.resolution = resolution
        self.domain_size = domain_size
        self.device = device if device is not None else DEVICE
        self.dx = domain_size / resolution
        
        # Konfigürasyon (set_*_geometry ile doldurulur)
        self.config: Optional[BoundaryConfig] = None
        
        # Kayıtlı nöronlar
        self.registered_neurons = []
        
        # Grid koordinatları
        x = torch.linspace(0, domain_size, resolution, device=self.device)
        y = torch.linspace(0, domain_size, resolution, device=self.device)
        self.X, self.Y = torch.meshgrid(x, y, indexing='ij')
    
    def set_cavity_geometry(self, lid_velocity: float = 1.0, 
                            wall_thickness: int = 1) -> 'ImmersedBoundary':
        """
        Lid-driven cavity geometrisi oluştur.
        
        Üst duvar lid_velocity ile hareket eder, diğer duvarlar sabit.
        
        Args:
            lid_velocity: Üst duvar hızı (x yönünde)
            wall_thickness: Duvar kalınlığı (grid noktası)
        
        Returns:
            self (zincirleme çağrı için)
        """
        N = self.resolution
        wt = wall_thickness
        
        # Maske: Duvarlar True, akışkan False
        mask = torch.zeros(N, N, dtype=torch.bool, device=self.device)
        
        # Sol duvar
        mask[:wt, :] = True
        # Sağ duvar
        mask[-wt:, :] = True
        # Alt duvar
        mask[:, :wt] = True
        # Üst duvar (lid) - bu da katı ama hareket ediyor
        mask[:, -wt:] = True
        
        # Duvar hızları tensörü (her nokta için farklı olabilir)
        u_wall = torch.zeros(N, N, device=self.device)
        v_wall = torch.zeros(N, N, device=self.device)
        
        # Üst duvarda (lid) x-yönünde hız var
        u_wall[:, -wt:] = lid_velocity
        
        self.config = BoundaryConfig(
            mask=mask,
            wall_velocity=(u_wall, v_wall),
            geometry_type='cavity'
        )
        
        # Kayıtlı nöronları güncelle
        self._update_registered_neurons()
        
        return self
    
    def set_channel_geometry(self, wall_velocity: float = 0.0,
                             wall_thickness: int = 1,
                             moving_wall: str = 'none') -> 'ImmersedBoundary':
        """
        Kanal (Couette/Poiseuille) geometrisi oluştur.
        
        Args:
            wall_velocity: Hareketli duvar hızı
            wall_thickness: Duvar kalınlığı
            moving_wall: 'top', 'bottom', 'both', 'none'
        
        Returns:
            self
        """
        N = self.resolution
        wt = wall_thickness
        
        mask = torch.zeros(N, N, dtype=torch.bool, device=self.device)
        
        # Alt duvar
        mask[:, :wt] = True
        # Üst duvar
        mask[:, -wt:] = True
        
        u_wall = torch.zeros(N, N, device=self.device)
        v_wall = torch.zeros(N, N, device=self.device)
        
        if moving_wall in ['top', 'both']:
            u_wall[:, -wt:] = wall_velocity
        if moving_wall in ['bottom', 'both']:
            u_wall[:, :wt] = -wall_velocity if moving_wall == 'both' else wall_velocity
        
        self.config = BoundaryConfig(
            mask=mask,
            wall_velocity=(u_wall, v_wall),
            geometry_type='channel'
        )
        
        self._update_registered_neurons()
        return self
    
    def set_cylinder_geometry(self, center: Tuple[float, float], 
                              radius: float) -> 'ImmersedBoundary':
        """
        Silindir geometrisi oluştur.
        
        Args:
            center: Silindir merkezi (x, y)
            radius: Silindir yarıçapı
        
        Returns:
            self
        """
        cx, cy = center
        
        # Mesafe hesapla
        dist = torch.sqrt((self.X - cx)**2 + (self.Y - cy)**2)
        
        # Silindir içi maske
        mask = dist <= radius
        
        # Silindir sabit (hız = 0)
        u_wall = torch.zeros_like(self.X)
        v_wall = torch.zeros_like(self.Y)
        
        self.config = BoundaryConfig(
            mask=mask,
            wall_velocity=(u_wall, v_wall),
            geometry_type='cylinder'
        )
        
        self._update_registered_neurons()
        return self
    
    def set_custom_geometry(self, mask: torch.Tensor,
                            u_wall: Optional[torch.Tensor] = None,
                            v_wall: Optional[torch.Tensor] = None) -> 'ImmersedBoundary':
        """
        Özel geometri tanımla.
        
        Args:
            mask: [H, W] bool tensor - True = katı
            u_wall: Duvar u-hızı (opsiyonel, varsayılan 0)
            v_wall: Duvar v-hızı (opsiyonel, varsayılan 0)
        
        Returns:
            self
        """
        if u_wall is None:
            u_wall = torch.zeros(self.resolution, self.resolution, device=self.device)
        if v_wall is None:
            v_wall = torch.zeros(self.resolution, self.resolution, device=self.device)
        
        self.config = BoundaryConfig(
            mask=mask.to(self.device),
            wall_velocity=(u_wall.to(self.device), v_wall.to(self.device)),
            geometry_type='custom'
        )
        
        self._update_registered_neurons()
        return self
    
    def register_neuron(self, neuron: nn.Module) -> None:
        """
        Nörona sınır bilgisi kaydet.
        
        Args:
            neuron: Sınır bilgisini alacak nöron (Projection, Advection, vb.)
        """
        if self.config is None:
            raise ValueError("Önce geometri tanımlanmalı (set_*_geometry)")
        
        # Nörona boundary bilgilerini ekle
        neuron.boundary_mask = self.config.mask
        neuron.wall_velocity = self.config.wall_velocity
        
        if neuron not in self.registered_neurons:
            self.registered_neurons.append(neuron)
    
    def unregister_neuron(self, neuron: nn.Module) -> None:
        """Nörondan sınır bilgisini kaldır."""
        if neuron in self.registered_neurons:
            self.registered_neurons.remove(neuron)
            neuron.boundary_mask = None
            neuron.wall_velocity = None
    
    def _update_registered_neurons(self) -> None:
        """Tüm kayıtlı nöronları güncelle."""
        for neuron in self.registered_neurons:
            self.register_neuron(neuron)
    
    def apply_forcing(self, u: torch.Tensor, v: torch.Tensor, 
                      dt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Direct Forcing IBM uygula - sınırda hızı zorla.
        
        Bu yöntem explicit forcing kullanır:
        f = (u_wall - u) / dt
        u_new = u + dt * f = u_wall (sınırda)
        
        Args:
            u: x-hız bileşeni [B, H, W]
            v: y-hız bileşeni [B, H, W]
            dt: Zaman adımı
        
        Returns:
            (u_new, v_new): Sınır koşulları uygulanmış hızlar
        """
        if self.config is None:
            return u, v
        
        mask = self.config.mask
        u_wall, v_wall = self.config.wall_velocity
        
        # Batch boyutu ekle (mask 2D, u/v 3D)
        if mask.dim() == 2 and u.dim() == 3:
            mask = mask.unsqueeze(0)
            u_wall = u_wall.unsqueeze(0)
            v_wall = v_wall.unsqueeze(0)
        
        # Sınırda duvar hızını zorla
        u_new = torch.where(mask, u_wall.expand_as(u), u)
        v_new = torch.where(mask, v_wall.expand_as(v), v)
        
        return u_new, v_new
    
    def get_fluid_mask(self) -> torch.Tensor:
        """Akışkan bölgesi maskesi (sınırın tersi)."""
        if self.config is None:
            return torch.ones(self.resolution, self.resolution, 
                            dtype=torch.bool, device=self.device)
        return ~self.config.mask
    
    def forward(self, u: torch.Tensor, v: torch.Tensor,
                dt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """nn.Module forward - apply_forcing ile aynı."""
        return self.apply_forcing(u, v, dt)

    def extra_repr(self) -> str:
        geom = self.config.geometry_type if self.config else 'none'
        return f"resolution={self.resolution}, geometry={geom}"


class ImmersedBoundary3D(nn.Module):
    """
    Hibrit Immersed Boundary Method - 3D
    
    3D akışlar için Direct Forcing IBM.
    
    Kullanım:
        ib = ImmersedBoundary3D(resolution=64)
        ib.set_channel_geometry()
        u, v, w = ib.apply_forcing(u, v, w, dt)
    """
    
    def __init__(self, resolution: int, domain_size: float = 2 * math.pi,
                 device: Optional[torch.device] = None):
        super().__init__()
        self.resolution = resolution
        self.domain_size = domain_size
        self.device = device if device is not None else DEVICE
        self.dx = domain_size / resolution
        
        self.config: Optional[BoundaryConfig] = None
        self.registered_neurons = []
        
        # 3D Grid
        x = torch.linspace(0, domain_size, resolution, device=self.device)
        y = torch.linspace(0, domain_size, resolution, device=self.device)
        z = torch.linspace(0, domain_size, resolution, device=self.device)
        self.X, self.Y, self.Z = torch.meshgrid(x, y, z, indexing='ij')
    
    def set_channel_geometry(self, wall_velocity: float = 0.0,
                             wall_thickness: int = 1,
                             direction: str = 'z') -> 'ImmersedBoundary3D':
        """
        3D kanal geometrisi (duvarlar z yönünde).
        
        Args:
            wall_velocity: Hareketli duvar hızı
            wall_thickness: Duvar kalınlığı
            direction: Duvar normal yönü ('x', 'y', 'z')
        
        Returns:
            self
        """
        N = self.resolution
        wt = wall_thickness
        
        mask = torch.zeros(N, N, N, dtype=torch.bool, device=self.device)
        
        if direction == 'z':
            mask[:, :, :wt] = True
            mask[:, :, -wt:] = True
        elif direction == 'y':
            mask[:, :wt, :] = True
            mask[:, -wt:, :] = True
        elif direction == 'x':
            mask[:wt, :, :] = True
            mask[-wt:, :, :] = True
        
        u_wall = torch.zeros(N, N, N, device=self.device)
        v_wall = torch.zeros(N, N, N, device=self.device)
        w_wall = torch.zeros(N, N, N, device=self.device)
        
        # Üst duvarda x-yönünde hız
        if direction == 'z':
            u_wall[:, :, -wt:] = wall_velocity
        
        self.config = BoundaryConfig(
            mask=mask,
            wall_velocity=(u_wall, v_wall, w_wall),
            geometry_type='channel'
        )
        
        self._update_registered_neurons()
        return self
    
    def set_sphere_geometry(self, center: Tuple[float, float, float],
                            radius: float) -> 'ImmersedBoundary3D':
        """
        Küre geometrisi oluştur.
        
        Args:
            center: Küre merkezi (x, y, z)
            radius: Küre yarıçapı
        
        Returns:
            self
        """
        cx, cy, cz = center
        
        dist = torch.sqrt((self.X - cx)**2 + (self.Y - cy)**2 + (self.Z - cz)**2)
        mask = dist <= radius
        
        N = self.resolution
        u_wall = torch.zeros(N, N, N, device=self.device)
        v_wall = torch.zeros(N, N, N, device=self.device)
        w_wall = torch.zeros(N, N, N, device=self.device)
        
        self.config = BoundaryConfig(
            mask=mask,
            wall_velocity=(u_wall, v_wall, w_wall),
            geometry_type='sphere'
        )
        
        self._update_registered_neurons()
        return self
    
    def set_custom_geometry(self, mask: torch.Tensor,
                            u_wall: Optional[torch.Tensor] = None,
                            v_wall: Optional[torch.Tensor] = None,
                            w_wall: Optional[torch.Tensor] = None) -> 'ImmersedBoundary3D':
        """Özel 3D geometri tanımla."""
        N = self.resolution
        if u_wall is None:
            u_wall = torch.zeros(N, N, N, device=self.device)
        if v_wall is None:
            v_wall = torch.zeros(N, N, N, device=self.device)
        if w_wall is None:
            w_wall = torch.zeros(N, N, N, device=self.device)
        
        self.config = BoundaryConfig(
            mask=mask.to(self.device),
            wall_velocity=(u_wall.to(self.device), v_wall.to(self.device), 
                          w_wall.to(self.device)),
            geometry_type='custom'
        )
        
        self._update_registered_neurons()
        return self
    
    def register_neuron(self, neuron: nn.Module) -> None:
        """Nörona sınır bilgisi kaydet."""
        if self.config is None:
            raise ValueError("Önce geometri tanımlanmalı")
        
        neuron.boundary_mask = self.config.mask
        neuron.wall_velocity = self.config.wall_velocity
        
        if neuron not in self.registered_neurons:
            self.registered_neurons.append(neuron)
    
    def unregister_neuron(self, neuron: nn.Module) -> None:
        """Nörondan sınır bilgisini kaldır."""
        if neuron in self.registered_neurons:
            self.registered_neurons.remove(neuron)
            neuron.boundary_mask = None
            neuron.wall_velocity = None
    
    def _update_registered_neurons(self) -> None:
        """Tüm kayıtlı nöronları güncelle."""
        for neuron in self.registered_neurons:
            self.register_neuron(neuron)
    
    def apply_forcing(self, u: torch.Tensor, v: torch.Tensor, w: torch.Tensor,
                      dt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Direct Forcing IBM uygula - 3D.
        
        Args:
            u, v, w: Hız bileşenleri [B, Nx, Ny, Nz]
            dt: Zaman adımı
        
        Returns:
            (u_new, v_new, w_new): Sınır koşulları uygulanmış hızlar
        """
        if self.config is None:
            return u, v, w
        
        mask = self.config.mask
        u_wall, v_wall, w_wall = self.config.wall_velocity
        
        # Batch boyutu ekle
        if mask.dim() == 3 and u.dim() == 4:
            mask = mask.unsqueeze(0)
            u_wall = u_wall.unsqueeze(0)
            v_wall = v_wall.unsqueeze(0)
            w_wall = w_wall.unsqueeze(0)
        
        u_new = torch.where(mask, u_wall.expand_as(u), u)
        v_new = torch.where(mask, v_wall.expand_as(v), v)
        w_new = torch.where(mask, w_wall.expand_as(w), w)
        
        return u_new, v_new, w_new
    
    def get_fluid_mask(self) -> torch.Tensor:
        """Akışkan bölgesi maskesi."""
        if self.config is None:
            return torch.ones(self.resolution, self.resolution, self.resolution,
                            dtype=torch.bool, device=self.device)
        return ~self.config.mask
    
    def forward(self, u: torch.Tensor, v: torch.Tensor, w: torch.Tensor,
                dt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """nn.Module forward."""
        return self.apply_forcing(u, v, w, dt)

    def extra_repr(self) -> str:
        geom = self.config.geometry_type if self.config else 'none'
        return f"resolution={self.resolution}, geometry={geom}"


# =============================================================================
# BÖLÜM 8: KULLANIM ÖRNEĞİ
# =============================================================================

def demo():
    """INNATE kütüphane demo"""
    print("=" * 70)
    print("INNATE: Intrinsic Navier-Stokes Neural Architecture")
    print("        for Temporal Evolution")
    print("=" * 70)
    print("\nPhysics-Native Neural Operators for 2D Incompressible Flow")
    
    # Model oluştur
    device = DEVICE
    print(f"\nDevice: {device}")
    
    model = INNATE(
        resolution=64,
        re_range=(100, 1500),
        bc_type='periodic'
    )
    
    print(f"\nModel parametreleri: {sum(p.numel() for p in model.parameters()):,}")
    
    # Trainer
    trainer = INNATETrainer(model, device)
    
    # Kısa eğitim
    print("\n--- Eğitim (Curriculum) ---")
    trainer.curriculum_train(num_epochs=500, max_steps=50)
    
    # Stabilite testi
    print("\n--- Stabilite Testi ---")
    results = trainer.test_stability(num_steps=200)
    
    print(f"\nSonuçlar:")
    print(f"  Reynolds: {results['reynolds']:.1f} ({results['regime']})")
    print(f"  Enerji sönümü: {results['energy_decay']:.4f}")
    print(f"  Enstrofi oranı: {results['enstrophy_ratio']:.4f}")
    print(f"  Stabil: {results['stable']}")
    print(f"  Türbülans korundu: {results['turbulence_sustained']}")
    
    return model, trainer


if __name__ == "__main__":
    demo()