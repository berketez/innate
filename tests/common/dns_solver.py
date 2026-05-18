"""
Pseudo-Spectral DNS Referans Çözücü

Yüksek çözünürlüklü Direct Numerical Simulation (DNS) çözücü.
INNATE modeli ile karşılaştırma için referans veri üretir.

Özellikler:
    - 2D ve 3D pseudo-spectral yöntem
    - RK4 zaman entegrasyonu
    - 2/3 dealiasing kuralı
    - Spektral downsampling

Kullanım:
    from tests.common.dns_solver import PseudoSpectralDNS
    
    dns = PseudoSpectralDNS(resolution=256, Re=1000)
    result = dns.solve(u0, v0, t_final=2.0)
    
    # INNATE çözünürlüğüne downsample
    u_ref = dns.downsample(result['u'], target_resolution=64)
"""

import numpy as np
from scipy.fft import fft2, ifft2, fftn, ifftn, fftfreq
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import math


@dataclass
class DNSResult:
    """DNS çözüm sonucu."""
    u: np.ndarray  # x-hız bileşeni [Nt, Nx, Ny]
    v: np.ndarray  # y-hız bileşeni [Nt, Nx, Ny]
    p: np.ndarray  # Basınç [Nt, Nx, Ny]
    t: np.ndarray  # Zaman dizisi [Nt]
    energy: np.ndarray  # Kinetik enerji [Nt]
    enstrophy: np.ndarray  # Enstrofi [Nt]


class PseudoSpectralDNS:
    """
    2D Pseudo-Spectral DNS Çözücü
    
    Navier-Stokes denklemlerini yüksek doğrulukla çözer.
    INNATE modeli için referans veri üretir.
    
    Denklemler (velocity-vorticity formülasyonu):
        ∂ω/∂t = -u·∇ω + ν∇²ω
        ∇²ψ = -ω
        u = ∂ψ/∂y, v = -∂ψ/∂x
    
    Args:
        resolution: Grid çözünürlüğü (tipik: 256, 512)
        Re: Reynolds sayısı
        domain_size: Domain boyutu (varsayılan 2π)
    """
    
    def __init__(self, resolution: int, Re: float, domain_size: float = 2 * np.pi):
        self.N = resolution
        self.Re = Re
        self.nu = 1.0 / Re
        self.L = domain_size
        self.dx = domain_size / resolution
        
        # Wavenumber dizileri
        self._setup_wavenumbers()
        
        # Dealiasing maskesi (2/3 kuralı)
        self._setup_dealiasing_mask()
    
    def _setup_wavenumbers(self) -> None:
        """Wavenumber dizilerini oluştur."""
        N = self.N
        L = self.L
        
        # 1D wavenumbers
        k = fftfreq(N, d=1/N) * 2 * np.pi / L
        
        # 2D wavenumber grids
        self.kx, self.ky = np.meshgrid(k, k, indexing='ij')
        
        # k² (Laplacian için)
        self.k_squared = self.kx**2 + self.ky**2
        self.k_squared[0, 0] = 1.0  # Divide by zero önle
        
        # Poisson çözücü için 1/k²
        self.inv_k_squared = 1.0 / self.k_squared
        self.inv_k_squared[0, 0] = 0.0  # DC bileşen
    
    def _setup_dealiasing_mask(self) -> None:
        """2/3 dealiasing maskesi oluştur."""
        N = self.N
        k_max = N // 3
        
        kx_abs = np.abs(self.kx) * self.L / (2 * np.pi)
        ky_abs = np.abs(self.ky) * self.L / (2 * np.pi)
        
        self.dealias_mask = (kx_abs < k_max) & (ky_abs < k_max)
    
    def _dealias(self, field_hat: np.ndarray) -> np.ndarray:
        """Dealiasing uygula."""
        return field_hat * self.dealias_mask
    
    def _compute_velocity_from_vorticity(self, omega_hat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Vortisite'den hız hesapla (stream function ile).
        
        ∇²ψ = -ω → ψ_hat = ω_hat / k²
        u = ∂ψ/∂y → u_hat = i*ky*ψ_hat
        v = -∂ψ/∂x → v_hat = -i*kx*ψ_hat
        """
        psi_hat = omega_hat * self.inv_k_squared
        psi_hat[0, 0] = 0  # Ortalama stream function = 0
        
        u_hat = 1j * self.ky * psi_hat
        v_hat = -1j * self.kx * psi_hat
        
        return u_hat, v_hat
    
    def _compute_rhs(self, omega_hat: np.ndarray) -> np.ndarray:
        """
        Vortisite denkleminin sağ tarafını hesapla.
        
        ∂ω/∂t = -u·∇ω + ν∇²ω
        """
        # Hız alanı
        u_hat, v_hat = self._compute_velocity_from_vorticity(omega_hat)
        u = np.real(ifft2(u_hat))
        v = np.real(ifft2(v_hat))
        
        # Vortisite gradyanları
        domega_dx_hat = 1j * self.kx * omega_hat
        domega_dy_hat = 1j * self.ky * omega_hat
        domega_dx = np.real(ifft2(domega_dx_hat))
        domega_dy = np.real(ifft2(domega_dy_hat))
        
        # Adveksiyon terimi (fiziksel uzayda)
        advection = u * domega_dx + v * domega_dy
        advection_hat = fft2(advection)
        advection_hat = self._dealias(advection_hat)
        
        # Difüzyon terimi (spektral uzayda)
        diffusion_hat = -self.nu * self.k_squared * omega_hat
        
        # Toplam RHS
        rhs_hat = -advection_hat + diffusion_hat
        
        return rhs_hat
    
    def _rk4_step(self, omega_hat: np.ndarray, dt: float) -> np.ndarray:
        """RK4 zaman adımı."""
        k1 = self._compute_rhs(omega_hat)
        k2 = self._compute_rhs(omega_hat + 0.5 * dt * k1)
        k3 = self._compute_rhs(omega_hat + 0.5 * dt * k2)
        k4 = self._compute_rhs(omega_hat + dt * k3)
        
        omega_hat_new = omega_hat + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        return omega_hat_new
    
    def _compute_cfl_dt(self, u: np.ndarray, v: np.ndarray, 
                        cfl: float = 0.5) -> float:
        """CFL koşuluna göre dt hesapla."""
        u_max = np.max(np.abs(u))
        v_max = np.max(np.abs(v))
        vel_max = max(u_max, v_max, 1e-10)
        
        # Advektif CFL
        dt_adv = cfl * self.dx / vel_max
        
        # Difüzif CFL
        dt_diff = cfl * self.dx**2 / (4 * self.nu)
        
        return min(dt_adv, dt_diff)
    
    def solve(self, u0: np.ndarray, v0: np.ndarray, 
              t_final: float, 
              dt: Optional[float] = None,
              save_every: int = 10,
              cfl: float = 0.5) -> DNSResult:
        """
        DNS çözümü yap.
        
        Args:
            u0: Başlangıç x-hız alanı [Nx, Ny]
            v0: Başlangıç y-hız alanı [Nx, Ny]
            t_final: Bitiş zamanı
            dt: Zaman adımı (None ise CFL'den hesaplanır)
            save_every: Her kaç adımda bir kaydet
            cfl: CFL sayısı
        
        Returns:
            DNSResult: Çözüm sonuçları
        """
        # Başlangıç vortisitesi
        u0_hat = fft2(u0)
        v0_hat = fft2(v0)
        omega0 = np.real(ifft2(1j * self.kx * v0_hat - 1j * self.ky * u0_hat))
        omega_hat = fft2(omega0)
        
        # Zaman adımı
        if dt is None:
            dt = self._compute_cfl_dt(u0, v0, cfl)
        
        n_steps = int(t_final / dt)
        
        # Sonuç dizileri
        n_saves = n_steps // save_every + 1
        u_history = np.zeros((n_saves, self.N, self.N))
        v_history = np.zeros((n_saves, self.N, self.N))
        p_history = np.zeros((n_saves, self.N, self.N))
        t_history = np.zeros(n_saves)
        energy_history = np.zeros(n_saves)
        enstrophy_history = np.zeros(n_saves)
        
        # Başlangıç durumu kaydet
        u_hat, v_hat = self._compute_velocity_from_vorticity(omega_hat)
        u = np.real(ifft2(u_hat))
        v = np.real(ifft2(v_hat))
        
        u_history[0] = u
        v_history[0] = v
        t_history[0] = 0.0
        energy_history[0] = 0.5 * np.mean(u**2 + v**2)
        enstrophy_history[0] = 0.5 * np.mean(omega0**2)
        
        save_idx = 1
        t = 0.0
        
        # Zaman ilerlemesi
        for step in range(n_steps):
            # Adaptif dt
            if dt is None:
                u_hat, v_hat = self._compute_velocity_from_vorticity(omega_hat)
                u = np.real(ifft2(u_hat))
                v = np.real(ifft2(v_hat))
                current_dt = self._compute_cfl_dt(u, v, cfl)
            else:
                current_dt = dt
            
            # RK4 adımı
            omega_hat = self._rk4_step(omega_hat, current_dt)
            t += current_dt
            
            # Kaydet
            if (step + 1) % save_every == 0 and save_idx < n_saves:
                u_hat, v_hat = self._compute_velocity_from_vorticity(omega_hat)
                u = np.real(ifft2(u_hat))
                v = np.real(ifft2(v_hat))
                omega = np.real(ifft2(omega_hat))
                
                # Basınç (Poisson'dan)
                # Basınç Poisson: ∇²p = -∇·(u·∇u)
                # Basit yaklaşım: p ≈ -0.5 * (u² + v²) + sabit
                p = -0.5 * (u**2 + v**2)
                p -= np.mean(p)
                
                u_history[save_idx] = u
                v_history[save_idx] = v
                p_history[save_idx] = p
                t_history[save_idx] = t
                energy_history[save_idx] = 0.5 * np.mean(u**2 + v**2)
                enstrophy_history[save_idx] = 0.5 * np.mean(omega**2)
                
                save_idx += 1
        
        return DNSResult(
            u=u_history[:save_idx],
            v=v_history[:save_idx],
            p=p_history[:save_idx],
            t=t_history[:save_idx],
            energy=energy_history[:save_idx],
            enstrophy=enstrophy_history[:save_idx]
        )
    
    def downsample(self, field: np.ndarray, target_resolution: int) -> np.ndarray:
        """
        Spektral downsampling.
        
        Args:
            field: Yüksek çözünürlüklü alan [Nt, Nx, Ny] veya [Nx, Ny]
            target_resolution: Hedef çözünürlük
        
        Returns:
            Düşük çözünürlüklü alan
        """
        if field.ndim == 2:
            return self._downsample_2d(field, target_resolution)
        elif field.ndim == 3:
            result = np.zeros((field.shape[0], target_resolution, target_resolution))
            for i in range(field.shape[0]):
                result[i] = self._downsample_2d(field[i], target_resolution)
            return result
        else:
            raise ValueError(f"Beklenmeyen boyut: {field.ndim}")
    
    def _downsample_2d(self, field: np.ndarray, target_resolution: int) -> np.ndarray:
        """2D spektral downsampling."""
        N_high = field.shape[0]
        N_low = target_resolution
        
        # FFT
        field_hat = fft2(field)
        
        # Düşük modları al
        half_low = N_low // 2
        
        # Yeni spektral array
        field_low_hat = np.zeros((N_low, N_low), dtype=complex)
        
        # Düşük frekans bileşenlerini kopyala
        field_low_hat[:half_low, :half_low] = field_hat[:half_low, :half_low]
        field_low_hat[:half_low, -half_low:] = field_hat[:half_low, -half_low:]
        field_low_hat[-half_low:, :half_low] = field_hat[-half_low:, :half_low]
        field_low_hat[-half_low:, -half_low:] = field_hat[-half_low:, -half_low:]
        
        # Normalizasyon
        field_low_hat *= (N_low / N_high) ** 2
        
        # IFFT
        field_low = np.real(ifft2(field_low_hat))
        
        return field_low


class PseudoSpectralDNS3D:
    """
    3D Pseudo-Spectral DNS Çözücü
    
    3D Navier-Stokes denklemlerini çözer.
    
    Args:
        resolution: Grid çözünürlüğü (tipik: 64, 128)
        Re: Reynolds sayısı
        domain_size: Domain boyutu (varsayılan 2π)
    """
    
    def __init__(self, resolution: int, Re: float, domain_size: float = 2 * np.pi):
        self.N = resolution
        self.Re = Re
        self.nu = 1.0 / Re
        self.L = domain_size
        self.dx = domain_size / resolution
        
        self._setup_wavenumbers()
        self._setup_dealiasing_mask()
    
    def _setup_wavenumbers(self) -> None:
        """3D wavenumber dizileri."""
        N = self.N
        L = self.L
        
        k = fftfreq(N, d=1/N) * 2 * np.pi / L
        self.kx, self.ky, self.kz = np.meshgrid(k, k, k, indexing='ij')
        
        self.k_squared = self.kx**2 + self.ky**2 + self.kz**2
        self.k_squared[0, 0, 0] = 1.0
        
        self.inv_k_squared = 1.0 / self.k_squared
        self.inv_k_squared[0, 0, 0] = 0.0
    
    def _setup_dealiasing_mask(self) -> None:
        """3D dealiasing maskesi."""
        N = self.N
        k_max = N // 3
        
        kx_abs = np.abs(self.kx) * self.L / (2 * np.pi)
        ky_abs = np.abs(self.ky) * self.L / (2 * np.pi)
        kz_abs = np.abs(self.kz) * self.L / (2 * np.pi)
        
        self.dealias_mask = (kx_abs < k_max) & (ky_abs < k_max) & (kz_abs < k_max)
    
    def _dealias(self, field_hat: np.ndarray) -> np.ndarray:
        """Dealiasing uygula."""
        return field_hat * self.dealias_mask
    
    def _project_divergence_free(self, u_hat: np.ndarray, v_hat: np.ndarray, 
                                  w_hat: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Hız alanını divergence-free yap (Helmholtz projeksiyon)."""
        # Divergence
        div_hat = 1j * (self.kx * u_hat + self.ky * v_hat + self.kz * w_hat)
        
        # Basınç (Poisson)
        p_hat = div_hat * self.inv_k_squared
        p_hat[0, 0, 0] = 0
        
        # Projeksiyon
        u_hat_proj = u_hat - 1j * self.kx * p_hat
        v_hat_proj = v_hat - 1j * self.ky * p_hat
        w_hat_proj = w_hat - 1j * self.kz * p_hat
        
        return u_hat_proj, v_hat_proj, w_hat_proj
    
    def _compute_rhs(self, u_hat: np.ndarray, v_hat: np.ndarray, 
                     w_hat: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """3D momentum denkleminin sağ tarafı."""
        # Fiziksel uzay
        u = np.real(ifftn(u_hat))
        v = np.real(ifftn(v_hat))
        w = np.real(ifftn(w_hat))
        
        # Gradyanlar
        du_dx = np.real(ifftn(1j * self.kx * u_hat))
        du_dy = np.real(ifftn(1j * self.ky * u_hat))
        du_dz = np.real(ifftn(1j * self.kz * u_hat))
        
        dv_dx = np.real(ifftn(1j * self.kx * v_hat))
        dv_dy = np.real(ifftn(1j * self.ky * v_hat))
        dv_dz = np.real(ifftn(1j * self.kz * v_hat))
        
        dw_dx = np.real(ifftn(1j * self.kx * w_hat))
        dw_dy = np.real(ifftn(1j * self.ky * w_hat))
        dw_dz = np.real(ifftn(1j * self.kz * w_hat))
        
        # Adveksiyon
        adv_u = u * du_dx + v * du_dy + w * du_dz
        adv_v = u * dv_dx + v * dv_dy + w * dv_dz
        adv_w = u * dw_dx + v * dw_dy + w * dw_dz
        
        adv_u_hat = self._dealias(fftn(adv_u))
        adv_v_hat = self._dealias(fftn(adv_v))
        adv_w_hat = self._dealias(fftn(adv_w))
        
        # Difüzyon
        diff_u_hat = -self.nu * self.k_squared * u_hat
        diff_v_hat = -self.nu * self.k_squared * v_hat
        diff_w_hat = -self.nu * self.k_squared * w_hat
        
        # Toplam RHS
        rhs_u_hat = -adv_u_hat + diff_u_hat
        rhs_v_hat = -adv_v_hat + diff_v_hat
        rhs_w_hat = -adv_w_hat + diff_w_hat
        
        return rhs_u_hat, rhs_v_hat, rhs_w_hat
    
    def _rk4_step(self, u_hat: np.ndarray, v_hat: np.ndarray, w_hat: np.ndarray,
                  dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """3D RK4 adımı."""
        # k1
        k1_u, k1_v, k1_w = self._compute_rhs(u_hat, v_hat, w_hat)
        
        # k2
        k2_u, k2_v, k2_w = self._compute_rhs(
            u_hat + 0.5*dt*k1_u, v_hat + 0.5*dt*k1_v, w_hat + 0.5*dt*k1_w
        )
        
        # k3
        k3_u, k3_v, k3_w = self._compute_rhs(
            u_hat + 0.5*dt*k2_u, v_hat + 0.5*dt*k2_v, w_hat + 0.5*dt*k2_w
        )
        
        # k4
        k4_u, k4_v, k4_w = self._compute_rhs(
            u_hat + dt*k3_u, v_hat + dt*k3_v, w_hat + dt*k3_w
        )
        
        # Güncelleme
        u_hat_new = u_hat + (dt/6) * (k1_u + 2*k2_u + 2*k3_u + k4_u)
        v_hat_new = v_hat + (dt/6) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
        w_hat_new = w_hat + (dt/6) * (k1_w + 2*k2_w + 2*k3_w + k4_w)
        
        # Projeksiyon
        u_hat_new, v_hat_new, w_hat_new = self._project_divergence_free(
            u_hat_new, v_hat_new, w_hat_new
        )
        
        return u_hat_new, v_hat_new, w_hat_new
    
    def solve(self, u0: np.ndarray, v0: np.ndarray, w0: np.ndarray,
              t_final: float, dt: float = 0.01,
              save_every: int = 10) -> Dict:
        """
        3D DNS çözümü.
        
        Args:
            u0, v0, w0: Başlangıç hız alanları [Nx, Ny, Nz]
            t_final: Bitiş zamanı
            dt: Zaman adımı
            save_every: Kayıt sıklığı
        
        Returns:
            Dict: Çözüm sonuçları
        """
        u_hat = fftn(u0)
        v_hat = fftn(v0)
        w_hat = fftn(w0)
        
        # Divergence-free yap
        u_hat, v_hat, w_hat = self._project_divergence_free(u_hat, v_hat, w_hat)
        
        n_steps = int(t_final / dt)
        n_saves = n_steps // save_every + 1
        
        # Sonuç dizileri
        results = {
            'u': np.zeros((n_saves, self.N, self.N, self.N)),
            'v': np.zeros((n_saves, self.N, self.N, self.N)),
            'w': np.zeros((n_saves, self.N, self.N, self.N)),
            't': np.zeros(n_saves),
            'energy': np.zeros(n_saves),
        }
        
        # Başlangıç
        u = np.real(ifftn(u_hat))
        v = np.real(ifftn(v_hat))
        w = np.real(ifftn(w_hat))
        
        results['u'][0] = u
        results['v'][0] = v
        results['w'][0] = w
        results['t'][0] = 0.0
        results['energy'][0] = 0.5 * np.mean(u**2 + v**2 + w**2)
        
        save_idx = 1
        t = 0.0
        
        for step in range(n_steps):
            u_hat, v_hat, w_hat = self._rk4_step(u_hat, v_hat, w_hat, dt)
            t += dt
            
            if (step + 1) % save_every == 0 and save_idx < n_saves:
                u = np.real(ifftn(u_hat))
                v = np.real(ifftn(v_hat))
                w = np.real(ifftn(w_hat))
                
                results['u'][save_idx] = u
                results['v'][save_idx] = v
                results['w'][save_idx] = w
                results['t'][save_idx] = t
                results['energy'][save_idx] = 0.5 * np.mean(u**2 + v**2 + w**2)
                
                save_idx += 1
        
        # Son boyutu kırp
        for key in results:
            results[key] = results[key][:save_idx]
        
        return results
    
    def downsample(self, field: np.ndarray, target_resolution: int) -> np.ndarray:
        """3D spektral downsampling."""
        if field.ndim == 3:
            return self._downsample_3d(field, target_resolution)
        elif field.ndim == 4:
            result = np.zeros((field.shape[0], target_resolution, 
                              target_resolution, target_resolution))
            for i in range(field.shape[0]):
                result[i] = self._downsample_3d(field[i], target_resolution)
            return result
        else:
            raise ValueError(f"Beklenmeyen boyut: {field.ndim}")
    
    def _downsample_3d(self, field: np.ndarray, target_resolution: int) -> np.ndarray:
        """3D spektral downsampling."""
        N_high = field.shape[0]
        N_low = target_resolution
        half_low = N_low // 2
        
        field_hat = fftn(field)
        field_low_hat = np.zeros((N_low, N_low, N_low), dtype=complex)
        
        # Düşük frekans bileşenlerini kopyala (8 köşe)
        field_low_hat[:half_low, :half_low, :half_low] = field_hat[:half_low, :half_low, :half_low]
        field_low_hat[:half_low, :half_low, -half_low:] = field_hat[:half_low, :half_low, -half_low:]
        field_low_hat[:half_low, -half_low:, :half_low] = field_hat[:half_low, -half_low:, :half_low]
        field_low_hat[:half_low, -half_low:, -half_low:] = field_hat[:half_low, -half_low:, -half_low:]
        field_low_hat[-half_low:, :half_low, :half_low] = field_hat[-half_low:, :half_low, :half_low]
        field_low_hat[-half_low:, :half_low, -half_low:] = field_hat[-half_low:, :half_low, -half_low:]
        field_low_hat[-half_low:, -half_low:, :half_low] = field_hat[-half_low:, -half_low:, :half_low]
        field_low_hat[-half_low:, -half_low:, -half_low:] = field_hat[-half_low:, -half_low:, -half_low:]
        
        field_low_hat *= (N_low / N_high) ** 3
        
        return np.real(ifftn(field_low_hat))


# Analitik çözümler (validasyon için)
def taylor_green_analytical(x: np.ndarray, y: np.ndarray, t: float, 
                            Re: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Taylor-Green vortex analitik çözümü (2D).
    
    Args:
        x, y: Koordinat gridleri
        t: Zaman
        Re: Reynolds sayısı
    
    Returns:
        (u, v, omega): Hız ve vortisite alanları
    """
    nu = 1.0 / Re
    decay = np.exp(-2 * nu * t)
    
    u = np.sin(x) * np.cos(y) * decay
    v = -np.cos(x) * np.sin(y) * decay
    omega = -2 * np.sin(x) * np.sin(y) * decay
    
    return u, v, omega


def lamb_oseen_vortex(x: np.ndarray, y: np.ndarray, t: float,
                      Re: float, Gamma: float = 1.0,
                      x0: float = np.pi, y0: float = np.pi) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Lamb-Oseen vortex analitik çözümü.
    
    Args:
        x, y: Koordinat gridleri
        t: Zaman
        Re: Reynolds sayısı
        Gamma: Sirkülasyon
        x0, y0: Vortex merkezi
    
    Returns:
        (u, v, omega): Hız ve vortisite alanları
    """
    nu = 1.0 / Re
    r2 = (x - x0)**2 + (y - y0)**2
    r = np.sqrt(r2 + 1e-10)
    
    # Core radius
    r_c = np.sqrt(4 * nu * (t + 0.1))  # t=0'da singularite önle
    
    # Vortisite
    omega = (Gamma / (np.pi * r_c**2)) * np.exp(-r2 / r_c**2)
    
    # Hız (azimuthal)
    v_theta = (Gamma / (2 * np.pi * r)) * (1 - np.exp(-r2 / r_c**2))
    
    # Kartezyen bileşenler
    theta = np.arctan2(y - y0, x - x0)
    u = -v_theta * np.sin(theta)
    v = v_theta * np.cos(theta)
    
    return u, v, omega


if __name__ == "__main__":
    # Basit test
    print("DNS Solver Test")
    
    N = 128
    Re = 1000
    L = 2 * np.pi
    
    dns = PseudoSpectralDNS(N, Re, L)
    
    # Taylor-Green başlangıç
    x = np.linspace(0, L, N, endpoint=False)
    y = np.linspace(0, L, N, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    u0, v0, _ = taylor_green_analytical(X, Y, 0, Re)
    
    print(f"Çözünürlük: {N}x{N}")
    print(f"Reynolds: {Re}")
    print(f"Çözülüyor...")
    
    result = dns.solve(u0, v0, t_final=1.0, save_every=100)
    
    print(f"Tamamlandı!")
    print(f"Kaydedilen adım: {len(result.t)}")
    print(f"Enerji decay: {result.energy[-1] / result.energy[0]:.4f}")
    
    # Downsampling test
    u_low = dns.downsample(result.u, 64)
    print(f"Downsampled shape: {u_low.shape}")

