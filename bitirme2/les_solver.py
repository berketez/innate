#!/usr/bin/env python3
"""
LES Validation Solver -- Bagimsiz Large Eddy Simulation cozucusu.

INNATE 3D modelinin urettigi sonuclari dogrulamak icin "ground truth" metrikleri
uretir. INNATE koduna bagimli DEGIL; sadece spectral operatorleri (safe_fftn/ifftn)
icin innate.py'den import yapar.

Yontem:
    - Pseudospectral (Fourier) -- anizotropik grid destegi
    - RK4 zaman entegrasyonu
    - Smagorinsky LES: nu_t = (Cs * Delta)^2 * |S|
    - Adaptive dt (CFL <= 0.5)
    - Spectral Leray projection (div-free garanti)
    - Boussinesq yaklasimi (sicaklik -> kaldirma kuvveti)
    - Kolmogorov forcing: F_x = A * sin(k_f * 2*pi*y / Ly)
    - Periodic BC (tum yonler)
    - 2/3 dealiasing kurali

Fizik:
    Momentum:  du/dt = -u.nabla(u) + (nu + nu_t)*lap(u) - nabla(p) + Ri*theta*e_y + F
    Enerji:    dtheta/dt = -u.nabla(theta) + (kappa + kappa_t)*lap(theta) + v*(dT/Ly)

    nu      = 1/Re = 2e-4
    kappa   = 1/(Re*Pr)
    Ri      = Ra/(Re^2 * Pr) = 0.0563
    nu_t    = (Cs*Delta)^2 * |S|        (Smagorinsky)
    kappa_t = nu_t / Pr_t               (turbulansi Prandtl = 0.85)

Precision: float32 (LES icin yeterli).
Device:    CUDA > MPS > CPU (otomatik secim).

Kullanim:
    python bitirme2/les_solver.py
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

# ── innate.py'den device-agnostik FFT import ─────────────────────────────
# safe_fftn/safe_ifftn: CUDA direkt, MPS fallback CPU, CPU direkt.
# Geri kalan tum operatorler bu dosyada tanimli.
sys.path.insert(0, "/Users/apple/Desktop/nsneuron")
from innate import safe_fftn, safe_ifftn


# ══════════════════════════════════════════════════════════════════════════
# YARDIMCI FONKSIYONLAR
# ══════════════════════════════════════════════════════════════════════════

def get_device() -> torch.device:
    """CUDA > MPS > CPU otomatik secim."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ══════════════════════════════════════════════════════════════════════════
# ANA SOLVER SINIFI
# ══════════════════════════════════════════════════════════════════════════

class LESSolver:
    """
    Pseudospectral LES cozucusu -- 3D mixed convection (Boussinesq).

    Grid: Nx x Ny x Nz (varsayilan: 96 x 160 x 64)
    Domain: Lx x Ly x Lz (varsayilan: 6 x 10 x 4)

    Smagorinsky SGS modeli ile kapatma.
    RK4 zaman entegrasyonu, her adimda spectral Leray projeksiyonu.
    Adaptive CFL kontrol ile dt ayarlamasi.
    """

    def __init__(
        self,
        # -- Grid --
        Nx: int = 96,
        Ny: int = 160,
        Nz: int = 64,
        Lx: float = 6.0,
        Ly: float = 10.0,
        Lz: float = 4.0,
        # -- Fizik --
        Re: float = 5000.0,
        Ra: float = 1e6,
        Pr: float = 0.71,
        # -- Forcing --
        forcing_amplitude: float = 0.005,
        k_f: int = 4,
        # -- Termal --
        T_hot: float = 20.0,
        T_cold: float = 0.0,
        # -- LES --
        Cs: float = 0.17,           # Smagorinsky sabiti
        Pr_t: float = 0.85,         # Turbulansi Prandtl sayisi
        # -- Numerik --
        cfl_target: float = 0.5,    # Hedef CFL sayisi
        dt_max: float = 0.02,       # Maksimum dt (CFL'den bagimsiz ust sinir)
        dt_min: float = 1e-5,       # Minimum dt (CFL cok kuculdugunde)
        # -- Buoyancy instabilite kontrolu --
        buoyancy_damping: bool = True,   # Anizotropik sigma-profile damping
        damping_safety: float = 3.5,     # Safety factor (>1.0, Re=5000 icin 3.5 gerekli)
        # -- Device --
        device: Optional[str] = None,
    ):
        # ── Device ──
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = get_device()

        # ── Grid parametreleri ──
        self.Nx, self.Ny, self.Nz = Nx, Ny, Nz
        self.Lx, self.Ly, self.Lz = Lx, Ly, Lz

        # Grid spacing
        self.dx = Lx / Nx
        self.dy = Ly / Ny
        self.dz = Lz / Nz
        self.dx_min = min(self.dx, self.dy, self.dz)

        # ── Fiziksel parametreler ──
        self.Re = Re
        self.Ra = Ra
        self.Pr = Pr
        self.nu = 1.0 / Re                     # Kinematik viskozite
        self.kappa = 1.0 / (Re * Pr)           # Termal difuzivite
        self.Ri = Ra / (Re**2 * Pr)            # Richardson sayisi

        # ── Termal ──
        self.dT = T_hot - T_cold
        # Boyutsuz formulasyonda baz profil gradyani = 1/Ly (dT kullanilmaz)
        self.dT_over_Ly = 1.0 / Ly             # = 0.1 (boyutsuz)

        # ── Forcing ──
        self.forcing_amplitude = forcing_amplitude
        self.k_f = k_f

        # ── LES parametreleri ──
        self.Cs = Cs
        self.Pr_t = Pr_t
        # LES filtre genisligi: geometrik ortalama (anizotropik grid icin standart)
        self.Delta = (self.dx * self.dy * self.dz) ** (1.0 / 3.0)

        # ── CFL kontrol ──
        self.cfl_target = cfl_target
        self.dt_max = dt_max
        self.dt_min = dt_min
        self.dt = dt_max  # Baslangic dt, ilk adimda CFL'e gore ayarlanir

        # ── Buoyancy instabilite kontrolu ──
        self.buoyancy_damping = buoyancy_damping
        self.damping_safety = damping_safety

        # ── Dalga sayisi dizileri (float32) ──
        self._setup_wavenumbers()

        # ── Fiziksel uzay y-grid (forcing icin) ──
        # shape: [1, 1, Ny, 1] -> broadcasting icin
        y_1d = torch.linspace(0, Ly, Ny + 1, dtype=torch.float32)[:-1]
        self.y_grid = y_1d.view(1, 1, Ny, 1).to(self.device)

        # ── Ozet ──
        print(
            f"LES Solver baslatildi:\n"
            f"  Grid:    {Nx} x {Ny} x {Nz}\n"
            f"  Domain:  {Lx} x {Ly} x {Lz}\n"
            f"  Re={Re}, Ra={Ra:.1e}, Pr={Pr}\n"
            f"  nu={self.nu:.2e}, kappa={self.kappa:.2e}, Ri={self.Ri:.4f}\n"
            f"  Smagorinsky: Cs={Cs}, Delta={self.Delta:.4f}, Pr_t={Pr_t}\n"
            f"  CFL hedef={cfl_target}, dt_max={dt_max}\n"
            f"  Forcing: Kolmogorov k_f={k_f}, A={forcing_amplitude}\n"
            f"  Buoyancy damping: {buoyancy_damping}, safety={damping_safety}\n"
            f"  Device:  {self.device}\n"
        )

    # ------------------------------------------------------------------
    # Dalga Sayisi Altyapisi
    # ------------------------------------------------------------------

    def _setup_wavenumbers(self):
        """
        Anizotropik dalga sayisi dizileri olustur.

        Her yon icin farkli grid spacing: kx, ky, kz
        Nyquist modlari sifirlanir (spectral ambiguity onleme).
        2/3 dealiasing maskesi mode-index bazli.
        """
        Nx, Ny, Nz = self.Nx, self.Ny, self.Nz
        Lx, Ly, Lz = self.Lx, self.Ly, self.Lz

        # 1D dalga sayilari
        kx_1d = torch.fft.fftfreq(Nx, d=Lx / Nx).to(torch.float32) * 2 * math.pi
        ky_1d = torch.fft.fftfreq(Ny, d=Ly / Ny).to(torch.float32) * 2 * math.pi
        kz_1d = torch.fft.fftfreq(Nz, d=Lz / Nz).to(torch.float32) * 2 * math.pi

        # Nyquist modlarini sifirla
        # N/2 modu belirsiz (+N/2 veya -N/2), anti-simetri k[-m] = -k[m]'i bozar
        if Nx % 2 == 0:
            kx_1d[Nx // 2] = 0.0
        if Ny % 2 == 0:
            ky_1d[Ny // 2] = 0.0
        if Nz % 2 == 0:
            kz_1d[Nz // 2] = 0.0

        # 3D meshgrid
        kx, ky, kz = torch.meshgrid(kx_1d, ky_1d, kz_1d, indexing='ij')

        self.kx = kx.to(self.device)
        self.ky = ky.to(self.device)
        self.kz = kz.to(self.device)

        # k^2 -- Laplacian icin
        self.k_squared = (kx**2 + ky**2 + kz**2).to(self.device)

        # k^2 Poisson icin: sifir olan yerleri 1.0 yap (bolme hatasi onleme)
        k_sq_poisson = self.k_squared.clone()
        k_sq_poisson[k_sq_poisson == 0.0] = 1.0
        self.k_squared_poisson = k_sq_poisson

        # k buyuklugu (enerji spektrumu icin)
        self.k_mag = torch.sqrt(self.k_squared).to(self.device)

        # --- Dealias maskesi: mode-index bazli 2/3 kurali ---
        # Mode indexleri (0, 1, ..., N/2, -N/2+1, ..., -1) -> gercek frekanslar
        mx = torch.fft.fftfreq(Nx, d=1.0) * Nx
        my = torch.fft.fftfreq(Ny, d=1.0) * Ny
        mz = torch.fft.fftfreq(Nz, d=1.0) * Nz
        Mx, My, Mz = torch.meshgrid(mx, my, mz, indexing='ij')

        dealias_mask = (
            (torch.abs(Mx) < Nx // 3) &
            (torch.abs(My) < Ny // 3) &
            (torch.abs(Mz) < Nz // 3)
        )
        self.dealias_mask = dealias_mask.to(torch.float32).to(self.device)

        # ── Anizotropik buoyancy damping profili ──
        # Leray projection geometrik faktor: f = (kx^2 + kz^2) / k^2
        # Buoyancy sadece y-momentuma etki eder, projection bunu kx/kz'ye dagitir.
        # ky-only modlar (kx=kz=0) icin f=0: buoyancy coupling SIFIR, stabil.
        # ky=0 modlar icin f=1: max instabilite (elevator removal zaten kaldiriyor).
        #
        # Anizotropik dispersion: sigma^2 + (nu+kappa)*k^2*sigma + nu*kappa*k^4 - Ri*(dT/Ly)*f = 0
        # sigma = -0.5*(nu+kappa)*k^2 + sqrt(0.25*(nu-kappa)^2*k^4 + Ri*(dT/Ly)*f)
        #
        # gamma_damp = safety * max(0, sigma)  -- her moda kendi growth rate'i kadar damping
        if self.buoyancy_damping and self.Ri > 0:
            kx2 = self.kx ** 2
            kz2 = self.kz ** 2
            k_sq = self.k_squared
            f_aniso = (kx2 + kz2) / (k_sq + 1e-12)
            # k=0 modu icin f=0 yap (mean mode zaten mean-removal ile kontrol ediliyor)
            f_aniso[0, 0, 0] = 0.0

            disc = 0.25 * (self.nu - self.kappa) ** 2 * k_sq ** 2 \
                 + self.Ri * self.dT_over_Ly * f_aniso
            sigma_profile = -0.5 * (self.nu + self.kappa) * k_sq + torch.sqrt(disc)
            self.gamma_damp = (self.damping_safety * torch.clamp(sigma_profile, min=0.0)).to(self.device)

            n_unstable = int((sigma_profile > 0).sum().item())
            sigma_max = float(sigma_profile.max().item())
            gamma_max = float(self.gamma_damp.max().item())
            print(f"  Buoyancy damping: {n_unstable} unstable mod, "
                  f"sigma_max={sigma_max:.5f}, gamma_max={gamma_max:.5f}")
        else:
            self.gamma_damp = torch.zeros_like(self.k_squared)

    # ------------------------------------------------------------------
    # Spectral Operatorler
    # ------------------------------------------------------------------

    def _fftn(self, x: torch.Tensor) -> torch.Tensor:
        """3D FFT (MPS-uyumlu, safe_fftn kullanir)."""
        return safe_fftn(x, dim=(-3, -2, -1))

    def _ifftn(self, x: torch.Tensor) -> torch.Tensor:
        """3D inverse FFT (MPS-uyumlu)."""
        return safe_ifftn(x, dim=(-3, -2, -1))

    def _dealias(self, f: torch.Tensor) -> torch.Tensor:
        """2/3 kurali ile dealiasing -- nonlinear terimler icin kritik."""
        return self._ifftn(self._fftn(f) * self.dealias_mask).real

    def _gradient(self, f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Spectral gradient: df/dx, df/dy, df/dz

        Fourier uzayinda turev: F[df/dx] = i*kx * F[f]
        Tek FFT, uc IFFT.
        """
        f_hat = self._fftn(f)
        df_dx = self._ifftn(1j * self.kx * f_hat).real
        df_dy = self._ifftn(1j * self.ky * f_hat).real
        df_dz = self._ifftn(1j * self.kz * f_hat).real
        return df_dx, df_dy, df_dz

    def _laplacian(self, f: torch.Tensor) -> torch.Tensor:
        """
        Spectral Laplacian: nabla^2 f = -(kx^2 + ky^2 + kz^2) * F[f]
        """
        return self._ifftn(-self.k_squared * self._fftn(f)).real

    def _divergence(self, u: torch.Tensor, v: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        Spectral divergence: nabla . (u, v, w)

        Tek seferde: i*(kx*u_hat + ky*v_hat + kz*w_hat)
        """
        u_hat = self._fftn(u)
        v_hat = self._fftn(v)
        w_hat = self._fftn(w)
        return self._ifftn(
            1j * (self.kx * u_hat + self.ky * v_hat + self.kz * w_hat)
        ).real

    def _curl(self, u: torch.Tensor, v: torch.Tensor, w: torch.Tensor
              ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Spectral curl (vortisite): omega = nabla x (u, v, w)

        omega_x = dw/dy - dv/dz
        omega_y = du/dz - dw/dx
        omega_z = dv/dx - du/dy
        """
        u_hat = self._fftn(u)
        v_hat = self._fftn(v)
        w_hat = self._fftn(w)
        ox = self._ifftn(1j * (self.ky * w_hat - self.kz * v_hat)).real
        oy = self._ifftn(1j * (self.kz * u_hat - self.kx * w_hat)).real
        oz = self._ifftn(1j * (self.kx * v_hat - self.ky * u_hat)).real
        return ox, oy, oz

    # ------------------------------------------------------------------
    # Pressure Projection (Spectral Leray Projector)
    # ------------------------------------------------------------------

    def _project(self, u: torch.Tensor, v: torch.Tensor, w: torch.Tensor
                 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Spectral Leray projeksiyonu: hiz alanini div-free yapaya projekte et.

        Tum islem Fourier uzayinda yapilir (Hermitian simetri korunur).

        Leray projector:
            u_hat_i -= k_i * (k_j * u_hat_j) / |k|^2

        Basinc:
            p_hat = -i * (k . u_hat) / |k|^2

        Returns: (u_proj, v_proj, w_proj, pressure)
        """
        u_hat = self._fftn(u)
        v_hat = self._fftn(v)
        w_hat = self._fftn(w)

        # k . u_hat (Fourier uzayinda divergence, 1j carpimsiz)
        k_dot_u = self.kx * u_hat + self.ky * v_hat + self.kz * w_hat

        # Leray projeksiyonu
        factor = k_dot_u / self.k_squared_poisson
        u_proj_hat = u_hat - self.kx * factor
        v_proj_hat = v_hat - self.ky * factor
        w_proj_hat = w_hat - self.kz * factor

        # Sifir modu koru (ortalama hiz)
        u_proj_hat[..., 0, 0, 0] = u_hat[..., 0, 0, 0]
        v_proj_hat[..., 0, 0, 0] = v_hat[..., 0, 0, 0]
        w_proj_hat[..., 0, 0, 0] = w_hat[..., 0, 0, 0]

        # Basinc: lap(p) = div(u) => -k^2 * p_hat = i * k_dot_u
        # => p_hat = -i * k_dot_u / k^2
        p_hat = -1j * k_dot_u / self.k_squared_poisson
        p_hat[..., 0, 0, 0] = 0.0  # sifir ortalama basinc

        # Fiziksel uzaya don
        u_proj = self._ifftn(u_proj_hat).real
        v_proj = self._ifftn(v_proj_hat).real
        w_proj = self._ifftn(w_proj_hat).real
        p = self._ifftn(p_hat).real

        return u_proj, v_proj, w_proj, p

    # ------------------------------------------------------------------
    # Smagorinsky SGS Modeli
    # ------------------------------------------------------------------

    def _compute_smagorinsky_viscosity(
        self, u: torch.Tensor, v: torch.Tensor, w: torch.Tensor
    ) -> torch.Tensor:
        """
        Smagorinsky SGS modeli: nu_t = (Cs * Delta)^2 * |S|

        |S| = sqrt(2 * S_ij * S_ij)  (strain rate tensor buyuklugu)

        S_ij = 0.5 * (du_i/dx_j + du_j/dx_i)

        Acik bilesenler:
            S_11 = du/dx,  S_22 = dv/dy,  S_33 = dw/dz
            S_12 = 0.5*(du/dy + dv/dx)
            S_13 = 0.5*(du/dz + dw/dx)
            S_23 = 0.5*(dv/dz + dw/dy)

        |S|^2 = 2*(S_11^2 + S_22^2 + S_33^2 + 2*S_12^2 + 2*S_13^2 + 2*S_23^2)

        Returns:
            nu_t: SGS viskozite alani [B, Nx, Ny, Nz]
        """
        # Tum hiz gradyanlari
        du_dx, du_dy, du_dz = self._gradient(u)
        dv_dx, dv_dy, dv_dz = self._gradient(v)
        dw_dx, dw_dy, dw_dz = self._gradient(w)

        # Strain rate tensor bilesenleri
        S_11 = du_dx
        S_22 = dv_dy
        S_33 = dw_dz
        S_12 = 0.5 * (du_dy + dv_dx)
        S_13 = 0.5 * (du_dz + dw_dx)
        S_23 = 0.5 * (dv_dz + dw_dy)

        # |S|^2 = 2 * S_ij * S_ij
        # S_ij * S_ij = S_11^2 + S_22^2 + S_33^2 + 2*(S_12^2 + S_13^2 + S_23^2)
        S_ij_S_ij = (
            S_11**2 + S_22**2 + S_33**2
            + 2.0 * (S_12**2 + S_13**2 + S_23**2)
        )

        # |S| = sqrt(2 * S_ij * S_ij)
        S_mag = torch.sqrt(2.0 * S_ij_S_ij + 1e-10)

        # nu_t = (Cs * Delta)^2 * |S|
        nu_t = (self.Cs * self.Delta) ** 2 * S_mag

        return nu_t

    # ------------------------------------------------------------------
    # RHS Hesaplama
    # ------------------------------------------------------------------

    def _compute_rhs(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        w: torch.Tensor,
        theta: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Momentum ve enerji denklemlerinin sag tarafini hesapla.

        Momentum: du/dt = -u.nabla(u) + (nu + nu_t)*lap(u) - nabla(p) + Ri*theta*e_y + F_x
        Enerji:   dtheta/dt = -u.nabla(theta) + (kappa + kappa_t)*lap(theta) + v*(dT/Ly)

        Basinc projeksiyonu burada YAPILMAZ (RK4 adimlarinda yapilir).

        NOT: Difuzyon terimi (nu + nu_t)*lap(u) seklinde, nu_t uzaysal degisen.
        Bu nedenle difuzyon terimi iki kisma ayrilir:
        - Molekuler: nu * lap(u)  (spectral, tam dogru)
        - SGS: div(nu_t * grad(u))  (fiziksel uzayda, dealiased)

        SGS difuzyonu tam anlamda div(nu_t * S_ij) olmalidir ama
        Smagorinsky'de basitlestirilmis form: nu_t * lap(u) + grad(nu_t) . grad(u)
        kullanilir. Homojen turbulansta grad(nu_t) kucuk oldugu icin
        nu_t * lap(u) yaklasimi LES'te standart ve yeterlidir.
        """
        # ── 1. Adveksiyon (dealiased) ──
        du_dx, du_dy, du_dz = self._gradient(u)
        dv_dx, dv_dy, dv_dz = self._gradient(v)
        dw_dx, dw_dy, dw_dz = self._gradient(w)

        adv_u = self._dealias(u * du_dx + v * du_dy + w * du_dz)
        adv_v = self._dealias(u * dv_dx + v * dv_dy + w * dv_dz)
        adv_w = self._dealias(u * dw_dx + v * dw_dy + w * dw_dz)

        # ── 2. Molekuler difuzyon (spectral, tam dogru) ──
        diff_u = self.nu * self._laplacian(u)
        diff_v = self.nu * self._laplacian(v)
        diff_w = self.nu * self._laplacian(w)

        # ── 3. SGS difuzyon: nu_t * lap(u) ──
        nu_t = self._compute_smagorinsky_viscosity(u, v, w)

        sgs_u = self._dealias(nu_t * self._laplacian(u))
        sgs_v = self._dealias(nu_t * self._laplacian(v))
        sgs_w = self._dealias(nu_t * self._laplacian(w))

        # ── 4. Kolmogorov forcing: F_x = A * sin(k_f * 2*pi*y / Ly) ──
        Fx = self.forcing_amplitude * torch.sin(
            self.k_f * 2 * math.pi * self.y_grid / self.Ly
        )

        # ── 5. Boussinesq kaldirma kuvveti: Ri * theta * e_y ──
        buoy_v = self.Ri * theta

        # ── Momentum RHS ──
        rhs_u = -adv_u + diff_u + sgs_u + Fx
        rhs_v = -adv_v + diff_v + sgs_v + buoy_v
        rhs_w = -adv_w + diff_w + sgs_w

        # ── 6. Sicaklik adveksiyonu (dealiased) ──
        dT_dx, dT_dy, dT_dz = self._gradient(theta)
        adv_T = self._dealias(u * dT_dx + v * dT_dy + w * dT_dz)

        # ── 7. Sicaklik difuzyonu (molekuler + SGS) ──
        kappa_t = nu_t / self.Pr_t
        diff_T = self.kappa * self._laplacian(theta)
        sgs_T = self._dealias(kappa_t * self._laplacian(theta))

        # ── 8. Termal kaynak: v * dT/Ly (baz profil etkisi) ──
        source_T = v * self.dT_over_Ly

        rhs_theta = -adv_T + diff_T + sgs_T + source_T

        # ── 9. Anizotropik buoyancy damping ──
        # Periodic BC + Boussinesq'te dusuk-k modlar lineer unstable.
        # Elevator removal (ky=0) tek basina yetersiz: ky!=0 dusuk-k modlar da buyur.
        #
        # Cozum: Her moda kendi anizotropik growth rate'i kadar damping.
        # gamma(k) = safety * max(0, sigma(k))
        # sigma(k) anizotropik dispersion'dan (Leray projection geometrik faktor).
        #
        # Sadece v ve theta'ya uygulanir — u/w'ye damping forcing enerjisini oldurur!
        # Buoyancy instabilitesi v-theta coupled system'den kaynaklanir.
        # u/w'ya dokunmamak turbulans uretimini (Kolmogorov forcing) korur.
        # Ref: Calzavarini et al. (2005), Boffetta & Ecke (2012)
        if self.Ri > 0 and self.gamma_damp.max() > 0:
            for rhs_f, field in [(rhs_v, v), (rhs_theta, theta)]:
                f_hat = self._fftn(field)
                rhs_f -= self._ifftn(self.gamma_damp * f_hat).real

        return rhs_u, rhs_v, rhs_w, rhs_theta

    # ------------------------------------------------------------------
    # Elevator Mode Removal (ky=0 filtreleme)
    # ------------------------------------------------------------------

    def _remove_elevator_modes(
        self,
        u: torch.Tensor, v: torch.Tensor,
        w: torch.Tensor, theta: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        ky=0 elevator modlarini kaldir.

        Periodic BC'de Boussinesq buoyancy ile ky=0 modlari (y-yonunde
        uniform dikey kolonlar) eksponansiyel buyur. Nonlineer adveksiyon
        bu modlara etki etmez → lineer instabilite → blow-up.

        Fourier uzayinda ky=0 indeksini (dim=2, index=0) sifirlayarak
        bu modlari kaldiriyoruz. Mean mode (kx=0,ky=0,kz=0) zaten
        mean-removal ile sifirlanmis durumda.

        Ref: Calzavarini et al. (2005), Phys. Rev. E 73, 035301
        """
        fields_out = []
        for field in (u, v, w, theta):
            f_hat = safe_fftn(field)
            # ky=0 → dim=2 (shape: [batch, Nx, Ny, Nz]), index 0
            f_hat[:, :, 0, :] = 0.0
            fields_out.append(safe_ifftn(f_hat).real)
        return tuple(fields_out)

    # (Eski _apply_lowk_damping kaldirildi -- anizotropik damping _compute_rhs icinde)

    # ------------------------------------------------------------------
    # RK4 Zaman Entegrasyonu
    # ------------------------------------------------------------------

    def _rk4_step(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        w: torch.Tensor,
        theta: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Tek RK4 zaman adimi, her alt-adimda basinc projeksiyonu.

        RK4 sematik:
            k1 = f(t, y)
            k2 = f(t + dt/2, y + dt/2 * k1)
            k3 = f(t + dt/2, y + dt/2 * k2)
            k4 = f(t + dt, y + dt * k3)
            y_new = y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

        Her ara durumda hiz projeksiyon yapilir (div-free garanti).
        Sicaklik perturbasyonu sifir ortalamaya normalize edilir.
        """
        dt = self.dt

        # --- k1 ---
        k1_u, k1_v, k1_w, k1_t = self._compute_rhs(u, v, w, theta)

        # --- k2 ---
        u2 = u + 0.5 * dt * k1_u
        v2 = v + 0.5 * dt * k1_v
        w2 = w + 0.5 * dt * k1_w
        u2, v2, w2, _ = self._project(u2, v2, w2)
        t2 = theta + 0.5 * dt * k1_t
        t2 = t2 - t2.mean(dim=(-3, -2, -1), keepdim=True)
        k2_u, k2_v, k2_w, k2_t = self._compute_rhs(u2, v2, w2, t2)

        # --- k3 ---
        u3 = u + 0.5 * dt * k2_u
        v3 = v + 0.5 * dt * k2_v
        w3 = w + 0.5 * dt * k2_w
        u3, v3, w3, _ = self._project(u3, v3, w3)
        t3 = theta + 0.5 * dt * k2_t
        t3 = t3 - t3.mean(dim=(-3, -2, -1), keepdim=True)
        k3_u, k3_v, k3_w, k3_t = self._compute_rhs(u3, v3, w3, t3)

        # --- k4 ---
        u4 = u + dt * k3_u
        v4 = v + dt * k3_v
        w4 = w + dt * k3_w
        u4, v4, w4, _ = self._project(u4, v4, w4)
        t4 = theta + dt * k3_t
        t4 = t4 - t4.mean(dim=(-3, -2, -1), keepdim=True)
        k4_u, k4_v, k4_w, k4_t = self._compute_rhs(u4, v4, w4, t4)

        # --- Birlestir ---
        u_new = u + dt / 6.0 * (k1_u + 2 * k2_u + 2 * k3_u + k4_u)
        v_new = v + dt / 6.0 * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)
        w_new = w + dt / 6.0 * (k1_w + 2 * k2_w + 2 * k3_w + k4_w)
        theta_new = theta + dt / 6.0 * (k1_t + 2 * k2_t + 2 * k3_t + k4_t)

        # --- Son projeksiyon + gauge fix ---
        u_new, v_new, w_new, p = self._project(u_new, v_new, w_new)
        theta_new = theta_new - theta_new.mean(dim=(-3, -2, -1), keepdim=True)

        # --- Elevator mode removal (ky=0) ---
        # Periodic BC'de ky=0 modlari eksponansiyel buyur (Calzavarini et al. 2005).
        # Bu modlar y-yonunde uniform "dikey kolon" instabilitesi olusturur.
        # Nonlineer adveksiyon bu modlara etki etmez → lineer buyume → blow-up.
        if self.Ri > 0:
            u_new, v_new, w_new, theta_new = self._remove_elevator_modes(
                u_new, v_new, w_new, theta_new
            )

        # --- Theta soft clamp ---
        # Boyutsuz formulasyonda theta ~ O(1). dT_over_Ly=0.1 ile baz profil
        # degisimi 1.0. theta_max=0.5 konservatif sinir.
        _theta_max = 0.5
        theta_new = _theta_max * torch.tanh(theta_new / _theta_max)

        return u_new, v_new, w_new, theta_new, p

    # ------------------------------------------------------------------
    # Adaptive CFL Kontrolu
    # ------------------------------------------------------------------

    def _update_dt(self, u: torch.Tensor, v: torch.Tensor, w: torch.Tensor):
        """
        CFL kosuluna gore dt'yi guncelle.

        CFL = max(|u|/dx, |v|/dy, |w|/dz) * dt

        dt_new = cfl_target / max(|u|/dx, |v|/dy, |w|/dz)

        Kararlilik icin: dt en fazla 1.5x artabilir, aninda azalabilir.
        """
        # Her yondeki max courant sayisi
        cx = u.abs().max().item() / self.dx
        cy = v.abs().max().item() / self.dy
        cz = w.abs().max().item() / self.dz

        c_max = max(cx, cy, cz, 1e-10)  # sifira bolme onleme

        dt_cfl = self.cfl_target / c_max

        # Difuzyon kararliligi da kontrol et:
        # dt_diff = 0.5 * dx_min^2 / (nu_eff)
        # nu_eff tahmini icin nu + ortalama nu_t kullanabiliriz
        # Ama RK4'te explicit difuzyon CFL genellikle advection CFL'den gevselik
        # Yine de bir ust sinir koyalim
        # Not: Smagorinsky nu_t lokal degisiyor, max nu_t bilinmiyor.
        #       Guvenli taraf: nu * 2 ile tahmin (LES'te nu_t ~ nu mertebesi)
        nu_eff_est = self.nu * 3.0  # konservatif tahmin
        dt_diff = 0.2 * self.dx_min**2 / (nu_eff_est + 1e-10)

        dt_new = min(dt_cfl, dt_diff, self.dt_max)
        dt_new = max(dt_new, self.dt_min)

        # Ani artisi sinirla (max 1.5x)
        dt_new = min(dt_new, 1.5 * self.dt)

        self.dt = dt_new

    # ------------------------------------------------------------------
    # Baslangic Kosullari
    # ------------------------------------------------------------------

    def create_initial_condition(
        self, noise_scale: float = 0.01, seed: int = 42,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Rastgele perturbasyonlu baslangic kosulu, div-free projekte edilmis.

        Args:
            noise_scale: Perturbasyonun standart sapmasi
            seed: Tekrarlanabilirlik icin random seed

        Returns:
            (u, v, w, theta, p) hepsi [1, Nx, Ny, Nz] float32
        """
        torch.manual_seed(seed)
        shape = (1, self.Nx, self.Ny, self.Nz)

        u = noise_scale * torch.randn(shape, dtype=torch.float32, device=self.device)
        v = noise_scale * torch.randn(shape, dtype=torch.float32, device=self.device)
        w = noise_scale * torch.randn(shape, dtype=torch.float32, device=self.device)

        # Div-free projeksiyonu
        u, v, w, p = self._project(u, v, w)

        # Sicaklik perturbasyonu (sifir ortalamali)
        theta = noise_scale * torch.randn(shape, dtype=torch.float32, device=self.device)
        theta = theta - theta.mean(dim=(-3, -2, -1), keepdim=True)

        return u, v, w, theta, p

    # ------------------------------------------------------------------
    # Diagnostik Metrikler
    # ------------------------------------------------------------------

    def compute_metrics(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        w: torch.Tensor,
        theta: torch.Tensor,
        p: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Kapsamli akis metrikleri hesapla.

        Dondurulen metrikler:
            TKE:          Turbulansi kinetik enerji = 0.5 * <u_i * u_i>
            enstrophy:    Z = 0.5 * <omega_i * omega_i>
            dissipation:  epsilon = 2 * nu * Z  (DNS benzeri; LES'te nu_eff ile)
            max_velocity: max |u|
            cfl:          Mevcut CFL sayisi
            div_error:    max |nabla . u| (div-free kontrol)
            nusselt:      Nu = 1 + <v * theta> / (kappa * dT/Ly)
            mean_nu_t:    Ortalama SGS viskozitesi
        """
        # ── Temel buyuklukler ──
        speed_sq = u**2 + v**2 + w**2
        tke = 0.5 * speed_sq.mean().item()
        max_vel = torch.sqrt(speed_sq + 1e-10).max().item()

        # ── Vortisite ve enstrophy ──
        ox, oy, oz = self._curl(u, v, w)
        omega_sq = ox**2 + oy**2 + oz**2
        enstrophy = 0.5 * omega_sq.mean().item()

        # ── Dissipasyon ──
        # DNS'te: epsilon = 2 * nu * Z  (Z = 0.5 * <|omega|^2>)
        # yani epsilon = nu * <|omega|^2>
        dissipation_molecular = self.nu * omega_sq.mean().item()

        # SGS dissipasyonu (Smagorinsky)
        nu_t = self._compute_smagorinsky_viscosity(u, v, w)
        mean_nu_t = nu_t.mean().item()
        dissipation_sgs = mean_nu_t * omega_sq.mean().item()
        dissipation_total = dissipation_molecular + dissipation_sgs

        # ── Divergence hatasi ──
        div = self._divergence(u, v, w)
        div_error = div.abs().max().item()

        # ── CFL ──
        cx = u.abs().max().item() / self.dx
        cy = v.abs().max().item() / self.dy
        cz = w.abs().max().item() / self.dz
        cfl = max(cx, cy, cz) * self.dt

        # ── Konvektif isi flux'i ve Nusselt ──
        # <v*theta> = hacim ortalama konvektif termal transport
        # Nu = 1 + <v*theta> / (kappa * dT_over_Ly)  (periodic BC'de yaklasik)
        v_theta_mean = (v * theta).mean().item()
        if abs(self.kappa * self.dT_over_Ly) > 1e-12:
            nusselt = 1.0 + v_theta_mean / (self.kappa * self.dT_over_Ly)
        else:
            nusselt = 1.0

        # ── Forcing power input: P_in = <u * F> ──
        # Steady state'te P_in ≈ epsilon_total olmali (enerji dengesi)
        Fx = self.forcing_amplitude * torch.sin(
            self.k_f * 2 * math.pi * self.y_grid / self.Ly
        )
        forcing_power = (u * Fx).mean().item()

        # ── Helicity ──
        helicity = (u * ox + v * oy + w * oz).mean().item()

        # ── Sicaklik istatistikleri ──
        theta_rms = theta.pow(2).mean().sqrt().item()

        return {
            "TKE": tke,
            "enstrophy": enstrophy,
            "dissipation_mol": dissipation_molecular,
            "dissipation_sgs": dissipation_sgs,
            "dissipation_total": dissipation_total,
            "forcing_power": forcing_power,
            "max_velocity": max_vel,
            "cfl": cfl,
            "div_error": div_error,
            "nusselt": nusselt,
            "v_theta_flux": v_theta_mean,
            "helicity": helicity,
            "theta_rms": theta_rms,
            "mean_nu_t": mean_nu_t,
            "dt": self.dt,
        }

    # ------------------------------------------------------------------
    # Enerji Spektrumu
    # ------------------------------------------------------------------

    def compute_energy_spectrum(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        w: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Izotropik kabuk-ortalalanmis enerji spektrumu E(k).

        E(k) = sum_{k <= |k'| < k+1} 0.5 * |u_hat(k')|^2

        k tam sayidir (dalga sayisi indexi).
        Cikis: (k_bins, E_k) ikisi de 1D tensor.
        """
        u_hat = self._fftn(u)
        v_hat = self._fftn(v)
        w_hat = self._fftn(w)

        # Enerji yogunlugu: 0.5 * (|u_hat|^2 + |v_hat|^2 + |w_hat|^2)
        # FFT normalizasyonu: fft sonucu N ile olcekli,
        # enerji icin 1/N^2 ile normalize etmeliyiz
        N_total = self.Nx * self.Ny * self.Nz
        energy_hat = 0.5 * (
            u_hat.abs().pow(2) + v_hat.abs().pow(2) + w_hat.abs().pow(2)
        ) / N_total**2

        # Dalga sayisi buyuklugu (fiziksel birimlerden normalize edilmis index)
        # k_mag = sqrt(kx^2 + ky^2 + kz^2) zaten hesapli
        # Ama biz "mode index" bazli k istiyoruz (tam sayi binler icin)
        # k_index = k_mag * L / (2*pi) gibi bir sey
        # Daha temiz: mode indexleri kullanalim
        mx = torch.fft.fftfreq(self.Nx, d=1.0) * self.Nx
        my = torch.fft.fftfreq(self.Ny, d=1.0) * self.Ny
        mz = torch.fft.fftfreq(self.Nz, d=1.0) * self.Nz
        Mx, My, Mz = torch.meshgrid(mx, my, mz, indexing='ij')
        k_index = torch.sqrt(Mx**2 + My**2 + Mz**2).to(self.device)

        # Kabuk ortalama: k_max = min(N) // 2
        k_max_shell = min(self.Nx, self.Ny, self.Nz) // 2
        k_bins = torch.arange(0, k_max_shell, dtype=torch.float32, device=self.device)
        E_k = torch.zeros(k_max_shell, dtype=torch.float32, device=self.device)

        # squeeze batch boyutu
        energy_2d = energy_hat.squeeze(0)  # [Nx, Ny, Nz]

        for ki in range(k_max_shell):
            # Kabuk: ki <= |k| < ki + 1
            mask = (k_index >= ki) & (k_index < ki + 1)
            if mask.any():
                E_k[ki] = energy_2d[mask].sum()

        return k_bins, E_k

    # ------------------------------------------------------------------
    # Spectral Entropy
    # ------------------------------------------------------------------

    def compute_spectral_entropy(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        w: torch.Tensor,
    ) -> float:
        """
        Spektral entropi: enerji dagilimininin ne kadar genis oldugunu olcer.

        S = -sum(p_k * log(p_k))  burada p_k = E(k) / sum(E(k))

        S yaklasik log(N) ise enerji duz dagilmis (genis bant turbulansi).
        S yaklasik 0 ise enerji tek modda yogunlasmis (laminer).

        Bu metrik INNATE'in laminer collapse'ini tespit etmek icin kritik.
        """
        _, E_k = self.compute_energy_spectrum(u, v, w)

        # Sifir olmayan modlar
        E_k_pos = E_k[E_k > 0]
        if len(E_k_pos) < 2:
            return 0.0

        # Normalizasyon
        E_total = E_k_pos.sum()
        if E_total < 1e-20:
            return 0.0

        p_k = E_k_pos / E_total
        entropy = -(p_k * torch.log(p_k + 1e-30)).sum().item()

        return entropy

    # ------------------------------------------------------------------
    # Energy Spectrum Slope
    # ------------------------------------------------------------------

    def compute_spectrum_slope(
        self,
        k_bins: torch.Tensor,
        E_k: torch.Tensor,
    ) -> float:
        """
        Enerji spektrumunun inertial range egimini hesapla.

        log(E) = slope * log(k) + const

        Kolmogorov teorisi: slope = -5/3 (yaklasik -1.667)
        2D turbulansi: slope = -3 (enstrophy cascade)
        Laminer: slope >> -5/3 (cok dik dusus)

        Inertial range: k=6 ile k=15 arasi (train.py ile AYNI k_range)
        """
        # Inertial range secimi — train.py ile tutarli
        k_min_fit = 6
        k_max_fit = min(15, len(E_k) - 1)

        if k_max_fit <= k_min_fit + 2:
            return 0.0

        k_range = k_bins[k_min_fit:k_max_fit]
        E_range = E_k[k_min_fit:k_max_fit]

        # Sifir olmayan modlar
        mask = E_range > 1e-20
        if mask.sum() < 3:
            return 0.0

        k_log = torch.log(k_range[mask])
        E_log = torch.log(E_range[mask])

        # Lineer regresyon: y = ax + b
        # a = (N*sum(xy) - sum(x)*sum(y)) / (N*sum(x^2) - sum(x)^2)
        N = k_log.shape[0]
        sx = k_log.sum()
        sy = E_log.sum()
        sxy = (k_log * E_log).sum()
        sxx = (k_log * k_log).sum()

        denom = N * sxx - sx * sx
        if abs(denom.item()) < 1e-20:
            return 0.0

        slope = (N * sxy - sx * sy) / denom
        return slope.item()

    # ------------------------------------------------------------------
    # Ana Calistirma Dongusu
    # ------------------------------------------------------------------

    @torch.no_grad()
    def run(
        self,
        n_steps: int = 5000,
        log_interval: int = 100,
        snapshot_interval: int = 500,
        noise_scale: float = 0.01,
        save_dir: Optional[str] = None,
        seed: int = 42,
    ) -> Dict:
        """
        LES simulasyonunu calistir.

        Args:
            n_steps:           Toplam adim sayisi
            log_interval:      Her N adimda metrik logla
            snapshot_interval: Her N adimda tam alan kaydet
            noise_scale:       IC perturbasyonu
            save_dir:          Kayit dizini (None = varsayilan les_reference/)
            seed:              Rastgele tohum

        Returns:
            dict: 'metrics_history', 'spectra', 'final_state', 'params'
        """
        if save_dir is None:
            save_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "les_reference"
            )

        os.makedirs(save_dir, exist_ok=True)
        print(f"Kayit dizini: {save_dir}")

        # ── Baslangic kosulu ──
        u, v, w, theta, p = self.create_initial_condition(
            noise_scale=noise_scale, seed=seed
        )

        # ── Metrik gecmisi ──
        metrics_history: List[Dict] = []
        spectra_history: List[Dict] = []
        t_sim = 0.0

        # ── Zamanlama ──
        t_wall_start = time.time()

        print(f"\n{'='*80}")
        print(f"LES Simulasyonu Basliyor: {n_steps} adim")
        print(f"{'='*80}\n")

        for step in range(1, n_steps + 1):
            # ── CFL-adaptive dt ──
            self._update_dt(u, v, w)

            # ── RK4 adimi ──
            u, v, w, theta, p = self._rk4_step(u, v, w, theta)
            t_sim += self.dt

            # ── NaN kontrolu ──
            if torch.isnan(u).any() or torch.isnan(v).any() or torch.isnan(w).any():
                print(f"\n[HATA] NaN tespit edildi! Step {step}, t={t_sim:.4f}")
                print(f"  Son dt={self.dt:.6f}")
                break

            # ── Metrik loglama ──
            if step % log_interval == 0 or step == 1:
                metrics = self.compute_metrics(u, v, w, theta, p)
                metrics["step"] = step
                metrics["t"] = t_sim
                metrics_history.append(metrics)

                # Spektrum
                k_bins, E_k = self.compute_energy_spectrum(u, v, w)
                slope = self.compute_spectrum_slope(k_bins, E_k)
                entropy = self.compute_spectral_entropy(u, v, w)

                metrics["spectrum_slope"] = slope
                metrics["spectral_entropy"] = entropy

                # Ekrana yazdir
                t_wall_elapsed = time.time() - t_wall_start
                steps_per_sec = step / max(t_wall_elapsed, 1e-6)

                print(
                    f"Step {step:>6d}/{n_steps} | "
                    f"t={t_sim:>8.3f} | "
                    f"dt={self.dt:.5f} | "
                    f"TKE={metrics['TKE']:.4e} | "
                    f"Z={metrics['enstrophy']:.4e} | "
                    f"eps={metrics['dissipation_total']:.4e} | "
                    f"P_in={metrics['forcing_power']:.4e} | "
                    f"slope={slope:.2f} | "
                    f"S={entropy:.3f} | "
                    f"v_max={metrics['max_velocity']:.3f} | "
                    f"CFL={metrics['cfl']:.3f} | "
                    f"div={metrics['div_error']:.2e} | "
                    f"{steps_per_sec:.1f} it/s"
                )

            # ── Snapshot kaydi ──
            if step % snapshot_interval == 0:
                k_bins_snap, E_k_snap = self.compute_energy_spectrum(u, v, w)
                spectra_history.append({
                    "step": step,
                    "t": t_sim,
                    "k_bins": k_bins_snap.cpu(),
                    "E_k": E_k_snap.cpu(),
                })

                # Snapshot dosyasi
                snapshot_path = os.path.join(save_dir, f"snapshot_{step:06d}.pt")
                torch.save({
                    "u": u.cpu(),
                    "v": v.cpu(),
                    "w": w.cpu(),
                    "theta": theta.cpu(),
                    "p": p.cpu(),
                    "t": t_sim,
                    "step": step,
                    "dt": self.dt,
                }, snapshot_path)

        # ── Simulasyon bitti ──
        t_wall_total = time.time() - t_wall_start
        print(f"\n{'='*80}")
        print(f"Simulasyon tamamlandi: {step} adim, t_sim={t_sim:.3f}")
        print(f"Toplam sure: {t_wall_total:.1f} s ({t_wall_total/60:.1f} dk)")
        print(f"{'='*80}\n")

        # ── Son durum metrikleri ──
        final_metrics = self.compute_metrics(u, v, w, theta, p)
        k_bins_final, E_k_final = self.compute_energy_spectrum(u, v, w)
        final_slope = self.compute_spectrum_slope(k_bins_final, E_k_final)
        final_entropy = self.compute_spectral_entropy(u, v, w)
        final_metrics["spectrum_slope"] = final_slope
        final_metrics["spectral_entropy"] = final_entropy

        # ── Sonuclari kaydet ──
        # 1. Enerji spektrumu
        spectrum_path = os.path.join(save_dir, "spectrum.pt")
        torch.save({
            "k_bins": k_bins_final.cpu(),
            "E_k": E_k_final.cpu(),
            "slope": final_slope,
            "spectra_history": spectra_history,
        }, spectrum_path)
        print(f"Spektrum kaydedildi: {spectrum_path}")

        # 2. Metrikler (JSON)
        metrics_path = os.path.join(save_dir, "metrics.json")
        # Tum metrik gecmisini float'a cevir (JSON serializable)
        metrics_serializable = []
        for m in metrics_history:
            ms = {}
            for k, val in m.items():
                if isinstance(val, (int, float)):
                    ms[k] = val
                elif isinstance(val, torch.Tensor):
                    ms[k] = val.item()
                else:
                    ms[k] = str(val)
            metrics_serializable.append(ms)

        output_json = {
            "params": {
                "Nx": self.Nx, "Ny": self.Ny, "Nz": self.Nz,
                "Lx": self.Lx, "Ly": self.Ly, "Lz": self.Lz,
                "Re": self.Re, "Ra": self.Ra, "Pr": self.Pr,
                "nu": self.nu, "kappa": self.kappa, "Ri": self.Ri,
                "Cs": self.Cs, "Pr_t": self.Pr_t, "Delta": self.Delta,
                "k_f": self.k_f, "forcing_amplitude": self.forcing_amplitude,
                "dT": self.dT,
            },
            "final_metrics": {k: v for k, v in final_metrics.items()
                              if isinstance(v, (int, float))},
            "metrics_history": metrics_serializable,
            "simulation": {
                "n_steps": step,
                "t_final": t_sim,
                "wall_time_seconds": t_wall_total,
            },
        }

        with open(metrics_path, "w") as f:
            json.dump(output_json, f, indent=2)
        print(f"Metrikler kaydedildi: {metrics_path}")

        # 3. Son durum
        final_state_path = os.path.join(save_dir, "final_state.pt")
        torch.save({
            "u": u.cpu(),
            "v": v.cpu(),
            "w": w.cpu(),
            "theta": theta.cpu(),
            "p": p.cpu(),
            "t": t_sim,
        }, final_state_path)
        print(f"Son durum kaydedildi: {final_state_path}")

        # ── Ozet ──
        self._print_summary(final_metrics, final_slope, final_entropy, t_sim, step, t_wall_total)

        return {
            "metrics_history": metrics_history,
            "spectra": {"k_bins": k_bins_final.cpu(), "E_k": E_k_final.cpu()},
            "final_state": {
                "u": u.cpu(), "v": v.cpu(), "w": w.cpu(),
                "theta": theta.cpu(), "p": p.cpu(),
            },
            "params": output_json["params"],
        }

    # ------------------------------------------------------------------
    # Ozet Yazdirma
    # ------------------------------------------------------------------

    def _print_summary(
        self,
        metrics: Dict,
        slope: float,
        entropy: float,
        t_sim: float,
        n_steps: int,
        wall_time: float,
    ):
        """Simulasyon sonunda kapsamli ozet yazdir."""
        print(f"\n{'='*80}")
        print(f"  LES DOGRULAMA COZUCUSU - SONUC OZETI")
        print(f"{'='*80}")
        print(f"")
        print(f"  Fiziksel Parametreler:")
        print(f"    Re = {self.Re:.0f}")
        print(f"    Ra = {self.Ra:.1e}")
        print(f"    Pr = {self.Pr}")
        print(f"    Ri = {self.Ri:.4f}")
        print(f"    nu = {self.nu:.2e}")
        print(f"    kappa = {self.kappa:.2e}")
        print(f"")
        print(f"  Grid: {self.Nx} x {self.Ny} x {self.Nz}")
        print(f"  Domain: {self.Lx} x {self.Ly} x {self.Lz}")
        print(f"  LES: Smagorinsky Cs={self.Cs}, Delta={self.Delta:.4f}")
        print(f"")
        print(f"  Simulasyon Istatistikleri:")
        print(f"    Toplam adim:     {n_steps}")
        print(f"    Simulasyon suresi: t = {t_sim:.3f}")
        print(f"    Duvar suresi:    {wall_time:.1f} s ({wall_time/60:.1f} dk)")
        print(f"    Son dt:          {metrics.get('dt', self.dt):.6f}")
        print(f"")
        print(f"  Akis Metrikleri (Son Durum):")
        print(f"    TKE (Turbulansi Kinetik Enerji):  {metrics['TKE']:.6f}")
        print(f"    Enstrophy Z:                      {metrics['enstrophy']:.6f}")
        print(f"    Dissipation (molekuler):           {metrics['dissipation_mol']:.6f}")
        print(f"    Dissipation (SGS):                 {metrics['dissipation_sgs']:.6f}")
        print(f"    Dissipation (toplam):              {metrics['dissipation_total']:.6f}")
        print(f"    Forcing power <u*F>:               {metrics['forcing_power']:.6f}")
        print(f"    Enerji dengesi (P_in/eps):         {metrics['forcing_power'] / max(metrics['dissipation_total'], 1e-12):.3f}  (1.0 = denge)")
        print(f"    Nusselt Nu:                        {metrics['nusselt']:.4f}")
        print(f"    Konvektif flux <v*theta>:           {metrics['v_theta_flux']:.6e}")
        print(f"    Helicity:                          {metrics['helicity']:.6f}")
        print(f"    Theta RMS:                         {metrics['theta_rms']:.6f}")
        print(f"    Max hiz:                           {metrics['max_velocity']:.4f}")
        print(f"    CFL:                               {metrics['cfl']:.4f}")
        print(f"    Divergence hatasi:                 {metrics['div_error']:.2e}")
        print(f"    Ortalama nu_t:                     {metrics['mean_nu_t']:.6f}")
        print(f"")
        print(f"  Spektral Analiz:")
        print(f"    E(k) egimi (inertial range):       {slope:.3f}")
        print(f"      (Kolmogorov -5/3 = -1.667)")
        print(f"    Spektral entropi:                  {entropy:.3f}")
        print(f"      (Yuksek = genis bantli turbulansi, Dusuk = laminer)")
        print(f"")

        # Fiziksel tutarlilik kontrolleri
        print(f"  Fiziksel Tutarlilik Kontrolleri:")
        issues = []

        if metrics['TKE'] < 1e-8:
            issues.append("  [UYARI] TKE cok dusuk -- akis laminer!")
        if self.Ri > 0 and metrics['nusselt'] < 1.0:
            issues.append("  [UYARI] Nu < 1.0 -- fiziksel olarak imkansiz (konveksiyon varken)")
        if metrics['div_error'] > 1e-4:
            issues.append(f"  [UYARI] Yuksek divergence hatasi: {metrics['div_error']:.2e}")
        if slope > -1.0:
            issues.append(f"  [UYARI] Spektrum egimi {slope:.2f} -- turbulansi yok veya cok zayif")
        if entropy < 1.0:
            issues.append(f"  [UYARI] Dusuk spektral entropi ({entropy:.2f}) -- laminer collapse?")
        energy_ratio = metrics['forcing_power'] / max(metrics['dissipation_total'], 1e-12)
        if energy_ratio > 2.0:
            issues.append(f"  [UYARI] P_in/eps = {energy_ratio:.2f} -- enerji birikimi (blow-up riski)")
        elif energy_ratio < 0.5 and metrics['TKE'] > 1e-6:
            issues.append(f"  [UYARI] P_in/eps = {energy_ratio:.2f} -- enerji kaybediliyor (decay)")

        if not issues:
            print("    Tum kontroller basarili.")
        else:
            for issue in issues:
                print(issue)

        print(f"")
        print(f"{'='*80}")
        print(f"  INNATE modeli icin beklenen referans degerler:")
        print(f"    TKE:      {metrics['TKE']:.6f}  (O(0.01-1) arasinda olmali)")
        print(f"    Z:        {metrics['enstrophy']:.6f}  (sifirdan yukari, platoya ulasmali)")
        print(f"    epsilon:  {metrics['dissipation_total']:.6f}  (forcing = dissipation dengesinde)")
        print(f"    P_in:     {metrics['forcing_power']:.6f}  (P_in/eps = {metrics['forcing_power'] / max(metrics['dissipation_total'], 1e-12):.3f}, 1.0 hedef)")
        if self.Ri > 0:
            print(f"    Nu:       {metrics['nusselt']:.4f}  (> 1.0 sart)")
            print(f"    <vT>:     {metrics['v_theta_flux']:.6e}  (konvektif termal transport)")
        else:
            print(f"    <vT>:     {metrics['v_theta_flux']:.6e}  (Ra=0, Nu anlamsiz)")
        print(f"    E(k):     ~ k^({slope:.2f})  (-5/3 ile -3 arasi normal)")
        print(f"    S_ent:    {entropy:.3f}  (> 2.0 turbulansi, < 1.0 laminer)")
        print(f"{'='*80}\n")


# ══════════════════════════════════════════════════════════════════════════
# MAIN -- Dogrudan calistirma
# ══════════════════════════════════════════════════════════════════════════

def main():
    """
    LES validation solver'i calistir ve referans metrikleri uret.

    Varsayilan parametreler INNATE 3D modeliyle birebir eslesir:
    - Grid: 96 x 160 x 64
    - Domain: 6 x 10 x 4
    - Re=5000, Ra=1e6, Pr=0.71
    - Kolmogorov forcing k_f=4
    - Smagorinsky LES Cs=0.17

    Cikti dizini: bitirme2/les_reference/
    """
    import argparse

    parser = argparse.ArgumentParser(description="LES Validation Solver")
    parser.add_argument("--n_steps", type=int, default=5000,
                        help="Toplam adim sayisi (varsayilan: 5000)")
    parser.add_argument("--log_interval", type=int, default=100,
                        help="Metrik loglama araligi (varsayilan: 100)")
    parser.add_argument("--snapshot_interval", type=int, default=500,
                        help="Snapshot kayit araligi (varsayilan: 500)")
    parser.add_argument("--Re", type=float, default=5000.0,
                        help="Reynolds sayisi (varsayilan: 5000)")
    parser.add_argument("--Ra", type=float, default=1e6,
                        help="Rayleigh sayisi (varsayilan: 1e6)")
    parser.add_argument("--k_f", type=int, default=4,
                        help="Forcing dalga sayisi (varsayilan: 4)")
    parser.add_argument("--A", type=float, default=None,
                        help="Forcing amplitude (varsayilan: 0.005)")
    parser.add_argument("--Cs", type=float, default=0.17,
                        help="Smagorinsky sabiti (varsayilan: 0.17)")
    parser.add_argument("--noise_scale", type=float, default=0.01,
                        help="IC perturbasyonu (varsayilan: 0.01)")
    parser.add_argument("--cfl", type=float, default=0.5,
                        help="Hedef CFL sayisi (varsayilan: 0.5)")
    parser.add_argument("--dt_max", type=float, default=0.02,
                        help="Maksimum dt (varsayilan: 0.02)")
    parser.add_argument("--no_buoyancy_damping", action="store_true",
                        help="Anizotropik buoyancy damping'i kapat")
    parser.add_argument("--damping_safety", type=float, default=3.5,
                        help="Damping safety factor (varsayilan: 3.5)")
    parser.add_argument("--Nx", type=int, default=96,
                        help="Grid x (varsayilan: 96)")
    parser.add_argument("--Ny", type=int, default=160,
                        help="Grid y (varsayilan: 160)")
    parser.add_argument("--Nz", type=int, default=64,
                        help="Grid z (varsayilan: 64)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device: cuda, mps, cpu (varsayilan: otomatik)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (varsayilan: 42)")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Kayit dizini (varsayilan: bitirme2/les_reference/)")

    args = parser.parse_args()

    # ── Solver olustur ──
    # Forcing amplitude: default veya kullanici belirler
    forcing_amp = args.A if args.A is not None else 0.005

    solver = LESSolver(
        Nx=args.Nx, Ny=args.Ny, Nz=args.Nz,
        Re=args.Re,
        Ra=args.Ra,
        k_f=args.k_f,
        Cs=args.Cs,
        forcing_amplitude=forcing_amp,
        cfl_target=args.cfl,
        dt_max=args.dt_max,
        buoyancy_damping=not args.no_buoyancy_damping,
        damping_safety=args.damping_safety,
        device=args.device,
    )

    # ── Calistir ──
    results = solver.run(
        n_steps=args.n_steps,
        log_interval=args.log_interval,
        snapshot_interval=args.snapshot_interval,
        noise_scale=args.noise_scale,
        save_dir=args.save_dir,
        seed=args.seed,
    )

    return results


if __name__ == "__main__":
    main()
