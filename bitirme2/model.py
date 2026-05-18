"""
model.py - INNATE v2 3D Mixed Convection Modeli (555-param versiyon)

Hybrid MLP+Physics mimarisi. IMEX zaman entegrasyonu. Scale-similarity SGS.
20 katman, 555 ogrenilebilir parametre.

Parametre butcesi (v2):
  MLP SGS shared weights           = 330  (fc1: 6*32+32=224, fc2: 32*2+2=66, layer_bias: 20*2=40)
  advection_modulator ×20          = 20
  aniso_ratio_y/z ×20              = 40   (anisotropik SGS)
  backscatter_coeff ×20            = 20
  C_ss ×20                         = 20   (scale-similarity, per-layer)
  kappa_scale_x/y/z ×20           = 60   (anisotropik termal difuzyon)
  thermal_adv_modulator ×20        = 20
  buoyancy_strength ×20            = 20   (per-layer buoyancy)
  dt_mults ×19 + dt_scale          = 20   (per-layer zaman adimi)
  forcing (amplitude + 4 harm.)    = 5
  TOPLAM                           = 555

v1'den farklar:
  - MLP SGS: (|S|, |Omega|, R, Ri_g, Re, layer) -> (Cs, kappa) [330 param]
  - IMEX: implicit mol. diffusion, explicit SGS [0 ek param]
  - Scale-similarity: mixed model [1 param]
  - Buoyancy damping KALDIRILDI (IMEX sayesinde gereksiz)
  - cs_re_a/b KALDIRILDI (MLP Re'yi input olarak aliyor)
  - cs_low/mid/high KALDIRILDI (MLP Cs'yi tahmin ediyor)
  - cs_thermal KALDIRILDI (MLP kappa'yi tahmin ediyor)

Yazar: Berke Tezgocen (tasarim), Claude (implementasyon)
"""

from __future__ import annotations

import math
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp

# -- path setup --
_this_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_this_dir.parent))

from innate import (
    Advection3D,
    Buoyancy3D,
    ContinuityNeuron3D,
    DensityUpdate3D,
    EddyViscosity3D,
    SpectralCsField,
    FluidState3D,
    Forcing3D,
    MLPSGS,
    Projection3D,
    SpectralOps3DAniso,
    StateEquation3D,
    ThermalAdvection3D,
    ThermalDiffusion3D,
    VariableDensityAdvection3D,
    safe_irfftn,
    safe_rfftn,
)

from config import Config


# =====================================================================
# 1. ThermalFluidState
# =====================================================================


@dataclass
class ThermalFluidState:
    """
    Termal-akiskan state: momentum (u,v,w), basinc (p), sicaklik (theta).
    Opsiyonel: t (zaman), rho (yogunluk - Phase D).
    """
    u: torch.Tensor       # [B, Nx, Ny, Nz]
    v: torch.Tensor
    w: torch.Tensor
    p: torch.Tensor
    theta: torch.Tensor   # T' perturbation sicaklik
    t: Optional[torch.Tensor] = None   # [B] zaman
    rho: Optional[torch.Tensor] = None # [B, Nx, Ny, Nz] yogunluk (Phase D)

    def kinetic_energy(self) -> torch.Tensor:
        """Domain-ortalama kinetik enerji: 0.5 * <u^2 + v^2 + w^2>.  Returns [B]."""
        return 0.5 * (self.u**2 + self.v**2 + self.w**2).mean(dim=(-3, -2, -1))

    def enstrophy(self) -> torch.Tensor:
        """Dogru enstrophy icin SpectralOps gerekli. PhysicsLoss._enstrophy() kullanin."""
        raise NotImplementedError(
            "ThermalFluidState.enstrophy() ops.curl gerektirir. "
            "Dogru hesap icin PhysicsLoss._enstrophy(state) kullanin."
        )

    def nusselt_number(self, Ly: float, kappa: float) -> torch.Tensor:
        """
        Nusselt sayisi: Nu = 1 + <v*theta> / (kappa * dT_over_Ly).

        dT_over_Ly = 1/Ly (boyutsuz baz sicaklik gradyani).
        LES solver ile tutarli formul.
        Returns [B].
        """
        vT = (self.v * self.theta).mean(dim=(-3, -2, -1))
        dT_over_Ly = 1.0 / Ly
        return 1.0 + vT / (kappa * dT_over_Ly + 1e-10)


# =====================================================================
# 2. Helper: ThermalFluidState -> FluidState3D
# =====================================================================


def _to_fluid_state(state: ThermalFluidState, ops) -> FluidState3D:
    """ThermalFluidState'i Advection3D/EddyViscosity3D icin FluidState3D'ye cevir."""
    ox, oy, oz = ops.curl(state.u, state.v, state.w)
    t = state.t if state.t is not None else torch.zeros(state.u.shape[0], device=state.u.device)
    return FluidState3D(
        u=state.u, v=state.v, w=state.w, p=state.p,
        omega_x=ox, omega_y=oy, omega_z=oz, t=t,
    )


# =====================================================================
# 3. INNATE3D_MixedConvection
# =====================================================================


class INNATE3D_MixedConvection(nn.Module):
    """
    INNATE v2 3D Mixed Convection Modeli - 555 param versiyon.

    20 katman x fractional-step = 20 zaman adimi (internal time-stepping).
    IMEX entegrasyon: molecular diffusion implicit, SGS diffusion explicit.
    MLP SGS: (|S|, |Omega|, R, Ri_g, Re, layer) -> Cs field + scalar Pr_t
             (kappa_t = nu_t / Pr_t, Reynolds analogy)
    Scale-similarity mixed model: tau = tau_smagorinsky + C_ss * L_ij

    v1'den farklar:
      - MLP SGS (shared, 330 param) replaces frequency-band Cs + cs_thermal
      - IMEX integration (unconditionally stable for mol. diffusion)
      - Scale-similarity (natural backscatter)
      - Buoyancy damping REMOVED (IMEX makes it unnecessary)
      - cs_re_a/b REMOVED (MLP takes Re as input)
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        d = config.domain
        p = config.physics
        m = config.model

        self.n_layers = m.n_layers
        self.use_eddy = m.use_eddy_viscosity
        self.use_checkpointing = m.gradient_checkpointing
        self.non_boussinesq = False  # Phase D postponed in v2
        self.use_imex = getattr(m, 'use_imex', True)

        # Tier flags
        self._use_anisotropic_sgs = m.use_anisotropic_sgs
        self._use_per_layer_dt = m.use_per_layer_dt

        # -- Spectral Operators (paylasimli, learnable degil) --
        self.ops = SpectralOps3DAniso(
            Nx=d.Nx, Ny=d.Ny, Nz=d.Nz,
            Lx=d.Lx, Ly=d.Ly, Lz=d.Lz,
        )

        # -- IMEX: k_squared buffer for implicit diffusion --
        # u_hat_new = (u_hat + dt * RHS_hat) / (1 + dt * nu * k^2)
        # Pre-compute k_squared for spectral implicit step
        # (ops.k_squared already exists as buffer, but we register a convenience ref)

        # -- Elevator mode mask (ky=0 filter) --
        _elev = torch.ones(1, d.Nx, d.Ny, d.Nz // 2 + 1)
        _elev[:, :, 0, :] = 0.0
        self.register_buffer('_elevator_mask', _elev)

        # -- Physics parameters (non-learnable) --
        self.nu = p.nu          # kinematic viscosity = 1/Re
        self.kappa = p.kappa    # thermal diffusivity = 1/(Re*Pr)
        self.Ri = p.Ri          # Richardson number = Ra/(Re^2 * Pr)
        self._dt_base = p.dt    # base time step (non-learnable)
        self.Re = p.Re
        self.Re_normalized = p.Re / 20000.0  # MLP input icin

        # -- Learnable dt_scale: global zaman adimi modulatoru --
        self.dt_scale = nn.Parameter(torch.tensor(1.0))

        # -- Per-layer dt multipliers (Tier 2) --
        if m.use_per_layer_dt:
            self.dt_mults = nn.ParameterList([
                nn.Parameter(torch.tensor(1.0)) for _ in range(self.n_layers - 1)
            ])

        # -- dx_min for CFL --
        self.dx_min = d.dx_min

        # -- GLOBAL NORONLAR --
        self.forcing = Forcing3D(
            Ny=d.Ny, Ly=d.Ly, k_f=p.k_f, mode=p.forcing_mode,
            use_harmonics=m.use_forcing_harmonics,
        )

        # -- TKE-coupled forcing damping (negative feedback loop) --
        self.register_buffer('_tke_ref', torch.tensor(0.0))

        # -- SGS pathway: Saf-INNATE Spectral-Cs (yeni) VEYA MLP-SGS (eski) --
        _use_spectral = getattr(m, 'use_spectral_cs', False)
        _use_mlp = getattr(m, 'use_mlp_sgs', True)
        _mlp_hidden = getattr(m, 'mlp_hidden_dim', 32)
        _use_scale_sim = getattr(m, 'use_scale_similarity', True)
        # Spectral truncation (default 5x8x6 = 240 modes/layer × 20 = ~9.6K params)
        _kx_max = getattr(m, 'spectral_cs_kx_max', 5)
        _ky_max = getattr(m, 'spectral_cs_ky_max', 8)
        _kz_max = getattr(m, 'spectral_cs_kz_max', 6)

        if _use_spectral:
            # Saf-INNATE: per-layer SpectralCsField, MLP yok
            self.mlp_sgs = None
            self.spectral_cs_modules = nn.ModuleList([
                SpectralCsField(
                    Nx=d.Nx, Ny=d.Ny, Nz=d.Nz,
                    kx_max=_kx_max, ky_max=_ky_max, kz_max=_kz_max,
                    base=0.155, Pr_t_init=0.85,
                    use_anisotropic=m.use_anisotropic_sgs,
                ) for _ in range(self.n_layers)
            ])
            print(f"  [SAF-INNATE] SpectralCsField per-layer, "
                  f"k_trunc=({_kx_max}x{_ky_max}x{_kz_max}), "
                  f"total spectral params={sum(p.numel() for p in self.spectral_cs_modules.parameters())}")
        elif _use_mlp:
            self.mlp_sgs = MLPSGS(hidden_dim=_mlp_hidden, n_layers=self.n_layers)
            self.spectral_cs_modules = None
        else:
            self.mlp_sgs = None
            self.spectral_cs_modules = None

        # -- Per-layer buoyancy (Tier 2) veya global --
        # NOT (boyut analizi 2026-04-25):
        # Buoyancy3D forward: Fy = Ri · strength · θ
        # `strength` LES referansında YOK — INNATE-spesifik kapasite parametresi.
        # init=0.5 (Buoyancy3D dahili default), eğitim sırasında [0,50] aralığında öğrenilir.
        # Forced-dominant rejimde (Ri ~ 1e-3) etki minimal; ancak demolarda
        # Ri up to 5.6e4 olduğunda LES-referansından sapma oluşur.
        # Tezde "strength" dürüstçe rapor edilmeli ve eğitim "thermally-coupled
        # forced convection" diye etiketlenmeli; demolar "extrapolation".
        # apriori_from_les.py --buoyancy-scale override sağlıyor.
        if m.use_per_layer_buoyancy:
            self.buoyancies = nn.ModuleList([
                Buoyancy3D(Ri=self.Ri) for _ in range(self.n_layers)
            ])
        else:
            self.buoyancy = Buoyancy3D(Ri=self.Ri)

        # -- PER-LAYER NORONLAR --
        self.advections = nn.ModuleList()
        self.projections = nn.ModuleList()
        self.eddy_viscosities = nn.ModuleList()
        self.thermal_advections = nn.ModuleList()
        self.thermal_diffusions = nn.ModuleList()

        for i in range(self.n_layers):
            # Advection3D: her katmanda bagimsiz
            adv = Advection3D(resolution=d.Nx, diff_ops=self.ops, use_lamb=True)
            with torch.no_grad():
                adv.advection_modulator.fill_(
                    1.0 + 0.01 * (i - self.n_layers / 2) / (self.n_layers / 2)
                )
            self.advections.append(adv)

            # Projection3D: 0 param
            self.projections.append(
                Projection3D(resolution=d.Nx, diff_ops=self.ops)
            )

            # EddyViscosity3D: v2 with MLP + scale-similarity OR Saf-INNATE Spectral
            if self.use_eddy:
                _spectral_for_layer = (
                    self.spectral_cs_modules[i] if _use_spectral else None
                )
                self.eddy_viscosities.append(
                    EddyViscosity3D(
                        resolution=d.Nx, diff_ops=self.ops,
                        grid_spacings=d.grid_spacings,
                        use_frequency_bands=(not _use_mlp and not _use_spectral
                                             and getattr(m, 'use_frequency_band_cs', False)),
                        use_turbulent_prandtl=(not _use_mlp and not _use_spectral
                                               and m.use_turbulent_prandtl),
                        use_anisotropic=m.use_anisotropic_sgs,
                        use_backscatter=m.use_backscatter,
                        use_local_cs=False,
                        use_local_thermal=False,
                        mlp_sgs=self.mlp_sgs,
                        spectral_cs=_spectral_for_layer,
                        use_scale_similarity=(_use_scale_sim and not _use_spectral),
                    )
                )

            # ThermalAdvection3D: modulator opsiyonel
            self.thermal_advections.append(
                ThermalAdvection3D(
                    spectral_ops=self.ops,
                    use_modulator=m.use_thermal_adv_modulator,
                )
            )

            # ThermalDiffusion3D: anisotropik opsiyonel
            self.thermal_diffusions.append(
                ThermalDiffusion3D(
                    kappa=self.kappa, spectral_ops=self.ops,
                    use_anisotropic=m.use_anisotropic_kappa,
                )
            )

        # -- Anizotropik buoyancy damping (LES'ten port: Calzavarini 2005) --
        # Boussinesq+periodic BC'de dusuk-k modlar lineer kararsiz. Elevator
        # mask (ky=0) tek basina yetersiz; ky!=0 dusuk-k modlar da buyur.
        # gamma(k) = safety * max(0, sigma(k)) -- her moda kendi growth rate'i
        # kadar damping. Sadece v ve theta'ya uygulanir (u/w'ye dokunmamak
        # forcing enerjisini korur). Ref: Boffetta & Ecke (2012), JFM.
        self._damping_safety = 2.0
        self._init_buoyancy_damping()

    # ---------------------------------------------------------------- #
    # Physics parameter sweep                                          #
    # ---------------------------------------------------------------- #

    def set_physics(self, Re: float, Ra: float, Pr: float = 0.71):
        """Re/Ra sweep icin fizik parametrelerini guncelle."""
        self.Re = Re
        self.Re_normalized = Re / 20000.0  # MLP SGS input icin
        self.nu = 1.0 / Re
        self.kappa = 1.0 / (Re * Pr)
        self.Ri = Ra / (Re**2 * Pr)

        # Config'i de guncelle (PhysicsLoss config.physics kullanir)
        self.config.physics.Re = Re
        self.config.physics.Ra = Ra
        self.config.physics.Pr = Pr

        # Buoyancy Richardson number guncelle
        if hasattr(self, 'buoyancies'):
            for b in self.buoyancies:
                b.set_Ri(self.Ri)
        else:
            self.buoyancy.set_Ri(self.Ri)

        # Thermal diffusion kappa guncelle
        for td in self.thermal_diffusions:
            td.set_kappa(self.kappa)

        # Buoyancy damping profilini Re/Ra'ya gore yeniden hesapla
        self._init_buoyancy_damping()

    # ---------------------------------------------------------------- #
    # Buoyancy damping (anizotropik sigma-profile, LES'ten port)       #
    # ---------------------------------------------------------------- #

    def _init_buoyancy_damping(self):
        """Anizotropik buoyancy damping profilini hesapla.

        Leray projection geometrik faktor: f = (kx^2 + kz^2) / k^2
        (ky-only modlar f=0 stabil; kx/kz iceren modlar f=1 max instabil)

        Anizotropik dispersion (Boussinesq + periodic BC):
            sigma^2 + (nu+kappa)*k^2*sigma + nu*kappa*k^4 - Ri*(dT/Ly)*f = 0
            sigma = -0.5*(nu+kappa)*k^2 + sqrt(disc)

        gamma_damp = safety * max(0, sigma) -- buffer'a [Nx, Ny, Nz//2+1]
        olarak yaz (rFFT shape ile uyumlu).
        """
        ops = self.ops
        kx2 = ops.kx ** 2
        kz2 = ops.kz ** 2
        k_sq = ops.k_squared  # [Nx, Ny, Nz//2+1]

        f_aniso = (kx2 + kz2) / (k_sq + 1e-12)
        # k=0 modu: mean-removal kontrol ediyor, damping sifir
        f_aniso = f_aniso.clone()
        f_aniso[0, 0, 0] = 0.0

        dT_over_Ly = 1.0 / self.config.domain.Ly

        if self.Ri > 0:
            disc = 0.25 * (self.nu - self.kappa) ** 2 * k_sq ** 2 \
                 + self.Ri * dT_over_Ly * f_aniso
            sigma_profile = -0.5 * (self.nu + self.kappa) * k_sq \
                          + torch.sqrt(disc.clamp_min(0.0))
            gamma_damp = self._damping_safety * torch.clamp(sigma_profile, min=0.0)
            # 20-layer fractional-step: damping per-layer uygulaniyor.
            # LES (RK4) tek step icin tasarlanmis -> n_layers'a bol ki cumulative
            # damping LES'le esit olsun. (CFD-expert teshisi, 2026-04-18)
            gamma_damp = gamma_damp / float(self.n_layers)
        else:
            gamma_damp = torch.zeros_like(k_sq)

        # Buffer guncelle (re-init senaryosu icin)
        if hasattr(self, '_gamma_damp'):
            self._gamma_damp.data.copy_(gamma_damp.to(self._gamma_damp.device))
        else:
            self.register_buffer('_gamma_damp', gamma_damp)

    # ---------------------------------------------------------------- #
    # dt hesaplama                                                      #
    # ---------------------------------------------------------------- #

    def _get_layer_dt(self, layer_idx: int) -> torch.Tensor:
        """Layer-specific dt hesapla.

        Max efektif dt = dt_base * 2.0 * 1.5 = dt_base * 3.0
        (Eski: dt_base * 3.0 * 2.0 = dt_base * 6.0 -- CFL ihlali riski)
        """
        dt = self._dt_base * torch.clamp(self.dt_scale, 0.5, 2.0)
        if self._use_per_layer_dt and hasattr(self, 'dt_mults'):
            if layer_idx < len(self.dt_mults):
                mult = torch.clamp(self.dt_mults[layer_idx], 0.7, 1.5)
                dt = dt * mult
        return dt

    # ---------------------------------------------------------------- #
    # Initial condition                                                 #
    # ---------------------------------------------------------------- #

    def create_initial_condition(
        self, batch_size: int = 1, device: torch.device = torch.device("cpu")
    ) -> ThermalFluidState:
        """Rastgele perturbasyonlu ilk kosul olustur."""
        d = self.config.domain
        shape = (batch_size, d.Nx, d.Ny, d.Nz)

        noise_scale = 0.01
        u = noise_scale * torch.randn(shape, device=device)
        v = noise_scale * torch.randn(shape, device=device)
        w = noise_scale * torch.randn(shape, device=device)

        p = torch.zeros(shape, device=device)

        # LES ile tutarli: noise_scale=0.01 (onceden 0.1 -> 10x buyuk theta perturbation,
        # Nu %70 fazla sapmanin kok sebebi. LES solver line 728 ile hizalandi.)
        theta = noise_scale * torch.randn(shape, device=device)
        theta = theta - theta.mean(dim=(-3, -2, -1), keepdim=True)

        # IC'yi divergence-free yap
        u, v, w, p = self.projections[0](u, v, w)

        # Non-Boussinesq: rho hesapla
        rho = None
        if self.non_boussinesq:
            T_total = self._compute_T_total(theta, device)
            rho = self.density_update(T_total)

        return ThermalFluidState(
            u=u, v=v, w=w, p=p, theta=theta,
            t=torch.zeros(batch_size, device=device),
            rho=rho,
        )

    # ---------------------------------------------------------------- #
    # Forward: 20-layer fractional-step                                 #
    # ---------------------------------------------------------------- #

    def _layer_step(
        self, layer_idx: int,
        u: torch.Tensor, v: torch.Tensor, w: torch.Tensor,
        p: torch.Tensor, theta: torch.Tensor,
        rho: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Tek katman = tek fractional-step zaman adimi (IMEX v2).

        IMEX entegrasyon:
          - Molecular diffusion: IMPLICIT (spectral, unconditionally stable)
          - SGS diffusion + advection + forcing + buoyancy: EXPLICIT
          - u_hat_new = (u_hat + dt * RHS_explicit_hat) / (1 + dt * nu_mol * k^2)

        Optimizasyonlar:
          1. FFT Caching: u_hat/v_hat/w_hat/theta_hat 1 kez hesaplanir (4 rFFT)
          2. Gradient Sharing: 9 velocity gradyan 1 kez hesaplanir
          3. Lamb Form Advection: omega×u (6 FFT)
          4. Leray Projector: div-free projeksiyon tamamen Fourier'de (7 FFT)
          5. IMEX: mol. diffusion 0 ek FFT (spectral division)
        """
        dt = self._get_layer_dt(layer_idx)
        ops = self.ops

        # ---- FFT CACHING ----
        u_hat = safe_rfftn(u)
        v_hat = safe_rfftn(v)
        w_hat = safe_rfftn(w)
        theta_hat = safe_rfftn(theta)

        # ---- GRADIENT SHARING ----
        du_dx, du_dy, du_dz = ops.gradient_from_hat(u_hat)
        dv_dx, dv_dy, dv_dz = ops.gradient_from_hat(v_hat)
        dw_dx, dw_dy, dw_dz = ops.gradient_from_hat(w_hat)
        vel_grads = (du_dx, du_dy, du_dz, dv_dx, dv_dy, dv_dz, dw_dx, dw_dy, dw_dz)

        # -- 1. Advection --
        state_fs = FluidState3D(
            u=u, v=v, w=w, p=p,
            omega_x=torch.zeros_like(u),
            omega_y=torch.zeros_like(u),
            omega_z=torch.zeros_like(u),
            t=torch.zeros(u.shape[0], device=u.device),
        )
        adv_u, adv_v, adv_w = self.advections[layer_idx](
            state_fs, u_hat=u_hat, v_hat=v_hat, w_hat=w_hat,
            vel_grads=vel_grads,
        )

        # -- 2. Source terms: forcing + buoyancy --
        Fx, Fy_f, Fz = self.forcing()

        # TKE-coupled forcing damping
        if self._tke_ref > 0:
            tke_now = 0.5 * (u**2 + v**2 + w**2).mean(dim=(-3, -2, -1))
            damp = (self._tke_ref / (tke_now + 1e-6)).clamp(0.1, 2.0)
            damp = damp.view(-1, 1, 1, 1)
            Fx = Fx * damp
            Fy_f = Fy_f * damp
            Fz = Fz * damp

        if hasattr(self, 'buoyancies'):
            _, Fy_b, _ = self.buoyancies[layer_idx](theta)
        else:
            _, Fy_b, _ = self.buoyancy(theta)

        # -- 3. SGS: MLP eddy viscosity + scale-similarity --
        if self.use_eddy and len(self.eddy_viscosities) > 0:
            eddy = self.eddy_viscosities[layer_idx]

            # compute_all: MLP SGS pathway with Re_normalized and layer_idx
            kappa_t, nu_x, nu_y, nu_z = eddy.compute_all(
                state_fs, self.nu,
                u_hat=u_hat, v_hat=v_hat, w_hat=w_hat,
                vel_grads=vel_grads,
                theta_hat=theta_hat, Ri=self.Ri,
                Re_normalized=self.Re_normalized,
                layer_idx=layer_idx,
            )

            # SGS diffusion: EXPLICIT (nu_t varies spatially)
            # sgs_diff = (nu_eff - nu_mol) * laplacian = nu_t * laplacian
            if self._use_anisotropic_sgs:
                d2u_dx2, d2u_dy2, d2u_dz2 = ops.directional_laplacian_from_hat(u_hat)
                d2v_dx2, d2v_dy2, d2v_dz2 = ops.directional_laplacian_from_hat(v_hat)
                d2w_dx2, d2w_dy2, d2w_dz2 = ops.directional_laplacian_from_hat(w_hat)

                # SGS part only (subtract molecular, which goes into IMEX implicit)
                nu_t_x = nu_x - self.nu
                nu_t_y = nu_y - self.nu
                nu_t_z = nu_z - self.nu

                sgs_diff_u = nu_t_x * d2u_dx2 + nu_t_y * d2u_dy2 + nu_t_z * d2u_dz2
                sgs_diff_v = nu_t_x * d2v_dx2 + nu_t_y * d2v_dy2 + nu_t_z * d2v_dz2
                sgs_diff_w = nu_t_x * d2w_dx2 + nu_t_y * d2w_dy2 + nu_t_z * d2w_dz2

                # Cross-diffusion KAPALI (LES uyumu, ML-expert tespiti):
                # LES sadece nu_t*lap(u) kullaniyor (les_solver:494). Asimetrik
                # form rate-of-strain (div(2*nu_t*S)) DEGIL, momentum korumuyor.
                # Test: kapatilinca slope cok daha iyilesirse buradan suclu.
                # (2026-04-18, fix #5)
            else:
                nu_t = nu_x - self.nu  # isotropic: nu_x = nu_y = nu_z = nu_mol + nu_t
                lap_u = ops.laplacian_from_hat(u_hat)
                lap_v = ops.laplacian_from_hat(v_hat)
                lap_w = ops.laplacian_from_hat(w_hat)
                sgs_diff_u = nu_t * lap_u
                sgs_diff_v = nu_t * lap_v
                sgs_diff_w = nu_t * lap_w
                # Cross-diffusion KAPALI (yukarida aciklama)
        else:
            kappa_t = None
            sgs_diff_u = sgs_diff_v = sgs_diff_w = 0.0

        # -- 4. IMEX velocity update --
        # Explicit RHS: -advection + forcing + buoyancy + SGS_diffusion
        rhs_u = -adv_u + Fx + sgs_diff_u
        rhs_v = -adv_v + Fy_f + Fy_b + sgs_diff_v
        rhs_w = -adv_w + Fz + sgs_diff_w

        # -- 4a. Anizotropik buoyancy damping (sadece v) --
        # rhs_v -= ifft(gamma_damp * fft(v)) -- LES'ten port (Calzavarini 2005)
        if self.Ri > 0 and float(self._gamma_damp.max().item()) > 0.0:
            rhs_v = rhs_v - ops.from_hat(self._gamma_damp * v_hat)

        if self.use_imex:
            # IMEX: molecular diffusion IMPLICIT in spectral space
            # u_hat_new = (u_hat + dt * rhs_hat) / (1 + dt * nu_mol * k^2)
            rhs_u_hat = safe_rfftn(rhs_u)
            rhs_v_hat = safe_rfftn(rhs_v)
            rhs_w_hat = safe_rfftn(rhs_w)

            denom = 1.0 + dt * self.nu * ops.k_squared  # [Nx, Ny, Nz//2+1]

            u = ops.from_hat((u_hat + dt * rhs_u_hat) / denom)
            v = ops.from_hat((v_hat + dt * rhs_v_hat) / denom)
            w = ops.from_hat((w_hat + dt * rhs_w_hat) / denom)
        else:
            # Fallback: fully explicit (legacy)
            mol_diff_u = self.nu * ops.laplacian_from_hat(u_hat)
            mol_diff_v = self.nu * ops.laplacian_from_hat(v_hat)
            mol_diff_w = self.nu * ops.laplacian_from_hat(w_hat)
            u = u + dt * (rhs_u + mol_diff_u)
            v = v + dt * (rhs_v + mol_diff_v)
            w = w + dt * (rhs_w + mol_diff_w)

        # -- 4.5. Soft clamp (SADECE son katman, safety only) --
        # 20-layer cumulative tanh attenuation: (1 - x²/75)^20 ~ %5-10 kayıp/kip.
        # Sadece son layer'da uygula. (ML-expert tespiti, 2026-04-18)
        if layer_idx == self.n_layers - 1:
            _vel_max = 5.0
            u = _vel_max * torch.tanh(u / _vel_max)
            v = _vel_max * torch.tanh(v / _vel_max)
            w = _vel_max * torch.tanh(w / _vel_max)

        # -- 5. Pressure projection (div-free) --
        u, v, w, p = self.projections[layer_idx].forward_leray(u, v, w, dt=dt)

        # -- 6. Thermal advection (cached theta_hat ile) --
        adv_T = self.thermal_advections[layer_idx](
            u, v, w, theta, theta_hat=theta_hat
        )

        # -- 7. Thermal SGS diffusion (explicit) + IMEX molecular --
        # IMEX splitting for thermal:
        #   - Base molecular kappa: IMPLICIT (denom_T = 1 + dt*kappa*k^2)
        #   - kappa_scale deviation from 1.0: EXPLICIT (small correction)
        #   - kappa_t (SGS from MLP): EXPLICIT (spatially varying)
        # This avoids double-counting molecular diffusion that the old code had
        # (ThermalDiffusion3D includes kappa_mol in its output + IMEX denom also has kappa_mol)
        if self.use_imex:
            td = self.thermal_diffusions[layer_idx]

            if td.use_anisotropic:
                sx = torch.clamp(td.kappa_scale_x, 0.1, 20.0)
                sy = torch.clamp(td.kappa_scale_y, 0.1, 20.0)
                sz = torch.clamp(td.kappa_scale_z, 0.1, 20.0)
                d2T_dx2, d2T_dy2, d2T_dz2 = ops.directional_laplacian_from_hat(theta_hat)
                # Molecular base (kappa * 1.0 * lap) goes into IMEX implicit
                # Scale deviation (kappa * (s-1) * lap) + SGS (kappa_t * lap): explicit
                diff_T_explicit = self.kappa * (sx - 1.0) * d2T_dx2 \
                                + self.kappa * (sy - 1.0) * d2T_dy2 \
                                + self.kappa * (sz - 1.0) * d2T_dz2
                if kappa_t is not None:
                    diff_T_explicit = diff_T_explicit + kappa_t * (d2T_dx2 + d2T_dy2 + d2T_dz2)
            else:
                scale = torch.clamp(td.kappa_scale, 0.1, 20.0)
                lap_theta = ops.laplacian_from_hat(theta_hat)
                # Scale deviation + SGS: explicit
                diff_T_explicit = self.kappa * (scale - 1.0) * lap_theta
                if kappa_t is not None:
                    diff_T_explicit = diff_T_explicit + kappa_t * lap_theta

            # Theta explicit RHS
            source_T = v * (1.0 / self.config.domain.Ly)
            rhs_theta = -adv_T + diff_T_explicit + source_T

            # Anizotropik buoyancy damping (theta da, LES ile uyumlu)
            if self.Ri > 0 and float(self._gamma_damp.max().item()) > 0.0:
                rhs_theta = rhs_theta - ops.from_hat(self._gamma_damp * theta_hat)

            # IMEX: implicit molecular thermal diffusion (base kappa only)
            rhs_theta_hat = safe_rfftn(rhs_theta)
            denom_T = 1.0 + dt * self.kappa * ops.k_squared
            theta = ops.from_hat((theta_hat + dt * rhs_theta_hat) / denom_T)
        else:
            # Fallback: fully explicit thermal
            diff_T = self.thermal_diffusions[layer_idx](
                theta, kappa_t=kappa_t, theta_hat=theta_hat
            )
            source_T = v * (1.0 / self.config.domain.Ly)
            rhs_theta_explicit = -adv_T + diff_T + source_T
            if self.Ri > 0 and float(self._gamma_damp.max().item()) > 0.0:
                rhs_theta_explicit = rhs_theta_explicit \
                    - ops.from_hat(self._gamma_damp * theta_hat)
            theta = theta + dt * rhs_theta_explicit

        # -- 9. Gauge fix: mean removal + theta clamp (son katman) --
        theta = theta - theta.mean(dim=(-3, -2, -1), keepdim=True)
        if layer_idx == self.n_layers - 1:
            _theta_max = 2.0
            theta = _theta_max * torch.tanh(theta / _theta_max)

        # -- 10. Elevator mode removal (SADECE son katman, ky=0 filter) --
        # LES tek-step icin tasarlanmis; 20× uygulamak dusuk-k pompalanmasini
        # tamamen koparir. Sadece son layer'da uygula. (ML-expert, 2026-04-18)
        if layer_idx == self.n_layers - 1:
            u = ops.from_hat(safe_rfftn(u) * self._elevator_mask)
            v = ops.from_hat(safe_rfftn(v) * self._elevator_mask)
            w = ops.from_hat(safe_rfftn(w) * self._elevator_mask)
            theta = ops.from_hat(safe_rfftn(theta) * self._elevator_mask)

        return u, v, w, p, theta, rho

    def forward(
        self, state: ThermalFluidState, return_intermediates: bool = False,
    ) -> ThermalFluidState | List[ThermalFluidState]:
        """
        20-layer fractional-step forward pass.

        Args:
            state: Giris ThermalFluidState
            return_intermediates: True ise tum ara state'leri dondur

        Returns:
            Son ThermalFluidState (veya liste)
        """
        u, v, w, p, theta = state.u, state.v, state.w, state.p, state.theta
        rho = state.rho
        intermediates = []

        for i in range(self.n_layers):
            if self.training and self.use_checkpointing:
                # rho'yu da checkpoint'a dahil et
                if rho is not None:
                    u, v, w, p, theta, rho = cp.checkpoint(
                        self._layer_step, i, u, v, w, p, theta, rho,
                        use_reentrant=False,
                    )
                else:
                    result = cp.checkpoint(
                        self._layer_step, i, u, v, w, p, theta,
                        use_reentrant=False,
                    )
                    u, v, w, p, theta, rho = result
            else:
                u, v, w, p, theta, rho = self._layer_step(
                    i, u, v, w, p, theta, rho
                )

            if return_intermediates:
                intermediates.append(ThermalFluidState(
                    u=u, v=v, w=w, p=p, theta=theta,
                    t=state.t + (i + 1) * self._dt_base if state.t is not None else None,
                    rho=rho,
                ))

        t_new = None
        if state.t is not None:
            t_new = state.t + self.n_layers * self._dt_base

        final_state = ThermalFluidState(
            u=u, v=v, w=w, p=p, theta=theta, t=t_new, rho=rho
        )

        if return_intermediates:
            return intermediates
        return final_state

    # ---------------------------------------------------------------- #
    # Helpers                                                           #
    # ---------------------------------------------------------------- #

    def count_parameters(self) -> int:
        """Toplam ogrenilebilir parametre sayisi."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def parameter_summary(self) -> Dict[str, int]:
        """Noron bazinda parametre sayilari."""
        summary = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                prefix = name.split('.')[0]
                summary[prefix] = summary.get(prefix, 0) + param.numel()
        return summary

    def _compute_T_total(self, theta: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Non-Boussinesq: toplam sıcaklık T_total = T_base(y) + θ·dT_nondim.

        Skala konvansiyonu (kritik!)
        ----------------------------
        Phase D'de iki farklı non-dim sıcaklık birleşiyor:

          - T_base = T(y)/T_ref_kelvin: mutlak Kelvin skalasında, ~1.0 mertebesinde
          - θ      = (T - T_ref)/ΔT:    Boussinesq fluctuation skalası, ~O(0.01)

        Ham toplam (eski hatalı form): T_total = T_base + θ
          → Birim karışımı! T_base ~1.0 ile θ ~0.01 toplanırsa
            efektif "gradient hijack": yoğunluk ρ = ρ₀·T₀/T_total ana
            olarak θ tarafından dövülür → patlama riski.

        Doğru form (boyutsal tutarlı):
          T_total* = T_base* + θ · (ΔT/T_ref) = T_base* + θ · dT_nondim

        2026-04-25 düzeltmesi: `theta` çarpan dT_nondim ile boyutlandırıldı.
        non_boussinesq=False default olduğu için runtime'da daha önce
        tetiklenmiyordu, ama Phase D başladığında EOS sapması ciddi olur.
        """
        p = self.config.physics
        d = self.config.domain
        y = torch.linspace(0, d.Ly, d.Ny, device=device)
        dT_nondim = p.dT / p.T_ref_kelvin  # = 20/293.15 ≈ 0.0682
        T_base = 1.0 - y / d.Ly * dT_nondim
        T_base = T_base.view(1, 1, d.Ny, 1)
        # KRİTİK: θ Boussinesq skalasında (ΔT-normalize), T_base T_ref-normalize.
        # Topluyorken θ'yı da T_ref-normalize'a çevirmek için dT_nondim ile çarp.
        return T_base + theta * dT_nondim

    def load_state_dict_compat(self, state_dict: dict, strict: bool = False):
        """
        Eski checkpoint'lardan v2 modele yukle (best-effort).
        v2'de kaldirilan parametreleri atlar:
          cs_low, cs_mid, cs_high, cs_thermal, cs_re_a, cs_re_b,
          buoyancy_damping_strength, local_alpha, local_R_crit,
          thermal_beta, thermal_Ri_crit
        """
        # v2'de kaldirilan parametre isimleri
        _SKIP_KEYS = {
            'cs_re_a', 'cs_re_b', 'buoyancy_damping_strength',
        }
        _SKIP_SUBSTRINGS = [
            'cs_low', 'cs_mid', 'cs_high', 'cs_thermal',
            'local_alpha', 'local_R_crit', 'thermal_beta', 'thermal_Ri_crit',
            'smagorinsky_coeff', 'pr_t',
        ]

        new_sd = OrderedDict()

        for key, val in state_dict.items():
            # Skip removed params
            if key in _SKIP_KEYS:
                continue
            if key == '_elevator_mask':
                continue
            if any(sub in key for sub in _SKIP_SUBSTRINGS):
                continue

            # buoyancy.buoyancy_strength → buoyancies.N.buoyancy_strength
            if key == "buoyancy.buoyancy_strength" and hasattr(self, 'buoyancies'):
                for i in range(self.n_layers):
                    new_sd[f"buoyancies.{i}.buoyancy_strength"] = val.clone()
                continue

            # kappa_scale → kappa_scale_x migration
            if "kappa_scale" in key and "kappa_scale_x" not in key \
               and "kappa_scale_y" not in key and "kappa_scale_z" not in key:
                new_key_x = key.replace("kappa_scale", "kappa_scale_x")
                new_key_y = key.replace("kappa_scale", "kappa_scale_y")
                new_key_z = key.replace("kappa_scale", "kappa_scale_z")
                model_keys = set(k for k, _ in self.named_parameters())
                found = False
                for nk in (new_key_x, new_key_y, new_key_z):
                    if nk in model_keys:
                        new_sd[nk] = val.clone()
                        found = True
                if not found:
                    new_sd[key] = val
                continue

            new_sd[key] = val

        return self.load_state_dict(new_sd, strict=strict)

    def extra_repr(self) -> str:
        return (
            f"n_layers={self.n_layers}, nu={self.nu:.2e}, kappa={self.kappa:.2e}, "
            f"Ri={self.Ri:.2e}, eddy={self.use_eddy}, checkpoint={self.use_checkpointing}, "
            f"non_boussinesq={self.non_boussinesq}, params={self.count_parameters()}"
        )
