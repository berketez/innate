# INNATE3D Matematiksel Formul Dogrulama Raporu

**Tarih:** 2026-02-23
**Hazirlayan:** Codex Consultant Agent
**Dosyalar:** `innate.py` (4744 satir), `train.py` (1045 satir), `model.py` (566 satir)

---

## 1. SPEKTRAL TUREVLER

### 1.1 Dalga Sayisi Hesabi (Wavenumber Vectors)

**Referans:** `torch.fft.fftfreq(N, d=dx)` fonksiyonu n/N dizisi uretir (n = 0, 1, ..., N/2-1, -N/2, ..., -1). `d` parametresi ornek araligini belirtir. Fiziksel dalga sayisi icin `2*pi` carpani gerekir.

**SpectralOps3D (innate.py:848):**
```python
k = fftfreq(resolution, d=domain_size/resolution) * 2 * math.pi
```
- `d = L/N = dx` (grid spacing)
- `fftfreq(N, d=dx)` uretir: `n / (N * dx) = n / L`
- `* 2*pi` ile: `k = 2*pi*n / L`

**SONUC: DOGRU.** Standart Fourier dalga sayisi `k_n = 2*pi*n/L`.

**SpectralOps3DAniso (innate.py:989-991):**
```python
kx_1d = torch.fft.fftfreq(Nx, d=Lx / Nx) * 2 * math.pi
ky_1d = torch.fft.fftfreq(Ny, d=Ly / Ny) * 2 * math.pi
kz_1d = torch.fft.fftfreq(Nz, d=Lz / Nz) * 2 * math.pi
```

**SONUC: DOGRU.** Her yon icin dogru `d = L_i / N_i` kullanilmis. Anisotropik grid icin `kx = 2*pi*n/Lx`, `ky = 2*pi*m/Ly`, `kz = 2*pi*l/Lz`.

### 1.2 Nyquist Modu Sifirlama (innate.py:993-999)

```python
if Nx % 2 == 0:
    kx_1d[Nx // 2] = 0.0
```

**SONUC: DOGRU.** Cift N icin Nyquist modu (`k = N/2`) belirsizdir (cos(N*pi*x/L) olabilir, sin olamaz). Turev alinirken `i*k*f_hat` formulu sin/cos karisimi uretir. Nyquist modunu sifirlamak standart practice'tir. Orsag & Patterson (1972) referansi.

### 1.3 Birinci Turev (Gradient)

**Referans formul:** `d/dx f(x) <--> i*k_x * f_hat(k)`

**SpectralOps3D (innate.py:871-874):**
```python
f_hat = safe_fftn(f)
df_dx = safe_ifftn(1j * self.kx * f_hat).real
df_dy = safe_ifftn(1j * self.ky * f_hat).real
df_dz = safe_ifftn(1j * self.kz * f_hat).real
```

**SONUC: DOGRU.** `i*k_x * f_hat` sonra IFFT ve `.real` almak standart spectral turev yontemidir.

**SpectralOps3DAniso (innate.py:1046-1050):** Ayni formul. **DOGRU.**

### 1.4 Ikinci Turev (Laplacian)

**Referans formul:** `nabla^2 f <--> -(kx^2 + ky^2 + kz^2) * f_hat`

**SpectralOps3D (innate.py:887-888):**
```python
f_hat = safe_fftn(f)
lap_f = safe_ifftn(-self.k_squared * f_hat).real
```

**SONUC: DOGRU.** `k_squared = kx^2 + ky^2 + kz^2`, carpim `-k^2 * f_hat`.

**Directional Laplacian (innate.py:1063-1067):**
```python
d2f_dx2 = safe_ifftn(-(self.kx ** 2) * f_hat).real
d2f_dy2 = safe_ifftn(-(self.ky ** 2) * f_hat).real
d2f_dz2 = safe_ifftn(-(self.kz ** 2) * f_hat).real
```

**SONUC: DOGRU.** Her yon icin ayri `d^2f/dx_i^2 = -k_i^2 * f_hat`.

### 1.5 Curl (Vortisite)

**Referans:**
```
omega_x = dw/dy - dv/dz
omega_y = du/dz - dw/dx
omega_z = dv/dx - du/dy
```

**SpectralOps3D (innate.py:906-913):**
```python
du_dx, du_dy, du_dz = self.gradient(u)
dv_dx, dv_dy, dv_dz = self.gradient(v)
dw_dx, dw_dy, dw_dz = self.gradient(w)
omega_x = dw_dy - dv_dz
omega_y = du_dz - dw_dx
omega_z = dv_dx - du_dy
```

**SONUC: DOGRU.** Ancak verimlilik notu: 3 gradient cagirisi = 3 FFT + 9 IFFT = toplam 12 FFT/IFFT islem. Oysa SpectralOps3DAniso.curl (innate.py:1081-1087) bunu 3 FFT + 3 IFFT = 6 islemde yapiyor. SpectralOps3D.curl gereksiz yere 2x daha yavas. Matematiksel sonuc ayni, performans farki.

**SpectralOps3DAniso.curl (innate.py:1081-1087):**
```python
u_hat = safe_fftn(u)
v_hat = safe_fftn(v)
w_hat = safe_fftn(w)
ox = safe_ifftn(1j * (self.ky * w_hat - self.kz * v_hat)).real
oy = safe_ifftn(1j * (self.kz * u_hat - self.kx * w_hat)).real
oz = safe_ifftn(1j * (self.kx * v_hat - self.ky * u_hat)).real
```

**SONUC: DOGRU.** `omega_x = i*ky*w_hat - i*kz*v_hat` dogru. Daha verimli implementasyon.

### 1.6 Divergence

**Referans:** `div(u) = du/dx + dv/dy + dw/dz`

**SpectralOps3D (innate.py:927-930):**
```python
du_dx, _, _ = self.gradient(u)
_, dv_dy, _ = self.gradient(v)
_, _, dw_dz = self.gradient(w)
return du_dx + dv_dy + dw_dz
```

**SONUC: DOGRU, ama VERIMSIZ.** Her gradient 1 FFT + 3 IFFT yapiyor. Toplam 3 FFT + 9 IFFT, ama sadece 3 IFFT sonucu kullaniliyor. 6 gereksiz IFFT. SpectralOps3DAniso.divergence (innate.py:1089-1097) bunu 3 FFT + 1 IFFT ile yapiyor. Matematiksel olarak ayni sonuc.

### 1.7 Dealiasing (2/3 Kurali)

**Referans:** Nonlineer terimlerde aliasing onlemek icin Fourier modlari `|k| >= N/3` icin kesilir (Orsag, 1971).

**SpectralOps3D (innate.py:856-859):**
```python
k_max = resolution // 3
dealias_mask = (torch.abs(kx) < k_max) & (torch.abs(ky) < k_max) & (torch.abs(kz) < k_max)
```

**DIKKAT - POTANSIYEL SORUN:** Bu maskeleme `kx` ve `ky` ve `kz` uzerinden yapiliyor, ama `kx` burada fiziksel dalga sayisi (2*pi*n/L boyutlu), `k_max` ise `resolution // 3` yani bir tamsayi. Eger `domain_size = 2*pi` ise:
- `kx` degerleri: 0, 1, 2, ..., N/2-1, -N/2, ..., -1 (cunku `fftfreq(N, d=1) * N` = mod indeksleri)
- `k_max = N//3`
- `|kx| < N//3` dogru calisir.

Ama eger `domain_size != 2*pi` ise, `kx = 2*pi*n/L` ve `k_max = N//3` -> birimler tutarsiz!

Ornek: `L = 10.0`, `N = 64`:
- `kx` = `fftfreq(64, d=10/64) * 2*pi` = `n * 2*pi / 10` -> max kx ~ 20
- `k_max = 64 // 3 = 21`
- Bu durumda `|kx| < 21` -> hemen hemen tumu giriyor, dealiasing etkisiz olur.

**SONUC: SpectralOps3D dealiasing domain_size != 2*pi icin HATALI.**

**SpectralOps3DAniso (innate.py:1016-1027):** Bu sinif bunu DOGRU yapiyor:
```python
mx = torch.fft.fftfreq(Nx, d=1.0) * Nx  # mod indeksleri: 0, 1, ..., N/2-1, -N/2, ...
Mx, My, Mz = torch.meshgrid(mx, my, mz, indexing='ij')
dealias_mask = (
    (torch.abs(Mx) < Nx // 3) &
    (torch.abs(My) < Ny // 3) &
    (torch.abs(Mz) < Nz // 3)
)
```

**SONUC: SpectralOps3DAniso dealiasing DOGRU.** Mod indeksleri uzerinden yapiliyor, boyutsuz. `fftfreq(N, d=1.0) * N` = `n` mod indeksleri.

**ONEM:** Model (model.py) SpectralOps3DAniso kullaniyor, dolayisiyla mevcut egitim pipeline'i bu hatadan etkilenmez. Ama SpectralOps3D generic kullanim icin hata iceriyor.

### 1.8 Poisson Cozumu

**Referans:** `nabla^2 p = rhs` --> `p_hat = rhs_hat / (-k^2)`, `k=0` icin `p_hat[0,0,0] = 0` (ortalama basinc keyfi).

**SpectralOps3D (innate.py:949-957):**
```python
rhs_hat = safe_fftn(rhs)
k_sq = self.k_squared.clone()
k_sq[0, 0, 0] = 1.0       # sifira bolme onle
p_hat = rhs_hat / (-k_sq + 1e-10)
p_hat[..., 0, 0, 0] = 0    # ortalama basinc = 0
```

**SUPHE: ISARET VE EPSILON PROBLEMI.**

Sorun 1: `(-k_sq + 1e-10)` yerine `-k_sq` olmali. `1e-10` eklenmesi gereksiz cunku `k_sq[0,0,0] = 1.0` zaten ayarlandi. Daha kotu: buyuk `k_sq` icin sorun yok ama kucuk `k_sq` icin (asla 0 olmayacak ama 1e-10 mertebesinde olabilir) yanlis sonuc uretebilir. Pratikte zarar vermez ama gereksiz.

Sorun 2: `p_hat[..., 0, 0, 0] = 0` dogrudan batch boyutuna uygulanacak mi? `...` ile evet, her batch elemaninin [0,0,0] modu sifirlanir. **DOGRU.**

**SpectralOps3DAniso (innate.py:1104-1109):**
```python
rhs_hat = safe_fftn(rhs)
p_hat = rhs_hat / (-self.k_squared_poisson)  # k_squared_poisson: k^2=0 noktalari 1.0
p_hat[..., 0, 0, 0] = 0.0
```

**SONUC: DOGRU ve TEMIZ.** `k_squared_poisson` init'te hazirlandi, epsilon gerekmez. `rhs_hat / (-k^2)` dogru isaret.

---

## 2. NAVIER-STOKES DENKLEMLERI

### 2.1 Advection: (u.nabla)u

**Referans:**
```
adv_u = u * du/dx + v * du/dy + w * du/dz
adv_v = u * dv/dx + v * dv/dy + w * dv/dz
adv_w = u * dw/dx + v * dw/dy + w * dw/dz
```

**Advection3D (innate.py:2437-2439):**
```python
adv_u = u * du_dx + v * du_dy + w * du_dz
adv_v = u * dv_dx + v * dv_dy + w * dv_dz
adv_w = u * dw_dx + v * dw_dy + w * dw_dz
```

**SONUC: DOGRU.** Convective form `(u.nabla)u` dogru uygulanmis. Dealiasing da sonrasinda yapilmis (2437-2444). Not: Conservative form `div(u otimes u)` degil, convective form. Incompressible akim icin ikisi esittir (`div(u) = 0` oldugunda).

### 2.2 Diffusion: nu * nabla^2(u)

**model.py:398-405:**
```python
# Izotropik
nu_eff = eddy(state_fs, self.nu)
diff_u = nu_eff * self.ops.laplacian(u)

# Anisotropik
diff_u = nu_x * d2u_dx2 + nu_y * d2u_dy2 + nu_z * d2u_dz2
```

**SONUC: DOGRU.** Izotropik mod: `nu_eff * nabla^2(u)`. Anisotropik mod: `sum_i nu_i * d^2u/dx_i^2`. Anisotropik SGS icin standart formul.

**NOT:** Anisotropik difuzyon `div(nu * grad(u))` seklinde yazildiginda, eger `nu` uzaya bagli ise (EddyViscosity3D verisi), tam formul:
```
div(nu * grad(u)) = nu * nabla^2(u) + grad(nu) . grad(u)
```
Koddaki implementasyon `grad(nu)` terimini IHMAL EDIYOR. Bu, `nu_t(x)` uzayda degistiginde bir yaklasim hatasidir. Ancak LES pratikte bu terimi genellikle ihmal eder (sifirinci mertebe yaklasim). **KABUL EDILEBILIR ama yaklasim.**

### 2.3 Pressure Projection

**Referans:**
1. Intermediate velocity: `u* = u + dt*RHS` (advection + diffusion + forcing)
2. Pressure Poisson: `nabla^2 p = (1/dt) * nabla.u*`
3. Correction: `u^{n+1} = u* - dt * nabla(p)`

**Projection3D (innate.py:2595-2622):**
```python
div_u = self.diff_ops.divergence(u, v, w)
# dt verilirse: fractional-step
div_for_poisson = div_u / (dt + 1e-10)    # nabla^2 p = nabla.u* / dt
p = self.diff_ops.solve_poisson(div_for_poisson)
dp_dx, dp_dy, dp_dz = self.diff_ops.gradient(p)
# Correction
u_proj = u - dt * dp_dx
v_proj = v - dt * dp_dy
w_proj = w - dt * dp_dz
```

**DOGRULAMA:**
- `nabla^2 p = nabla.u* / dt` --> `p_hat = (nabla.u*)_hat / (-k^2 * dt)`
- `u^{n+1} = u* - dt * nabla(p)`
- `nabla.u^{n+1} = nabla.u* - dt * nabla^2(p) = nabla.u* - dt * (nabla.u*/dt) = 0`

**SONUC: DOGRU.** Chorin fractional-step projeksiyonu matematiksel olarak tutarli. Helmholtz modu (dt=None) da dogru: `nabla^2 p = nabla.u`, `u_new = u - nabla(p)`.

### 2.4 Velocity Update (Fractional Step)

**model.py:408-413:**
```python
u = u + dt * (-adv_u + Fx + diff_u)
v = v + dt * (-adv_v + Fy_f + Fy_b + diff_v)
w = w + dt * (-adv_w + Fz + diff_w)
u, v, w, p = self.projections[layer_idx](u, v, w, dt=dt)
```

**Referans NS (non-dimensionalized):**
```
du/dt = -(u.nabla)u - nabla(p) + nu*nabla^2(u) + F + Ri*theta*e_y
```

Fractional step:
1. `u* = u + dt * [-(u.nabla)u + nu*nabla^2(u) + F]`
2. Pressure Poisson + correction

**SONUC: DOGRU.** Adveksiyon negatif isaret ("-adv_u") dogru: NS denkleminde adveksiyon terimi `-(u.nabla)u`. Buoyancy Fy_b y-bilesenine eklenmis.

---

## 3. TURBULANSI (SGS Modeli)

### 3.1 Strain Rate Tensor

**Referans:**
```
S_ij = 0.5 * (du_i/dx_j + du_j/dx_i)
```

**EddyViscosity3D._compute_strain_magnitude (innate.py:3742-3759):**
```python
S_xx = du_dx
S_yy = dv_dy
S_zz = dw_dz
S_xy = 0.5 * (du_dy + dv_dx)
S_xz = 0.5 * (du_dz + dw_dx)
S_yz = 0.5 * (dv_dz + dw_dy)
```

**SONUC: DOGRU.** Diyagonal: `S_ii = du_i/dx_i`. Off-diagonal: `S_ij = 0.5*(du_i/dx_j + du_j/dx_i)`.

### 3.2 Strain Rate Magnitude |S|

**Referans:**
```
|S| = sqrt(2 * S_ij * S_ij)
S_ij S_ij = S_xx^2 + S_yy^2 + S_zz^2 + 2*(S_xy^2 + S_xz^2 + S_yz^2)
```

**Kodda (innate.py:3757-3759):**
```python
S_sq = S_xx**2 + S_yy**2 + S_zz**2 + 2*(S_xy**2 + S_xz**2 + S_yz**2)
return torch.sqrt(2 * S_sq + 1e-8)
```

**DOGRULAMA:**
Tam kontraksiyon: `S_ij S_ij = sum_{i,j} S_ij^2`
= `S_xx^2 + S_yy^2 + S_zz^2 + S_xy^2 + S_yx^2 + S_xz^2 + S_zx^2 + S_yz^2 + S_zy^2`
= `S_xx^2 + S_yy^2 + S_zz^2 + 2*S_xy^2 + 2*S_xz^2 + 2*S_yz^2` (simetri: `S_ij = S_ji`)

Kod: `S_sq = S_xx^2 + ... + 2*(S_xy^2 + ...)` -> bu `S_ij S_ij`.
Sonuc: `|S| = sqrt(2 * S_ij S_ij)`.

**SONUC: DOGRU.** `|S| = sqrt(2 * S_ij * S_ij)` standart Smagorinsky formulasyonudur.

### 3.3 Eddy Viscosity

**Referans (Smagorinsky, 1963):**
```
nu_t = (C_s * Delta)^2 * |S|
Delta = (dx * dy * dz)^(1/3)
```

**Tek Cs modu (innate.py:3686-3687):**
```python
C_s = torch.clamp(self.smagorinsky_coeff, 0.05, 0.3)
nu_t = (C_s * self.delta) ** 2 * strain_mag
```

**SONUC: DOGRU.** `(Cs*Delta)^2 * |S|`. Clamp [0.05, 0.3] makul aralik (literatur: Cs ~ 0.1-0.2).

**Filter width (innate.py:3638-3642):**
```python
if grid_spacings is not None:
    dx, dy, dz = grid_spacings
    self.delta = (dx * dy * dz) ** (1.0 / 3.0)
else:
    self.delta = 2 * math.pi / resolution
```

**SONUC: DOGRU.** Anisotropik grid: `Delta = (dx*dy*dz)^(1/3)`. Isotropik fallback: `Delta = L/N = 2*pi/N` (domain_size=2*pi varsayimi).

### 3.4 Frekans-Band SGS (innate.py:3676-3688)

```python
nu_t_low  = (cs_l * self.delta)**2 * self.diff_ops.band_filter(strain_mag, 'low')
nu_t_mid  = (cs_m * self.delta)**2 * self.diff_ops.band_filter(strain_mag, 'mid')
nu_t_high = (cs_h * self.delta)**2 * self.diff_ops.band_filter(strain_mag, 'high')
nu_t = nu_t_low + nu_t_mid + nu_t_high
```

**ANALIZ:** Bu orijinal bir yaklasim. Standart Smagorinsky'de tek Cs var. Burada strain magnitude frekans bandlarina ayrilip her biri icin farkli Cs kullaniliyor. Mantigi: dusuk frekanslarda (buyuk yapilar) az viskozite, yuksek frekanslarda (kucuk yapilar) cok viskozite.

**POTANSIYEL SORUN:** `strain_mag` kendisi zaten nonlineer (`|S| = sqrt(2*S_ij*S_ij)`). Band filter uygulayinca `band_filter(|S|)` elde ediyoruz, ama `|S|` nin Fourier uzayinda band filtrelenmesi fiziksel olarak `|S|` nin o frekans bilesenlerini verir. Bu fiziksel olarak sorgulanabilir -- standart yaklasim velocity field'i filtreleyip sonra strain hesaplamak olurdu. Ama bu bir yaklasimdir ve egitim sirasinda ogrenilebilir parametreler bunu telafi edebilir.

**SONUC: KABUL EDILEBILIR YAKLASIM.** Standart degil ama calisabilir.

### 3.5 Backscatter (innate.py:3698-3701)

```python
bs = torch.clamp(self.backscatter_coeff, -0.02, 0.0)
nu_t = nu_t + bs * self.delta**2 * strain_mag
```

**ANALIZ:** `nu_t_final = nu_t + bs * Delta^2 * |S|` = `(Cs^2 + bs) * Delta^2 * |S|`. `bs in [-0.02, 0]`, yani `Cs^2 - 0.02` olabilir. `Cs_min = 0.05` iken `Cs^2 = 0.0025`. Bu durumda: `0.0025 - 0.02 = -0.0175 < 0` --> **NEGATIF EFEKTIF VISKOZITE MUMKUN!**

`forward()` fonksiyonunda backscatter sonrasi `nu_t`'ye herhangi bir pozitiflik clampi yok. Doner deger: `nu_molecular + nu_t`, burada `nu_t` negatif olabilir. Eger `nu_molecular + nu_t < 0` ise, efektif negatif difuzyon = ANTI-DIFFUSION = numerik kararsizlik.

**SONUC: TEHLIKELI.** Backscatter clampi [-0.02, 0] ile nu_t negatif olabilir ve hicbir guard yok. Debugger notu (discussion.md) bu sorunu zaten tanimlamis.

### 3.6 SGS Dissipation (innate.py:3761-3766)

```python
def sgs_dissipation(self, state, nu_molecular=0.0):
    nu_t = nu_eff - nu_molecular
    strain_mag = self._get_strain_mag(state)
    return 2 * nu_t * strain_mag**2
```

**Referans:** `eps_sgs = 2 * nu_t * S_ij * S_ij = nu_t * |S|^2` (cunku `|S| = sqrt(2*S_ij*S_ij)` -> `|S|^2 = 2*S_ij*S_ij`).

Kod: `2 * nu_t * strain_mag^2` = `2 * nu_t * 2 * S_ij*S_ij` = `4 * nu_t * S_ij*S_ij`.

**SONUC: HATALI (2x fazla).** Dogru formul: `eps_sgs = 2 * nu_t * S_ij * S_ij`. Koddaki `strain_mag = |S| = sqrt(2*S_ij*S_ij)`, yani `strain_mag^2 = 2*S_ij*S_ij`. Bu durumda `2 * nu_t * strain_mag^2 = 2 * nu_t * 2 * S_ij*S_ij = 4*nu_t*S_ij*S_ij`. Bu 2 KATI fazla.

Dogru implementasyon: `return nu_t * strain_mag**2` (cunku strain_mag icinde zaten 2 var).

**ONEM:** Bu fonksiyon su anda loss hesabinda (train.py) kullanilmiyor gibi gorunuyor -- dissipation_loss dogrudan enstrophy uzerinden hesaplaniyor. Ama diagnostic araclardan cagrilirsa yanlis sonuc verir.

---

## 4. TERMAL NORONLAR

### 4.1 Buoyancy Kuvveti

**Referans (Boussinesq):** `F_buoy = Ri * theta * e_y`

**Buoyancy3D (innate.py:3908-3911):**
```python
strength = torch.clamp(self.buoyancy_strength, 0.0, 50.0)
Fy = self.Ri * strength * theta
return zeros, Fy, zeros
```

**SONUC: DOGRU.** `F_y = Ri * strength * T'`, `F_x = F_z = 0`. Buoyancy sadece dikey (y-yonu). `buoyancy_strength` ogrenilebilir skaler carpan (init=1, clamp [0,50]).

### 4.2 Thermal Diffusion

**Referans:** `kappa * nabla^2(T')`

**ThermalDiffusion3D izotropik (innate.py:3973-3979):**
```python
scale = torch.clamp(self.kappa_scale, 0.1, 5.0)
lap = self.ops.laplacian(theta)
if has_kappa_t:
    return (self.kappa * scale + kappa_t) * lap
else:
    return self.kappa * scale * lap
```

**SONUC: DOGRU.** `(kappa*scale + kappa_t) * nabla^2(T')`. `kappa_t` = turbulent termal difuzivite (EddyViscosity3D'den).

**Anisotropik (innate.py:3961-3972):**
```python
diff = (self.kappa * sx + kappa_t) * d2T_dx2 \
     + (self.kappa * sy + kappa_t) * d2T_dy2 \
     + (self.kappa * sz + kappa_t) * d2T_dz2
```

**SONUC: DOGRU.** `kappa_eff_i * d^2T'/dx_i^2` her yon icin ayri efektif difuzivite.

### 4.3 Thermal Advection

**Referans:** `(u.nabla)T' = u*dT'/dx + v*dT'/dy + w*dT'/dz`

**ThermalAdvection3D (innate.py:4022-4028):**
```python
dT_dx, dT_dy, dT_dz = self.ops.gradient(theta)
adv = u * dT_dx + v * dT_dy + w * dT_dz
adv = self.ops.dealias(adv)
```

**SONUC: DOGRU.** Convective form + dealiasing.

### 4.4 Thermal Update

**model.py:422:**
```python
theta = theta + dt * (-adv_T + diff_T)
```

**Referans:** `dT'/dt = -(u.nabla)T' + kappa_eff * nabla^2(T')`

**SONUC: DOGRU.** Negatif isaret adveksiyon icin dogru.

### 4.5 Density Update (Non-Boussinesq)

**Referans (ideal gaz, sabit p0):** `rho = rho_0 * T_0 / T_total`

**DensityUpdate3D (innate.py:4060-4072):**
```python
T_safe = torch.clamp(T_total, min=0.01)
rho = self.rho_0 * self.T_0 / T_safe
return torch.clamp(rho, 0.5 * self.rho_0, 2.0 * self.rho_0)
```

**SONUC: DOGRU.** Ideal gaz durum denklemi `p = rho*R*T`, sabit `p_0 = rho_0*R*T_0` ile `rho = rho_0*T_0/T`. Nondim: `R = 1`.

---

## 5. NUSSELT NUMBER

**Referans (konvektif Nusselt):**
```
Nu = 1 + <v * T'> / (kappa * Delta_T / L_y)
```
Burada `Delta_T` ust-alt sicaklik farki.

**ThermalFluidState.nusselt_number (model.py:91-97):**
```python
def nusselt_number(self, Ly: float, kappa: float) -> torch.Tensor:
    vT = (self.v * self.theta).mean(dim=(-3, -2, -1))
    return 1.0 + vT / (kappa + 1e-10)
```

**SORUN: `Delta_T / L_y` EKSIK.**

Referans formul: `Nu = 1 + <v*T'> / (kappa * dT/dy_wall)`. Burada `dT/dy_wall = Delta_T / L_y`.

Koddaki formul: `Nu = 1 + <v*T'> / kappa`. Bu sadece `Delta_T / L_y = 1` oldugunda dogru.

Nondimensionalization'a bagli: eger sicaklik `T_0` ile boyutsuzlastirilmis ve `L_y = 1` alinmissa sorun yok. Ama config'te `Ly != 1` (ornegin `Ly = 10.0`) ve `dT != 1` ise, Nusselt number YANLIS olacaktir.

Nusselt loss'undaki `nusselt_number(Ly, kappa)` cagrisinda `Ly` parametresi veriliyor ama fonksiyon icinde KULLANILMIYOR.

**SONUC: SUPHE - NONDIM'E BAGIMLI.** Eger problem tamamen boyutsuz (`Ly=1`, `Delta_T=1`) ise dogru. Aksi halde `Nu = 1 + <v*T'> * Ly / (kappa * Delta_T)` olmali. Fonksiyon `Ly` parametresini aliyor ama kullanmiyor -- bu bir hataya isaret ediyor.

---

## 6. LOSS FONKSIYONLARI

### 6.1 Divergence Loss (train.py:168-174)

```python
div = self.ops.divergence(state.u, state.v, state.w)
return div.pow(2).mean() + 0.1 * div.abs().mean()
```

**SONUC: DOGRU.** L2 + 0.1*L1 penalty. Standart incompressibility constraint.

### 6.2 Energy Balance Loss (train.py:176-223)

```python
dEdt = (E1 - E0) / dt
Z = 0.5 * (self._enstrophy(s0) + self._enstrophy(s1))
eps = 2.0 * phys.nu * Z
P_f = (Fx * u_mid + Fy * v_mid + Fz * w_mid).mean(dim=(-3, -2, -1))
P_b = phys.Ri * (v_mid * theta_mid).mean(dim=(-3, -2, -1))
residual = (dEdt + eps - P_f - P_b).abs()
```

**Referans:**
```
dE/dt = -2*nu*Z + P_forcing + P_buoyancy
=>  dE/dt + 2*nu*Z - P_f - P_b = 0
```

Burada `E = 0.5*<u^2>`, `Z = <omega^2>` (enstrophy), `eps = 2*nu*Z`.

**DIKKAT:** `_enstrophy` fonksiyonu (train.py:152-154):
```python
ox, oy, oz = self.ops.curl(state.u, state.v, state.w)
return (ox**2 + oy**2 + oz**2).mean(dim=(-3, -2, -1))
```
Bu `<omega^2>` veriyor, `0.5*<omega^2>` degil. Standart enstrophy tanimi `Z = 0.5*<omega^2>` veya `Z = <omega^2>` olabilir (konvansiyona bagli).

`eps = 2*nu*Z = 2*nu*<omega^2>`. Dissipation rate `epsilon = nu*<omega^2>` (Tennekes & Lumley) veya `epsilon = 2*nu*S_ij*S_ij`. Homogeneous isotropic turbulansta `<omega^2> = 2*S_ij*S_ij`, dolayisiyla `nu*<omega^2> = 2*nu*S_ij*S_ij = epsilon`.

Koddaki: `eps = 2*nu*<omega^2>`. Eger `epsilon = nu*<omega^2>` ise bu **2 KATI FAZLA**. Eger enstrophy `Z = 0.5*<omega^2>` tanimiyla kullaniliyorsa `2*nu*Z = nu*<omega^2> = epsilon` **DOGRU** olur.

`_enstrophy` fonksiyonu `<omega^2>` donuyor (0.5 carpani yok). Energy balance'ta `eps = 2*nu*Z` ile `Z = <omega^2>` ise `eps = 2*nu*<omega^2>` = **2*epsilon**. Bu HATALI.

**Kiyaslama:** `FluidState3D.enstrophy()` (innate.py:353-356):
```python
omega_sq = self.omega_x**2 + self.omega_y**2 + self.omega_z**2
return 0.5 * omega_sq.mean(dim=(-3, -2, -1))
```
Bu `0.5*<omega^2>` = standart enstrophy. Ama `PhysicsLoss._enstrophy` **0.5 CARPANI YOK**.

**SONUC: TUTARSIZLIK.** `PhysicsLoss._enstrophy()` ve `FluidState3D.enstrophy()` farkli tanimlar kullaniyor. Energy balance icin:
- Eger dissipation = `nu * <omega^2>` ise, `eps = 2*nu*_enstrophy()` = `2*nu*<omega^2>` = **2x HATALI**
- Dogru kullanim: `eps = nu * _enstrophy()` (cunku `_enstrophy = <omega^2>`)
- VEYA `_enstrophy`'yi `0.5*<omega^2>` yap, sonra `eps = 2*nu*_enstrophy = nu*<omega^2>` dogru olur.

**KRITIK HATA: Energy balance loss'ta dissipation 2 kati fazla hesaplaniyor.** Bu, modelin dissipation'i underestimate etmesine, yani daha az turbulansi damplatmasina neden olur -- fiziksel olarak yanlis yone iter.

### 6.3 Spectrum Loss (train.py:56-131)

```python
E_hat = 0.5 * (u_hat.abs()**2 + v_hat.abs()**2 + w_hat.abs()**2)
E_hat = E_hat / N_total**2
```

**Referans (Parseval):** 3D DFT'nin unnormalized versiyonunda `sum |x[n]|^2 = (1/N)*sum |X[k]|^2`. PyTorch'un `fftn` unnormalized DFT kullanir: `X[k] = sum_n x[n] * exp(-2*pi*i*n*k/N)`. Parseval teoremi: `sum |x|^2 = (1/N)*sum |X|^2`. 3D'de: `sum |x|^2 = (1/(Nx*Ny*Nz))*sum |X|^2`.

Enerji yogunlugu: `E = 0.5*<|u|^2>` = `0.5/(Nx*Ny*Nz) * sum |u|^2`
Fourier'de: `= 0.5/(Nx*Ny*Nz)^2 * sum |u_hat|^2`

Kod: `E_hat = 0.5 * |u_hat|^2 / N_total^2` = `0.5 * |u_hat|^2 / (Nx*Ny*Nz)^2`

**SONUC: DOGRU.** Normalizasyon dogru uygulanmis.

**Spectrum slope (train.py:104-131):**
```python
slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2 + 1e-10)
target_slope = -5.0 / 3.0
return (slope - target_slope).pow(2)
```

**SONUC: DOGRU.** Log-log uzayinda linear regression ile slope fit. Kolmogorov -5/3 yasasi hedef. Standart yaklasim.

### 6.4 Dissipation Loss (train.py:238-260)

```python
Z = self._enstrophy(state)   # <omega^2>
eps = 2.0 * phys.nu * Z      # 2*nu*<omega^2>

eps_spectral = 2.0 * phys.nu * (self.ops.k_squared * E_hat).sum(dim=(-3,-2,-1))
```

**SORUN:** Ayni 2x carpan hatasi (6.2'deki gibi). `eps` fiziksel-uzay hesaplamasi enstrophy bazli, `eps_spectral` ise `2*nu*sum(k^2*E_hat)`.

Spectral dissipation: `epsilon = 2*nu*sum_k k^2 * E_hat(k)`. Burada `E_hat(k) = 0.5*|u_hat|^2/N^2`. Yani `sum(k^2 * E_hat)` = `sum(k^2 * 0.5*|u_hat|^2/N^2)`.

Parseval'dan: `sum_k k^2 * |u_hat|^2 / N^2 = <|nabla u|^2>`.
Isotropik turbulansta: `<|nabla u|^2> = <omega^2>` (incompressible icin).

Yani `2*nu*sum(k^2 * E_hat)` = `2*nu*0.5*<|nabla u|^2>` = `nu*<omega^2>` = `epsilon`.

Ama `eps = 2*nu*_enstrophy = 2*nu*<omega^2>` = `2*epsilon`.

**SONUC: TUTARSIZ ama kendisiyle TUTARLI.** Her iki taraf da ayni yanlis 2x carpani kullaniyorsa, fark = 0 olur. Ama burada bir taraf `2*nu*<omega^2>` (yanlis), diger taraf `2*nu*sum(k^2*E_hat)` = `2*nu*0.5*<omega^2>` = `nu*<omega^2>` (dogru). **FARK SIFIR OLMAZ.**

Detayli: `eps - eps_spectral` = `2*nu*<omega^2> - nu*<omega^2>` = `nu*<omega^2>` = `epsilon`.
Bu hiçbir zaman sifir olmayacak! Loss her zaman epsilon kadar olacak.

**SONUC: HATALI.** Dissipation loss her zaman sifirdan farkli olacak cunku iki taraf farkli normalizations kullaniyor. Fix: her iki tarafta da ayni normalization kullan.

### 6.5 Nusselt Loss (train.py:262-279)

```python
Nu = state.nusselt_number(Ly, kappa)
Nu_target = max(0.069 * Ra**(1/3) * Pr**0.074, 2.0)
loss_floor = relu(1 - Nu).pow(2).mean()
loss_target = 0.1 * ((Nu - Nu_target) / Nu_target).pow(2).mean()
```

**SONUC: FORMUL DOGRU, ama Nusselt hesabi kusurlu (Bolum 5'e bak).** Globe-Dropkin korelasyonu `Nu = 0.069*Ra^(1/3)*Pr^0.074` standart RB konveksiyonu korelasyonudur. Bidirectional loss (floor + target) iyi tasarlanmis.

---

## 7. FORCING

### 7.1 Kolmogorov Forcing

**Referans:** `F_x = A * sin(k_f * y)`, `F_y = F_z = 0`

**Forcing3D (innate.py:3844-3846):**
```python
arg = self.k_f * 2.0 * math.pi * self.y_grid / self.Ly + self.phi
Fx = A * torch.sin(arg)
```

**SONUC: DOGRU.** `F_x = A * sin(k_f * 2*pi*y/Ly + phi)`. `2*pi/Ly` carpani dalga sayisini fiziksel moda cevirir: `k_f` modluk Kolmogorov forcing.

### 7.2 Harmonikler (innate.py:3853-3856)

```python
y_norm = 2.0 * math.pi * self.y_grid / self.Ly
Fx = Fx + self.amplitude_k2 * torch.sin(2 * self.k_f * y_norm + self.phase_k2)
Fx = Fx + self.amplitude_k3 * torch.sin(3 * self.k_f * y_norm + self.phase_k3)
```

**SONUC: DOGRU.** Ikinci ve ucuncu harmonikler. Init=0 ile baslangicta etkisiz.

---

## 8. VORTICITY (Vortisite Denklemi)

### 8.1 Vortex Stretching

**Referans:**
```
d(omega)/dt = -(u.nabla)omega + (omega.nabla)u + nu*nabla^2(omega)
```

**Vorticity3D (innate.py:2515-2525):**
```python
# Adveksiyon: -u.nabla(omega)
adv_omega_x = -(u * domega_x_dx + v * domega_x_dy + w * domega_x_dz)
# Stretching: omega.nabla(u)
stretch_x = omega_x * du_dx + omega_y * du_dy + omega_z * du_dz
stretch_y = omega_x * dv_dx + omega_y * dv_dy + omega_z * dv_dz
stretch_z = omega_x * dw_dx + omega_y * dw_dy + omega_z * dw_dz
```

**SONUC: DOGRU.** Vortisite adveksiyonu negatif isaret ve vortex stretching formulleri dogru. Stretching terimi 3D'ye ozgu (2D'de sifir).

---

## 9. MODEL FORWARD PASS (model.py)

### 9.1 Fractional Step Sirasi

```
model.py:_layer_step sirasi:
1. Density update (Non-Boussinesq)
2. Advection
3. Source terms (forcing + buoyancy)
4. Effective viscosity + diffusion
5. Velocity update: u* = u + dt*(-adv + F + diff)
6. Pressure projection: u^{n+1} = u* - dt*nabla(p)
7. Thermal advection
8. Thermal diffusion
9. Theta update: theta^{n+1} = theta + dt*(-adv_T + diff_T)
10. Gauge fix: mean removal
```

**SONUC: DOGRU SIRALAMA.** Standart fractional-step: once intermediate velocity, sonra pressure correction. Thermal equation ayri integrate ediliyor (operator splitting). Bu Chorin-type projeksiyonun standart uzantisidir.

### 9.2 dt Hesabi

```python
def _get_layer_dt(self, layer_idx):
    dt = self._dt_base * torch.clamp(self.dt_scale, 0.5, 2.0)
    if self._use_per_layer_dt:
        mult = torch.clamp(self.dt_mults[layer_idx], 0.7, 1.5)
        dt = dt * mult
    return dt
```

Max efektif dt = `dt_base * 2.0 * 1.5 = dt_base * 3.0`.

**SONUC: MAKUL.** Onceki versiyon `dt_base * 6.0` olasiligi vardi (discussion.md'de debugger belirtmis). Simdi max 3.0x, daha guvenli.

---

## 10. OZET TABLOSU

| # | Formul | Dosya:Satir | Sonuc | Aciklama |
|---|--------|-------------|-------|----------|
| 1 | Wavenumber k = 2*pi*n/L | innate.py:848,989 | DOGRU | fftfreq kullanimi dogru |
| 2 | Nyquist sifirlama | innate.py:993-999 | DOGRU | Standart practice |
| 3 | df/dx = IFFT(i*k*f_hat) | innate.py:871-874 | DOGRU | Spectral turev |
| 4 | nabla^2 f = IFFT(-k^2*f_hat) | innate.py:887-888 | DOGRU | Spectral Laplacian |
| 5 | Curl bileselenleri | innate.py:910-913 | DOGRU | Standart curl formulu |
| 6 | Divergence | innate.py:927-930 | DOGRU | Standart divergence |
| 7a | Dealiasing 2/3 (SpectralOps3D) | innate.py:856-859 | **HATALI** | L != 2*pi'de yanlis (boyut uyumsuzlugu) |
| 7b | Dealiasing 2/3 (Aniso) | innate.py:1016-1027 | DOGRU | Mod indeksi bazli |
| 8 | Poisson cozumu | innate.py:1104-1109 | DOGRU | -rhs_hat/k^2, k=0 -> p=0 |
| 9 | Advection (u.nabla)u | innate.py:2437-2439 | DOGRU | Convective form |
| 10 | Diffusion nu*nabla^2(u) | model.py:398-405 | DOGRU* | *grad(nu) terimi ihmal |
| 11 | Pressure projection | innate.py:2595-2622 | DOGRU | Chorin method |
| 12 | Strain rate S_ij | innate.py:3750-3755 | DOGRU | 0.5*(du_i/dx_j + du_j/dx_i) |
| 13 | \|S\| = sqrt(2*S_ij*S_ij) | innate.py:3757-3759 | DOGRU | Standart formul |
| 14 | nu_t = (Cs*Delta)^2*\|S\| | innate.py:3686-3687 | DOGRU | Smagorinsky |
| 15 | Filter width Delta | innate.py:3638-3642 | DOGRU | (dx*dy*dz)^(1/3) |
| 16 | Backscatter | innate.py:3698-3701 | **TEHLIKELI** | Negatif nu_t mumkun, guard yok |
| 17 | SGS dissipation | innate.py:3761-3766 | **HATALI** | 2x fazla (2*nu_t*\|S\|^2 degil, nu_t*\|S\|^2 olmali) |
| 18 | Buoyancy F = Ri*theta*e_y | innate.py:3908-3911 | DOGRU | Boussinesq |
| 19 | Thermal diffusion | innate.py:3949-3979 | DOGRU | kappa*nabla^2(T') |
| 20 | Thermal advection | innate.py:4022-4028 | DOGRU | (u.nabla)T' |
| 21 | Density rho = rho_0*T_0/T | innate.py:4060-4072 | DOGRU | Ideal gaz EOS |
| 22 | Nusselt number | model.py:91-97 | **SUPHE** | Ly/DeltaT normalization eksik |
| 23 | Energy balance loss | train.py:176-223 | **HATALI** | eps = 2*nu*<omega^2> yerine nu*<omega^2> olmali |
| 24 | Dissipation loss | train.py:238-260 | **HATALI** | Iki taraf farkli normalization |
| 25 | Spectrum E(k) | train.py:56-101 | DOGRU | Parseval normalization dogru |
| 26 | Kolmogorov -5/3 | train.py:104-131 | DOGRU | Log-log slope fit |
| 27 | Nusselt loss | train.py:262-279 | DOGRU* | *Nusselt hesabina bagimli |
| 28 | Vortex stretching | innate.py:2520-2525 | DOGRU | omega.nabla(u) |
| 29 | Forcing (Kolmogorov) | innate.py:3844-3856 | DOGRU | A*sin(k_f*2*pi*y/Ly) |
| 30 | Velocity update | model.py:408-413 | DOGRU | Fractional-step |

---

## 11. KRITIK HATALAR VE ONERILER

### Oncelik 1: Energy Balance Loss (KRITIK)
- **Sorun:** `_enstrophy()` -> `<omega^2>`, sonra `eps = 2*nu*<omega^2>` = 2*epsilon
- **Fix:** `eps = phys.nu * Z` (2.0 carpanini kaldir)
- **Etki:** Model dissipation'i overestimate ediyor -> turbulansi yeterince damplatmiyor

### Oncelik 2: Dissipation Loss (KRITIK)
- **Sorun:** Fiziksel ve spectral dissipation farkli normalization
- **Fix:** Her ikisinde de ayni normalization kullan. En basiti:
  ```python
  eps = phys.nu * Z  # <omega^2>
  eps_spectral = phys.nu * (self.ops.k_squared * E_hat).sum(...) * 2  # veya normalization duzelt
  ```
- **Etki:** Loss hiçbir zaman sifir olamaz, model surekli gradient aliyor

### Oncelik 3: Nusselt Number (ORTA)
- **Sorun:** `Ly` parametresi alinip kullanilmiyor
- **Fix:** `return 1.0 + vT * Ly / (kappa * delta_T + 1e-10)` (delta_T bilgisi gerekli)
- **Etki:** Nondimensionalization'a bagimli. Eger problem `Ly=1`, `dT=1` ise sorun yok.

### Oncelik 4: Backscatter Guard (ORTA)
- **Sorun:** `nu_t` negatif olabilir
- **Fix:** `nu_t = torch.clamp(nu_t, min=0.0)` ekle (veya `min=1e-8`)
- **Etki:** Anti-diffusion -> numerik kararsizlik -> NaN

### Oncelik 5: SGS Dissipation Fonksiyonu (DUSUK)
- **Sorun:** 2x fazla hesaplama
- **Fix:** `return nu_t * strain_mag**2`
- **Etki:** Su anda loss'ta kullanilmiyor, sadece diagnostic

### Oncelik 6: SpectralOps3D Dealiasing (DUSUK)
- **Sorun:** `domain_size != 2*pi` icin yanlis
- **Fix:** Mod indeksi bazli maskeleme yap (SpectralOps3DAniso'daki gibi)
- **Etki:** Model SpectralOps3DAniso kullaniyor, bu sinif kullanilmiyor

---

## 12. GENEL DEGERLENDIRME

Projenin matematiksel altyapisi genel olarak saglamdir. Spectral turevler, Poisson cozucusu, adveksiyon, projeksiyon ve SGS model formulleri dogru uygulanmis. **Kritik sorunlar loss fonksiyonlarinda:** enstrophy normalization tutarsizligi energy balance ve dissipation loss'larini bozuyor. Bu loss'lar modelin fizik ogrenme yetenegini dogrudan etkiler -- duzeltilmesi yuksek onceliklidir.

Fizik noronlarinin kendileri (Advection3D, Projection3D, EddyViscosity3D, Buoyancy3D vb.) matematiksel olarak dogrudur. Tasarim kararlari (learnable modulators, clamp araliklari) makuldur.
