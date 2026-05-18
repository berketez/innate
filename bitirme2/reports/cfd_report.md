# INNATE3D Mixed Convection -- CFD Fizik Dogruluk Raporu

**Tarih:** 2026-02-23
**Hazirlayan:** CFD Expert Agent
**Proje:** 3D Mixed Convection (Rayleigh-Benard + Forced Convection)
**Parametreler:** Re=5000, Ra=1e6, Pr=0.71, Domain: 6x10x4, Grid: 96x160x64

---

## 0. Yonetici Ozeti

285 parametreli saf INNATE mimarisi genel olarak **fiziksel bakimdan saglikli** bir yapi sergiliyor. Navier-Stokes, Boussinesq, pressure Poisson, termal denklem ve SGS model formulasyonlarinin hepsi matematiksel olarak dogru. Ancak asagida onem sirasina gore siralanan **7 sorun** tespit ettim. Bunlarin 2 tanesi **kritik** (simülasyonu bozar), 3 tanesi **orta** (sonuclari bozar ama patlamaz), 2 tanesi **dusuk** (iyilestirme onerisi).

---

## 1. KRITIK SORUNLAR

### 1.1 CFL IHLALI -- dt Cok Buyuk, Patlamaya Acik

**Dosya:** `/Users/apple/Desktop/nsneuron/bitirme2/model.py`, satirlar 275-286
**Dosya:** `/Users/apple/Desktop/nsneuron/bitirme2/config.py`, satir 57

**Mevcut durum:**
```python
# config.py
dt: float = 0.01

# model.py _get_layer_dt()
dt = self._dt_base * torch.clamp(self.dt_scale, 0.5, 2.0)
# per-layer: mult clamp [0.7, 1.5]
# Max efektif dt = 0.01 * 2.0 * 1.5 = 0.03
```

**CFL analizi:**

| Parametre | Deger |
|-----------|-------|
| dx = dy = dz | 0.0625 |
| nu = 1/Re | 2.0e-4 |
| dt_base | 0.01 |
| dt_max (2.0 * 1.5) | 0.03 |
| Advective CFL (u=1) | 0.16 |
| Advective CFL (u=5) | 0.80 |
| Advective CFL (u=10, dt_max) | **4.80** |
| Diffusive CFL | 5.12e-4 (ihmal edilebilir) |

u_max=20 clamp sinirinda ve dt_max=0.03 ile CFL = 20 * 0.03 / 0.0625 = **9.6**. Bu acik explicit Euler icin tam bir felaket. CFL > 1 ise explicit zaman integrasyonu **kacinilmaz olarak kararsizidir**.

**Ne olmasi gerekiyor:**
Explicit Euler (forward Euler) icin advektif CFL < 1 sarttir. Ideal olarak CFL ~ 0.3-0.5. Bu da dt_max ~ 0.003 gerektirir (u_max=10 varsayimiyla).

Alternatif: Implicit difuzyon (IMEX sema) ile advektif CFL ~ 1.0'a izin verilebilir, ama mevcut kod tam explicit.

**Etki:** Hiz alani buyudukce (turbulans gelistikce) CFL > 1 olur ve sayi patlamasi (NaN) baslar. Bu, 63 saatlik egitimin step 280'de NaN vermesinin **bir numarali sebebi**. Debugger'in tespiti ile uyumlu.

**Onerilen fix:**
```python
# dt_base'i 0.003'e dusur VEYA
# dt_scale clamp araligi [0.1, 1.0] yap
# Ya da adaptif CFL-based dt kullan:
def _get_layer_dt(self, layer_idx, u, v, w):
    u_max = torch.amax(torch.abs(u) + torch.abs(v) + torch.abs(w))
    dt_cfl = 0.3 * self.dx_min / (u_max + 1e-8)
    dt = torch.minimum(self._dt_base * self.dt_scale, dt_cfl)
    return dt
```

---

### 1.2 Advection3D Skew-Symmetric Formdan Yoksun -- Enerji Korunumu Bozuk

**Dosya:** `/Users/apple/Desktop/nsneuron/innate.py`, satirlar 2429-2457

**Mevcut durum (Advection3D):**
```python
# Sadece convective form: (u . nabla) u
adv_u = u * du_dx + v * du_dy + w * du_dz
adv_v = u * dv_dx + v * dv_dy + w * dv_dz
adv_w = u * dw_dx + v * dw_dy + w * dw_dz
```

**Karsilastirma (2D Advection class):**
```python
# 2D versiyonda skew-symmetric form VAR (satirlar 1321-1343):
# 0.5 * (convective + divergence form)
conv_u = u * du_dx + v * du_dy
div_form_u = d(uu)/dx + d(vu)/dy
adv_u = 0.5 * (conv_u + div_form_u)
```

**Ne olmasi gerekiyor:**
Incompressible akista (nabla . u = 0), convective ve divergence formlar matematiksel olarak esdegerdir. Ancak **ayriktirilmis** (discretized) durumda, yuvarlama hatalari nedeniyle divergence tam olarak sifir degildir. Skew-symmetric form:

    N_s(u) = 0.5 * [ (u . nabla)u + nabla . (u otimes u) ]

kinetik enerjiyi **TAM OLARAK** korur (ayrik seviyede bile), cunku ek anti-simetri korunur. Bu, uzun sureli (binlerce adim) autoregressive unrolling'de **enerji drift'ini** onler.

2D versiyon bunu dogru yapiyor, 3D versiyon yapmiyOR.

**Etki:** Uzun unrolling'de (20 layer * 10 step = 200 adim) kinetik enerjide sahte birikim ya da kayip olusur. Turbulans istatistiklerini bozar. Backscatter ile birlesince kontrol edilemez enerji artisi soz konusu olabilir.

**Onerilen fix:**
```python
def forward(self, state: FluidState3D):
    u, v, w = state.u, state.v, state.w
    # Convective form
    du_dx, du_dy, du_dz = self.diff_ops.gradient(u)
    dv_dx, dv_dy, dv_dz = self.diff_ops.gradient(v)
    dw_dx, dw_dy, dw_dz = self.diff_ops.gradient(w)
    conv_u = u*du_dx + v*du_dy + w*du_dz
    conv_v = u*dv_dx + v*dv_dy + w*dv_dz
    conv_w = u*dw_dx + v*dw_dy + w*dw_dz

    # Divergence form: nabla . (u otimes u)
    d_uu_dx, d_vu_dy, d_wu_dz = (self.diff_ops.gradient(u*u)[0],
                                   self.diff_ops.gradient(v*u)[1],
                                   self.diff_ops.gradient(w*u)[2])
    div_u = d_uu_dx + d_vu_dy + d_wu_dz
    # (ayni sekilde v ve w icin)
    ...
    # Skew-symmetric
    adv_u = 0.5 * (conv_u + div_u)
    adv_v = 0.5 * (conv_v + div_v)
    adv_w = 0.5 * (conv_w + div_w)
    # Dealias + modulate
    ...
```

---

## 2. ORTA ONCELIKLI SORUNLAR

### 2.1 Nusselt Sayisi Formulu Eksik/Hatali

**Dosya:** `/Users/apple/Desktop/nsneuron/bitirme2/model.py`, satirlar 91-97

**Mevcut durum:**
```python
def nusselt_number(self, Ly: float, kappa: float) -> torch.Tensor:
    vT = (self.v * self.theta).mean(dim=(-3, -2, -1))
    return 1.0 + vT / (kappa + 1e-10)
```

**Ne olmasi gerekiyor:**
Rayleigh-Benard konveksiyonda Nusselt sayisi:

    Nu = 1 + (Ly / (kappa * Delta_T)) * <v * T'>

burada Delta_T = T_hot - T_cold sicaklik farki, Ly = domain yuksekligi, <.> domain ortalamasi.

Eger nondimensional T' kullaniliyorsa (sicaklik farkina normalize), o zaman:

    Nu = 1 + <v * theta> * Ly / kappa

ya da eger T' zaten Delta_T ile normalize edilmisse:

    Nu = 1 + <v * theta> / kappa

Mevcut kodda `Ly` boleni eksik. Nondimensionalization konvansiyonuna bagli, ama mevcut formul:
- `kappa = 1/(Re*Pr) = 2.82e-4`
- Tipik `<v*theta>` degeri ~ O(1e-3)
- Nu ~ 1 + 1e-3 / 2.82e-4 ~ 4.5 (makul gibi gorunuyor)

**Ancak** theta'nin normalizasyonu belirsiz. Eger theta boyutlu (Delta_T ~ 20K) ise Nu patlar. Eger theta boyutsuz (theta ~ O(0.1)) ise formul yaklasik dogru ama `Ly` boleni hala gerekli olabilir. Bu, nondimensionalization konvansiyonunun **acikca belirtilmemis** olmasindan kaynaklanan bir belirsizlik.

**Oneri:** Nondimensionalization tanimini bir yere yazin ve Nu formulunu ona gore duzeltin. theta_init = 0.1*randn gorunce boyutsuz oldugunu dusunuyorum, bu durumda formul yaklasik ama kesin dogru degil.

---

### 2.2 Forcing Amplitude Kalibrasyonu Dusuk

**Dosya:** `/Users/apple/Desktop/nsneuron/innate.py`, satirlar 3817-3818, 3841

**Mevcut durum:**
```python
self.amplitude = nn.Parameter(torch.tensor(0.001))
# forward'da:
A = self.amplitude.clamp(1e-5, 0.1)
```

**Fiziksel denge analizi:**
Kolmogorov forcing altinda enerji dengesi: `F . u ~ epsilon ~ nu * <|nabla u|^2>`

Re=5000'de beklenen hiz buyuklugu O(1). Tipik forcing amplitude:
- DNS literaturunde: A ~ 1/Re ila A ~ O(0.1) arasinda degisir
- Eger F ~ A * sin(k_f * y) ve beklenen kinetik enerji E ~ O(1) ise:
  A ~ nu * k_f^2 * U ~ 2e-4 * (2*pi/10)^2 * 1 ~ 8e-5

Init=0.001 makul bir baslangic. Ancak **clamp(1e-5, 0.1)** ust siniri 0.1 ise ve egitim A'yi buyutmeye calisirsa, bu A*sin(k_f*y) ile pumplanan enerji hizi:

    dE/dt ~ A * <u * sin(k_f*y)> ~ A * U ~ 0.1 * 1 = 0.1

Bu bircok turbulanss scenario icin cok fazla. Dissipation rate epsilon ~ nu * Re_lambda^2 / 15 ~ O(0.01) olabilir. A=0.1 ile forcing dissipation'dan hizli enerji pompalar -> enerji birikimi -> patlama.

**Oneri:** Ust clamp sinirini 0.01'e dusur veya Re'ye oranli yap: `A_max = C / Re` (C ~ 10-50).

---

### 2.3 Backscatter Anti-Diffusion Riski

**Dosya:** `/Users/apple/Desktop/nsneuron/innate.py`, satirlar 3698-3701

**Mevcut durum:**
```python
if self.use_backscatter:
    bs = torch.clamp(self.backscatter_coeff, -0.02, 0.0)
    nu_t = nu_t + bs * self.delta ** 2 * strain_mag
```

**Problem:**
backscatter_coeff init=0, clamp [-0.02, 0.0]. Gradient-based optimization baslarken, gradient buyuk olasilikla negatif yonde iter (loss'u azaltmak icin turbulans dagilimini genisletmek ister).

bs = -0.02 oldugunda:
```
nu_t_backscatter = -0.02 * 0.0625^2 * |S| = -7.8e-5 * |S|
```

nu_molecular = 2e-4 iken, yuksek strain bolgelerde |S| > 3 oldugunda:
```
nu_t_backscatter = -2.3e-4 > nu_molecular
```
Bu da yerel olarak **negatif efektif viskozite** olusturur. Negatif difuzyon = anti-diffusion = katastrofik instabilite.

Debugger'in tespiti (#2) ile TAM UYUMLU.

**Oneri:** init=0 dogru, ama kullanim sirasinda toplam nu_eff >= nu_min guvenlik siniri konmali:
```python
nu_eff = nu_molecular + nu_t
nu_eff = torch.clamp(nu_eff, min=0.5 * nu_molecular)  # asla negatif difuzyon olmasin
```

---

## 3. DUSUK ONCELIKLI SORUNLAR / IYILESTIRME ONERILERI

### 3.1 model.py'de FluidState3D'ye Sahte omega=0 Gonderiliyor

**Dosya:** `/Users/apple/Desktop/nsneuron/bitirme2/model.py`, satirlar 350-356

**Mevcut durum:**
```python
state_fs = FluidState3D(
    u=u, v=v, w=w, p=p,
    omega_x=torch.zeros_like(u),  # <-- sahte sifir
    omega_y=torch.zeros_like(u),
    omega_z=torch.zeros_like(u),
    t=torch.zeros(u.shape[0], device=u.device),
)
```

**Analiz:**
Advection3D ve EddyViscosity3D aslinda omega_x/y/z kullanMIyor (sadece u,v,w kullaniyorlar). Dolayisiyla bu sifir degerler SIMDILIK fiziksel bir hata yaratmiyor. Ancak:

1. Her adimda 3 adet zeros_like tensor olusturma = gereksiz memory allocation (96*160*64*4 byte * 3 = ~11.8 MB/layer)
2. Gelecekte vorticity-based diagnostics kullanilacaksa yanlis sonuc verir

**Oneri:** FluidState3D'yi Advection3D/EddyViscosity3D icin ZORUNLU olmayan sekilde yeniden tasarla, ya da her adimda `ops.curl(u,v,w)` ile gercek vorticity hesapla. model.py satirlar 107-112'deki `_to_fluid_state` fonksiyonu zaten bunu yapiyor ama `_layer_step` icinde KULLANILMIYOR.

---

### 3.2 Thermal Advection Convective Form -- Skew-Symmetric Degil

**Dosya:** `/Users/apple/Desktop/nsneuron/innate.py`, satirlar 4011-4028

**Mevcut durum:**
```python
dT_dx, dT_dy, dT_dz = self.ops.gradient(theta)
adv = u * dT_dx + v * dT_dy + w * dT_dz  # convective form
```

**Ne olmasi gerekiyor:**
Momentum icin skew-symmetric form oneriyorken, sicaklik denklemi icin de tutarli olmak lazim. Sicaklik icin skew-symmetric:

    N_s(T) = 0.5 * [ u . nabla(T) + nabla . (u*T) ]

Incompressible akista (nabla . u = 0) iki form esdegerdir, ama ayrik sayisal duzlemde simetri bozulur. Skew-symmetric form, sicaklik alaninin L2 normunu (termal enerjiyi) tam korur.

**Etki:** Momentum kadar kritik degil cunku theta clamp [-10, 10] ile sinirli ve mean removal uygulamyor. Ama uzun simulasyonlarda theta'da drift gorulebilir.

---

## 4. DOGRULANAN (DOGRU) FORMULASYONLAR

### 4.1 SpectralOps3DAniso -- DOGRU

- Dalga sayilari: `kx = fftfreq(Nx, d=Lx/Nx) * 2*pi` -- DOGRU
- Nyquist sifirlama: DOGRU (spectral ambiguity onleme)
- Gradient: `df/dx = IFFT(i*kx * FFT(f))` -- DOGRU
- Laplacian: `nabla^2 f = IFFT(-k^2 * FFT(f))` -- DOGRU
- Directional laplacian: `d2f/dx2 = IFFT(-kx^2 * FFT(f))` -- DOGRU
- Curl: `omega = nabla x u` -- DOGRU (isaret ve sira dogru)
- Divergence: `nabla . u = IFFT(i*(kx*u_hat + ky*v_hat + kz*w_hat))` -- DOGRU
- Poisson solver: `p_hat = -rhs_hat / k^2, p_hat[0,0,0]=0` -- DOGRU
- Dealias 2/3 rule: Mode index bazli, her yon bagimisiz -- DOGRU

**Detayli kontrol:**
```
kx = fftfreq(96, d=6.0/96) * 2*pi = fftfreq(96, d=0.0625) * 2*pi
kx[1] = (1/96) / 0.0625 * 2*pi = 0.16667 * 2*pi = 1.0472
Bu da 2*pi/Lx = 2*pi/6 = 1.0472 -- DOGRU
```

### 4.2 Pressure Poisson Projection -- DOGRU

**Dosya:** innate.py satirlar 2571-2640

Fractional-step (Chorin) yontemi:
1. `div_u = nabla . u*` (intermediate velocity'nin divergence'i)
2. `nabla^2 p = div_u / dt`
3. `u_new = u* - dt * nabla p`

Bu standart fractional-step projectiondir. Spectral Poisson cozumu:
- `p_hat = -rhs_hat / k^2` (Poisson cozumu)
- `p_hat[0,0,0] = 0` (mean pressure gauge)
- Sonuc: `nabla . u_new = 0` makine hassasiyetinde

**Dogrulama:** dt ile bolme ve carpma birbirini IPTAL EDER. Yani `u_new = u* - nabla(nabla^{-2}(nabla . u*))`. Bu Helmholtz projeksiyonun kendisidir. DOGRU.

### 4.3 Boussinesq Buoyancy -- DOGRU

**Dosya:** innate.py satirlar 3883-3916

```python
Fy = Ri * strength * theta
```

Boussinesq yaklasiminda: `F_buoy = Ri * beta * Delta_T * theta' * e_y`

Burada Ri = Ra/(Re^2 * Pr) = 1e6 / (25e6 * 0.71) = 0.0563.

Kuvvet y-yonunde (dikey) ve sicak parcacik yukselir (theta > 0 => Fy > 0 => yukari). Bu **DOGRU** (y ekseninin yukari oldugu varsayimiyla, ki domain 6x10x4'te y=10 en uzun eksen).

### 4.4 Thermal Equation -- DOGRU

model.py satirlar 416-422:
```python
adv_T = self.thermal_advections[layer_idx](u, v, w, theta)
diff_T = self.thermal_diffusions[layer_idx](theta, kappa_t=kappa_t)
theta = theta + dt * (-adv_T + diff_T)
```

Bu: `d(theta)/dt = -u . nabla(theta) + kappa_eff * nabla^2(theta)`

Isaret konvansiyonu dogru: adveksiyon negatif (tasima), difuzyon pozitif (yayilma).

### 4.5 Smagorinsky SGS Model -- DOGRU

**Dosya:** innate.py satirlar 3742-3759, 3669-3688

Strain rate tensor:
```python
S_ij = 0.5 * (du_i/dx_j + du_j/dx_i)
|S| = sqrt(2 * S_ij * S_ij)
```

Smagorinsky modeli:
```python
nu_t = (Cs * delta)^2 * |S|
```

delta = (dx * dy * dz)^(1/3) = 0.0625 (bu durumda isotropik grid oldugu icin)

Frekans-band bazli Cs (cs_low, cs_mid, cs_high) fiziksel olarak mantikli: yuksek frekanslarda daha buyuk Cs (daha cok dissipation), dusuk frekanslarda daha kucuk Cs. Init degerleri (0.08, 0.15, 0.22) literaturle uyumlu (Germano dynamic model genelde 0.1-0.2 arasinda).

### 4.6 Anisotropik Diffusion -- DOGRU

```python
diff_u = nu_x * d2u/dx2 + nu_y * d2u/dy2 + nu_z * d2u/dz2
```

Bu, anisotropik viskozite tensorunun diyagonal bilesenleriyle difuzyondur. Cross-diffusion terimleri (d2u/dxdy vb.) ihmal ediliyor ki bu Smagorinsky tipi eddy viscosity modelleri icin STANDART bir yaklasimdir.

### 4.7 Non-Boussinesq Yogunluk Modeli -- DOGRU (Kisitlamali)

```python
rho = rho_0 * T_0 / T_total  # ideal gaz, sabit p_0
```

Bu: p = rho * R * T, p = p_0 = sabit => rho = p_0 / (R*T) = rho_0 * T_0 / T

Low-Mach limit'te (Ma << 1) termodynamik basinc sabit varsayimi DOGRU. Clamp [0.5*rho_0, 2.0*rho_0] makul guvenlik siniri (2x yogunluk degisimi ~ 2x sicaklik degisimi).

### 4.8 Dealias 2/3 Rule -- DOGRU

```python
dealias_mask = (|Mx| < Nx/3) & (|My| < Ny/3) & (|Mz| < Nz/3)
```

Her yon icin Nyquist modunun 2/3'une kadar korunuyor, gerisi sifirlanipyor. Bu 3D'de standart Orszag 2/3 kurali. Kuadratik nonlinearite icin aliasing-free garanti saglar.

### 4.9 Gauge Fix (Mean Removal) -- DOGRU

```python
theta = theta - theta.mean(dim=(-3, -2, -1), keepdim=True)
```

Periodic BC'de sicaklik perturbasyonunun ortalamasinin sifir olmasi gerekir (aksi halde theta gauge drift yapar). Her adimda mean removal yapmak DOGRU.

### 4.10 Kolmogorov Forcing -- DOGRU

```python
arg = k_f * 2*pi * y / Ly
Fx = A * sin(arg)
```

Standart Kolmogorov forcing: `F = A * sin(k_f * y) * e_x`. x-yonunde zorlanma, y-yonunde degisken. Bu shear instability uretir ve turbulans gelistirir. k_f=1 en dusuk mod.

### 4.11 Periodic BC -- DOGRU

FFT dogal olarak periodic boundary condition varsayar. Tum yonlerde periodic BC uygulanmis (Fourier bazli). Mixed convection icin gercekci degil (gercekte alt=sicak, ust=soguk Dirichlet BC olmali) ama **sicaklik perturbasyonu** uzerinden calisildigi icin (T' = T - T_base), periodic BC kabul edilebilir bir yaklasim.

---

## 5. NAVIER-STOKES DENKLEM KONTROLU

Tam fractional-step formülasyonu (`model.py` `_layer_step`):

**Adim 1 -- Advection:**
```
u* = u^n + dt * (-advection + forcing + buoyancy + diffusion)
```

Acik yazarsak:
```
u* = u^n + dt * [ -(u . nabla)u + F + nu_eff * nabla^2 u ]
v* = v^n + dt * [ -(u . nabla)v + Fy_forcing + Ri*strength*theta + nu_eff * nabla^2 v ]
w* = w^n + dt * [ -(u . nabla)w + nu_eff * nabla^2 w ]
```

**Adim 2 -- Projection:**
```
nabla^2 p = nabla . u* / dt
u^{n+1} = u* - dt * nabla p
```

Bu standart Chorin (1968) / Temam (1969) projection yontemidir. **Birinci dereceden zaman dogruluguna sahiptir** (pressure splitting error O(dt)). Ikinci derece icin Brown, Cortez & Minion (2001) yontemi gerekir (pressure increment form), ama INNATE icin bu yeterli.

**Termal denklem:**
```
theta^{n+1} = theta^n + dt * [ -(u . nabla)theta + kappa_eff * nabla^2 theta ]
```

**UYARI:** Termal adveksiyon, YENI (projected) hiz alani (u^{n+1}) ile yapiliyor (model.py satir 416). Bu dogru: projection SONRASI hiz alani divergence-free, ve sicaklik bu temiz hiz alaniyla tasinmali.

---

## 6. PARAMETRE BUTCESI VE FIZIKSEL ANLAMLILIGI

| Parametre | Adet | Fiziksel Anlam | Dogru mu? |
|-----------|------|----------------|-----------|
| advection_modulator x20 | 20 | Sayisal hata kompanzasyonu | Evet |
| cs_low/mid/high x20 | 60 | Frekans-band SGS | Evet |
| pr_t x20 | 20 | Turbulent Prandtl | Evet |
| aniso_ratio_y/z x20 | 40 | Buoyancy anisotropisi | Evet |
| backscatter x20 | 20 | Enerji geri aktarimi | RISKLI |
| kappa_scale_x/y/z x20 | 60 | Termal anisotropi | Evet |
| thermal_adv_modulator x20 | 20 | Termal adveksiyon mod. | Evet |
| buoyancy_strength x20 | 20 | Per-layer buoyancy | Evet |
| dt_mults x19 + dt_scale | 20 | Adaptif zaman adimi | RISKLI |
| forcing (A + 4 harm.) | 5 | Dis kuvvet | Evet |
| **TOPLAM** | **285** | | |

285 parametre cok makul. Her parametrenin fiziksel yorumu var. MLP'siz tasarim DOGRU karar.

---

## 7. OZET VE ONCELIK SIRASI

| # | Sorun | Onem | Etki | Fix Kolayligi |
|---|-------|------|------|---------------|
| 1.1 | CFL ihlali (dt cok buyuk) | KRITIK | NaN patlama | Kolay |
| 1.2 | Skew-symmetric advection eksik | KRITIK | Enerji drift | Orta |
| 2.1 | Nusselt formulu belirsiz | ORTA | Yanlis Nu | Kolay |
| 2.2 | Forcing amplitude ust siniri | ORTA | Enerji birikim | Kolay |
| 2.3 | Backscatter anti-diffusion | ORTA | Instabilite | Kolay |
| 3.1 | Sahte omega=0 memory waste | DUSUK | Performans | Orta |
| 3.2 | Thermal skew-sym. eksik | DUSUK | Termal drift | Orta |

**Ilk yapilmasi gereken:** 1.1 (CFL fix) ve 1.2 (skew-symmetric advection). Bu ikisi patlamaya dogrudan yol aciyor.

---

## 8. GRID YETERLILIGI

| Metrik | Deger | Yorum |
|--------|-------|-------|
| Grid: Nx x Ny x Nz | 96 x 160 x 64 | |
| dx = dy = dz | 0.0625 | Izotropik grid (tesadufen) |
| Kolmogorov eta (Re=5000) | ~0.0017 | epsilon~1 varsayimiyla |
| dx / eta | ~37 | LES icin iyi (DNS dx/eta ~ 2 gerekir) |
| LES skoru (grid_analysis.py) | 80/100 | Makul |

96x160x64 grid, Re=5000 icin LES **makul**. DNS degil (ona 37^3 ~ 50.000x daha fazla nokta lazim). EddyViscosity3D (SGS model) ZORUNLU.

---

*Rapor sonu. Sorular icin CFD Expert Agent'a danisin.*
