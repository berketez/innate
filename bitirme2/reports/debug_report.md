# INNATE3D Debug Raporu
**Tarih:** 2026-02-23
**Debugger Agent:** Python debug uzmani
**Kapsam:** model.py, train.py, innate.py (SpectralOps3DAniso, Advection3D, Projection3D, EddyViscosity3D, Forcing3D, Buoyancy3D, ThermalDiffusion3D, ThermalAdvection3D, DensityUpdate3D)

---

## Ozet

285 parametreli saf INNATE 3D mixed convection modelinde toplam **14 bug/risk** tespit edildi:
- **4 KRITIK** (NaN/patlama/yanlis fizik)
- **5 YUKSEK** (sessiz hata/dogru olmayan gradient)
- **3 ORTA** (verimlilik/ince mantik hatasi)
- **2 DUSUK** (kod hijyeni)

---

## KRITIK SEVIYE

---

### BUG-01: dt_scale Clamp Tutarsizligi (model.py vs train.py)

**Dosya/Satir:**
- `model.py:281` -- `torch.clamp(self.dt_scale, 0.5, 2.0)`
- `train.py:193` -- `torch.clamp(self.model.dt_scale, 0.1, 3.0)`

**Nedir:**
Model forward pass'ta dt_scale'i [0.5, 2.0] araliginda clamp'liyor, ama PhysicsLoss.energy_balance_loss icinde dt'yi [0.1, 3.0] araliginda clamp'liyor.

**Neden sorun:**
Forward pass dt_scale=2.0 ile calisirken, energy balance loss dt_scale=3.0 ile hesap yapiyor olabilir. Veya dt_scale'in gercek degeri 0.3 olsa, forward'da 0.5'e clamp'lenir ama loss'ta 0.3 olarak kullanilir. Bu, loss fonksiyonunun fizik dengesini **yanlis dt** ile hesaplamasi demek.

Sonuc: energy balance loss yanlis residual hesaplar, gradient yanlis yonde iter, model "denge" yerine "chaos"a yakinsar.

**Etki:** Loss fonksiyonu yalan soyluyor -- gradient yanlis yonde.

**Onerilen fix:**
```python
# train.py satir 193'u model ile tutarli yap:
dt = self.model._get_layer_dt(0)  # model'in kendi dt hesabini kullan
# VEYA en azindan ayni clamp:
dt = self.model._dt_base * torch.clamp(self.model.dt_scale, 0.5, 2.0)
```

En temiz cozum: `PhysicsLoss`'un dt'yi `model._get_layer_dt(layer_idx)` uzerinden almasidir. Farkli katmanlarin farkli dt'si var, ama energy balance tek bir dt kullaniyor. Bu da ayri bir fiziksel tutarsizlik (bkz BUG-08).

---

### BUG-02: Vorticity Sifir Geciliyor -- EddyViscosity3D Icin Strain Hesabi Uyumsuz Olabilir

**Dosya/Satir:** `model.py:350-357`, `model.py:372-378`

**Nedir:**
`_layer_step` icinde `FluidState3D` olusturulurken:
```python
state_fs = FluidState3D(
    u=u, v=v, w=w, p=p,
    omega_x=torch.zeros_like(u),  # <-- SIFIR
    omega_y=torch.zeros_like(u),  # <-- SIFIR
    omega_z=torch.zeros_like(u),  # <-- SIFIR
    t=torch.zeros(u.shape[0], device=u.device),
)
```
Vorticity (omega) alanlari sifir olarak veriliyor, gercek curl(u) hesaplanmiyor.

**Neden sorun:**
Simdilik `EddyViscosity3D._compute_strain_magnitude()` ve `Advection3D.forward()` dogrudan `state.u, state.v, state.w`'den gradient hesapliyor, `omega` alanlarina bakmiyorlar. Dolayisiyla **su an icin bu bir bug degil, ama bir tuzak.**

Eger gelecekte bir noron `state.omega_x`'e erisirse (ornegin diagnostik Vorticity3D noronu, veya bir enstrophy hesabi), sifir deger gorur ve **sessizce yanlis sonuc uretir.** Hata mesaji bile gelmez.

Ayrica `_to_fluid_state()` helper'i (model.py:105-112) dogru sekilde curl hesapliyor, AMA bu helper `_layer_step` icinde KULLANILMIYOR. Iki farkli kod yolu var.

**Etki:** Su an dogrudan bug degil, gelecekte gizli hata kaynagi.

**Onerilen fix:**
Ya `_to_fluid_state()` helper'ini `_layer_step` icinde kullanin:
```python
state_fs = _to_fluid_state(
    ThermalFluidState(u=u, v=v, w=w, p=p, theta=theta), self.ops
)
```
Ya da `omega_*` alanlarini None yapabilecek bir `FluidState3D` variant'i olusturun. Ama ekstra FFT (curl) hesabi her layer'da 3 FFT + 3 IFFT = 6 ek islem demek. 20 layer icin 120 FFT. Performans karari.

---

### BUG-03: Backscatter Negatif Difuzyon -- Garantili Kararsizlik

**Dosya/Satir:** `innate.py:3698-3701`

**Nedir:**
```python
if self.use_backscatter:
    bs = torch.clamp(self.backscatter_coeff, -0.02, 0.0)
    nu_t = nu_t + bs * self.delta ** 2 * strain_mag
```
`bs` her zaman negatif veya sifir. Bu demek ki `nu_t`'den cikariliyor.

**Neden sorun:**
Negatif eddy viscosity = **anti-diffusion** = fiziksel olarak kararsiz. Kucuk `bs=-0.02` bile strain_mag buyukse (turbulansta kolayca O(100)) onemsiz olmayan negatif difuzyon uretir. 20 layer x 10 step = 200 adimda bu birikerek patlar.

Daha da kotusu: `nu_t + bs * delta^2 * strain_mag` sonucu negatif olabilir:
- `nu_t = (Cs * delta)^2 * strain_mag`, Cs ~ 0.15, delta ~ 0.1 icin `nu_t ~ 0.000225 * strain_mag`
- `bs * delta^2 * strain_mag = -0.02 * 0.01 * strain_mag = -0.0002 * strain_mag`
- Net `nu_t` neredeyse sifira inebilir, hatta negatif

Negatif toplam viskozite (`nu + nu_t < 0`) forward Euler icin kosusluz kararsiz. CFL'den bagimsiz.

**Etki:** 200 adimlik unrolling'de ustel buyume ve NaN.

**Onerilen fix:**
```python
if self.use_backscatter:
    bs = torch.clamp(self.backscatter_coeff, -0.02, 0.0)
    nu_t = nu_t + bs * self.delta ** 2 * strain_mag
    nu_t = torch.clamp(nu_t, min=0.0)  # TOPLAM nu_t negatif olamaz!
```
Veya daha konservatif: backscatter'i nu_t'nin %10'u ile sinirla:
```python
backscatter = bs * self.delta ** 2 * strain_mag
nu_t = nu_t + torch.clamp(backscatter, min=-0.1 * nu_t)
```

---

### BUG-04: torch.clamp Gradient Olduruyor -- 200 Adimda Birikimli Etki

**Dosya/Satir:** `model.py:431-436`

**Nedir:**
```python
_vel_max = 20.0
u = torch.clamp(u, -_vel_max, _vel_max)
v = torch.clamp(v, -_vel_max, _vel_max)
w = torch.clamp(w, -_vel_max, _vel_max)
theta = torch.clamp(theta, -10.0, 10.0)
```

**Neden sorun:**
`torch.clamp` flat bolgede (sinir degerlere ulastiginda) gradient = 0 dondurur. Tek basina sorun degil ama:

1. 20 layer x 10 step = 200 seri clamp islemi
2. Herhangi bir noktada `|u| > 20` olursa, o noktanin gradient'i olur (sifir gradient geri yayilir)
3. Egitimin basinda velocity'ler kucuk, sorun yok. Ama egitim ilerledikce turbulansla birlikte O(10) velocity'ler olusuyor
4. Clamp aktive olunca oradaki gradient olur, model o bolge hakkinda ogrenmeyi birakir

Bu effect "dead neuron" benzetmesiyle dusunulebilir: ReLU'da sifirin altinda kalan noronlar gibi, clamp sinirinda kalan grid noktalari gradient uretmiyor.

**Etki:** Egitim ilerledikce ogrenme yavaslar, belirli bolgelerde gradient kaybi.

**Onerilen fix:**
Soft clamp (tanh-based) kullan:
```python
def soft_clamp(x, limit):
    """Gradient-friendly clamp: tanh scaling."""
    return limit * torch.tanh(x / limit)

u = soft_clamp(u, 20.0)
```
Veya clamp'i sadece eval modda (inference) kullan, training'de kaldir:
```python
if not self.training:
    u = torch.clamp(u, -20.0, 20.0)
```
Ama ikinci secenek patlama riskini geri getirir. En saglikli cozum: soft clamp + loss-based penalty.

---

## YUKSEK SEVIYE

---

### BUG-05: Energy Balance Loss Tek dt Kullaniyor -- Per-Layer dt Yok Sayiliyor

**Dosya/Satir:** `train.py:193-194`

**Nedir:**
```python
dt = self.model._dt_base * torch.clamp(self.model.dt_scale, 0.1, 3.0)
```
Tum katmanlar icin tek bir dt kullaniliyor. Ama model `_use_per_layer_dt=True` ise her katmanin kendi `dt_mults[i]` carpani var.

**Neden sorun:**
Energy balance hesabi `dE/dt = (E1 - E0) / dt` yapiyor. Gercek dt her katmanda farkli ise bu hesap yanlis.

Ornek: Katman 0'in dt'si 0.007, katman 19'un dt'si 0.014 ise, energy balance hepsini dt=0.01 ile hesapliyor. Bu %40'a varan hata demek.

**Etki:** Energy balance loss yanlis gradient uretiyor, dt_mults parametreleri dogru ogrenemez.

**Onerilen fix:**
Energy balance'i katman bazinda hesapla veya `intermediates` listesinden her cift state arasi dogru dt kullan:
```python
# Her intermediate state cifti icin dogru dt:
for i in range(1, len(states)):
    dt_i = self.model._get_layer_dt(i - 1)
    dEdt = (E1 - E0) / dt_i
    ...
```

---

### BUG-06: ThermalDiffusion3D has_kappa_t Kontrolu Kirilgan

**Dosya/Satir:** `innate.py:3959`

**Nedir:**
```python
has_kappa_t = kappa_t is not None and not (
    isinstance(kappa_t, torch.Tensor) and kappa_t.dim() == 0 and kappa_t.item() == 0.0
)
```

**Neden sorun:**
`.item()` cagirisi gradient iceren bir tensor'de:
1. 0-dim tensor'un gradient'ini koparir (`.item()` Python float dondurur)
2. Ama asil sorun: bu kontrol **sadece tam olarak 0.0** degerini yakaliyor

`compute_thermal_eddy_diffusivity` fonksiyonu `use_turbulent_prandtl=False` ise `torch.tensor(0.0)` donduruyor (satir 3735). Bu durumda 0-dim tensor, `.item() == 0.0` True, `has_kappa_t = False`. Burasi dogru.

ANCAK: eger `kappa_t` cok kucuk ama sifir olmayan bir deger ise (ornegin 1e-20), has_kappa_t=True olur ve gereksiz yere `(kappa * sx + kappa_t)` hesabi yapilir. Bu kritik bir bug degil ama floating point hassasiyetinde sinir durum.

Daha onemli sorun: bu kontrolun **autograd graph'ini degistirmesi**. `has_kappa_t` True vs False farkli hesaplama yollari (`if/else`) demek. Gradient checkpointing sirasinda bu kontrolun sonucu degisirse, forward ve recomputation farkli yollara gidebilir. Su an icin pratikte sorun yok cunku `kappa_t` deterministik hesaplaniyor, ama kirilgan.

**Etki:** Su an calisir durumda, ama kirilgan tasarim.

**Onerilen fix:**
```python
# has_kappa_t kontrolu yerine her zaman ekle, kappa_t=0 ise etkisi yok:
if kappa_t is None:
    kappa_t = torch.zeros(1, device=theta.device)
# Sonra her zaman (kappa * sx + kappa_t) kullan
```

---

### BUG-07: Advection3D'de Advection Modulator Clamp Olunca Gradient Kesilir

**Dosya/Satir:** `innate.py:2456`

**Nedir:**
```python
mod = torch.clamp(self.advection_modulator, 0.5, 1.5)
return mod * adv_u, mod * adv_v, mod * adv_w
```

**Neden sorun:**
`advection_modulator` init'te `fill_()` ile 0.9-1.1 arasinda ayarlaniyor (model.py:199-201). Clamp [0.5, 1.5] genellikle aktive olmaz.

AMA: gradient optimizer modulator'u sinirin disina iterse, clamp gradient'i sifirlar. BUG-04 ile ayni mekanizma ama parametrelerin uzerinde. 20 advection modulator var, optimizer hepsini ayni anda gunceller. Herhangi birinin clamp'e takildigi anda o katman advection ogrenimini durdurur.

**Etki:** Egitim sirasinda belirli katmanlarin advection ogrenimi durabilir.

**Onerilen fix:**
Clamp yerine sigmoid-based parametreleme:
```python
# __init__'te:
self._advection_modulator_raw = nn.Parameter(torch.zeros(1))

# forward'da:
mod = 0.5 + 1.0 * torch.sigmoid(self._advection_modulator_raw)  # [0.5, 1.5] araligi
```
Sigmoid her zaman gradient verir, keskin sinir yok.

---

### BUG-08: Forcing Ciktisi [1,1,Ny,1] Broadcasting ile Tum Grid'e Yayiliyor -- Fiziksel Olarak Dogru Ama Dikkat

**Dosya/Satir:** `innate.py:3839-3858`, `model.py:360-364`

**Nedir:**
Forcing3D `(Fx, Fy, Fz)` shape=[1,1,Ny,1] donduruyor. `model.py` satir 408:
```python
u = u + dt * (-adv_u + Fx + diff_u)
```
Burada `u` shape=[B,Nx,Ny,Nz], `Fx` shape=[1,1,Ny,1]. PyTorch broadcasting ile calisir.

**Neden sorun:**
Kolmogorov forcing sadece y-yonunde degisen sinusoidal bir profil. x ve z'de uniform. Bu fiziksel olarak dogru -- klasik Kolmogorov flow tam olarak boyle.

AMA: Fy ve Fz `torch.zeros_like(self.y_grid)` yani shape=[1,1,Ny,1]. Buoyancy ise `Fy = self.Ri * strength * theta` shape=[B,Nx,Ny,Nz]. Satir 409:
```python
v = v + dt * (-adv_v + Fy_f + Fy_b + diff_v)
```
Burada `Fy_f` (forcing, shape=[1,1,Ny,1]) + `Fy_b` (buoyancy, shape=[B,Nx,Ny,Nz]) broadcasting ile toplanir. Broadcasting dogru calisiyor ama Fy_f SIFIR oldugu icin etkisi yok. Sorun yok.

**Gercek risk:** Eger gelecekte `Fy_f` sifir olmayan bir deger alirsa (ornegin stochastic modda) ve shape uyumsuzlugu olursa, broadcasting sessizce yanlis sonuc verebilir.

**Etki:** Su an sorun yok, ama kodu daha robust yapmak gerekir.

**Onerilen fix:** Forcing output'unu [B,Nx,Ny,Nz]'ye expand et:
```python
Fx, Fy_f, Fz = self.forcing()
Fx = Fx.expand_as(u)
Fy_f = Fy_f.expand_as(u)
Fz = Fz.expand_as(u)
```

---

### BUG-09: Thermal Advection Sonrasi Updated Velocity Kullaniliyor -- Operator Splitting Sirasi

**Dosya/Satir:** `model.py:416`

**Nedir:**
`_layer_step` icindeki islem sirasi:
```
1. Advection (eski u,v,w ile)
2. Source terms (forcing + buoyancy)
3. Diffusion
4. Velocity update: u_new = u_old + dt*(...)
5. Pressure projection: u_new -> div-free
6. Thermal advection: u.nabla(T') -- BURADA u_new KULLANILIYOR
7. Thermal diffusion
8. Theta update
```

Satir 416:
```python
adv_T = self.thermal_advections[layer_idx](u, v, w, theta)
```
Burada `u, v, w` **zaten guncelenmis** (adim 4 ve 5 sonrasi). Momentum advection (adim 1) eski velocity ile yapiliyor ama thermal advection yeni velocity ile yapiliyor.

**Neden sorun:**
Klasik fractional-step yonteminde genellikle AYNI zaman adimindaki velocity ile tum advection'lar yapilir (Strang splitting veya Lie splitting). Burada momentum advection n anindaki velocity ile, thermal advection n+1 anindaki velocity ile hesaplaniyor. Bu **birinci derece splitting hatasi** yaratir.

Bu, kendi basina NaN yapmaz ama fiziksel tutarsizlik yaratir:
- Enerji korunumu bozulur (momentum ve thermal farkli velocity "gorur")
- dt buyudukce hata buyur (splitting hatasi O(dt))

**Etki:** Fiziksel tutarsizlik, O(dt) mertebesinde splitting hatasi. NaN yapmaz ama dogru cozumden saptirir.

**Onerilen fix (iki secenek):**

A) Thermal advection'i da ESKi velocity ile yap (advection splitting oncesi kaydet):
```python
# Adim 1 oncesinde kaydet:
u_old, v_old, w_old = u, v, w
# ... velocity update ...
# Adim 6: eski velocity ile thermal advection
adv_T = self.thermal_advections[layer_idx](u_old, v_old, w_old, theta)
```

B) Ikinci derece Strang splitting kullan (daha dogru ama daha pahali).

---

## ORTA SEVIYE

---

### BUG-10: Strain Magnitude'da Cift Hesaplama

**Dosya/Satir:** `innate.py:3695-3696` ve `innate.py:3737-3738`

**Nedir:**
`_layer_step` icinde once `compute_thermal_eddy_diffusivity` (satir 382), sonra `compute_anisotropic_nu` (satir 386) cagriliyor. Her ikisi de `_get_strain_mag(state)` cagiriyor. Eski cache mekanizmasi kaldirildi (checkpoint uyumsuzlugu yuzunden), dolayisiyla strain magnitude **iki kez** hesaplaniyor.

Her strain hesabi: 3x gradient (9 FFT + 9 IFFT) = 18 FFT islemi. Cift hesaplama = fazladan 18 FFT/layer.

**Etki:** Performans kaybi, 20 layer icin 360 gereksiz FFT.

**Onerilen fix:**
`_layer_step` icinde strain'i bir kez hesapla, sonra her iki fonksiyona parametre olarak gec:
```python
strain_mag = eddy._compute_strain_magnitude(state_fs)
kappa_t = eddy.compute_thermal_eddy_diffusivity_with_strain(state_fs, strain_mag)
nu_x, nu_y, nu_z = eddy.compute_anisotropic_nu_with_strain(state_fs, self.nu, strain_mag)
```

---

### BUG-11: _Z_ref Her Epoch'ta Ilk State'den Resetleniyor

**Dosya/Satir:** `train.py:424`

**Nedir:**
```python
self._Z_ref = self._enstrophy(states[0]).detach().mean().clamp(min=1.0)
```
Her `compute_all` cagrisinda `_Z_ref` ilk state'in enstrophy'sine resetleniyor. Ilk state random IC'den geliyor ve enstrophy'si cok kucuk olabilir (noise_scale=0.01 ile Z ~ O(0.01)).

Sonra `stability_loss` bunu kullanir:
```python
Z_ratio = Z / self._Z_ref  # Z_ref cok kucuk ise Z_ratio cok buyuk
```

**Neden sorun:**
Eger `_Z_ref = 1.0` (clamp sonrasi), son state'in enstrophy'si Z=500 (turbulans icin normal) ise:
`Z_ratio = 500/1.0 = 500`. Threshold 10000 oldugu icin sorun yok.

AMA: Eger bir epoch'ta Z = 50000 (bu gercekten patlama) olursa, `Z_ratio = 50000` ve stability loss aktive olur. Bu dogru.

Aslinda sorun sart degil, ama `_Z_ref`'in her epoch'ta rastgele IC'ye bagli olmasi istatistiksel olarak gurulutusu loss'a bias ekliyor.

**Etki:** Stability loss'un tutarsiz davranisi, ozellikle erken epoch'larda.

**Onerilen fix:**
`_Z_ref`'i running average ile guncelle:
```python
Z_current = self._enstrophy(states[0]).detach().mean().clamp(min=1.0)
if self._Z_ref is None:
    self._Z_ref = Z_current
else:
    self._Z_ref = 0.99 * self._Z_ref + 0.01 * Z_current
```

---

### BUG-12: compute_energy_spectrum safe_fftn Kullaniyor (Full FFT) -- rfftn Daha Verimli

**Dosya/Satir:** `train.py:70-76`

**Nedir:**
```python
u_hat = safe_fftn(u)
v_hat = safe_fftn(v)
w_hat = safe_fftn(w)
```
`safe_fftn` = `torch.fft.fftn` = full complex FFT. Girdiler reel oldugu icin output Hermitian simetrik: negatif frekanslar pozitifin kompleks eslenigi. Yani bilginin yarisi redundant.

`torch.fft.rfftn` son boyutu Nz//2+1'e indirerek bellek ve zaman tasarrufu saglar. 96x160x64 grid icin:
- fftn output: 96x160x64 complex = 12.5M complex sayisi
- rfftn output: 96x160x33 complex = 6.3M complex sayisi (yaklasik %50 tasarruf)

**Etki:** Her spectrum hesabinda gereksiz bellek ve zaman kullanimi.

**Onerilen fix:**
Asil performance-critical yol (`SpectralOps3DAniso.gradient`, `.laplacian` vs.) zaten `safe_fftn` kullaniyor ve dalga sayilari full FFT'ye gore ayarlanmis. Burayi degistirmek buyuk bir refactor gerektirir.

Ama `compute_energy_spectrum` **bagimsiz bir fonksiyon** ve loss hesabinda kullaniliyor. Bunda `rfftn` kullanmak kolay:
```python
u_hat = torch.fft.rfftn(u, dim=(-3,-2,-1))
# Sonra k_squared'i da rfft boyutuna uygun sec
```

---

## DUSUK SEVIYE

---

### BUG-13: TBPTT'de t Alani Detach Edilmiyor

**Dosya/Satir:** `train.py:977`

**Nedir:**
```python
current = ThermalFluidState(
    u=intermediates[-1].u.detach(),
    v=intermediates[-1].v.detach(),
    ...
    t=intermediates[-1].t,  # <-- detach yok!
    ...
)
```

**Neden sorun:**
`t` alani su an `torch.zeros` ile olusturuluyor ve sadece aritmetik (sabit ekleme) ile gunceleniyor, gradient tasimiyor. Dolayisiyla pratikte sorun yok. Ama eger gelecekte `t` gradient-tasiyan bir hesaplamadan gecirilirse (ornegin time-dependent forcing), graph leak olur.

**Etki:** Su an etkisi yok, gelecek icin temizlenmeli.

**Onerilen fix:**
```python
t=intermediates[-1].t.detach() if intermediates[-1].t is not None else None,
```

---

### BUG-14: DensityUpdate3D Clamp Gradient Kesiyor

**Dosya/Satir:** `innate.py:4070-4072`

**Nedir:**
```python
T_safe = torch.clamp(T_total, min=0.01)
rho = self.rho_0 * self.T_0 / T_safe
return torch.clamp(rho, 0.5 * self.rho_0, 2.0 * self.rho_0)
```

**Neden sorun:**
Iki katmanli clamp: once T_total, sonra rho. BUG-04 ile ayni mekanizma: clamp sinirlarinda gradient = 0.

Non-Boussinesq modda rho her layer'da hesaplaniyor. Eger T_total fiziksel olarak 0.01'in altina duserse veya rho 0.5 ya da 2.0 sinirini asarsa, o bolgede gradient kaybi olur.

Boussinesq modda (Phase A-C) bu kod calismiyor, dolayisiyla su an etkisi yok. Phase D icin risk.

**Etki:** Phase D'de potansiyel gradient kaybi.

**Onerilen fix:**
Soft clamp veya log-space parametreleme:
```python
T_safe = T_total.clamp(min=0.01) + 0.01 * torch.sigmoid(T_total)
# veya
rho = self.rho_0 * self.T_0 / (T_safe + 1e-6)
# rho clamp yerine loss-based penalty
```

---

## EK GOZLEMLER (Bug Degil, Tasarim Kararlari)

### G-01: Projection Sonrasi Velocity Divergence-Free, Ama Theta Icin Bu Garanti Yok

`_layer_step` adim 5'te `u,v,w` divergence-free yapiliyor ama theta (sicaklik) icin boyle bir constraint yok. Fiziksel olarak dogru -- sicaklik solenoidal bir alan degil. Sadece kaydedilmeli.

### G-02: Forcing amplitude Clamp [1e-5, 0.1]

`Forcing3D.forward` satir 3841: `A = self.amplitude.clamp(1e-5, 0.1)`. Re=5000 icin steady-state forcing amplitude ~ O(1e-4). Clamp [1e-5, 0.1] genis bir aralik, dogru gozukuyor. Ama Re=10000'e geciste gereken forcing degisebilir.

### G-03: Poisson Cozucu k_squared_poisson k^2=0'i 1.0 ile Degistiriyor

`innate.py:1012-1013`:
```python
k_sq_poisson = k_squared.clone()
k_sq_poisson[k_sq_poisson == 0.0] = 1.0
```
Ve satir 1107-1108:
```python
p_hat = rhs_hat / (-k_sq_poisson)
p_hat[..., 0, 0, 0] = 0.0  # mean pressure = 0
```
Bu dogru: once k=0'da bolme hatasini onle (1.0 ile bol), sonra mean pressure'i sifirla. Standart spectral Poisson cozucu.

### G-04: 20 Layer x Forward Euler = 1. Derece Zaman Integratoru

Model `u = u + dt * (RHS)` kullaniyor (forward Euler). Bu 1. dereceden dogruluk demek. Turbulans icin genellikle en az RK2 (2. derece) veya RK4 kullanilir. Ancak INNATE'in yaklasimi farkli: "katmanlar" aslinda ogrenilebilir zaman integratoru gibi davranir. dt_mults ve advection_modulator bu hatanin bir kismini kompanse ediyor. Yine de O(dt) splitting hatasi var.

---

## ONCELIK SIRASI

| # | Seviye | Bug | Hemen Fix? | Effort |
|---|--------|-----|-----------|--------|
| 1 | KRITIK | BUG-01: dt_scale clamp tutarsizligi | EVET | 5 dk |
| 2 | KRITIK | BUG-03: Backscatter negatif difuzyon | EVET | 5 dk |
| 3 | KRITIK | BUG-04: Hard clamp gradient oldurme | Sonra | 30 dk |
| 4 | YUKSEK | BUG-05: Energy balance tek dt | EVET | 15 dk |
| 5 | YUKSEK | BUG-09: Thermal advection splitting sirasi | Sonra | 20 dk |
| 6 | YUKSEK | BUG-07: Advection modulator clamp | Sonra | 15 dk |
| 7 | KRITIK | BUG-02: Vorticity sifir | Sonra | 10 dk |
| 8 | ORTA | BUG-10: Cift strain hesabi | Sonra | 30 dk |
| 9 | YUKSEK | BUG-06: has_kappa_t kontrolu | Sonra | 10 dk |
| 10 | ORTA | BUG-11: Z_ref reset | Sonra | 10 dk |
| 11 | ORTA | BUG-12: rfftn optimizasyon | Sonra | 20 dk |
| 12 | DUSUK | BUG-13: t detach | Hemen | 1 dk |
| 13 | DUSUK | BUG-14: DensityUpdate clamp | Phase D | 15 dk |
| 14 | YUKSEK | BUG-08: Forcing shape | Sonra | 5 dk |

---

## DERS: Bu Tip Hatalari Gelecekte Nasil Yakalarsin

1. **Clamp tutarsizligi:** Ayni degiskeni birden fazla yerde clamp'liyorsan, HEP AYNI clamp degerlerini kullan. Ideali: clamp'i tek bir yerde yap, digerlerinde o fonksiyonu cagir.

2. **Hard clamp vs soft clamp:** Training loop'ta `torch.clamp` kullanmak gradient oldurur. Kural: eger bir parametrenin ustune clamp koyuyorsan ve o parametre ogrenilecekse, soft clamp (sigmoid, tanh, softplus) kullan.

3. **Operator splitting sirasi:** Fractional-step yonteminde tum advection'lari ayni zaman adimindaki velocity ile yap. Guncelenmis velocity ile thermal advection yapmak splitting hatasini arttirir.

4. **Negatif difuzyon:** Fiziksel olarak kararsiz. Backscatter gibi mekanizmalar MUTLAKA toplam difuzyon >= 0 garantisi ile kullanilmali.

5. **Loss-model tutarliligi:** Loss fonksiyonunun kullandigi parametreler (dt, nu, vs.) model'in forward pass'ta kullandiklari ile BIREBIR ayni olmali. Farkli clamp degerler = farkli fizik.
