# INNATE3D Mixed Convection - Kod Inceleme Raporu

**Tarih:** 2026-02-23
**Reviewer:** Kod Inceleme Agent (Claude Opus 4.6)
**Proje:** INNATE3D - Saf Fizik Parametre Mimarisi (285 param, MLP yok)
**Incelenen dosyalar:**
- `/Users/apple/Desktop/nsneuron/bitirme2/model.py`
- `/Users/apple/Desktop/nsneuron/bitirme2/train.py`
- `/Users/apple/Desktop/nsneuron/bitirme2/config.py`
- `/Users/apple/Desktop/nsneuron/bitirme2/evaluate.py`
- `/Users/apple/Desktop/nsneuron/innate.py` (ilgili class'lar ve forward metodlari)

---

## Ozet

Toplam **18 bulgu**, oncelik dagilimi:
- CRITICAL: 2
- HIGH: 6
- MEDIUM: 6
- LOW: 4

---

## CRITICAL

### C-1. evaluate.py: `state.enstrophy()` NotImplementedError firlatir

**Dosya:** `/Users/apple/Desktop/nsneuron/bitirme2/evaluate.py`, satir 80
**Sorun:** `eval_energy_balance()` fonksiyonu `s0.enstrophy()` ve `s1.enstrophy()` cagiriyor. Ancak `ThermalFluidState.enstrophy()` metodu (model.py satir 84-89) acikca `NotImplementedError` firlatacak sekilde yazilmis. Docstring'de bile "PhysicsLoss._enstrophy() kullanin" deniliyor.

```python
# evaluate.py satir 80
Z = 0.5 * (s0.enstrophy() + s1.enstrophy())
```

```python
# model.py satir 84-89
def enstrophy(self) -> torch.Tensor:
    raise NotImplementedError(
        "ThermalFluidState.enstrophy() ops.curl gerektirir. "
        "Dogru hesap icin PhysicsLoss._enstrophy(state) kullanin."
    )
```

**Etki:** `evaluate.py` calistirildiginda `eval_energy_balance()` ANINDA crash olur. Hicbir evaluation tamamlanamaz.
**Oneri:** `eval_energy_balance` icinde `PhysicsLoss._enstrophy()` kullanilmali veya model ops'a erisilerek enstrophy dogrudan hesaplanmali:

```python
def _calc_enstrophy(state, ops):
    ox, oy, oz = ops.curl(state.u, state.v, state.w)
    return (ox**2 + oy**2 + oz**2).mean(dim=(-3, -2, -1))

# eval_energy_balance icinde:
Z = 0.5 * (_calc_enstrophy(s0, model.ops) + _calc_enstrophy(s1, model.ops))
```

**Oncelik:** CRITICAL

---

### C-2. evaluate.py: eval_stability icinde de enstrophy crash

**Dosya:** `/Users/apple/Desktop/nsneuron/bitirme2/evaluate.py`, satir 325
**Sorun:** Ayni problem `eval_stability()` icinde de var:

```python
Z = state.enstrophy().mean().item()  # satir 325
```

Bu da `NotImplementedError` firlatir. Stabilite testi de calistirildiginda crash olur.

**Etki:** `eval_stability()` fonksiyonu calistirildiginda ilk adimda crash. Tum evaluation pipeline kirilir.
**Oneri:** C-1 ile ayni fix. `model.ops.curl()` kullanilarak enstrophy hesaplanmali.

**Oncelik:** CRITICAL

---

## HIGH

### H-1. model.py: Vorticity hesabi atlanip zeros veriliyor - EddyViscosity'ye yanlis girdi

**Dosya:** `/Users/apple/Desktop/nsneuron/bitirme2/model.py`, satir 350-356 ve 372-378
**Sorun:** `_layer_step` icinde `FluidState3D` olusturulurken `omega_x`, `omega_y`, `omega_z` hep `torch.zeros_like(u)` olarak veriliyor. Gercek vorticity hesabi yapilmiyor.

```python
state_fs = FluidState3D(
    u=u, v=v, w=w, p=p,
    omega_x=torch.zeros_like(u),  # YANLIS: gercek vorticity degil
    omega_y=torch.zeros_like(u),
    omega_z=torch.zeros_like(u),
    t=torch.zeros(u.shape[0], device=u.device),
)
```

**Analiz:** `Advection3D.forward()` (innate.py satir 2419-2457) vorticity degerlerini kullanmiyor, dogrudan u/v/w uzerinden calisyor. `EddyViscosity3D` de `_compute_strain_magnitude` icinde dogrudan gradyanlardan hesap yapiyor, omega kullanmiyor. Dolayisiyla sifir omega SUAN bir bug degil, ama gelecekte `Vorticity3D`, `Helicity3D` gibi noronlar eklendiginde sessiz hata kaynagi olacak.

**Ek sorun:** Gereksiz 3 adet `torch.zeros_like(u)` alokasyonu yapiliyor -- her biri 96x160x64 tensor. Her layer step'te 6 gereksiz tensor (2 FluidState3D olusumu x 3 tensor). 20 layer x 10 step = 200 adim x 6 tensor = 1200 gereksiz alokasyon.

**Oneri:** Ya gercek vorticity hesaplanmali (`self.ops.curl(u,v,w)`), ya da FluidState3D'nin omega gerektirmeyecek sekilde revize edilmesi lazim (Optional yapilmali). Minimum duzeltme olarak `torch.zeros_like` yerine `torch.empty_like` bile daha ucuz.

**Oncelik:** HIGH

---

### H-2. train.py: energy_balance_loss dt hesabi model._get_layer_dt ile TUTARSIZ

**Dosya:** `/Users/apple/Desktop/nsneuron/bitirme2/train.py`, satir 193
**Sorun:** `PhysicsLoss.energy_balance_loss()` icinde dt su sekilde hesaplaniyor:

```python
dt = self.model._dt_base * torch.clamp(self.model.dt_scale, 0.1, 3.0)
```

Ancak model icinde `_get_layer_dt()` (model.py satir 281):

```python
dt = self._dt_base * torch.clamp(self.dt_scale, 0.5, 2.0)
```

Clamp araliklari farkli: loss'ta `[0.1, 3.0]`, model'de `[0.5, 2.0]`. Ayrica loss hesabi per-layer `dt_mults` carpanini ihmal ediyor.

**Etki:** Energy balance loss'un hesapladigi dE/dt, model'in gercekte kullandigi dt'ye uymaz. Model `dt_base * 2.0 * 1.5 = dt_base * 3.0` max kullanabilirken, loss `dt_base * 3.0` hesapliyor -- bu durumda per-layer multiplier olmadan kabaca dogru ama kesin degil. Per-layer dt aktifken her layer farkli dt kullanir, loss'taki tek dt ile hesap yanlis olur.

**Oneri:** Loss hesabi her layer icin o layer'in efektif dt'sini kullanmali. Ya da en azindan clamp araliklari ayni olmali.

**Oncelik:** HIGH

---

### H-3. train.py: Intermediate states'te t hesabi per-layer dt'yi yansitmiyor

**Dosya:** `/Users/apple/Desktop/nsneuron/bitirme2/model.py`, satir 479
**Sorun:** `return_intermediates=True` durumunda timestamp hesabi:

```python
t=state.t + (i + 1) * self._dt_base if state.t is not None else None,
```

Bu sabit `_dt_base` ile carpim yapiyor. Ancak model per-layer degisken dt kullaniyor (`_get_layer_dt(i)` her layer icin farkli olabilir). `t` degeri cumulative olarak tum onceki layer'larin dt toplami olmali, sabit `_dt_base` ile degil.

**Etki:** Intermediate state'lerin zaman damgasi yanlis. Energy balance loss'ta `dEdt = (E1-E0)/dt` hesabi yanlis dt kullanir.

**Oneri:** Layer loop icinde `cumulative_t` degiskeni tutulup her layer'in efektif dt'si eklenmeli:

```python
cumulative_t = 0.0
for i in range(self.n_layers):
    layer_dt = self._get_layer_dt(i)
    # ... layer step ...
    cumulative_t = cumulative_t + layer_dt
    if return_intermediates:
        intermediates.append(ThermalFluidState(
            ..., t=state.t + cumulative_t, ...
        ))
```

**Oncelik:** HIGH

---

### H-4. model.py: dt_mults off-by-one / coverage gap

**Dosya:** `/Users/apple/Desktop/nsneuron/bitirme2/model.py`, satir 166-169
**Sorun:** `dt_mults` `n_layers - 1 = 19` eleman iceriyor. `_get_layer_dt()` satir 283 `layer_idx < len(self.dt_mults)` kontrolu yapiyor. Bu durumda layer 0-18 icin dt_mults uygulanir, layer 19 (son layer) icin uygulanmaz.

```python
# __init__:
self.dt_mults = nn.ParameterList([
    nn.Parameter(torch.tensor(1.0)) for _ in range(self.n_layers - 1)
])

# _get_layer_dt:
if layer_idx < len(self.dt_mults):  # layer 0..18: dt_mults var, layer 19: YOK
    mult = torch.clamp(self.dt_mults[layer_idx], 0.7, 1.5)
```

Yorum diyor ki "n_layers-1 cunku dt_scale zaten global 1 tane". Mantik: dt_scale global + 19 per-layer = 20 toplam parametre. ANCAK bu son layer'in dt'sini kontrol edememe demek. Son layer'in dt'si sadece `dt_scale`'e bagli.

**Etki:** Son layer'in dt'si digerlerinden farkli davranir. Eger model son layer'da buyuk bir dt istiyor olsaydi bunu ayarlayamaz.

**Oneri:** Tutarlilik icin n_layers adet dt_mult kullanilmali veya dt_scale kaldirilip tamamiyla per-layer'a gecilmeli. Docstring'deki parametre butcesi ile tutarli olacak sekilde.

**Oncelik:** HIGH

---

### H-5. evaluate.py: DNS comparison icinde model.set_physics cagrildiktan sonra geri alinmiyor

**Dosya:** `/Users/apple/Desktop/nsneuron/bitirme2/evaluate.py`, satir 399
**Sorun:** `eval_dns_comparison()` fonksiyonu `model.set_physics(Re_val=1000, Ra_val=1e5)` cagiriyor. Bu model'in nu, kappa, Ri degerlerini KALICI olarak degistirir. Eger evaluation pipeline'da bu fonksiyon ortada cagiriliyorsa, sonraki metrikler yanlis Re/Ra ile hesaplanir.

```python
model.set_physics(Re_val, Ra_val)  # satir 399 - KALICI DEGISIKLIK
```

`evaluate()` fonksiyonunda DNS comparison 3. sirada calistiriliyor (satir 571). Sonrasinda Phase 2 metrikleri calistiriliyor (satir 594+). Ancak stability test DNS'ten once yapildigi icin (satir 534), ve DNS separate `dns_model` kullaniyor. Bu durumda ana `model` degismiyor. AMA `dns_model` uzerinde `set_physics` cagriliyor.

Gercekte tehlike su: `eval_dns_comparison()` parametresi olarak NORMAL `model` gecirilebilir. Fonksiyon kendi icinde `model.set_physics()` cagiriyor. Fonksiyondan sonra model'in fizik parametreleri degismis oluyor.

**Etki:** `eval_dns_comparison()` fonksiyonunu dogru model ile cagiranlar icin side-effect var.

**Oneri:** Fonksiyon sonunda orjinal Re/Ra'ya geri donulmeli veya kopyasi uzerinde calisilmali.

**Oncelik:** HIGH

---

### H-6. model.py: forward() return_intermediates=True ise final_state donmez

**Dosya:** `/Users/apple/Desktop/nsneuron/bitirme2/model.py`, satir 491-493
**Sorun:**

```python
if return_intermediates:
    return intermediates   # SON STATE BURADA
return final_state
```

`intermediates` listesi 20 eleman iceriyor (her layer icin bir state). Son eleman zaten final state. Ancak `train.py` satir 953'te:

```python
step_states = [current] + intermediates  # 1 + 20 = 21 state
```

Burada `current` initial state, `intermediates[0]` layer 0 sonrasi, ..., `intermediates[19]` layer 19 sonrasi (final). Bu dogru calisiyor.

AMA, `return_intermediates=True` iken final_state AYRI OLARAK donmuyor. Eger biri hem intermediates hem final_state isterse ikisini birlikte alamaz. Bu bir API tutarsizligi.

**Etki:** Fonksiyonel bir hata degil su anda, ama `return_intermediates=True` iken `t_new` hesabi `final_state` icinde yapilmis, intermediates'te ise her layer icin ayri `t` var. Intermediates kullanilirken final state'in ayri t_new hesabi kaybolur.

**Oncelik:** HIGH (API tasarim sorunu)

---

## MEDIUM

### M-1. train.py: compute_energy_spectrum icinde k_squared device mismatch riski

**Dosya:** `/Users/apple/Desktop/nsneuron/bitirme2/train.py`, satir 84
**Sorun:**

```python
k_mag = torch.sqrt(ops.k_squared)  # ops bufferlari farkli device'ta olabilir
```

`ops.k_squared` bir register_buffer. Model `.to(device)` ile tasindiysa buffer da tasiniyor. Ancak `E_hat` batch-ortalamalanmis `[Nx, Ny, Nz]` tensor -- fonksiyon basinda `u_hat` hesabi `u`'nun device'inda oluyor. Eger `ops` farkli device'ta ise scatter_add crash verir.

**Etki:** Normal kullanmda sorun yok (ops model ile ayni device'ta). Ancak eger biri cpu'da ops olusturup gpu'da u gecirirse crash olur.

**Oneri:** Acik device kontrolu eklenebilir veya dokumante edilebilir.

**Oncelik:** MEDIUM

---

### M-2. train.py: spectrum_slope_loss requires_grad=False tensoru donebilir

**Dosya:** `/Users/apple/Desktop/nsneuron/bitirme2/train.py`, satir 116
**Sorun:**

```python
if mask.sum() < 3:
    return torch.tensor(0.0, device=spectrum.device, requires_grad=False)
```

Eger yeterli inertial range noktasi yoksa, requires_grad=False tensor donuyor. Bu loss terimine eklendiginde gradient graph kirilir. `sum(step_loss_dict.values())` cagrisinda bu terim gradient akismaz.

**Etki:** Egitim basinda (random IC, dusuk enerji) spectrum loss'un gradient vermeme olasiligi yuksek. Ama `PhysicsLoss.spectrum_loss()` icinde (satir 234-235) zaten "return 0.0" kontrolu var. Asil sorun: loss dict'e 0.0 deger olarak girse bile `sum()` icinde diger terimlerle toplaniyor ve `backward()` cagrisi sorun cikarmiyor cunku diger terimler grad_enabled.

**Oneri:** `torch.tensor(0.0, device=..., requires_grad=True)` veya `torch.zeros(1, device=..., requires_grad=True).squeeze()` kullanilmali. Ya da `weights` kontrolu ile hesaplanmasin.

**Oncelik:** MEDIUM

---

### M-3. config.py: Config.set_physics ile PhysicsConfig guncelleniyor ama model.set_physics ayri

**Dosya:** `/Users/apple/Desktop/nsneuron/bitirme2/config.py`, satir 166-170 ve model.py satir 254-269
**Sorun:** Iki ayri `set_physics` metodu var:
1. `Config.set_physics(Re, Ra, Pr)` -- sadece config degerlerini degistirir
2. `INNATE3D_MixedConvection.set_physics(Re, Ra, Pr)` -- model parametrelerini VE noron parametrelerini gunceller, AMA config'i guncellemez

Train loop'ta sadece `model.set_physics()` cagriliyor (satir 928). Config degismiyor. Bu durumda `PhysicsLoss` icinde `self.config.physics.nu` kullanilirken ESKI nu kullaniliyor:

```python
# train.py satir 207 (energy_balance_loss):
eps = 2.0 * phys.nu * Z  # phys = self.config.physics -- ESKI DEGER
```

```python
# train.py satir 928 (train loop):
model.set_physics(Re, Ra)  # model.nu degisir, config.physics.nu DEGISMEZ
```

**Etki:** Re sweep sirasinda (Phase B/C), energy_balance_loss ve dissipation_loss HER ZAMAN config'teki default Re=5000 ile hesaplaniyor. Model icinde gercek nu degismis olsa bile loss yanlis nu kullanir.

**Oneri:** `model.set_physics()` icinde `self.config.physics.Re = Re` seklinde config de guncellenmeli. Veya loss hesabi `self.model.nu` kulllanmali (config degil).

**Oncelik:** MEDIUM -- Re sweep yapilana kadar gorunmez ama Phase B/C'de ciddi sorun.

---

### M-4. train.py: _create_phase_d_optimizer keyword'leri model yapisiyla uyumsuz

**Dosya:** `/Users/apple/Desktop/nsneuron/bitirme2/train.py`, satir 732-734
**Sorun:** Phase D optimizer parameter gruplama keyword'leri:

```python
frozen_keywords = ("layers", "projection")
finetune_keywords = ("forcing", "buoyancy", "thermal")
new_keywords = ("density", "var_density", "continuity", "state_equation")
```

Ancak model'deki gercek parametre isimleri:
- `advections.0.advection_modulator` -- "layers" icermiyor, "advection" iceriyor
- `eddy_viscosities.0.cs_low` -- "layers" icermiyor
- `dt_scale`, `dt_mults.0` -- hicbir keyword'e uymuyor -> finetune grubuna duser

**Sonuc:** frozen_keywords'de "layers" ve "projection" var ama model'de "layers" isminde parametre YOK (moduleler `advections`, `projections`, `eddy_viscosities` vs. isimli). `Projection3D` 0 parametre -- zaten frozen grubuna dusmez.

Tum momentum ile ilgili parametreler (advection_modulator, cs_low/mid/high, aniso_ratio, backscatter, dt_scale, dt_mults) hicbir frozen/finetune keyword'une uymuyor ve default olarak finetune grubuna dusuyor.

**Etki:** Phase D'de "frozen" grubu BOS olabilir. Tum parametreler "finetune" veya "new" grubuna duser. Transfer learning stratejisi calismaz.

**Oneri:** Keyword'ler model'deki gercek isimlerle eslesmeli:

```python
frozen_keywords = ("advections", "projections", "eddy_viscosities")
finetune_keywords = ("forcing", "buoyancy", "thermal", "dt_scale", "dt_mults")
new_keywords = ("density_update", "var_density_advections", "continuity", "state_equation")
```

**Oncelik:** MEDIUM

---

### M-5. model.py: Non-Boussinesq modda advection icin FluidState3D omega yerine zeros

**Dosya:** `/Users/apple/Desktop/nsneuron/bitirme2/model.py`, satir 369-378
**Sorun:** Non-Boussinesq branch'ta `VariableDensityAdvection3D` cagriliyor (satir 346-348). Ancak satir 369-378'de eddy viscosity icin FluidState3D olusturulurken yine zeros omega:

```python
if not (self.non_boussinesq and rho is not None):
    pass  # state_fs zaten yukarda olusturuldu
else:
    state_fs = FluidState3D(
        omega_x=torch.zeros_like(u), ...
    )
```

Bu durumda non-Boussinesq branch icin eddy viscosity'nin FluidState3D'si ayri olusturuluyor ama yine zeros omega ile. Fonksiyonel olarak strain_magnitude omega kullanmiyor, ama gereksiz alokasyon ve potansiyel gelecek bug.

**Oncelik:** MEDIUM

---

### M-6. evaluate.py: eval_temperature_bounds y grid Ny+1 aliyor, model Ny aliyor

**Dosya:** `/Users/apple/Desktop/nsneuron/bitirme2/evaluate.py`, satir 142
**Sorun:**

```python
y = torch.linspace(0, dom.Ly, dom.Ny + 1, device=state.theta.device)[:-1]
```

Model'deki `_compute_T_total` (model.py satir 516):

```python
y = torch.linspace(0, d.Ly, d.Ny, device=device)
```

Fark: evaluate `linspace(0, Ly, Ny+1)[:-1]` kullanir (Ny noktali, [0, ..., Ly*(Ny-1)/Ny]), model `linspace(0, Ly, Ny)` kullanir (Ny noktali, [0, ..., Ly]). Son nokta: evaluate'da `Ly*159/160 = 9.9375`, model'de `Ly = 10.0`.

**Etki:** T_base profili evaluate ve model'de FARKLI. Cok kucuk fark (son nokta 10.0 vs 9.9375) ama prensipte tutarsiz. Periyodik grid icin `linspace(0, Ly, Ny+1)[:-1]` dogru olan (Forcing3D de bunu kullaniyor, innate.py satir 3836).

**Oneri:** model.py satir 516 `torch.linspace(0, d.Ly, d.Ny+1, device=device)[:-1]` olmali.

**Oncelik:** MEDIUM

---

## LOW

### L-1. model.py: `_to_fluid_state` helper kullanilmiyor

**Dosya:** `/Users/apple/Desktop/nsneuron/bitirme2/model.py`, satir 105-112
**Sorun:** `_to_fluid_state()` fonksiyonu tanimlanmis ama hicbir yerde cagirilmiyor. `_layer_step` icinde FluidState3D her seferinde inline olusturuluyor. Bu fonksiyon gercek vorticity hesaplayan dogru implementasyon (`ops.curl(u,v,w)` kullanir) ama kullanilmiyor.

**Etki:** Dead code. Ayrica IRONIK bir sekilde H-1'deki sorunun cozumunu iceriyor ama cagrilmiyor.

**Oneri:** Ya kullanilmali ya da kaldirilmali. Kullanildiginda H-1 de cozulur.

**Oncelik:** LOW

---

### L-2. train.py / config.py: Kullanilmayan import'lar

**Dosya:** `/Users/apple/Desktop/nsneuron/bitirme2/model.py`, satir 27-28
**Sorun:** `OrderedDict` sadece `load_state_dict_compat` icinde kullaniliyor (dogru). Ancak `from dataclasses import dataclass` import ediliyor, `ThermalFluidState` icin. `List, Optional, Tuple` da kullaniliyor. Bunlar temiz.

Ama `train.py` satir 37'de `Tuple` import ediliyor ve gercekten de kullaniliyor. Temiz.

Gercek dead import: model.py satir 32 `from typing import Dict, List, Optional, Tuple` -- `Dict` model.py icinde sadece `parameter_summary` ve type hint'lerde kullaniliyor. Temiz.

Bir dead import: `config.py` satir 6: `asdict` import ediliyor ve kullaniliyor (satir 175). Temiz.

Sonuc: Buyuk dead import yok. Minor.

**Oncelik:** LOW

---

### L-3. train.py: LR scheduler step_ou cagirilmiyor (stochastic forcing)

**Dosya:** `/Users/apple/Desktop/nsneuron/bitirme2/train.py`
**Sorun:** `Forcing3D` stochastic modda `step_ou(dt)` cagirilarak Ornstein-Uhlenbeck process'i ilerletilmeli. Ancak training loop icinde sadece `reset_phase()` cagriliyor (satir 931), `step_ou` hic cagirilmiyor. Stochastic forcing modunda bu, OU state'inin her epoch basinda sifirlandigi ama adimlar icinde ilerletilmedigi anlamina gelir.

**Etki:** Su an forcing_mode="kolmogorov" (default), dolayisiyla stochastic mod kullanilmiyor. Ileride stochastic mode gecilirse bug olur.

**Oncelik:** LOW

---

### L-4. model.py: Advection3D'ye resolution=Nx geciriliyor ama anisotropik grid'de Nx != Ny != Nz

**Dosya:** `/Users/apple/Desktop/nsneuron/bitirme2/model.py`, satir 197
**Sorun:**

```python
adv = Advection3D(resolution=d.Nx, diff_ops=self.ops)
```

`Advection3D.__init__` icinde `resolution` sadece `diff_ops` None ise fallback `SpectralOps3D(resolution)` olusturmak icin kullaniliyor. `self.ops` zaten disaridan verildigi icin resolution parametresi kullanilmiyor. Sorun yok ama yaniltici. Ayni durum `Projection3D` ve `EddyViscosity3D` icin de gecerli.

**Oncelik:** LOW

---

## Genel Degerlendirme

### Iyi Yapilmis Yonler

1. **Mimari karar dogru:** MLP kaldirma ve saf fizik parametrelerine gecis fiziksel olarak mantikli. 285 parametre ile yorumlanabilir bir model.

2. **Clamp'ler eklenmis:** Debugger'in onceki raporundaki advection_modulator clamp sorunu duzeltilmis (innate.py satir 2456: `torch.clamp(self.advection_modulator, 0.5, 1.5)`).

3. **TBPTT stratejisi dogru:** Autoregressive unrolling'de gradient graph derinligini 20'de tutmak icin detach kullanilmasi dogru yaklasim.

4. **Curriculum tasarimi iyi dusunulmus:** 4 fazli egitim, linear ramp-up, Re/Ra sweep mantikli.

5. **Checkpoint/resume mekanizmasi saglam:** Phase D transfer learning icin per-parameter-group optimizer iyi bir fikir.

### Acil Yapilmasi Gerekenler (Blocker)

1. **C-1 ve C-2:** evaluate.py enstrophy hatasi duzeltilmeli. Bu olmadan hicbir evaluation calistirilamaz.

2. **M-3:** `config.physics.nu` vs `model.nu` tutarsizligi. Re sweep sirasinda tum physics loss'lar yanlis nu kullanir. Bu egitimin kalitesini dogrudan etkiler.

3. **H-2 + H-3:** dt tutarsizligi. Energy balance loss'un dogru calisabilmesi icin dt hesabi model ile ayni olmali.

### Mimari Oneri

Model'deki `FluidState3D` zorunlu omega alanlari gereksiz maliyet ve karmasiklik yaratiyiyor. `Optional[torch.Tensor] = None` yapilmasi veya model icinde `_to_fluid_state` helper'inin kullanilmasi onerilir. Bu hem H-1'i cozer, hem gereksiz alokasyonu engeller.

---

*Rapor sonu. Sorular icin reviewer agent'a ulasabilirsiniz.*
