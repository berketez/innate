# INNATE3D ML/Training Pipeline Analiz Raporu

**Tarih:** 2026-02-23
**Analist:** ML Expert Agent (Claude Opus)
**Proje:** INNATE3D Mixed Convection -- 285 param saf fizik modeli
**Dosyalar:** train.py, model.py, config.py, evaluate.py, innate.py

---

## Ozet Tablo

| Kategori | Durum | Oncelik |
|----------|-------|---------|
| A. Loss Fonksiyonlari | 5 sorun (2 ciddi, 3 orta) | YUKSEK |
| B. Gradient Flow | 2 sorun (1 ciddi, 1 orta) | YUKSEK |
| C. Optimizer & LR | 2 sorun (1 ciddi, 1 orta) | ORTA |
| D. Performance | 4 iyilestirme firsati | DUSUK |
| E. Curriculum Learning | 1 sorun (orta) | ORTA |
| F. Numerical Stability | 2 sorun (1 ciddi, 1 orta) | YUKSEK |

---

## A. Loss Fonksiyonlari Kalitesi

### A1. L_divergence -- SORUNSUZ (projeksiyondan dolayi)

```python
div = self.ops.divergence(state.u, state.v, state.w)
return div.pow(2).mean() + 0.1 * div.abs().mean()
```

L2 + L1 hibrit tasarimi mantikli. L1 terimi kucuk sapmalara hassasiyet katarken L2 buyuk sapmalari cezalandirir. Ancak onemli bir nokta: Projection3D sonrasinda div(u) zaten makine hassasiyetinde ~0. Bu demek ki:

**Sorun (ORTA):** Projection sonrasi divergence neredeyse sifir. Bu loss teriminin gradient'i hemen hemen sifir olacak. Loss weight 20-100 arasinda olmasina ragmen, gradient contribution minimumal. Divergence loss fiilen "dead loss" durumunda.

**Etki:** Gradient israf ediliyor ama zarar vermiyor. Weight'i 1.0'a dusursen ayni sey olur.

**Oneri:** Weight'i 1.0'a dusur veya loss'u kaldirip sadece metric olarak izle. Gradient budget'i diger loss'lara ver.


### A2. L_energy_balance -- ISINTERNALLY CONSISTENT AMA BIR SORUN VAR

```python
dt = self.model._dt_base * torch.clamp(self.model.dt_scale, 0.1, 3.0)
```

**Sorun (CIDDI): dt clamp araligi uyumsuz.** Energy balance loss'ta `dt_scale` [0.1, 3.0] arasinda clamp edilirken, model._get_layer_dt()'de `dt_scale` [0.5, 2.0] arasinda clamp ediliyor. Bu ikisi farkli degerler uretir:

- Energy balance loss: dt = dt_base * clamp(dt_scale, 0.1, 3.0)
- Model forward: dt = dt_base * clamp(dt_scale, 0.5, 2.0) * clamp(dt_mults[i], 0.7, 1.5)

Energy balance, modelin gercekte kullandigi dt'den farkli bir dt ile hesaplaniyor. Bu enerji muhafaza denklemini bozar -- loss sifir olsa bile enerji gercekte korunmuyor olabilir.

**Fix:** energy_balance_loss icinde de model._get_layer_dt() kullanilmali, veya en azindan ayni clamp araliklari kullanilmali.


### A3. L_spectrum -- FIZIKSEL OLARAK DOGRU, IMPLEMENTASYONDA INCE SORUN

Spectrum hesabi dogru: shell-averaged E(k), log-log linear regression, slope target -5/3.

**Sorun (ORTA): Spectrum loss sadece son state uzerinden hesaplaniyor.** Turbulans henuz gelismemis bir IC'den 20 layer sonra inertial range olusmus olmayi beklemek agresif. Erken epoch'larda spectrum anlamli olmayabilir ve gradient noise verebilir.

**Ek sorun:** `k_range=(4, 20)` sabit. Grid 96x160x64 icin Nyquist k_max ~ 48-80 arasi. Inertial range bu grid'de k=4-20 arasi makul, ama erken epoch'larda enerji hala buyuk olceklerde ve spectrum flat olabilir. Bu durumda slope fit random noise uzerinden slope cikariyor.

**Oneri:** Spectrum loss'u Phase B'ye kadar (epoch < 1500) devre disi birak veya weight'i 0.01 gibi cok kucuk tut. Phase A'da zaten weight=0.5 ama bu bile erken epoch'larda zarar verebilir.


### A4. L_nusselt -- ONCEKI SORUN DUZELTILMIS, YENI SORUN VAR

Eski versiyon `relu(1-Nu)^2` idi -- Nu >= 1 olunca gradient sifir. Yeni versiyon:

```python
loss_floor = torch.relu(1.0 - Nu).pow(2).mean()
loss_target = 0.1 * ((Nu - Nu_target) / Nu_target).pow(2).mean()
```

Bidirectional yapilmis, her zaman gradient var. Bu iyi.

**Sorun (ORTA): Nu_target hesabi egitim sirasinda Re/Ra sweep'e bagli degil.**

```python
Ra = self.config.physics.Ra  # Config'deki STATIK Ra
```

Ama model.set_physics(Re, Ra) ile her epoch farkli Re/Ra kullaniliyor. Config.physics.Ra degismiyor (set_physics sadece model uzerinde). Dolayisiyla Nu_target her zaman config default Ra=1e6 ile hesaplaniyor, ama model farkli Ra ile calisiyor olabilir.

**Fix:** `nusselt_loss` metoduna Re/Ra parametresi gecirilmeli veya model'den guncel Ri okunmali.


### A5. L_dissipation -- FIZIKSEL OLARAK REDUNDANT

```python
eps = 2.0 * phys.nu * Z           # physical space
eps_spectral = 2*nu * sum(k^2 * E_hat)  # spectral space
return |eps - eps_spectral|
```

Enstrophy ve spectral dissipation arasindaki tutarlilik. Parseval teoremi geregi bunlar MATEMATIKSEL OLARAK ESIT olmali (surekli limitte). Discrete'te kucuk farklar olabilir ama bu loss fiilen spectral method'un internal consistency'sini test ediyor, modelin parametrelerini degil.

**Sorun (ORTA):** Bu loss modelin fizik parametrelerine gradient vermez -- sadece spectral ops'un hassasiyetini olcer. Parametrelere gradient ancak Z uzerinden gider ki Z zaten energy_balance_loss'ta da var. Redundant gradient.

**Oneri:** Weight'i Phase A'da 0 yap, Phase C'de 1.0 gibi tut. Critical path'te degil.


### A6. L_thermal_var ve L_stability -- GUARD LOSS'LAR, SORUNSUZ

Threshold'lar makul:
- `var_max = 100.0` -- cok yuksek, sadece gercek blowup'i yakalar
- Stability: `relu(Z_ratio - 10000)^2` -- 10000x enstrophy artisi, sadece patlama senaryosu

Onceki versiyon `relu(Z_ratio - 10)^2` idi, turbulansi aktif olarak engelleyen bir threshold. 10000 makul.

### A7. L_energy_balance -- AYRI BIR SORUN: FORCING MISMATCH

```python
Fx, Fy, Fz = self.model.forcing()  # Bu her cagrildiginda phi degisebilir
```

Energy balance loss'ta forcing `model.forcing()` ile alinirken, model forward'daki forcing de ayni fonksiyonu cagiriyor. Her ikisi de ayni `phi` kullanir (cunku `reset_phase()` epoch basinda bir kez cagrilir). Bu tutarli.

Ancak amplitude clamp farki:
- Forcing3D.forward(): `A = self.amplitude.clamp(1e-5, 0.1)`
- Model layer_step'te forcing cagrilir ve velocity update'te kullanilir.
- Energy balance loss'ta da ayni forcing cagrilir.
- Bu tutarli. **SORUNSUZ.**


### A8. Phase D Loss'lari (Non-Boussinesq)

- L_continuity_rho: Fiziksel olarak dogru. Forward Euler yaklasimi.
- L_state: Devre disi (weight=0). Dogru karar -- kinematik basinc ile termodinamik basinci karsilastirmak yanlis.
- L_mass: Global kutle korunumu. Basit ama etkili.

**Sorun (DUSUK):** continuity_loss'ta `dt = self.config.physics.dt` kullaniliyor, model'in efektif dt'si degil. Energy balance loss ile ayni sorun ama daha az kritik (Phase D'de).


---

## B. Gradient Flow

### B1. TBPTT Implementasyonu -- DOGRU

```python
for _step in range(config.training.num_steps):  # 10 step
    intermediates = model(current, return_intermediates=True)  # 20 layer
    step_loss_dict = physics_loss.compute_all(step_states, weights)
    (step_loss / config.training.num_steps).backward()

    # DETACH -- gradient graph burada kesilir
    current = ThermalFluidState(
        u=intermediates[-1].u.detach(),
        ...
    )
```

TBPTT dogru implement edilmis:
1. Her step icinde 20 layer uzerinden gradient akiyor (checkpointing ile)
2. Step'ler arasi detach -- gradient graph max derinlik 20 layer
3. Loss her step'te backward edilip accumulate ediliyor
4. Grad norm tek seferde clip ediliyor

**Bu iyi.** num_steps=10, n_layers=20 ile efektif grad depth 20 (200 degil).

### B2. Gradient Checkpointing -- DOGRU

```python
cp.checkpoint(self._layer_step, i, u, v, w, p, theta, rho, use_reentrant=False)
```

`use_reentrant=False` kullanilmis (dogru -- MPS ve modern PyTorch icin). Strain cache bug'i daha once fix edilmis (discussion'da bahsediliyor).

**Memory kazanici:** 20 layer icin ~20x memory tasarrufu (intermediate activation'lar atilir, recompute edilir).

### B3. Vanishing Gradient Riski -- CIDDI

20 layer sequential fractional step. Her layer'da:
1. Advection (nonlinear, quadratic)
2. Viscous diffusion (linear, damping)
3. Projection (linear, zeroing divergent components)
4. Thermal advection + diffusion

**Sorun (CIDDI): Projection operatoru gradient'i keser.**

Projection3D, Helmholtz decomposition yapar: `u_new = u - grad(p)` nerede `lap(p) = div(u)`. Bu islem DIVERGENT bileseni tamamen sifilar. Eger gradient bilgisi o bilesenlerde ise, 20 projeksiyon sonrasi vanishing gradient kacinilmaz.

Advection modulator, eddy viscosity Cs, buoyancy strength gibi parametrelere gradient su yoldan akar:
```
loss -> state_final -> ... -> layer_i output -> layer_i projection -> layer_i velocity_update
```

Projection, gradient'i `u_div-free = u - nabla*p` uzerinden geri iletir. `nabla*p = nabla * (lap_inv(div(u)))` oldugu icin bu bir lineer projeksiyon operatoru (I - nabla*lap_inv*div). Bu operatorun eigenvalue'lari 0 ve 1. Divergent modlarda eigenvalue = 0, yani o modlardaki gradient kaybolur. Solenoid (div-free) modlarda eigenvalue = 1, gradient korunur.

**Pratik etki:** Advection ve forcing gibi parametreler hiz alaninin solenoid kismini etkiler, dolayisiyla gradient'leri korunur. Ancak 20 sequential projection sonrasi, her birinde kucuk numerical error birikebilir ve gradient giderek zayiflar.

**Oneri:**
- Gradient norm per-layer takibi ekle (her katmandaki grad norm'u logla)
- Eger son katmanlardaki grad norm cok dusukse, skip connection dusunulebilir (ama mimariye mudahale)


### B4. Gradient Clipping -- YETERLI AMA KABA

```python
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip)
# grad_clip = 1.0
```

285 parametre icin global norm 1.0 agresif. Her parametrenin gradient contribution'i ~1/sqrt(285) ~ 0.06 olur. Bu kucuk parametrelerde gradient'i bastirabilir.

**Oneri:** Gradient clipping'i 5.0 veya 10.0'a cikar. 285 parametre cok kucuk bir model -- grad norm nadiren patlayacak (TBPTT sayesinde). Log'dan grad_norm degerlerini izle, ona gore ayarla.


---

## C. Optimizer & LR

### C1. AdamW 285 Parametre icin -- UYGUN MU?

AdamW, per-parameter adaptive learning rate kullanir. 285 parametre icin:
- Momentum (m_t): 285 float
- Variance (v_t): 285 float
- Toplam ek hafiza: 285 * 2 * 4 bytes = ~2.3 KB (ihmal edilir)

Adam 285 parametre icin gayet uygun. Alternatif LBFGS olabilir (fizik problemlerinde iyi) ama TBPTT + multi-loss yapisinda LBFGS kullanmak zor (closure fonksiyonu lazim, multi-step backward ile uyumsuz).

**Karar: AdamW dogru secim.**

### C2. Weight Decay -- CIDDI SORUN

```python
weight_decay: float = 1e-4
```

**Sorun (CIDDI): Fizik parametreleri weight decay'den muaf olmali.**

Weight decay, tum parametreleri sifira dogru iter. Fizik parametreleri icin bu yanlis:
- `advection_modulator` ~ 1.0 olmali (adveksiyon siddeti)
- `buoyancy_strength` ~ 1.0 olmali (buoyancy kuvveti)
- `dt_scale` ~ 1.0 olmali (zaman adimi)
- `kappa_scale_x/y/z` ~ 1.0 olmali (termal difuzyon)

Weight decay bunlari 0'a itmeye calisir. Bu fiziksel olarak yanlis -- adveksiyon sifir olmamali, difuzyon sifir olmamali.

Ornegin `amplitude` parametresi (forcing): init=0.001, clamp [1e-5, 0.1]. Weight decay bunu 0'a iter. Forcing olmadan turbulansi surduremezsin.

**Backscatter icin de sorun:** init=0, weight decay zaten 0'da tutuyor. Backscatter asla ogrenemez (ama bu parametrenin ogrenmemesi iyi olabilir, cunku anti-diffusion tehlikeli).

**Fix (KRITIK):**

```python
# Fizik parametreleri icin weight decay = 0
param_groups = [
    {"params": physics_params, "weight_decay": 0.0},
    {"params": other_params, "weight_decay": 1e-4},
]
```

Aslinda 285 parametrenin HEPSI fizik parametresi. MLP yok. Dolayisiyla `weight_decay=0` olmali. 285 parametre ile overfitting riski zaten yok.


### C3. Warmup -- YETERLI

```python
warmup_epochs: int = 200
lr: float = 3e-4
```

200 epoch warmup, 5000 total epoch icin %4. min_lr=1e-5'ten 3e-4'e linear ramp. Bu makul.

**Ama:** Phase A sadece 1500 epoch. Warmup bunun %13'u. Phase A'da model temel dinamikleri ogrenmeli. 200 epoch warmup ile 1300 epoch gercek ogrenme var. Yeterli.


### C4. LR Schedule -- SORUNSUZ

Cosine annealing, lr -> min_lr=1e-5. Standard ve iyi calisan bir schedule. Phase gecislerinde LR restart dusunulebilir ama mevcut continuous schedule de calisir.


---

## D. Performance Optimizasyonlari

### D1. FFT Tekrar Hesaplama

**Sorun (ORTA):** Her layer'da advection, eddy viscosity, thermal advection, dissipation loss icin ayri ayri FFT yapiliyor. Ornegin tek bir layer'da:

1. Advection3D: `ops.gradient(u)` -> 3 FFT (u, v, w icin 9 gradient = 3 FFT + 9 IFFT)
2. EddyViscosity3D: `_compute_strain_magnitude` -> ayni gradient'ler tekrar
3. ThermalAdvection3D: `ops.gradient(theta)` -> 1 FFT + 3 IFFT
4. ThermalDiffusion3D: `ops.laplacian(theta)` veya `directional_laplacian` -> 1 FFT + 1-3 IFFT
5. Projection3D: `ops.divergence` + `ops.solve_poisson` -> ek FFT'ler

Bir layer icinde velocity field'in FFT'si en az 2 kez hesaplaniyor (advection + eddy viscosity). Theta'nin FFT'si de 2 kez (thermal advection + thermal diffusion).

**Cache mumkun mu?** Zor. Gradient checkpointing recomputation yapiyor, cache invalidate olur. Ancak layer ICINDE (checkpointing bir butun layer'i sarar) gradient ile eddy viscosity arasinda paylasim mumkun.

**Pratik kazanim:** ~%15-20 hizlanma mumkun ama mimari degisiklik gerektirir (layer_step icinde gradients'i bir kez hesapla, sonra advection ve eddy icin paylas).

**Oneri:** Su anki yapi calistiktan sonra optimizasyona gec. Premature optimization yapma.

### D2. torch.compile()

```python
# MPS'te torch.compile erken asamada (PyTorch 2.6)
```

MPS backend'de torch.compile kisitli destek. CUDA'da ise %20-50 hizlanma saglar. Spectral ops (FFT) compile ile iyi calisir.

**Oneri:** CUDA'da egitim yapilacaksa `model = torch.compile(model)` dene. MPS'te deneme -- crash edebilir ama denemekte fayda var.

### D3. Mixed Precision

**Sorun:** Spectral methods float32 hassasiyeti gerektirir. FFT'de float16 numerical error biriktirir ve divergence-free garantisi bozulur. bfloat16 daha iyi ama MPS'te kisitli destek.

**Oneri:** Mixed precision KULLANMA. 285 parametre + 96x160x64 grid. Bottleneck compute degil memory. Float32 ile kalsin.

### D4. Gereksiz .clone() veya .contiguous()

Kod taramasi yapildi. Acik `.clone()` veya `.contiguous()` cagrisi yok. PyTorch'un internal tensor operations zaten gerektiginde contiguous yapar. Temiz.

**Peak Memory Tahmini:**
- Grid: 96 x 160 x 64 = 983,040 float per field
- Fields per state: u, v, w, p, theta = 5 x 983K x 4 bytes = ~19.7 MB
- 20 layer checkpointing ile: ~2 state aktif + recompute = ~40 MB fields
- FFT intermediate: ~60 MB (complex64 buffers)
- Gradient buffers: ~100 MB
- **Toplam tahmini peak: ~300-500 MB** (M-chip unified memory icin rahat)


---

## E. Curriculum Learning

### E1. Phase Gecisleri -- SMOOTH MU?

```python
PHASE_BOUNDARIES = {
    "A": (0, 1500),
    "B": (1500, 3000),
    "C": (3000, 5000),
    "D": (5000, 8000),
}
```

Phase B'de A -> C arasinda linear interpolasyon var. Bu smooth.

Phase A -> B gecisi: epoch 1500'de A weights'ten B interpolasyonuna gecis. B'nin t=0 noktasinda A weights ile baslar, t=1'de C weights'e ulasir. Bu SMOOTH -- ani degisim yok.

Phase B -> C gecisi: epoch 3000'de t=1.0, yani zaten C weights'teyiz. Smooth.

Phase C -> D gecisi: Burada Boussinesq loss'lari C seviyesinden D hedefine ramp-up ile degisir, non-Boussinesq loss'lari 0'dan ramp-up yapar. Bu da smooth.

**Sorun (ORTA): L_divergence 100 -> 10 arasinda Phase D'de duser.** Bu loss zaten gradient vermiyor (A1'de bahsedildi) o yuzden sorun degil. Ama L_continuity_rho 0 -> 50 ve L_mass 0 -> 100 artisi, eger bu loss'larin gradient'leri buyukse, optimizer momentum'unu bozabilir.

**Oneri:** Phase D ramp-up suresini 2000'den 3000'e cikar (daha yavas acilis).

### E2. Re/Ra Sweep

```python
RE_RA_TABLE = {
    "A": {"Re": [5000], "Ra": [1e6]},
    "B": {"Re": [5000, 7000], "Ra": [1e5, 1e6]},
    ...
}
```

Phase A'da tek Re/Ra. Bu iyi -- once bir noktayi ogren. Phase B'de coklu. Phase C'de genis sweep. Bu mantikli curriculum.

**Ama:** `random.choice` kullaniliyor. Bu PyTorch reproducibility'yi bozar (numpy/torch seed ayri). Kucuk sorun ama bilmekte fayda var.


---

## F. Numerical Stability

### F1. Clamp Degerleri -- GENEL OLARAK UYGUN

| Parametre | Clamp | Init | Yorum |
|-----------|-------|------|-------|
| advection_modulator | [0.5, 1.5] | 1.0 | OK |
| dt_scale | [0.5, 2.0] | 1.0 | OK |
| dt_mults | [0.7, 1.5] | 1.0 | OK |
| cs_low/mid/high | [0.01, 0.4] | 0.08/0.15/0.22 | OK (literature range) |
| pr_t | [0.3, 2.0] | 0.85 | OK |
| aniso_ratio_y/z | [0.3, 3.0] | 1.0 | OK |
| backscatter_coeff | [-0.02, 0.0] | 0.0 | OK (tartismali, asagida) |
| buoyancy_strength | [0.0, 50.0] | 1.0 | Ust sinir yuksek ama OK |
| amplitude (forcing) | [1e-5, 0.1] | 0.001 | OK |
| kappa_scale_x/y/z | [0.5, 3.0] | 1.0 | OK |
| thermal_adv_modulator | [0.5, 2.0] | 1.0 | OK |
| velocity clamp | [-20, 20] | -- | OK (Re=5000'de |u|~1-10) |
| theta clamp | [-10, 10] | -- | OK |

### F2. Backscatter -- POTANSIYEL SORUN

```python
bs = torch.clamp(self.backscatter_coeff, -0.02, 0.0)
nu_t = nu_t + bs * self.delta**2 * strain_mag
```

**Sorun (CIDDI): Negatif difuzyon.** `bs * delta^2 * strain_mag` negatif bir terim. Bu nu_t'yi azaltir. Eger nu_t < 0 olursa anti-diffusion -- ustel buyume.

Hesap yapalim: nu_molecular = 1/5000 = 2e-4. nu_t (Smagorinsky) ~ (0.15 * delta)^2 * |S|. delta = (dx*dy*dz)^(1/3) ~ (0.0625 * 0.0625 * 0.0625)^(1/3) = 0.0625. Yani nu_t ~ (0.15 * 0.0625)^2 * |S| ~ 8.8e-5 * |S|. Backscatter: bs * delta^2 * |S| = -0.02 * 0.0625^2 * |S| = -7.8e-5 * |S|. Bu, nu_t'nin neredeyse tamami kadar negatif deger!

**Nu_eff = nu_mol + nu_t + bs_term = 2e-4 + 8.8e-5*|S| - 7.8e-5*|S| = 2e-4 + 1e-5*|S|.** Bu durumda net nu_t cok kucuk kalir. Yuksek |S| bolgelerinde bile pozitif kalir (nu_mol kurtarir). Ama eger nu_mol kuculdugunde (yuksek Re) sorun olabilir.

**Mevcut durum:** backscatter init=0, weight decay sifira iter. Fiilen kapalida kalacak. Ama gradient onu negatife itebilir.

**Oneri:** Backscatter'i Phase A-B'de tamamen kapat (`requires_grad=False`). Phase C'de ac.


### F3. Division by Zero -- KORUNMALAR YETERLI

- `spectrum / (counts + 1e-10)` -- OK
- `slope denominator + 1e-10` -- OK
- `nusselt: / (kappa + 1e-10)` -- OK
- `strain_mag: sqrt(2*S_sq + 1e-8)` -- OK

### F4. Log/Exp Overflow -- YOK

Kodda `torch.exp` veya `torch.log` direkt kullanimi yok (spectrum loss'taki log haric, orada `spectrum > 1e-20` guard var).

### F5. NaN Guard -- MODEL ICINDE VAR, LOSS ICINDE YOK

```python
# model.py layer_step sonu:
u = torch.clamp(u, -20.0, 20.0)
v = torch.clamp(v, -20.0, 20.0)
w = torch.clamp(w, -20.0, 20.0)
theta = torch.clamp(theta, -10.0, 10.0)
```

**Sorun (ORTA):** Loss hesabinda NaN check yok. Eger herhangi bir loss NaN olursa, total loss NaN olur, backward NaN gradient uretir, tum parametreler NaN olur.

**Oneri:** Loss hesabi sonrasinda NaN guard ekle:

```python
if torch.isnan(step_loss):
    print(f"  WARNING: NaN loss at step {_step}, skipping backward")
    continue
```


---

## G. Evaluate.py -- SORUNLAR

### G1. eval_energy_balance'da enstrophy hatasi

```python
Z = 0.5 * (s0.enstrophy() + s1.enstrophy())
```

**HATA:** `ThermalFluidState.enstrophy()` `NotImplementedError` raise ediyor! Bu fonksiyon calistirildiginda crash verecek.

```python
def enstrophy(self) -> torch.Tensor:
    raise NotImplementedError(
        "ThermalFluidState.enstrophy() ops.curl gerektirir. "
    )
```

PhysicsLoss._enstrophy(state) kullanilmali, ama evaluate.py bunu import etmiyor.

**Fix:** evaluate.py'de `eval_energy_balance` icinde model.ops.curl kullanarak enstrophy hesapla.


### G2. DNS Comparison Grid Mismatch

DNS comparison'da grid kucultme yapiliyor (32x48x24) ama model spectral ops grid-bagli. Checkpoint'taki model farkli grid boyutuyla yuklenmez. Kod bunu try/except ile yakaliyor ama sonuc: untrained model ile karsilastirma. Anlamsiz.

**Oneri:** DNS comparison icin model'i ayni grid boyutuyla olustur veya resolution-invariant test yaz.


---

## H. Genel Mimari Degerlendirme

### H1. 285 Parametre Yeterli mi?

3D turbulans (96x160x64 grid, Re=5000-10000, Ra=1e5-1e7): Bu COKLU fizik problemi. 285 parametre ile:

- Her layer'da ~14 parametre (20 layer)
- Per-layer: 1 advection_mod + 3 cs + 1 pr_t + 2 aniso + 1 backscatter + 3 kappa_scale + 1 thermal_adv_mod + 1 buoyancy = 13 param
- Global: 1 dt_scale + 19 dt_mults + 5 forcing = 25 param

Fizik tabanlı model oldugu icin bu YETERLI olabilir. MLP'siz yaklaşim dogru -- fizik operatorleri kendi iclerinde yuzlerce denklem cozmekle gectikleri icin parametreler sadece "ne kadar" kararini veriyor. Bu "saf INNATE" felsefi olarak dogru.

**Ama risk:** Eger bazı parametreler gradient almazsa (dead parameters), efektif parametre sayisi 285'ten cok daha az olabilir. Ozellikle divergence sonrasi ve backscatter (init=0, weight decay) endiselerim var.


### H2. Training Suresi Tahmini

- 1 epoch: 20 layer * 10 step * (advection + eddy + projection + thermal) per layer
- Tahmini per-epoch: 96x160x64 gridde MPS'te ~0.5-2 saniye
- 5000 epoch: ~40 dakika - 2.5 saat (MPS)
- 8000 epoch (Phase D dahil): ~1-4 saat

Bu M-chip icin makul. Eger CUDA varsa 2-5x hizlanir.


---

## I. Oncelikli Aksiyon Plani

### KRITIK (hemen yap):

1. **Weight decay = 0 yap.** Tum parametreler fizik parametresi, weight decay zarar veriyor.
   ```python
   weight_decay: float = 0.0  # config.py'de
   ```

2. **Energy balance loss'ta dt uyumsuzlugunu fix et.** model._get_layer_dt() ile ayni hesabi kullan.

3. **evaluate.py'deki enstrophy bug'ini fix et.** `state.enstrophy()` yerine `ops.curl` bazli hesap.

### YUKSEK (bu hafta):

4. **Nusselt loss'ta Ra'yi guncel model parametresinden al.** `self.model.Ri` uzerinden.

5. **NaN guard ekle** training loop'a (loss NaN ise skip).

6. **Gradient clipping'i 5.0'a cikar.** 285 parametre icin 1.0 cok agresif.

### ORTA (daha sonra):

7. Divergence loss weight'i 1.0'a dusur (veya kaldır).
8. Dissipation loss weight'i Phase A'da 0, Phase B+ da kucuk tut.
9. Backscatter'i Phase A-B'de kapat.
10. Phase D ramp-up suresini 3000 epoch'a cikar.

### DUSUK (optimizasyon fazinda):

11. Layer icinde FFT cache paylas (advection + eddy).
12. torch.compile() CUDA'da dene.
13. Per-layer gradient norm monitoring ekle.


---

## J. Discussion Notu

Debugger agentin NaN patlamasi analizi tutarli. `_get_layer_dt()` clamp araligi (eski: [0.1, 3.0] * [0.5, 2.0] = max 6x) duzeltilmis ([0.5, 2.0] * [0.7, 1.5] = max 3x). Advection modulator clamp [0.5, 1.5] eklenmis. Bu fix'ler dogru.

Ancak energy_balance_loss'taki dt HALA eski clamp araligini kullaniyor ([0.1, 3.0]). Bu fix edilmeli.

---

*Rapor sonu. Dosya: `/Users/apple/Desktop/nsneuron/bitirme2/reports/ml_report.md`*
