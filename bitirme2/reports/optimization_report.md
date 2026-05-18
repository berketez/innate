# INNATE3D Optimizasyon Raporu

**Tarih:** 2026-02-23
**Hazirlayan:** Gemini Consultant Agent (Gemini 3 Pro + Claude Opus analizi)
**Proje:** INNATE3D Mixed Convection -- 285 parametre, saf fizik tabanli 3D turbulan simuelasyonu

---

## Yonetici Ozeti

Bu rapor INNATE3D projesinin mimari, performans, egitim stratejisi, fizik constraint'leri ve olceklenebilirlik boyutlarini analiz eder. Gemini 3 Pro'dan alinan gorusler Claude Opus'un kod analizi ile karsilastirilarak sentezlenmistir.

**En kritik 5 oneri (oncelik sirasina gore):**

1. **Shared parametreler kullan** -- Per-layer parametreler fizik kurallarinin zamansal invariansina aykiri. 285 -> ~15 parametre, cok daha iyi generalization.
2. **Spectral uzayda kal** -- FFT cagrilarini %80 azalt. Lineer terimleri spectral uzayda hesapla, sadece nonlineer advection icin fiziksel uzaya don.
3. **Uncertainty weighting** -- Elle curriculum agirliklari yerine ogrenilebilir sigma_i parametreleri (Kendall et al.). Gradient conflict azalir.
4. **Energy flux constraint** -- Slope-5/3 yerine Pi(k) = epsilon zorlama. Fiziksel olarak cok daha guclu constraint.
5. **Dusuk cozunurluklu pre-training + spectral zero-padding** -- 48x80x32'de egit, 96x160x64'e transfer et. Egitim suresi 4-8x azalir.

---

## 1. Mimari Alternatifler

### 1.1 Per-Layer vs Shared Parametreler

| Ozellik | Per-Layer (mevcut) | Shared |
|---------|-------------------|--------|
| Parametre sayisi | 285 | ~15 |
| Fizik tutarliligi | Dusuk (Pr_t her adimda farkli?) | Yuksek (fizik sabiti = sabit) |
| Generalization | Zayif (20 adim sonrasi extrapolation) | Guclu (ayni operator sonsuz tekrar) |
| Overfitting riski | Yuksek | Dusuk |
| Egitim kolayligi | Zor (285 param arasi etkilesim) | Kolay |

**Gemini'nin Gorasu:**
> Fizik kurallari zaman adamindan zaman adimina degismez. Per-layer Pr_t PDE solver'in ruhuna aykiri ve long-term stability'yi bozar. Kesinlikle shared parametreler daha iyi.

**Claude'un Gorasu:**
> Per-layer parametreler "egitim sirasinda farkli katmanlarin farkli zaman olceklerini yakalasin" amacli tasarlanmis. Ancak pratikte bu, fiziksel anlamsiz farklar olusturuyor. Shared yaklasima gecis mantikli.

**Ortak Karara:** Shared parametreler kullanilmali. Eger zaman-bagimli davranis isteniyorsa, tek parametre setini `cos(pi * layer_idx / n_layers)` gibi deterministik bir moduelasyon ile scale edin.

**Uygulama:**
```python
# ONCE (285 parametre)
self.eddy_viscosities = nn.ModuleList([EddyViscosity3D(...) for _ in range(20)])

# SONRA (~15 parametre)
self.eddy_viscosity = EddyViscosity3D(...)  # tek instance, 20 kez kullan
self.advection = Advection3D(...)           # tek instance
self.thermal_advection = ThermalAdvection3D(...)
# ... diger noronlar
```

### 1.2 Multi-Scale / Residual Connections

**Gemini:** Onermiyor. FFT zaten multi-scale: frekans bandlari (low/mid/high) bu isi yapiyor.

**Claude:** Katiliyorum. Band-filtered SGS (cs_low, cs_mid, cs_high) zaten multi-scale yaklasim. Residual connection eklemeye gerek yok -- fractional-step zaten Euler integrator formunda u_{n+1} = u_n + dt * RHS, bu implicit bir residual connection.

### 1.3 Gradient Checkpointing

**Mevcut Durum:** Tum `_layer_step` checkpoint ediliyor (layer bazinda).

**Gemini Onerisi:** Sadece nonlineer terimleri (Advection, dealiasing) checkpoint yap. Lineer terimlerin (Diffusion, Projection) turevleri analitik ve bellek ayak izi dusuk.

**Claude Degerlendirmesi:** Bu oneri pratikte zor. PyTorch'un `checkpoint` mekanizmasi fonksiyon bazli calisir, terim bazli degil. Mevcut strateji (layer bazinda) zaten optimal. Iyilestirme icin `use_reentrant=False` (zaten kullaniliyor) yeterli.

**Sonuc:** Mevcut gradient checkpointing stratejisi yeterli. Shared parametrelere gecis bellegi %90 azaltacagindan checkpointing bile gereksiz hale gelebilir.

---

## 2. Performans

### 2.1 FFT Optimizasyonu (EN KRITIK)

**Mevcut Durum:** Her layer'da su FFT cagrilari var:
- Advection: 3x gradient = 3 FFT + 9 IFFT + 3 dealias = 3+9+3+3=18 FFT/IFFT
- EddyViscosity (strain): 3x gradient = 3 FFT + 9 IFFT = 12
- EddyViscosity (band_filter): 3x = 3 FFT + 3 IFFT = 6
- Anisotropic diffusion: 3x directional_laplacian = 3 FFT + 9 IFFT = 12
- Projection: divergence + solve_poisson + gradient = 3+1+2 FFT + 1+1+3 IFFT = 11
- ThermalAdvection: gradient + dealias = 1+3+1+1 = 6
- ThermalDiffusion: directional_laplacian = 1+3 = 4

**Toplam: ~70 FFT/IFFT islem PER LAYER = 1400 islem per forward pass (20 layer)**

**Gemini Onerisi: "Spectral uzayda kal"**
> Lineer terimlerin (viskozite, difuzyon, basinc projeksiyonu) tumu spektral uzayda sadece bir matris carpimidir. FFT/IFFT'yi sadece nonlineer advection terimi icin kullanin.

**Claude Degerlendirmesi:**
Bu oneride dogru ve yanlis yonler var:
- DOGRU: Laplacian spectral uzayda -k^2 * f_hat carpimi. FFT+IFFT tasarruf edilir.
- DOGRU: Projection spectral uzayda tamamen yapilabilir (zaten oyle).
- YANLIS: Advection (u * du/dx) nonlineer oldugu icin fiziksel uzaya donmek ZORUNLU. Ancak mevcut kodda `u` zaten fiziksel uzayda, dolayisiyla optimizasyon "ortak FFT" yonunde olmali.

**Pratik Optimizasyon Onerileri:**

```
ONCELIK 1: Ortak FFT hesapla, sonucu paylas

# Bir kez hesapla:
u_hat, v_hat, w_hat = FFT(u), FFT(v), FFT(w)

# Gradient: ik * f_hat (IFFT gerekli, carpim fiziksel uzayda)
du_dx = IFFT(i*kx * u_hat)  # zaten boyle ama u_hat'i tekrar hesaplama

# Laplacian: -k^2 * f_hat (IFFT gerekli AMA diffusion = nu*laplacian
#   fiziksel uzayda toplanacaksa IFFT'yi ertele)

# Projection: tamamen spectral: p_hat = -div_hat / k^2, u_hat -= i*kx*p_hat
```

**Tahmini Kazanc:** FFT cagrilarini ~70'den ~25-30'a dusurur (%60 azalma).

```python
# Onerilern pseudo-implementasyonu
def _layer_step_optimized(self, layer_idx, u, v, w, p, theta, rho=None):
    dt = self._get_layer_dt(layer_idx)

    # (1) ORTAK FFT: tum alanlar icin tek sefer
    u_hat = torch.fft.fftn(u, dim=(-3,-2,-1))
    v_hat = torch.fft.fftn(v, dim=(-3,-2,-1))
    w_hat = torch.fft.fftn(w, dim=(-3,-2,-1))
    theta_hat = torch.fft.fftn(theta, dim=(-3,-2,-1))

    # (2) Gradient (fiziksel uzayda carpim icin IFFT gerekli)
    du_dx = torch.fft.ifftn(1j*self.ops.kx * u_hat, dim=(-3,-2,-1)).real
    # ... (9 gradient bilesen -- bunlar kacinilmaz)

    # (3) Advection (fiziksel uzayda -- nonlineer)
    adv_u = u*du_dx + v*du_dy + w*du_dz  # IFFT zaten yapildi
    # dealias (1 FFT + 1 IFFT per bilesen)

    # (4) Diffusion SPECTRAL UZAYDA (IFFT'yi momentum update'de yap)
    # diff_u_hat = nu_eff * (-k_squared) * u_hat  -- NO IFFT needed yet

    # (5) Momentum update SPECTRAL UZAYDA
    # u_hat_new = u_hat + dt * (-FFT(adv_u) + FFT(Fx) + diff_u_hat)
    # (6) Projection SPECTRAL UZAYDA
    # div_hat = i*kx*u_hat_new + i*ky*v_hat_new + i*kz*w_hat_new
    # p_hat = -div_hat / k_squared_poisson
    # u_hat_new -= dt * i*kx * p_hat

    # (7) SON IFFT: spectral'den fiziksel uzaya
    u = torch.fft.ifftn(u_hat_new, dim=(-3,-2,-1)).real
    # ...
```

### 2.2 torch.compile()

**Gemini:** Kullanilabilir ama FFT graph break yapabilir. Sadece nonlineer parcalari derleyin.

**Claude:** PyTorch 2.x'te `torch.compile(mode="reduce-overhead")` FFT ile calisir. Ancak MPS backend'de compiler destegi sinirli. CUDA'da deneyin.

**Oneri:** CUDA'ya gecildiginde `torch.compile(model, mode="reduce-overhead")` deneyin. MPS'te kullanmayin.

### 2.3 Mixed Precision

**Gemini:** "Hayir, turbulansta tehlikeli. Float16 yuksek frekanstaki enerjiyi sifira yuvarlar (underflow), simuelasyon patlar."

**Claude:** Katiliyorum. 96x160x64 gridde en yuksek frekans bilesenlerinin enerjisi ~1e-8 seviyesinde olabilir. Float16 min subnormal ~6e-8, bu underflow'a dogru. **Kesinlikle float32 kullanilmali.** bfloat16'nin exponent range'i yeterli olsa da 285 parametre icin hiz kazanci sifir.

### 2.4 Batch Size

**Gemini:** Batch size = 1 tutup num_steps artirin. GPU bellegi yetmeyecektir.

**Claude:** Hesaplama: 96x160x64 x float32 = 3.93 MB/alan x 5 alan (u,v,w,p,theta) x 2 (grad) = ~40 MB/state. 20 intermediate + grad = ~2 GB. Checkpoint ile ~500 MB. Batch=2 mumkun ama fayda sinirli. Stokastik IC + Re/Ra sweep zaten varyans sagliyor.

**Oneri:** Batch size = 1 tutun. num_steps'i (suan 10) 15-20'ye cikarin.

### 2.5 MPS vs CUDA

| Ozellik | MPS (Apple Silicon) | CUDA (NVIDIA) |
|---------|-------------------|---------------|
| 3D FFT hizi | Yavas (2-5x) | Hizli (cuFFT) |
| Complex tensor destegi | Sinirli | Tam |
| torch.compile | Sinirli | Tam |
| Bellek | Unified (paylasilir) | Dedicated VRAM |
| Gelistirme | Uygun | Prod. icin sart |

**Oneri:** Gelistirme MPS'te, ciddi egitim CUDA'da. Colab/Lambda/Paperspace gibi bulut GPU kullanin.

---

## 3. Egitim Stratejisi

### 3.1 Optimizer: Adam vs L-BFGS

**Gemini:** "L-BFGS teorik olarak harika ama stokastik kayiplari sevmez. AdamW mevcut curriculum icin dogru secim."

**Claude:** Katiliyorum. 285 parametre icin L-BFGS cazip ama:
- Her epoch'ta farkli IC (stokastik)
- Her epoch'ta farkli Re/Ra (curriculum)
- Loss landscape cok degisken

AdamW mevcut yaklasim icin uygun.

**Ancak: Hibrit yaklasim dikkate deger:**
```python
# Ilk 3000 epoch: AdamW (stokastik curriculum)
# Son 2000 epoch: L-BFGS (sabit Re/Ra, deterministik IC)
if epoch > 3000:
    # L-BFGS icin sabit kosullar
    Re, Ra = 5000, 1e6  # sabit
    state = model.create_initial_condition(seed=42)  # deterministik
    optimizer = torch.optim.LBFGS(model.parameters(), lr=1e-3, max_iter=20)
```

### 3.2 Data-Free vs DNS Pre-Training

**Gemini:** "Data-free training kaotik sistemlerde trivial cozume (u=0) gitme egilimindedir. Kaba DNS verisi ile pre-training yapin."

**Claude:** Bu cok onemli bir tespit. Mevcut loss yapisi (divergence + energy balance + spectrum) trivial cozumu destekleyebilir:
- u=0 => divergence=0 (mukemmel)
- u=0 => energy balance: dE/dt=0, eps=0, P=0 (mukemmel)
- u=0 => spectrum loss: belirsiz (sifir enerji)

**Ancak:** Nusselt loss (Nu >= 1) ve buoyancy forcing bunu onlemeye calisiyor. Pratikte calisiyor mu, test edilmeli.

**Oneri:**
1. 32^3 veya 48^3 gridde basit bir pseudo-spectral DNS yazin (500 satir Python yeterli)
2. Re=1000, Ra=1e4 gibi dusuk parametrelerde 10-20 zaman adimi uretun
3. Tek forward-pass veri uzerinden MSE loss ile pre-train edin (50-100 epoch)
4. Sonra physics-only fine-tuning'e gecin

Bu "warm start" modeli dogru manifolda baslatir. **Maliyet: 1 saatlik is, fayda: gunlerce curriculum yerine saatlerce egitim.**

### 3.3 TBPTT vs Adjoint / Shooting Methods

**Gemini:** "TBPTT'ye devam edin ama 'Pushforward' stratejisi uygulayin: Ilk epochlarda loss'u sadece t=1,2 adimlarinda hesaplayin, egitim ilerledikce artirin."

**Claude:** Pushforward (veya "unrolling schedule") cok iyi bir oneri:
```python
# Curriculum'a unrolling schedule ekle
UNROLL_SCHEDULE = {
    "A": 2,    # Ilk fazda 2 step (40 layer)
    "B": 5,    # 5 step (100 layer)
    "C": 10,   # 10 step (200 layer)
    "D": 10,
}
```

**Adjoint method** hakkinda: 285 parametre icin overkill. Adjoint'in faydasi bellek tasarrufu (O(1) vs O(T)), ancak gradient checkpointing zaten bunu yapiyorz. Implement etmeye degmez.

**Multiple Shooting** hakkinda: IC'leri birden fazla noktada baslatirak paralel egitim yapilabilir ama pratikte TBPTT + pushforward yeterli.

### 3.4 Otomatik Loss Agirliklama

**Gemini:** "Uncertainty Weighting (Kendall et al.) kullanin."

**Claude:** Bu en pratik ve etkili oneri. Mevcut elle ayarlanmis 7+ loss agirligi yerine:

```python
class UncertaintyWeighting(nn.Module):
    """Kendall et al. 2018 - Multi-Task Learning uncertainty weighting."""
    def __init__(self, n_losses: int):
        super().__init__()
        # log(sigma^2) parametreleri -- negatif olabilir
        self.log_vars = nn.ParameterList([
            nn.Parameter(torch.tensor(0.0)) for _ in range(n_losses)
        ])

    def forward(self, losses: List[torch.Tensor]) -> torch.Tensor:
        total = torch.tensor(0.0, device=losses[0].device)
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total = total + precision * loss + self.log_vars[i]
        return total
```

**Faydalari:**
- Farkli buyukluk siralarindaki loss'lar otomatik dengelenir
- Gradient conflict azalir (spectrum loss ~0.1 vs divergence loss ~1e-6)
- Phase A/B/C curriculum agirliklari gereksiz hale gelir

**Ek parametre maliyeti:** +7 (log_vars) = toplam 292 parametre. Ihmal edilir.

**UYARI:** Phase D (Non-Boussinesq) icin hala manual ramp-up gerekebilir cunku yeni loss terimleri (continuity_rho, mass) onceki parametreleri bozabilir.

---

## 4. Fizik-Informed Yaklasimlar

### 4.1 Spectral Loss Iyilestirmesi

**Mevcut:** Sadece inertial range slope -> -5/3.

**Gemini Onerisi: Energy Flux Constraint**
> Pi(k) = epsilon (inertial range'de sabit enerji akisi) zorlayici cok daha guclu constraint.

**Claude:** Bu dogru ve uygulanabilir. Implementasyon:

```python
def energy_flux_loss(u, v, w, ops, nu):
    """
    Spectral energy flux Pi(k) = -dT(k)/dk
    Inertial range'de Pi(k) = epsilon (sabit) olmali.
    """
    # Transfer spectrum T(k) hesapla
    # T(k) = sum_{|p|=k} Re[ u_hat(p) * (u x omega)_hat(-p) ]
    # Basitlestirmis versiyon: Pi(k) = epsilon - 2*nu*k^2*E(k)

    spectrum = compute_energy_spectrum(u, v, w, ops)
    k = torch.arange(len(spectrum), device=u.device, dtype=torch.float32) + 1

    # Inertial range'de Pi(k) ~ sabit
    k_min, k_max = 4, 20
    mask = (k >= k_min) & (k <= k_max) & (spectrum > 1e-20)
    if mask.sum() < 4:
        return torch.tensor(0.0, device=u.device)

    Pi_k = 2 * nu * (k[mask]**2) * spectrum[mask]  # dissipation at wavenumber k
    # Pi(k) varyansini minimize et (sabit olmali)
    return Pi_k.var() / (Pi_k.mean()**2 + 1e-10)
```

### 4.2 Energy Cascade Constraint

**Gemini:** "Yuksek frekans dissipasyonunun dusuk frekanstaki enerji uretimini karsilamasi gerekir."

**Claude:** Bu zaten energy_balance_loss'ta implicit var ama frekans bazinda explicit zorlanmasi faydali:

```python
def cascade_balance_loss(u, v, w, ops, nu):
    """
    Band-bazli enerji dengesi:
    P_low (production at low-k) ~= D_high (dissipation at high-k)
    """
    # Low-band kinetic energy
    u_low = ops.band_filter(u, 'low')
    v_low = ops.band_filter(v, 'low')
    w_low = ops.band_filter(w, 'low')
    E_low = 0.5 * (u_low**2 + v_low**2 + w_low**2).mean()

    # High-band dissipation
    u_high = ops.band_filter(u, 'high')
    v_high = ops.band_filter(v, 'high')
    w_high = ops.band_filter(w, 'high')
    lap_u_h = ops.laplacian(u_high)
    lap_v_h = ops.laplacian(v_high)
    lap_w_h = ops.laplacian(w_high)
    D_high = -nu * (u_high*lap_u_h + v_high*lap_v_h + w_high*lap_w_h).mean()

    # Oran ~ O(1) olmali
    ratio = E_low / (D_high + 1e-10)
    return (torch.log(ratio + 1e-10))**2  # log-scale balance
```

### 4.3 Nusselt Optimizasyonu

**Gemini:** "Globe-Dropkin regresyonu yerine duvar termal gradyani kullanin."

**Claude:** Periodikt BC'de "duvar" yok (tam periodikt domain). Bu oneri sinirli sinir kosullari (wall-bounded flow) icin gecerli. Mevcut mixed convection'da y-yonunde de periodikt BC var, dolayisiyla duvar gradyani tanimsiz.

**Alternatif Oneri:** Mevcut bidirectional Nusselt loss iyi tasarlanmis. Iyilestirme icin:
1. Globe-Dropkin hedefini Ra'ya gore interpolation yaparak daha hassas hedef olustur
2. Nusselt'i birden fazla y-kesitinde hesapla (y=Ly/4, Ly/2, 3Ly/4) ve tutarliligi zorla
3. Nusselt'in zaman serisindeki varyansini da constraint olarak ekle (stationary state'te sabit olmali)

---

## 5. Scaling

### 5.1 Yuksek Re'ye Olcekleme

**Mevcut:** Re=5000-10000, 96x160x64 grid.

| Re | Grid Gereksinimi (LES) | Mevcut Grid Yeterliligi | SGS Yukue |
|----|----------------------|----------------------|-----------|
| 5,000 | ~80x120x50 | Yeterli (80/100) | Orta |
| 10,000 | ~120x200x80 | Sinirda (74/100) | Yuksek |
| 50,000 | ~300x500x200 | Yetersiz | Cok yuksek |
| 100,000 | ~500x800x350 | Imkansiz | Asiri |

**Gemini:** "Re=50K ustu icin Dynamic Smagorinsky gerekli. Parametreleri f(Re, delta_x) olarak parametrize edin."

**Claude:** Katiliyorum. Mevcut sabit Cs clamp araliklari (0.01-0.4) Re=50K'da yetersiz kalir. Oneriler:

1. **Re-dependent parameterization:**
```python
# Cs = Cs_base * (Re / Re_ref)^alpha
# alpha ogrenilebilir parametre
self.cs_base = nn.Parameter(torch.tensor(0.15))
self.cs_re_exponent = nn.Parameter(torch.tensor(0.0))  # init=0 (Re-bagimsiz)
# Forward:
Cs_eff = self.cs_base * (Re / 5000.0) ** self.cs_re_exponent
```

2. **Grid-aware parametreler:**
```python
# delta / eta oranina gore adaptif Cs
# eta = (nu^3 / epsilon)^(1/4) Kolmogorov microscale
# delta / eta buyudukce (under-resolved) Cs artmali
```

### 5.2 Grid Refinement Stratejisi

**Gemini:** "Kucuk gridde egit, spectral zero-padding ile buyut."

**Claude:** Bu cok pratik ve dogru bir oneri. Spectral method'un en buyuk avantaji:

```python
def spectral_upsample(field_coarse, target_shape):
    """
    48x80x32 -> 96x160x64 spectral zero-padding.
    Fizik katsayilari (Cs, Pr_t vb.) degismez!
    """
    f_hat = torch.fft.fftn(field_coarse, dim=(-3,-2,-1))
    # Zero-pad: yuksek frekanslara sifir ekle
    Nx, Ny, Nz = target_shape
    f_hat_fine = torch.zeros((*f_hat.shape[:-3], Nx, Ny, Nz),
                             dtype=f_hat.dtype, device=f_hat.device)
    # Dusuk frekanslari kopyala
    nx, ny, nz = f_hat.shape[-3:]
    f_hat_fine[..., :nx//2, :ny//2, :nz//2] = f_hat[..., :nx//2, :ny//2, :nz//2]
    # Normalize et (boyut orani)
    scale = (Nx*Ny*Nz) / (nx*ny*nz)
    return torch.fft.ifftn(f_hat_fine * scale, dim=(-3,-2,-1)).real
```

**Onerilen Egitim Pipeline'i:**
1. Phase 0 (pre-training): 32^3, Re=1000, 500 epoch (dakikalar)
2. Phase A: 48x80x32, Re=5000, 1500 epoch (saatler)
3. Phase B-C: 96x160x64'e spectral transfer, 3500 epoch (saatler)
4. Phase D: Non-Boussinesq, ayni grid

---

## 6. Claude vs Gemini Karsilastirma Tablosu

| Konu | Claude | Gemini | Uzlasma |
|------|--------|--------|---------|
| Shared vs Per-layer | Shared tercih | Shared kesinlikle | SHARED |
| FFT optimizasyonu | Ortak FFT + lineer terimleri spectral'de tut | "Tamamen spectral uzaya tas" | Hibrit: lineer spectral, nonlineer fiziksel |
| Mixed precision | Hayir (float32 sart) | Hayir (float32 sart) | FLOAT32 |
| L-BFGS | Hibrit (son fazda) | Sadece deterministik batch'te | HIBRIT yaklasim |
| DNS pre-training | Faydali ama zorunlu degil | "Cok onemli, trivial cozum riski var" | ONERILIYOR (warm start) |
| Uncertainty weighting | Cok faydali (+7 param) | Oneriliyor | UYGULANMALI |
| Energy flux constraint | Uygulanabilir | "Cok daha guclu constraint" | EKLENECEK |
| Nusselt / duvar gradient | Periodikt BC'de gecersiz | Sinirli BC varsayiyor | MEVCUT YETERLI |
| Grid refinement | Spectral zero-padding | Spectral zero-padding | TAMAMEN UZLASTI |
| Dynamic Smagorinsky | Re-dependent parametrize | Gerekli Re>50K icin | PLAN'A ALINMALI |

---

## 7. Uygulama Oncelik Sirasi

### Dusuk Efor, Yuksek Etki (Hemen Uygulanabilir)
1. **Uncertainty weighting ekle** -- ~30 satir kod, curriculum agirliklari otomatik
2. **Pushforward unrolling schedule** -- num_steps'i curriculum'a bag (2->5->10)
3. **Ortak FFT hesaplama** -- u_hat/v_hat/w_hat once hesapla, paylas

### Orta Efor, Yuksek Etki (1-2 gun)
4. **Shared parametrelere gec** -- model.py refactor
5. **Energy flux loss ekle** -- yeni loss terimi (~50 satir)
6. **Dusuk cozunurluk pre-training** -- 48x80x32 config + spectral upsample

### Yuksek Efor, Orta Etki (Gelecek sprint)
7. **DNS warm start** -- basit spectral DNS yazilmali
8. **Re-dependent SGS parametrizasyonu** -- scaling icin
9. **Hibrit optimizer** -- AdamW -> L-BFGS gecis

---

## 8. Sonuc

INNATE3D saf fizik tabanli tasarimi ile benzersiz bir yaklasim. 285 parametre ile 3D turbulan simuelasyonu yapmak hem cesur hem de fizik-informed. Ancak:

1. Per-layer parametreler fiziksel olarak anlamsiz farklar yaratiyorz. Shared'a gecin.
2. FFT kullanimi optimize edilmeli -- mevcut haliyle %60'i gereksiz tekrar.
3. Loss agirliklari elle tuning yerine uncertainty weighting ile otomatiklestirin.
4. Energy flux constraint spectrum loss'tan cok daha guclu bir fiziksel zorlama.
5. Grid refinement (dusuk -> yuksek cozunurluk transfer) egitim suresini dramatik azaltir.

Bu 5 degisiklik birlikte uygulandiginda, egitim suresinin 3-5x kisaldigini, stability'nin artti ve fiziksel dogrulukgun iyilestigini bekliyorum.
