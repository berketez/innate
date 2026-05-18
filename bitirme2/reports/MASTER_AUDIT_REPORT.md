# INNATE3D Master Audit Report
**Tarih:** 2026-02-23 | **6 Uzman, 500K+ token analiz**

Reviewer (Opus) + CFD Expert (Opus) + ML Expert (Opus) + Debugger (Opus) + Codex (GPT-5.3) + Gemini (3 Pro)

---

## EXECUTIVE SUMMARY

30 formul kontrol edildi: 24 dogru, 4 hatali, 2 supheli.
18 kod inceleme bulgusi: 2 critical, 6 high, 6 medium, 4 low.
14 numerik bug/risk: 4 critical, 5 high, 5 medium.
15 ML/training bulgusi: 3 critical, 5 high, 7 medium.

**Toplam: 10 CRITICAL bug** (egitimi dogrudan patlatan veya yanlis ogreten)

---

## TIER 0: CRITICAL BUGS (Egitimi Patlatan / Yanlis Ogreten)

### BUG-01: Dissipation Rate 2x Hatasi [Codex]
**Dosya:** `train.py:207`
**Sorun:** `eps = 2.0 * phys.nu * Z` ama Z zaten `<omega^2>` donduruyor. Dogru: `eps = nu * <omega^2>`. Dissipation 2 kat fazla hesaplaniyor.
**Etki:** Energy balance loss ASLA dogru olamaz. Dissipation loss'un fiziksel ve spectral tarafi 2x farkli → loss hicbir zaman sifir olamaz → model surekli yanlis gradient aliyor.
**Fix:** `eps = phys.nu * Z` (2.0 carpanini kaldir)

### BUG-02: Dissipation Loss Tutarsizligi [Codex]
**Dosya:** `train.py:238-260`
**Sorun:** Fiziksel taraf `2*nu*<omega^2>` (yanlis), spectral taraf `nu*<omega^2>` (dogru). Fark = epsilon.
**Etki:** Loss asla sifir olamaz. Surekli gecersiz gradient.
**Fix:** Fiziksel tarafi da `nu*<omega^2>` yap (BUG-01 ile birlikte duzeltilir)

### BUG-03: CFL Ihlali [CFD Expert]
**Dosya:** `model.py:281, config.py:57`
**Sorun:** dt_base=0.01, max dt_mult=2.0 → eff_dt=0.02. Velocity clamp=20 → CFL = 20*0.02/0.0625 = 6.4. Explicit Euler'de CFL > 1 = patlama.
**Etki:** Yuksek hiz bolgeleri olusmaya baslayinca (epoch ~200+) NaN.
**Fix:** (a) dt_base=0.003 veya (b) adaptif CFL veya (c) velocity clamp=5

### BUG-04: Advection3D Skew-Symmetric Degil [CFD Expert]
**Dosya:** `innate.py:2429-2457` (Advection3D.forward)
**Sorun:** 2D Advection skew-symmetric form kullaniyor (1/2 convective + 1/2 divergence). 3D versiyonu SADECE convective form.
**Etki:** Kinetik enerjide drift, 200 adimlik unrolling'de birikerek patlama.
**Fix:** 3D'ye de skew-symmetric form ekle: `0.5*(u*du/dx + d(uu)/dx) + ...`

### BUG-05: Weight Decay Fizik Parametrelerini Olduryor [ML Expert]
**Dosya:** `train.py` optimizer kurulumu
**Sorun:** weight_decay=1e-4 TUM 285 parametreye uygulanyor. Bunlar fizik sabitleri (Cs~0.15, Pr_t~0.85). Weight decay bunlari 0'a iter.
**Etki:** `forcing.amplitude` (init=0.001) weight decay ile ogrenme sansi bulamadan sifirlanir. Tum parametreler fiziksel degerlerinden uzaklasir.
**Fix:** `weight_decay=0` yap. MLP yok, overfitting riski yok.

### BUG-06: dt_scale Clamp Tutarsizligi [Debugger + ML + Reviewer]
**Dosya:** `model.py:281` vs `train.py:193`
**Sorun:** Model: `clamp(dt_scale, 0.5, 2.0)`. Loss: `clamp(dt_scale, 0.1, 3.0)`. Farkli dt degerleri.
**Etki:** Enerji dengesi yanlis dt ile hesaplaniyor. Loss sifir olsa bile gercek enerji korunmuyor.
**Fix:** Tek bir `get_effective_dt()` metodu yap, her yerde onu kullan.

### BUG-07: Backscatter Negatif Difuzyon [Debugger + Codex + CFD]
**Dosya:** `innate.py:3698-3701`
**Sorun:** `bs in [-0.02, 0]`, `nu_t = nu_t + bs * delta^2 * S`. Cs_min^2=0.0025 ama bs=-0.02 → net negatif.
**Etki:** Negatif eddy viscosity = anti-difuzyon = kosulsuz kararsiz → NaN.
**Fix:** `nu_t = torch.clamp(nu_t, min=0.0)` ekle (backscatter sonrasi)

### BUG-08: torch.clamp Gradient Olduruyor [Debugger]
**Dosya:** `model.py:431-436`
**Sorun:** `torch.clamp(u, -20, 20)` × 20 layer × 10 step = 200 seri hard clamp. Sinirda gradient=0.
**Etki:** Turbulans gelistikce velocity O(10)'a ulasir, clamp aktivasyonu artar, gradient olur → dead parameters.
**Fix:** Soft clamp: `limit * torch.tanh(x / limit)`

### BUG-09: evaluate.py Enstrophy Crash [Reviewer + ML]
**Dosya:** `evaluate.py:80, 325`
**Sorun:** `state.enstrophy()` → `NotImplementedError`. Evaluation pipeline tamamen kirik.
**Fix:** `model.ops.curl()` ile enstrophy hesapla.

### BUG-10: config.physics.nu Re Sweep'te Guncellenmiyor [Reviewer]
**Dosya:** `model.py:set_physics()` vs `train.py:PhysicsLoss`
**Sorun:** `model.set_physics(Re, Ra)` cagirilinca model icindeki noronlar guncellenir ama `config.physics.Re` degismez. PhysicsLoss `config.physics.nu` kullanir → Phase B/C'de tum loss'lar default Re=5000 ile hesaplanir.
**Fix:** `set_physics()` icinde `self.config.physics.Re = Re` de guncelle.

---

## TIER 1: HIGH PRIORITY (Egitim Kalitesini Etkiler)

### HIGH-01: Per-layer dt Loss'ta Ihmal Ediliyor [Reviewer]
Per-layer dt_mults loss hesabinda kullanilmiyor. Model farkli dt ile forward yapiyor, loss tek dt ile kontrol ediyor.

### HIGH-02: Thermal Advection Splitting Hatasi [Debugger]
Momentum advection eski velocity ile, thermal advection guncel (post-projection) velocity ile yapiliyor. Enerji korunumunu bozar.

### HIGH-03: Gradient Clipping 1.0 Cok Agresif [ML Expert]
285 param icin gradient norm = sum(g^2) dusuk olur. 1.0 clip cok agresif, 5.0-10.0 oneriliyor.

### HIGH-04: Forcing Amplitude Ust Siniri Yuksek [CFD Expert]
clamp(1e-5, 0.1): A=0.1 ile pumplanan enerji >> dissipation. Ust sinir 0.01 olmali.

### HIGH-05: Divergence Loss Dead [ML Expert]
Projection sonrasi divergence ~1e-7. Loss asla anlamli gradient uretmiyor. Hesap maliyeti bosa gidiyor.

### HIGH-06: Spectrum Loss Erken Epoch'larda Noise [ML Expert]
Turbulans henuz gelismemisken spectrum loss gradient noise veriyor. Phase A'da weight=0 olmali (curriculum'da zaten oyle mi kontrol et).

### HIGH-07: Nusselt Loss Ra Statik [ML Expert]
Nusselt target'i `config.physics.Ra`'dan hesaplaniyor. Curriculum sweep'te Ra degisince target guncellenmiyor.

### HIGH-08: SGS Dissipation 2x Hatasi [Codex]
`innate.py:3761-3766`: `2*nu_t*strain_mag^2` ama strain_mag icinde zaten sqrt(2*S_ij*S_ij). Simdilik diagnostik, loss'ta kullanilmiyor.

---

## TIER 2: OPTIMIZATION OPPORTUNITIES

### OPT-01: Shared vs Per-Layer Parametreler [Gemini]
285 per-layer param fiziksel olarak anlamsiz (turbulent Prandtl her adimda degismez). Tek set param 20 kez tekrar → 285 → ~15 param. Generalization arttirir, egitim kolaylasir.
**Karar gerekli:** Mevcut egitim once denensin, sonra shared'e gecis yapilabilir.

### OPT-02: FFT Optimizasyonu [Gemini]
1400 FFT/IFFT per forward pass. Lineer terimleri spectral'de tutarak %60 azaltilabilir.
**Karar gerekli:** Kod karmasikligi arttirir. Performance-critical degilse ertelenebilir.

### OPT-03: Dusuk Cozunurluk Pre-Training [Gemini]
48x80x32'de egit, spectral zero-padding ile 96x160x64'e transfer. 4-8x hizlanma.
**Karar gerekli:** Fizik katsayilari resolution-independent mi? Olasilikla evet.

### OPT-04: Uncertainty Weighting [Gemini]
Kendall et al. - ogrenilebilir loss agirliklari. +7 param, curriculum yerine otomatik.
**Karar gerekli:** Curriculum zaten var, ikisi birlikte mi yoksa replacement mi?

### OPT-05: Energy Flux Constraint [Gemini]
Spectrum slope yerine Pi(k) = epsilon zorlayan loss. Fiziksel olarak daha guclu.
**Karar gerekli:** Implementasyon karmasikligi.

---

## TIER 3: LOW PRIORITY / CLEANUP

- Dead code: `_to_fluid_state()` metodu hic cagirilmiyor
- `stochastic` forcing modu icin `step_ou()` cagirilmiyor
- `return_intermediates=True` iken API tasarim sorunu
- Evaluate'de `model.set_physics()` yan etkisi geri alinmiyor

---

## FIX ONCELIK SIRASI

```
Asama 1 (HEMEN - Egitimi duzelt):
  BUG-01: Dissipation 2x → train.py:207
  BUG-02: Dissipation loss tutarsizligi → train.py:238-260
  BUG-05: weight_decay=0 → train.py optimizer
  BUG-06: dt clamp tutarliligi → model.py + train.py
  BUG-07: Backscatter nu_t guard → innate.py:3701
  BUG-10: config.physics Re guncelleme → model.py

Asama 2 (YAKIN - Stabilite):
  BUG-03: CFL → dt_base=0.005 veya velocity clamp=5
  BUG-04: Skew-symmetric advection → innate.py
  BUG-08: Soft clamp → model.py
  HIGH-03: Gradient clip 5.0 → train.py
  HIGH-04: Forcing clamp(1e-5, 0.01) → innate.py

Asama 3 (KISA VADE - Kalite):
  BUG-09: Enstrophy fix → evaluate.py
  HIGH-01: Per-layer dt loss'ta → train.py
  HIGH-02: Thermal splitting → model.py
  HIGH-07: Nusselt Ra dynamic → train.py

Asama 4 (ORTA VADE - Optimizasyon):
  OPT-01: Shared params tartismasi
  OPT-03: Low-res pre-training
  OPT-04: Uncertainty weighting
```

---

## KAYNAK RAPORLAR

| Rapor | Ajan | Dosya |
|-------|------|-------|
| Code Review | Reviewer (Opus) | `reports/review_report.md` |
| CFD Physics | CFD Expert (Opus) | `reports/cfd_report.md` |
| ML/Training | ML Expert (Opus) | `reports/ml_report.md` |
| Numerical Debug | Debugger (Opus) | `reports/debug_report.md` |
| Math Formulas | Codex (GPT-5.3) | `reports/math_report.md` |
| Optimization | Gemini (3 Pro) | `reports/optimization_report.md` |
