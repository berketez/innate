# INNATE Projesi - Kapsamlı Döküman

**Son Güncelleme:** 2026-01-08
**Proje Sahibi:** Berke Tezgöçen
**Ana Klasör:** `/Users/apple/Desktop/nsneuron`
**Hedef:** Startup & Ticari Ürün

---

## 1. PROJENİN VİZYONU

### 1.1 Büyük Resim

```
┌─────────────────────────────────────────────────────────────────────┐
│                           INNATE                                     │
│         "Fizik-Gömülü Nöronlarla Evrensel Akışkan Çözücü"           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐           │
│   │Advection │  │Vorticity │  │Projection│  │  Diğer   │  ...      │
│   │  u·∇u    │  │   ∇×u    │  │  ∇·u=0   │  │ Nöronlar │           │
│   └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘           │
│        │             │             │             │                   │
│        └─────────────┴─────────────┴─────────────┘                   │
│                           │                                          │
│                           ▼                                          │
│                  ┌─────────────────┐                                 │
│                  │  HER AKIŞKAN    │                                 │
│                  │   PROBLEMİ      │                                 │
│                  │                 │                                 │
│                  │ • Periodic ✓    │                                 │
│                  │ • Non-periodic ✓│                                 │
│                  │ • 2D/3D ✓       │                                 │
│                  └─────────────────┘                                 │
│                                                                      │
│   Rakipler (ANSYS, OpenFOAM): Günler/Haftalar                       │
│   INNATE: Saniyeler + Öğreniyor + Genelleştiriyor                   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 Startup Hedefi

**Vizyon:** Fizik-gömülü yapay zeka ile CFD pazarını dönüştürmek.

**Değer Önerisi:**
- ANSYS lisansı: ~$50,000/yıl
- INNATE: Daha hızlı, daha ucuz, öğrenen sistem
- Hedef müşteriler: Otomotiv, havacılık, enerji, HVAC

### 1.3 Ana Hedefler

1. **INNATE** adlı physics-native neural operator kütüphanesi geliştirmek
2. **Tüm akışkan problemlerini** çözebilen evrensel nöronlar oluşturmak
3. Benchmark problemlerle validasyon yapmak (TGV3D, Cavity, Cylinder, vb.)
4. Ticari ürün haline getirmek

### 1.2 INNATE Felsefesi

INNATE bir **PINN (Physics-Informed Neural Network) DEĞİL**.

| Özellik | PINN | INNATE |
|---------|------|--------|
| Fizik nasıl eklenir? | Loss fonksiyonuna PDE residual eklenir | Nöronun kendisi fiziksel operatör |
| Türev hesaplama | Autograd (yavaş) | Spektral/FFT (hızlı, tam doğru) |
| Enerji korunumu | Garanti yok | Yapısal olarak garanti (skew-symmetric) |
| Parametre sayısı | Yüzlerce bin | Çok az (9 temel), ağda binlerce |

**Analoji:** INNATE nöronları = PyTorch'un `nn.Conv2d`, `nn.Linear` katmanları gibi.
- `nn.Conv2d` → görüntü için temel yapı taşı
- `Advection`, `Vorticity`, `Projection` → fluid için temel yapı taşları

Kütüphaneyi DEĞİŞTİRMİYORSUN. Uygulamada bu yapı taşlarını birleştirip ağ oluşturuyorsun.

---

## 2. INNATE KÜTÜPHANESİ (innate.py)

### 2.1 Temel 9 Parametre (2D)

INNATE'in çekirdeğinde sadece **9 öğrenilebilir parametre** var:

| Nöron | Parametre | Açıklama |
|-------|-----------|----------|
| Advection | `advection_modulator` (1) | u·∇u şiddetini modüle eder |
| Vorticity | `circulation_preservation` (1) | Sirkülasyon korunumunu ayarlar |
| Diffusion | `diffusion_coefficient` (1) | ν∇²u difüzyon katsayısı |
| Projection | `pressure_weight` (1) | Basınç projeksiyon ağırlığı |
| TimeMarcher | `dt_scale` (1) | Zaman adımı ölçekleme |
| TimeMarcher | `stability_weight` (1) | Stabilite ağırlığı |
| TimeMarcher | `cfl_factor` (1) | CFL koşulu faktörü |
| Boundary | `wall_damping` (1) | Duvar sönümleme |
| Boundary | `bc_strength` (1) | Sınır koşulu kuvveti |

Bu 9 parametre **tek bir nöron** için. Uygulama seviyesinde bunları binlerce kez birleştiriyorsun.

### 2.2 Mevcut 2D Nöronlar

```python
# innate.py'den
class Advection(nn.Module)      # u·∇u - skew-symmetric form
class Vorticity(nn.Module)      # ∇×u, sirkülasyon korunumu
class Projection(nn.Module)     # ∇²p = ∇·u, divergence-free projeksiyon
class TimeMarcher(nn.Module)    # RK4 entegrasyon, adaptif dt
class Boundary(nn.Module)       # Sınır koşulları (periodic, no-slip)
class DataInjector(nn.Module)   # Gözlem verisi enjeksiyonu
class Reynolds(nn.Module)       # Re sayısı öğrenme
class SpectralOps(nn.Module)    # FFT-tabanlı türevler (öğrenilmez)
```

### 2.3 Mevcut 3D Nöronlar

```python
# innate.py'den - 3D versiyonlar
class Advection3D(nn.Module)
class Vorticity3D(nn.Module)              # 3 bileşenli vortisite
class Projection3D(nn.Module)
class TimeMarcher3D(nn.Module)
class Helicity3D(nn.Module)               # 3D spesifik: H = u·ω
class SpectralEnergyFlux3D(nn.Module)     # Enerji kaskadı
class EddyViscosity3D(nn.Module)          # Türbülans modelleme
class EnergyPreservingIntegrator3D(nn.Module)
class StrainRate3D(nn.Module)
class SpectralOps3D(nn.Module)
```


---

## 3. BİTİRME PROJESİ: TGV3D

### 3.1 Taylor-Green Vortex 3D Problemi

**Domain:** [0, 2π]³ × [0, T]
**Reynolds:** Re ≈ 1000 (ν = 0.001)

**Başlangıç Koşulları (t=0):**
```
u = sin(x)·cos(y)·cos(z)
v = -cos(x)·sin(y)·cos(z)
w = 0
p = [cos(2x) + cos(2y)]·[cos(2z) + 2] / 16
```

**Önemli:** t > 0 için kapalı form analitik çözüm YOK! DNS referans verisi gerekli.

### 3.2 Mevcut Multi-Branch PINN Yaklaşımı

Dosya: `/Users/apple/Desktop/dosyalar/okul/güz26/bitirmeproje1/kodlar/bitirmeyaklasım1/tgv3d_multipinn.py`

**Yapı:**
- 4 branch (sine, tanh, swish, sine activations)
- ~400k parametre
- Quantum-inspired superposition (softmax ağırlıkları)
- Hard IC ansatz: `u = u0 + t * network_output`

**Fiziksel Kısıtlar:**
- Momentum (u, v, w): Navier-Stokes
- Continuity: ∇·u = 0 (100x ağırlık)
- Energy balance: ∂E/∂t + ν⟨ω²⟩ = 0
- Global divergence: ⟨∇·u⟩ = 0
- Mean momentum: ⟨u⟩ = ⟨v⟩ = ⟨w⟩ = 0
- TGV symmetry
- Vorticity transport
- Helicity: H ≈ 0

**Eğitim:**
- 18k epoch
- Curriculum learning
- Sobol resampling

### 3.3 DNS Referans Verisi

```
/Users/apple/Desktop/dosyalar/okul/güz26/bitirmeproje1/kodlar/bitirmeyaklasım1/
├── dns_results_64cubed.npz   (174 MB)
└── dns_results_128cubed.npz  (986 MB)
```

---

## 4. HEDEF: INNATE + MULTI-BRANCH = TGV3D

### 4.1 Önerilen Mimari: MultiINNATE3D

```
┌─────────────────────────────────────────────────────────────────┐
│                        MultiINNATE3D                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Branch 1: Advection-Dominant                                   │
│  ├── Advection3D (×N)                                          │
│  ├── Projection3D                                               │
│  └── TimeMarcher3D                                              │
│                                                                 │
│  Branch 2: Vorticity-Dominant                                   │
│  ├── Vorticity3D (×N)                                          │
│  ├── Helicity3D                                                 │
│  └── TimeMarcher3D                                              │
│                                                                 │
│  Branch 3: Energy-Preserving                                    │
│  ├── EnergyPreservingIntegrator3D                              │
│  ├── SpectralEnergyFlux3D                                      │
│  └── TimeMarcher3D                                              │
│                                                                 │
│  Branch 4: Turbulence-Aware                                     │
│  ├── StrainRate3D                                               │
│  ├── EddyViscosity3D                                            │
│  └── TimeMarcher3D                                              │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│  Softmax Fusion Layer (öğrenilebilir ağırlıklar)               │
├─────────────────────────────────────────────────────────────────┤
│  SpectralOps3D (paylaşılan, FFT-tabanlı türevler)              │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Parametre Hesabı

Her nöronun ~1-3 parametresi var. Eğer:
- Her branch'te 10 nöron zinciri
- 4 branch
- Her nöron 2 parametre ortalama

**Toplam:** 4 × 10 × 2 = 80 temel parametre

Ama bunları **spatial modulation** ile genişletebiliriz:
- Her nöron için 4×4×4 = 64 spatial weight
- 40 nöron × 64 = 2560 parametre

Veya **multi-scale** yaklaşım:
- Her nöron farklı resolution'larda çalışır
- Coarse-to-fine hierarchy

### 4.3 INNATE'in Avantajları

| Özellik | Multi-Branch PINN | MultiINNATE3D |
|---------|-------------------|---------------|
| Türev hesaplama | Autograd (yavaş) | FFT (10-100x hızlı) |
| Enerji korunumu | Loss ile zorlanıyor | Yapısal garanti |
| Divergence-free | Loss ile zorlanıyor | Projeksiyon ile garanti |
| Stabilite | Explode olabilir | CFL + skew-symmetric |
| Fiziksel yorumlanabilirlik | Düşük | Yüksek (her nöron bir operatör) |

---

## 5. MEVCUT DURUM VE YAPILACAKLAR


### 5.2 Yapılacaklar

1. **INNATE3D sınıfı oluştur** - 3D nöronları birleştiren ana model
2. **MultiINNATE3D tasarla** - Multi-branch yapısı
3. **TGV3D testi yaz** - DNS verisiyle karşılaştırma
4. **Eğitim pipeline** - Curriculum learning, checkpointing
5. **Validasyon metrikleri** - Enerji, enstrofi, divergence

### 5.3 Test Dosyaları

```
/Users/apple/Desktop/nsneuron/
├── innate.py                    # Ana kütüphane
├── tests/
│   ├── periodic_2d/
│   │   └── lamb_oseen.py        # Lamb-Oseen vortex testi
│   ├── noslip_2d/
│   │   └── couette_flow.py      # Couette flow testi
│   └── periodic_3d/
│       └── tgv3d.py             # [YAPILACAK] TGV3D testi
```

---

## 6. TEKNİK NOTLAR

### 6.1 Skew-Symmetric Advection

Enerji korunumu için kritik:
```python
# Convective form: (u·∇)u
conv = u * du_dx + v * du_dy

# Divergence form: ∇·(u⊗u)
div = d(uu)/dx + d(vu)/dy

# Skew-symmetric: 0.5 * (conv + div)
adv = 0.5 * (conv + div)  # Enerji koruyan!
```

### 6.2 RK4 Entegrasyon

```python
k1 = f(t, y)
k2 = f(t + dt/2, y + dt*k1/2)
k3 = f(t + dt/2, y + dt*k2/2)
k4 = f(t + dt, y + dt*k3)
y_new = y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
```

### 6.3 CFL Koşulları

```python
# Advective CFL
dt_adv = dx / |u|_max

# Diffusive CFL
dt_diff = dx² / (4ν)

# Toplam
dt = min(dt_adv, dt_diff) * safety_factor
```

### 6.4 Spektral Türevler

```python
# FFT-tabanlı türev (tam doğru, aliasing yok)
f_hat = fft(f)
df_dx = ifft(1j * kx * f_hat)

# 2/3 dealiasing kuralı
k_max = N // 3
mask = |k| < k_max
f_hat_filtered = f_hat * mask
```

---

## 7. KULLANIM

### 7.1 Lamb-Oseen Testi (2D)

```bash
cd /Users/apple/Desktop/nsneuron/tests/periodic_2d
python lamb_oseen.py --epochs 300 --steps 100

# Extended model (100 parametre) için:
python lamb_oseen.py --epochs 300 --steps 100 --extended
```

### 7.2 Couette Flow Testi (2D)

```bash
cd /Users/apple/Desktop/nsneuron/tests/noslip_2d
python couette_flow.py
```

### 7.3 TGV3D (Mevcut Multi-Branch PINN)

```bash
cd /Users/apple/Desktop/dosyalar/okul/güz26/bitirmeproje1/kodlar/bitirmeyaklasım1
python tgv3d_multipinn.py
```

---

## 8. ÖNCEKİ OTURUMDAN NOTLAR

### 8.1 Düzeltilen Hatalar

1. **Euler → RK4:** Enerji korunumu için kritik
2. **Skew-symmetric advection:** Enerji patlamasını önledi
3. **CFL koşulları:** Advective + diffusive
4. **nu Tensor hatası:** `torch.full_like` için scalar'a çevir
5. **Energy metric normalization:** Aynı formül predicted ve reference için
6. **Couette analytical formula:** `(-1)**n` eksikti

### 8.2 ExtendedINNATE Wrapper

Test dosyasında (lamb_oseen.py) ~100 parametre eklemek için wrapper:
- Kütüphaneyi DEĞİŞTİRME
- Wrapper ile ek parametreler ekle
- `--extended` flag ile kullan

### 8.3 Dikkat Edilecekler

- Multiplicative scaling birikir → explosion
- Additive correction kullan
- Her step'te çarpım yerine toplama

---

## 9. REFERANSLAR

1. **Taylor-Green Vortex:** Klasik benchmark, 1937
2. **Spectral Methods:** Canuto et al., "Spectral Methods in Fluid Dynamics"
3. **Energy-Preserving Schemes:** Morinishi et al., 1998
4. **PINNs:** Raissi et al., 2019
5. **Neural Operators:** Lu et al., FNO, 2021

---

## 10. İLETİŞİM VE DEVAM

Bu dökümanı okuyan yeni oturum için:

1. Önce bu dökümanı oku
2. `innate.py`'yi incele (ana kütüphane)
3. `tests/periodic_2d/lamb_oseen.py`'yi incele (örnek test)
4. TGV3D için `INNATE3D` ve `MultiINNATE3D` sınıflarını oluştur
5. DNS verisini kullanarak eğit ve validate et

**Ana hedef:** TGV3D'yi INNATE ile çöz, Multi-Branch PINN'den daha iyi sonuç al.

---

## 11. YENİ EKLENEN YETENEKLER (2025-12-28)

### 11.1 extra_repr() - Debug/Görselleştirme

Tüm nöronlar artık güzel yazdırma desteğine sahip:

```python
model = INNATE(64)
print(model)
# INNATE(
#   (advection): Advection(modulator=1.0000, has_boundary=False)
#   (vorticity): Vorticity(circulation_preservation=0.9900)
#   (projector): Projection(pressure_weight=1.0000, has_boundary=False)
#   ...
# )
```

### 11.2 Diagnostic Methods - INNATE'e Eklendi

```python
model = INNATE(64)
state = model.create_initial_state()

# Tek tek
energy = model.get_energy(state)
enstrophy = model.get_enstrophy(state)
divergence = model.get_divergence(state)
cfl = model.get_cfl_number(state)
re = model.get_reynolds()

# Hepsi birden
diagnostics = model.get_diagnostics(state)
# {'energy': 0.5, 'enstrophy': 1.2, 'divergence': 1e-6, ...}
```

### 11.3 Utils Modülü

Yeni `utils/` klasörü oluşturuldu:

```
/Users/apple/Desktop/nsneuron/utils/
├── __init__.py
├── composable.py   # Sequential, Parallel, Residual, MultiBranch
└── monitor.py      # PhysicsMonitor
```

#### composable.py - Nöron Birleştirme

```python
from utils import Sequential, Parallel, Residual, MultiBranch

# Sequential: Sıralı bağlantı
pipeline = Sequential(Advection(64), Projection(64))

# Parallel: Paralel + weighted fusion
ensemble = Parallel(Advection(64), Vorticity(64), fusion='weighted')

# Residual: Skip connection
block = Residual(Advection(64), scale=0.1)

# MultiBranch: TGV3D için multi-branch yapı
model = MultiBranch([
    Branch([Advection(64), Projection(64)], name='adv'),
    Branch([Vorticity(64), Projection(64)], name='vort'),
])
```

#### monitor.py - Fizik İzleme

```python
from utils import PhysicsMonitor

model = INNATE(64)
monitor = PhysicsMonitor(model, log_every=10)

for epoch in range(1000):
    states = model(initial_state, num_steps=100)
    warnings = monitor.log(states[-1], epoch)

    if warnings:
        print(f"Warnings: {warnings}")

# Özet ve görselleştirme
monitor.print_summary()
monitor.plot(save_path='physics_history.png')
monitor.save('physics_history.json')
```

---

## 12. DOSYA YAPISI (GÜNCEL)

```
/Users/apple/Desktop/nsneuron/
├── innate.py                    # Ana kütüphane (2D + 3D nöronlar)
├── PROJE_DOKUMANI.md            # Bu döküman
├── README.md                    # Kısa README
├── utils/
│   ├── __init__.py
│   ├── composable.py            # Sequential, Parallel, Residual, MultiBranch
│   └── monitor.py               # PhysicsMonitor
├── tests/
│   ├── common/
│   │   └── trainer_base.py      # Ortak trainer sınıfı
│   ├── periodic_2d/
│   │   └── lamb_oseen.py        # Lamb-Oseen vortex testi + ExtendedINNATE
│   ├── noslip_2d/
│   │   └── couette_flow.py      # Couette flow testi
│   └── periodic_3d/
│       └── tgv3d.py             # [YAPILACAK] TGV3D testi
```
