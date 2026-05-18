# INNATE — Fizik Tabanlı Sinir Operatörü ile Türbülanslı Akış Modellemesi

**İstanbul Teknik Üniversitesi · Fen-Edebiyat Fakültesi · Fizik Mühendisliği Bölümü**
**Lisans Bitirme Projesi 1 & 2 · 2025–2026**
**Berke Tezgöçen** — _Danışman: Prof. Dr. Emre Onur Kahya_

---

## Kısaca

3B türbülanslı karışık konveksiyon akışını çözen, fiziksel operatörleri (adveksiyon, projeksiyon, kaldırma, alt-ızgara viskozitesi) doğrudan sinir ağının nöronları olarak inşa eden bir **PyTorch** modeli.

- **9 905 parametre** — klasik operatör öğrenme modellerinden (FNO, DeepONet ~10⁶–10⁷) iki büyüklük mertebesi daha küçük.
- **Diverjans-serbest, sayısal kararlı**, 100 epoch boyunca tek bir NaN/Inf yok.
- **MLP içermez** — tüm öğrenilebilir parametreler fiziksel olarak yorumlanabilir (Smagorinsky Cs alanı, türbülans Prandtl, vs.).
- **20 katmanlı fraksiyonel-adım IMEX** zaman integratörü, spektral Leray projeksiyonu, anizotropik elevator-mode damping.
- Doğrulama: bağımsız LES referansı (Smagorinsky Cs=0.17, RK4, 60K step).

```bash
git clone https://github.com/berketez/innate.git
cd innate/bitirme2
python train.py --epochs 100 --Re 10000        # eğitim (~6 saat, RTX 4090)
python evaluate.py --ckpt results_v2/checkpoints/checkpoint_epoch000099.pt
```

---

## Bu Repoda Ne Var?

İki dönem süren bir lisans bitirme projesinin **tüm yazılım üretimi**:

| Dönem | Konu | Konum | Tez |
|---|---|---|---|
| **Bitirme 1** (Güz 25–26) | 3B Taylor–Green Girdabı (TGV3D), PINN + INNATE prototipi | [`bitirme/`](bitirme/) | [`sonuclar/bitirme1_PINN_TGV3D.pdf`](sonuclar/bitirme1_PINN_TGV3D.pdf) (62 sayfa) |
| **Bitirme 2** (Bahar 25–26) | 3B Karışık Konveksiyon, saf-fizik INNATE | [`bitirme2/`](bitirme2/) | [`bitirme2/tez_final/bitirme2.pdf`](bitirme2/tez_final/bitirme2.pdf) (65 sayfa) |

---

## Bitirme 1 — 3B Taylor–Green Girdabı (TGV3D) + PINN

İlk dönem çalışması, türbülanslı akış modellemesinin makine öğrenmesi ile yapılması için klasik bir kıyas problemi olan **Taylor–Green Girdabı**'nı ele aldı. TGV3D, periyodik küp içinde tanımlı, başlangıçta basit analitik bir konfigürasyondan başlayan ve hızla 3B türbülansa geçen bir test problemidir; analitik kapalı-form çözüm sadece t = 0'da vardır, dolayısıyla doğrulama doğrudan DNS referansları ve fiziksel büyüklüklerin izlenmesi ile yapılır.

### Yöntem
- **Problem:** 3B sıkıştırılamaz Navier–Stokes denklemleri, [0, 2π]³ periyodik küp, Re ≈ 1000, t ∈ [0, 0.8]
- **Yaklaşım:** Fizik Bilgili Sinir Ağları (PINN) — diferansiyel denklemlerin fiziksel kısıtlarını doğrudan kayıp fonksiyonuna entegre eden bir paradigma
- **Mimari:** Farklı aktivasyon fonksiyonlu dört dalın **softmax süperpozisyonu** ile birleştirildiği çok dallı (multi-branch) PINN — yaklaşık 4 × 10⁵ parametre
- **Tek ağdan dört çıkış:** (u, v, w, p) aynı ağ üzerinden eş zamanlı kestirim

### Anahtar Yenilikler
- **Sert (hard) başlangıç koşulu ansatzı:** t = 0 başlangıç koşulu **yapısal olarak** ağa gömülür, "sıfır çözüme" çökme önlenir
- **Müfredat öğrenmesi:** Fiziksel kısıtlar (süreklilik, momentum, enerji dengesi, vortisite taşınımı, helisite, simetri) **kademeli** olarak devreye alınır
- **FP32 türevler:** İkinci mertebe türevlerin sayısal kararlılığı için karma hassasiyet yerine FP32
- **Sobol tabanlı örnekleme + zaman yanlılığı:** t ≈ 0 bölgesine ağırlıklı örnekleme

### Sonuçlar (TGV3D, t = 0.8)

| Metrik | PINN | INNATE (ilk prototip) | Açıklama |
|---|---|---|---|
| u L² hatası | ~%11.3 | **~%0.31** | INNATE 36× daha iyi |
| v L² hatası | ~%14.1 | **~%0.11** | INNATE 128× daha iyi |
| w göreli hatası | ~%77.8 | **~%0.76** | w₀ = 0 nedeniyle PINN yanıltıcı; INNATE yine de yüksek doğrulukta |
| Diverjans | ~10⁻³ | **3.9 × 10⁻⁴** | INNATE'in yapısal projeksiyonu sayesinde |
| Enerji hata (ort.) | ~%5 | — | t = 0.8'de %7.7 maksimum |
| Parametre sayısı | ~400 K | ~10 K | INNATE 40× daha küçük |
| Eğitim süresi | ~60 saat (RTX 4090) | — | 5000 epoch ön sonuç |

### INNATE'in Doğuşu

Bitirme 1, PINN paradigmasının üç boyutlu türbülans problemlerinde gözlemlenen **yapısal sınırlarına** çözüm olarak **INNATE (Intrinsic Navier–Stokes Neural Architecture for Temporal Evolution)** yaklaşımını ilk kez kavramsal olarak ortaya koydu:

> *"Physics as structure, not penalty."*
> Fiziksel operatörler kayıp fonksiyonuna eklenmiş cezalar olarak değil, doğrudan ağı oluşturan nöronlar olarak tanımlanır.

İlk prototip — adveksiyon, difüzyon, projeksiyon, vortisite, helisite nöronları + PhysicsModulator MLP — TGV3D'de PINN'e kıyasla iki büyüklük mertebesi düşük L² hatası üretti. Bu kavramsal kanıt, **Bitirme 2'de** çok daha zorlu bir problemde (karışık konveksiyon, Re = 10 000, ısı transferi içeren) gerçek anlamda test edildi ve **MLP modülü tamamen kaldırılıp saf-fizik bir mimariye** dönüştürüldü.

---

## Probleme Bir Bakış (Bitirme 2)

- **Akış:** Boussinesq yaklaşımı altında 3B sıkıştırılamaz Navier–Stokes + enerji denklemi
- **Geometri:** Lx × Ly × Lz = 6 × 10 × 4 (dikey y), periyodik küp benzeri, 96 × 160 × 64 ızgara
- **Termal sınır:** alttan ısıtılan / üstten soğutulan, ΔT = 20 (Rayleigh–Bénard tipi kararsız katmanlanma)
- **Zorlama:** Kolmogorov, kf = 4
- **Çalışma noktaları:** Re ∈ {7 000, 10 000}, Ra = 10⁵, Pr = 0.71, Ri ≈ 1.4 × 10⁻³

İki türbülans üretme mekanizması aynı akışta birlikte aktif: **Kolmogorov mekanik zorlaması** ve **Rayleigh–Bénard termal kaldırma**. "Karışık konveksiyon" terimi bu birlikte aktif konfigürasyonu ifade eder.

---

## Mimari — Sayfa Sayısı Değil, Modül Sayısı

### 1. Saf-Fizik Nöron Kütüphanesi (`innate.py`)

Tüm operatörler bağımsız `nn.Module` olarak, paylaşımlı bir spektral türev modülü (`SpectralOps3DAniso`) üzerinden çalışır.

| Nöron | Operatör | Öğrenilebilir |
|---|---|---|
| `Advection3D` | (u · ∇) u — Lamb form: ω × u | advection_modulator (clamp [0.5, 1.5]) |
| `Projection3D` | Helmholtz projeksiyon (diverjans-serbest garanti) | — (yapısal) |
| `Buoyancy3D` | Ri · θ · ê_y (Boussinesq) | buoyancy_strength |
| `EddyViscosity3D` | Smagorinsky: νt = (Cs · Δ)² · |S| + anizotropik | — |
| `SpectralCsField` | Cs(x) Fourier serisi (5 × 8 × 6 mod) | 240 katsayı/katman × 20 |
| `ThermalAdvection3D` | (u · ∇) θ | thermal_adv_modulator |
| `ThermalDiffusion3D` | κ ∇² θ + anizotropik | kappa_scale |
| `Forcing3D` | Kolmogorov + harmonikler | A_F |

### 2. Fraksiyonel-Adım IMEX Zaman Integratörü

Tek katmanın bir zaman adımı:

```python
# 1. Açık RHS
adv_u, adv_v, adv_w = Advection3D(u)
nu_t                = SpectralCsField() * Delta**2 * |S|      # alt-ızgara
buoy_v              = Ri * theta                              # Boussinesq
F                   = Forcing3D()
R_u = -adv_u + div(nu_t * S) + buoy_v + F

# 2. Fourier uzayında yarı-örtük difüzyon (CFL kısıtı kalkar)
u_hat_new = (u_hat + dt * R_u_hat) / (1 + dt * nu * k_squared)

# 3. Spektral Leray projeksiyonu (div u = 0 yapısal garanti)
u_hat_div_free = (I - k @ k.T / k_squared) @ u_hat_new
```

20 katman ardışık çağrılarak tek ileri geçiş 20·Δt = 0.4 birim zamanı ilerletir.

### 3. Kademe 1+2+3 Eğitim Disiplini

| Kademe | Ne yapar | Niye gerekli |
|---|---|---|
| **1: Parametre Hijyeni** | 120 kanonik parametre (Boussinesq, advection mod., dt skaler) 1.0 değerinde dondurulur | Aksi takdirde optimizer kanonik denklem yapısından kaçar, anti-fizik Nu = 244 / 211 / 430 üretir |
| **2: PDE Artık Kaybı** | R = (u_{n+1} - u_n) / dt − RHS_canonical doğrudan kayıp fonksiyonunda | SGS kapanış kalıntısının kanonik DNS denkleminden sapması ölçülür |
| **3: Minimal 5-Terimli Kayıp** | 27 heterojen kayıp → 5 boyut-ayarlı terim (NS artığı, termal artık, diverjans, LES korelasyonu, spektrum şekli) | Optimizer en büyük gradyanı üreten terim tarafından sürüklenmez, dengeli yön |

---

## Final Sonuçlar (Spectral-Cs v3, 9 905 parametre)

| Metrik | Re = 7 000 | Re = 10 000 | LES Re = 7 000 | LES Re = 10 000 | INNATE / LES |
|---|---|---|---|---|---|
| TKE | 0.0158 | 0.0159 | 0.00848 | 0.00731 | 1.9× / 2.2× |
| Nusselt (zaman ort.) | 64.7 | 153.5 | 39.3 | 70.5 | 1.6× / 2.2× |
| theta_rms | 0.103 | 0.130 | 0.0240 | 0.0268 | 4.3× / 4.8× |
| Spektrum eğimi | −2.98 | −2.56 | −1.74 | −1.78 | — |
| Maks. diverjans hatası | 1.06e−5 | 9.87e−6 | 4.3e−6 | 4.5e−6 | 2× |
| NaN / Inf olayı | 0 | 0 | — | — | ✓ |
| Eğitim süresi | — | ~6 saat (RTX 4090) | — | — | — |

**Nitel:** Kararlı, diverjans-serbest, doğru-yönlü enerji-kaskat.
**Nicel:** TKE / Nu 2× sapma, theta_rms 4–5× sapma — kök sebepleri tezde Bölüm 4.5'te detaylandırılmıştır (termal Spectral-Cs eksikliği, termal spektrum kaybı yokluğu, A_F kaçışı).

---

## Görsel Çıktılar

| | LES (Smagorinsky) | INNATE (Spectral-Cs 9 905 par.) |
|---|---|---|
| 3B hacim render | `bitirme2/tez_final/figurler/ornek_kareler/les_3d_frame.png` | `bitirme2/tez_final/figurler/ornek_kareler/innate_3d_frame.png` |
| Zaman serisi (8 panel) | `bitirme2/tez_final/figurler/fig_11_slice_4zaman_clean.png` (üst LES, alt INNATE) | — |

Tezdeki tüm figürler `bitirme2/tez_final/figurler/` altında (PNG). 4K video dosyaları repo'ya dahil değildir; örnek kareler `ornek_kareler/` klasöründe.

---

## Kurulum

```bash
# 1. Repo
git clone https://github.com/berketez/innate.git
cd innate

# 2. Conda env (Python 3.11 önerilir)
conda create -n innate python=3.11 -y && conda activate innate

# 3. PyTorch + CUDA 11.8 (NVIDIA için)
pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# veya Apple Silicon için:
# pip install torch==2.5.1

# 4. Diğer bağımlılıklar
pip install numpy scipy matplotlib pyvista tqdm
```

### Donanım gereksinimi

| Görev | Tavsiye edilen | Tipik süre |
|---|---|---|
| Eğitim (100 epoch × 1000 step) | RTX 4090 (16 GB) / A100 / H100 | 4090: ~6 saat, H100: ~2 saat |
| LES referans (60K step) | RTX 4090 / A100 | 4090: ~4 saat |
| Smoke test (5 epoch × 100 step) | Apple Silicon MPS / herhangi GPU | 5–10 dk |

> **Not:** macOS MPS üzerinde 300+ step TBPTT için NaN bug'ı vardır. Mac'i sadece smoke testler için kullanın; tam eğitim mutlaka CUDA üzerinde.

---

## Hızlı Başlangıç

```bash
cd bitirme2

# LES referans veri seti üret
python les_solver.py --Re 10000 --Ra 1e5 --n_steps 60000

# INNATE modeli eğit (tam)
python train.py --epochs 100 --num_steps 1000 --Re 10000

# Smoke test (Mac veya hızlı doğrulama)
python smoke_test.py --epochs 5 --steps 100

# Eğitilmiş modeli değerlendir
python evaluate.py --ckpt results_v2/checkpoints/checkpoint_epoch000099.pt
```

---

## Repo Yapısı

```
innate/
├── README.md
├── innate.py                          # ana INNATE nöron kütüphanesi (~4500 satır)
├── PROJE_DOKUMANI.md                  # erken tasarım notları
├── INNATE_DEMO_PLANI.md
│
├── bitirme/                           # Bitirme 1 (PINN + TGV3D)
│
├── bitirme2/                          # Bitirme 2 (Karışık Konveksiyon)
│   ├── config.py                      # nested dataclass konfigürasyonu
│   ├── model.py                       # INNATE3D_MixedConvection (PyTorch nn.Module)
│   ├── train.py                       # Kademe 1+2+3 eğitim pipeline + NaN guard
│   ├── evaluate.py                    # Re sweep + LES karşılaştırma
│   ├── les_solver.py                  # bağımsız LES doğrulama çözücüsü
│   ├── loss_scales.py                 # boyut-ayarlı kayıp ölçekleri
│   ├── smoke_test.py                  # hızlı kararlılık testi
│   ├── grid_analysis.py               # DNS gereksinimi vs LES skoru
│   ├── apriori_from_les.py            # LES'ten önceden hesaplanmış SGS
│   ├── simulate.py
│   │
│   ├── viz_pipeline/                  # 2D/3D render scriptleri (matplotlib + PyVista)
│   ├── visualize/                     # plot yardımcıları
│   ├── tests/                         # birim testler
│   │
│   ├── tez_final/                     # Bitirme 2 PDF + figürler
│   │   ├── bitirme2.pdf               # ← son tez raporu
│   │   └── figurler/                  # tüm figürler (PNG)
│   │       └── ornek_kareler/         # videolardan temsili 3D render kareleri
│   │
│   └── literature/                    # referans makaleler (FNO, LES-Nets, PINN failures)
│
└── sonuclar/
    └── bitirme1_PINN_TGV3D.pdf        # Bitirme 1 PDF (62 sayfa)
```

**Repo'ya dahil edilmeyen** (`.gitignore`'da):
- 4K video dosyaları (~800 MB) — örnek kareler PNG olarak dahil
- LES referans `.npz` ve checkpoint `.pt` dosyaları (>100 MB)
- `bitirme2/archive/` (tarihsel kayıt, GB mertebesinde)
- `bitirme2/benchmarks/` (dış kaynak benchmark veri setleri)
- Tez LaTeX kaynak dosyaları (PDF kalır)

---

## Tezde Sunulan Temel Bulgular

1. **Saf-fizik mimari prensibi karışık konveksiyon problemlerinde uygulanabilirdir.** Diverjans-serbestlik ve sayısal kararlılık yapısal olarak korunmuştur (100 epoch boyunca sıfır NaN / Inf olayı).
2. **Eğitim disiplini, mimari kadar belirleyicidir.** Kademe 1+2+3 olmadan model 120 öğrenilebilir parametreyi kullanarak kanonik denklem yapısından kaçar (anti-fizik Nu = 244 / 211 / 430).
3. **Düşük parametre sayılı bir model ile kararlı çözüm üretilebilir.** 9 905 parametre, klasik operatör öğrenme modellerinden iki büyüklük mertebesi daha azdır.
4. **Termal alan modellemesi açık bir sınırdır.** theta_rms ve v·theta akı sapmaları, termal Spectral-Cs ve termal spektrum şekil kaybının eksikliğine bağlanmıştır.
5. **Spectral-Cs uzaysal kapasite tam aktive olmamıştır.** 100 epoch eğitim bütçesi altında Fourier katsayıları başlangıç değerlerinden anlamlı uzaklaşmamış; daha uzun eğitim (500–1000 epoch) ve daha güçlü donanım (A100 / H100) gerekli.

---

## Karşılaşılan Kritik Hatalar (ve Düzeltmeleri)

Final eğitim öncesi yapılan paralel kod review ile tespit edilen yedi kritik hata:

| # | Etki | Düzeltme |
|---|---|---|
| 1 | NaN guard eksik | Kayıp + grad NaN / Inf tespiti ile backward atlama |
| 2 | Checkpoint mantığı | Her 10 epoch + her epoch latest copy |
| 3 | LR scheduler orantısız | Warmup = max_epochs / 5, T0 = 0.6 × post-warmup |
| 4 | Tier 1 freeze eksik | dt_scale, dt_mults, backscatter da eklendi |
| 5 | Gradient routing uyumsuz | Tier 1 ile çakışan routing otomatik kapatıldı |
| 6 | gamma_damp residual'dan eksik | NS artığında elevator damping eklendi |
| 7 | Loss scale kalibrasyon | NS_RES_SCALE: 1e-4 → 1e-3, SPECTRUM_SHAPE_SCALE: 1 → 50 |

Detaylar: tezin Bölüm 4.2.3'ünde.

---

## Tasarım Felsefesi

1. **"Physics as structure, not penalty."** Operatörler ceza değil, mimari.
2. **Disiplinli parametre hijyeni.** Denklem-değiştirici parametreler öğrenilebilir bırakılmaz; sadece kapanış katsayıları öğrenilir.
3. **Akademik dürüstlük.** Başarısızlıkları gizlemek yerine sistematik biçimde belgelemek; 7 kritik kod hatası, 3 anti-fizik konfigürasyon ve 2 yapısal sınır final tezde açıkça raporlanmıştır.

---

## Atıf

```bibtex
@thesis{tezgocen2026_innate,
  author = {Berke Tezg{\"o}{\c{c}}en},
  title  = {{Ü}{\c{c}} Boyutlu Kar{\i}{\c{s}}{\i}k Konveksiyon Ak{\i}{\c{s}}{\i}n{\i}n
            Fizik Tabanl{\i} Sinir Operat{\"o}r{\"u} (INNATE) ile {\c{C}}{\"o}z{\"u}m{\"u}},
  school = {{\.I}stanbul Teknik {\"U}niversitesi},
  year   = {2026},
  type   = {Bitirme {\c{C}}al{\i}{\c{s}}mas{\i}},
}
```

---

## İletişim

- **Berke Tezgöçen** — btezgocen97@gmail.com
- LinkedIn: [Berke-Tezgöçen](https://www.linkedin.com/in/Berke-Tezgöçen)
- GitHub: [@berketez](https://github.com/berketez)

**Akademik kullanım için açıktır.** Soru, hata raporu veya katkı için issue açabilirsiniz.
