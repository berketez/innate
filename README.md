# INNATE — Fizik Tabanlı Sinir Operatörü ile Türbülanslı Akış Modellemesi

**İstanbul Teknik Üniversitesi · Fen-Edebiyat Fakültesi · Fizik Mühendisliği Bölümü**
**Lisans Bitirme Projesi 1 & 2 · 2025–2026**
**Berke Tezgöçen** — _Danışman: Prof. Dr. Emre Onur Kahya_

---

## TL;DR

3D türbülanslı karışık konveksiyon akışını çözen, fiziksel operatörleri (adveksiyon, projeksiyon, kaldırma, alt-ızgara viskozitesi) doğrudan sinir ağının nöronları olarak inşa eden bir **PyTorch** modeli.

- **9 905 parametre** — klasik operatör öğrenme modellerinden (FNO, DeepONet ~$10^{6}$–$10^{7}$) iki büyüklük mertebesi daha küçük.
- **Diverjans-serbest, sayısal kararlı**, 100 epoch boyunca tek NaN/Inf yok.
- **MLP içermez** — tüm öğrenilebilir parametreler fiziksel olarak yorumlanabilir (Smagorinsky $C_s$ alanı, türbülans Prandtl, vs.).
- **20 katmanlı fraksiyonel-adım IMEX** zaman integratörü, spektral Leray projeksiyonu, anizotropik elevator-mode damping.
- Doğrulama: bağımsız LES referansı (Smagorinsky $C_s=0.17$, RK4, 60K step).

```bash
git clone https://github.com/berketez/nsneuron1.git
cd nsneuron1/bitirme2
python train.py --epochs 100 --Re 10000        # eğitim (~6 saat, RTX 4090)
python evaluate.py --ckpt results_v2/checkpoints/checkpoint_epoch000099.pt
```

---

## Bu Repoda Ne Var?

İki dönem süren bir lisans bitirme projesinin **tüm yazılım üretimi**:

| Dönem | Konu | Konum | Tez |
|---|---|---|---|
| **Bitirme 1** (Güz 25–26) | 3D Taylor–Green Girdabı (TGV3D), PINN + INNATE prototipi | [`bitirme/`](bitirme/) | [`sonuclar/bitirme1_PINN_TGV3D.pdf`](sonuclar/bitirme1_PINN_TGV3D.pdf) (62 sayfa) |
| **Bitirme 2** (Bahar 25–26) | 3D Karışık Konveksiyon, saf-fizik INNATE | [`bitirme2/`](bitirme2/) | savunma sonrasında `sonuclar/` |

---

## Probleme Bir Bakış

**Akış**: Boussinesq yaklaşımı altında 3D sıkıştırılamaz Navier–Stokes + enerji denklemi.
**Geometri**: $L_{x}\!\times\!L_{y}\!\times\!L_{z} = 6\!\times\!10\!\times\!4$ (dikey $y$), periyodik küp benzeri, $96\!\times\!160\!\times\!64$ ızgara.
**Termal sınır**: alttan ısıtılan/üstten soğutulan, $\Delta T=20$ (Rayleigh–Bénard tipi kararsız katmanlanma).
**Zorlama**: Kolmogorov $F_{x} = A\sin(k_{f}y)\,\hat{\mathbf{e}}_{x}$, $k_{f}=4$.
**Çalışma noktaları**: $\mathrm{Re}\in\{7000,10\,000\}$, $\mathrm{Ra}=10^{5}$, $\mathrm{Pr}=0.71$, $\mathrm{Ri}\approx 1.4\!\times\!10^{-3}$.

> İki türbülans-üretme mekanizması aynı akışta birlikte aktif: Kolmogorov mekanik zorlaması ve Rayleigh–Bénard termal kaldırma. ‘‘Karışık konveksiyon’’ terimi bu birlikte-aktif konfigürasyonu ifade eder.

---

## Mimari — Sayfa Sayısı Değil, Modül Sayısı

### 1. Saf-Fizik Nöron Kütüphanesi (`innate.py`)

Tüm operatörler bağımsız `nn.Module` olarak, paylaşımlı bir spektral türev modülü (`SpectralOps3DAniso`) üzerinden çalışır.

| Nöron | Operatör | Öğrenilebilir |
|---|---|---|
| `Advection3D` | $(\mathbf{u}\cdot\nabla)\mathbf{u}$ Lamb form $\boldsymbol{\omega}\!\times\!\mathbf{u}$ | $\alpha_{\mathrm{adv}}^{(\ell)}\in[0.5,1.5]$ |
| `Projection3D` | $\mathcal{P} = I - kk^{T}/\|k\|^{2}$ Helmholtz | — (yapısal garanti) |
| `Buoyancy3D` | $\mathrm{Ri}\,\theta\,\hat{\mathbf{e}}_{y}$ | $\beta_{B}^{(\ell)}$ |
| `EddyViscosity3D` | Smagorinsky $\nu_{t}=(C_{s}\Delta)^{2}\|S\|$ + anizotropik | – |
| `SpectralCsField` | $C_{s}(\mathbf{x})$ Fourier serisi (5×8×6 mod) | 240 katsayı/katman × 20 |
| `ThermalAdvection3D` | $(\mathbf{u}\cdot\nabla)\theta$ | $\alpha_{\theta}^{(\ell)}$ |
| `ThermalDiffusion3D` | $\kappa\nabla^{2}\theta$ + anizotropik | $\kappa_{\mathrm{scale}}$ |
| `Forcing3D` | Kolmogorov + harmonikler | $A_{F}$ |

### 2. Fraksiyonel-Adım IMEX Zaman Integratörü

Tek katmanın bir zaman adımı:

```
# 1. Açık RHS (Fourier'de değil, fiziksel uzayda)
adv_u, adv_v, adv_w = Advection3D(u)
nu_t                = SpectralCsField * |S|^2     # alt-ızgara
buoy_v              = Ri * theta                  # Boussinesq
F                   = Forcing3D()
R_u = -adv_u + ∇·(nu_t·S) + buoy_v + F

# 2. Fourier uzayında yarı-örtük difüzyon (CFL kısıtı kalkar)
û*(k) = (û_n(k) + Δt·R̂_u(k)) / (1 + Δt·ν·|k|²)

# 3. Spektral Leray projeksiyonu (∇·u = 0 yapısal garanti)
û_{n+1}(k) = (I − k⊗k/|k|²) · û*(k)
```

20 katman ardışık çağrılarak tek bir ileri geçiş $20\Delta t = 0.4$ birim zamanı ilerletir.

### 3. Kademe 1+2+3 Eğitim Disiplini

| Kademe | Ne yapar | Niye gerekli |
|---|---|---|
| **1: Parametre Hijyeni** | 120 kanonik parametre (Boussinesq, advection modülatörü, dt skaler) `1.0` değerinde dondurulur | Aksi takdirde optimizer kanonik denklem yapısından kaçar, anti-fiziksel $\mathrm{Nu}=244$/$211$/$430$ üretir |
| **2: PDE Artık Kaybı** | $\mathcal{R}_{u} = (u_{n+1}-u_{n})/\Delta t - \mathrm{RHS}_{\mathrm{canonical}}$ doğrudan kayıp fonksiyonuna | SGS kapanış kalıntısının kanonik DNS denkleminden sapması ölçülür |
| **3: Minimal 5-Terimli Kayıp** | 27 heterojen kayıp → 5 boyut-ayarlı terim (NS artığı, termal artık, diverjans, LES korelasyonu, spektrum şekli) | Optimizer en büyük gradyanı üreten kayıp tarafından sürüklenmez, dengeli yön |

---

## Final Sonuçlar (Spectral-Cs v3, 9 905 parametre)

| Metrik | Re=7 000 | Re=10 000 | LES Re=7 000 | LES Re=10 000 | INNATE/LES |
|---|---|---|---|---|---|
| TKE | 0.0158 | 0.0159 | 0.00848 | 0.00731 | 1.9× / 2.2× |
| Nusselt (zaman ort.) | 64.7 | 153.5 | 39.3 | 70.5 | 1.6× / 2.2× |
| $\theta_{\mathrm{rms}}$ | 0.103 | 0.130 | 0.0240 | 0.0268 | 4.3× / 4.8× |
| Spektrum eğimi | −2.98 | −2.56 | −1.74 | −1.78 | — |
| Maks. diverjans hatası | 1.06e−5 | 9.87e−6 | 4.3e−6 | 4.5e−6 | 2× |
| NaN/Inf olayı | 0 | 0 | — | — | ✓ |
| Eğitim süresi | — | ~6 saat (RTX 4090) | — | — | — |

**Nitel:** Kararlı, diverjans-serbest, doğru-yönlü enerji-kaskat. **Nicel:** TKE/Nu 2× sapma, $\theta_{\mathrm{rms}}$ 4–5× sapma — kök sebepleri Bitirme 2 tezinde Bölüm 4.5'te detaylandırılmıştır.

---

## Görsel Çıktılar

| | LES (Smagorinsky) | INNATE (Spectral-Cs 9 905 par.) |
|---|---|---|
| 3D Hacim Render | `bitirme2/tez_final/figurler/ornek_kareler/les_3d_frame.png` | `bitirme2/tez_final/figurler/ornek_kareler/innate_3d_frame.png` |
| Zaman serisi (8 panel) | `bitirme2/tez_final/figurler/fig_11_slice_4zaman_clean.png` | (üst sıra LES, alt sıra INNATE) |

Tezdeki tüm figürler `bitirme2/tez_final/figurler/` altında (PNG). 4K video dosyaları repo'ya dahil değildir (boyut nedeniyle); örnek kareler `ornek_kareler/` klasöründe.

---

## Kurulum

```bash
# 1. Repo
git clone https://github.com/berketez/nsneuron1.git
cd nsneuron1

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

### LES referans veri seti üret
```bash
cd bitirme2
python les_solver.py --Re 10000 --Ra 1e5 --n_steps 60000 \
                     --save_dir les_reference/Re10000_Ra1e5_v2
```

### INNATE modeli eğit (tam)
```bash
python train.py --config configs/spectral_cs_v3.yaml \
                --epochs 100 --num_steps 1000 \
                --Re 10000 --Ra 1e5
```

### Smoke test (Mac veya hızlı doğrulama)
```bash
python smoke_test.py --epochs 5 --steps 100
```

### Eğitilmiş modeli değerlendir
```bash
python evaluate.py --ckpt results_v2/checkpoints/checkpoint_epoch000099.pt \
                   --re_sweep 7000,10000 --output_dir eval_output/
```

### Tezi derle
```bash
cd tez_final && pdflatex bitirme2.tex && pdflatex bitirme2.tex && open bitirme2.pdf
```

---

## Repo Yapısı

```
nsneuron1/
├── README.md                          # bu dosya
├── innate.py                          # ana INNATE nöron kütüphanesi (~4500 satır)
├── PROJE_DOKUMANI.md                  # erken tasarım notları
├── INNATE_DEMO_PLANI.md
│
├── bitirme/                           # Bitirme 1 (PINN + TGV3D)
│
├── bitirme2/                          # Bitirme 2 (Karışık Konveksiyon)
│   ├── config.py                      # nested dataclass konfigürasyonu
│   ├── model.py                       # INNATE3D_MixedConvection (PyTorch nn.Module)
│   ├── train.py                       # Kademe 1+2+3 eğitim pipeline'ı + NaN guard
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
│   ├── tez_final/                     # TEZ KAYNAĞI
│   │   ├── bitirme2.tex
│   │   ├── bolum1_giris.tex .. bolum5_sonuclar.tex
│   │   ├── kaynaklar.tex
│   │   ├── bitirme2.pdf               # ← derlenen son tez
│   │   └── figurler/                  # tüm figürler (PNG)
│   │       └── ornek_kareler/         # videolardan temsili 3D render kareleri
│   │
│   └── literature/                    # referans makaleler (FNO, LES-Nets, PINN failures)
│
└── sonuclar/                          # tezler + ek raporlar
    └── bitirme1_PINN_TGV3D.pdf        # Bitirme 1 PDF (62 sayfa)
    # bitirme2_INNATE_MixedConv.pdf — savunma sonrası eklenecek
```

**Repo'ya dahil edilmeyen** (boyut/lisans nedeniyle, `.gitignore`'da):
- 4K video dosyaları (`*.mp4`, ~800 MB) — örnek kareler PNG olarak dahil
- LES referans `.npz` ve checkpoint `.pt` dosyaları (>100 MB)
- `bitirme2/archive/` (tarihsel kayıt, GB mertebesinde)
- `bitirme2/benchmarks/` (dış kaynak benchmark veri setleri, ayrı repolar)

---

## Tezde Sunulan Temel Bulgular

1. **Saf-fizik mimari prensibi karışık konveksiyon problemlerinde uygulanabilirdir.** Diverjans-serbestlik ve sayısal kararlılık yapısal olarak korunmuştur (100 epoch boyunca sıfır NaN/Inf olayı).
2. **Eğitim disiplini, mimari kadar belirleyicidir.** Kademe 1+2+3 olmadan model 120 öğrenilebilir parametreyi kullanarak kanonik denklem yapısından kaçar (anti-fizik $\mathrm{Nu}=244/211/430$).
3. **Düşük parametre sayılı bir model ile kararlı çözüm üretilebilir.** 9 905 parametre, klasik operatör öğrenme modellerinden iki büyüklük mertebesi daha azdır.
4. **Termal alan modellemesi açık bir sınırdır.** $\theta_{\mathrm{rms}}$ ve $v\theta$ akısı sapmaları, termal Spectral-Cs ve termal spektrum şekil kaybının eksikliğine bağlanmıştır.
5. **Spectral-Cs uzaysal kapasite tam aktive olmamıştır.** 100 epoch eğitim bütçesi altında Fourier katsayıları başlangıç değerlerinden anlamlı uzaklaşmamış; daha uzun eğitim (500–1000 epoch) ve daha güçlü donanım (A100/H100) gerekli.

---

## Karşılaşılan Kritik Hatalar (ve Düzeltmeleri)

Final eğitim öncesi yapılan paralel kod review ile tespit edilen yedi kritik hata:

| # | Etki | Düzeltme |
|---|---|---|
| 1 | NaN guard eksik | Kayıp + grad NaN/Inf tespiti ile backward atlama (`train.py:2293`) |
| 2 | Checkpoint mantığı | Her 10 epoch + her epoch latest copy |
| 3 | LR scheduler orantısız | Warmup = max\_epochs/5, $T_{0} = 0.6\times$post-warmup |
| 4 | Tier 1 freeze eksik | $\alpha_{\mathrm{dt}}, \mu_{\mathrm{dt}}^{(\ell)}, \text{backscatter}$ de eklendi |
| 5 | Gradient routing uyumsuz | Tier 1 ile çakışan routing otomatik kapatıldı |
| 6 | `gamma_damp` residual'dan eksik | NS artığında elevator damping eklendi |
| 7 | Loss scale kalibrasyon | `NS_RES_SCALE: 1e-4→1e-3`, `SPECTRUM_SHAPE_SCALE: 1→50` |

Detaylar: Bitirme 2 tezi Bölüm 4.2.3.

---

## Tasarım Felsefesi

1. **"Physics as structure, not penalty."** Operatörler ceza değil, mimari.
2. **Disiplinli parametre hijyeni.** Denklem-değiştirici parametreler öğrenilebilir bırakılmaz; sadece kapanış katsayıları öğrenilir.
3. **Akademik dürüstlük.** Başarısızlıkları gizlemek yerine sistematik biçimde belgelemek; 7 kritik kod hatası, 3 anti-fizik konfigürasyon ve 2 yapısal sınır final tez yazımında açıkça raporlanmıştır.

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
