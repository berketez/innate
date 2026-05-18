# Bitirme Tezi - Tam Üretim Pipeline'ı

**Proje:** Saf-INNATE Spectral-Cs v3 (9905 parametre, 20-layer fractional-step IMEX neural operator)
**Problem:** Mixed Convection 3D, Re=10000, Ra=1e5, Pr=0.71, Boussinesq, periodik BC
**Grid:** 96 × 160 × 64, domain Lx=2π, Ly=10, Lz=2π
**Eğitim:** TAMAM (100 epoch, num_steps=2000 INNATE-step, dt_layer=0.005, curriculum A→D)
**LES referans:** TAMAM (60000 step × dt=0.02 = 1200 zaman birim, 60 full snapshot)

> NOT: Mevcut tez şekilleri (`tez_figurleri/fig_01..04`) ile eğitim/eval karşılaştırma hazır. Bu pipeline kalan **6 büyük görevi** ele alır.

---

## 1. Ortam ve Yol Sözleşmesi

| Makine | Yol | Python | Kullanım |
|--------|-----|--------|----------|
| Mac (lokal) | `/Users/apple/Desktop/nsneuron1/bitirme2` | sistem `python3` | analiz, plot, hafif iş, indirme |
| 4090 (uzak) | `C:\Users\berke\Desktop\nsneuron1\bitirme2` | anaconda3 (3.11.9, torch 2.5.1+cu121, pyvista 0.48.2) | rollout, 3D render, video encode |

**SSH:** `~/.claude/remote-exec.sh "<komut>"` — ControlMaster persist, cd otomatik kalıcı.
**Schtasks zorunlu** uzun işler için (SSH oturum koparsa Start-Process child ölür).
**Türkçe karakter:** Her Python script başında `sys.stdout.reconfigure(encoding="utf-8")` zaten var (`02_innate_rollout.py`, `03_render_ansys.py`, `04_render_3d.py` doğrulandı).
**Log redirect:** `> log.txt 2>&1` ve `python -u` + `PYTHONUNBUFFERED=1` ZORUNLU.

**Ortak değişkenler (kopyala-yapıştır kolaylığı için):**
```
PROJ_WIN=C:\Users\berke\Desktop\nsneuron1\bitirme2
PROJ_MAC=/Users/apple/Desktop/nsneuron1/bitirme2
CKPT_WIN=results_v2\checkpoints\checkpoint_epoch000099.pt
CKPT_SPEC=results_v2\4090_sync\checkpoint_epoch000099_spectral.pt   # spectral varyantı varsa
LES_NPZ_WIN=data\sim_states\les_real_60k.npz
IC_WIN=data\sim_states\shared_ic_seed42.npz
```
> Hangi checkpoint? `results_v2/4090_sync/checkpoint_epoch000099_spectral.pt` mevcutsa onu kullan (saf-INNATE Spectral v3). Yoksa `checkpoints/checkpoint_epoch000099.pt`. Adım 0'da doğrulanacak.

---

## 2. Eğitim/Rollout Zaman Hesabı (KRİTİK)

| Kavram | Değer |
|--------|-------|
| INNATE-step | 1 model() çağrısı = 20 layer = 20 × dt_layer |
| dt_layer | **0.005** (CFL için zorunlu — config default 0.02 NaN üretir) |
| 1 INNATE-step zaman | 20 × 0.005 = **0.1 birim** |
| LES dt | 0.02 |
| 1 LES-step zaman | **0.02 birim** |
| Eğitimde görülen mesafe | 2000 INNATE-step × 0.1 = **200 zaman birimi** |
| Script `--n-steps` argümanı | **LES-step birimi bekler**, içeride `n_innate = n_steps // n_layers` (20) yapar |
| 200 birim rollout için | `--n-steps 40000` (40000 × 0.005 = 200) |
| LES verisi (mevcut) | 60000 step × 0.02 = **1200 zaman birimi** |
| Karşılaştırma penceresi | **t ∈ [0, 200]** (eğitim mesafesi) — LES'ten ilk 10000 step alınır |

---

## 3. Pipeline Adımları

### Adım 0 — Hazırlık ve Doğrulama (Mac, 5 dk)

| Ne | Komut |
|----|-------|
| Spectral ckpt var mı | `~/.claude/remote-exec.sh "if exist %PROJ_WIN%\results_v2\4090_sync\checkpoint_epoch000099_spectral.pt echo SPECTRAL_VAR else echo SPECTRAL_YOK"` |
| Disk alanı | `~/.claude/remote-exec.sh --status` |
| Mac viz_pipeline | `ls /Users/apple/Desktop/nsneuron1/bitirme2/viz_pipeline/` |
| Mac data dizini | `ls /Users/apple/Desktop/nsneuron1/bitirme2/data/sim_states/ 2>/dev/null || mkdir -p /Users/apple/Desktop/nsneuron1/bitirme2/data/sim_states` |
| Schtasks test | `~/.claude/remote-exec.sh "schtasks /query /tn DummyTest 2>nul & echo SCHTASKS_OK"` |

**Karar noktası:**
- `SPECTRAL_VAR` ise → `CKPT_USE=results_v2\4090_sync\checkpoint_epoch000099_spectral.pt` + `--use-spectral-cs` flag
- `SPECTRAL_YOK` ise → `CKPT_USE=results_v2\checkpoints\checkpoint_epoch000099.pt` (spectral flag yok)

---

### Adım 1 — INNATE Rollout (4090, ~10-15 dk)

| Alan | Değer |
|------|-------|
| Ne | Eğitilmiş ckpt ile shared IC'den 200 birim rollout, slice+snapshot+metric kaydet |
| Nerede | 4090, Schtasks (background) |
| Süre | ~10-15 dk (40000 LES-step / 2000 INNATE-step, RTX 4090) |
| Çıktı | `C:\Users\berke\...\bitirme2\data\sim_states\innate_rollout_200.npz` (~150-300 MB) |
| Log | `C:\Users\berke\...\bitirme2\logs\innate_rollout_200.log` |
| Bağımlılık | Adım 0 (ckpt seçimi) |

**Komut (Spectral ckpt varsa):**
```bash
~/.claude/remote-exec.sh cd C:\Users\berke\Desktop\nsneuron1\bitirme2

~/.claude/remote-exec.sh "schtasks /create /tn INNATE_ROLLOUT /sc once /st 00:00 /tr \"cmd /c cd /d C:\Users\berke\Desktop\nsneuron1\bitirme2 && set PYTHONUNBUFFERED=1 && C:\Users\berke\anaconda3\python.exe -u viz_pipeline\02_innate_rollout.py --checkpoint results_v2\4090_sync\checkpoint_epoch000099_spectral.pt --use-spectral-cs --spectral-kx 5 --spectral-ky 8 --spectral-kz 6 --Re 10000 --Ra 1e5 --Pr 0.71 --n-steps 40000 --dt 0.005 --shared-ic data\sim_states\shared_ic_seed42.npz --output data\sim_states\innate_rollout_200.npz --device cuda > logs\innate_rollout_200.log 2>&1\" /f /ru SYSTEM"

~/.claude/remote-exec.sh "schtasks /run /tn INNATE_ROLLOUT"
```

**Spectral ckpt YOKSA (`--use-spectral-cs` ve `--spectral-*` flag'lerini çıkar, ckpt yolunu değiştir):**
```
... C:\Users\berke\anaconda3\python.exe -u viz_pipeline\02_innate_rollout.py --checkpoint results_v2\checkpoints\checkpoint_epoch000099.pt --Re 10000 --Ra 1e5 --Pr 0.71 --n-steps 40000 --dt 0.005 ...
```

**İzleme:**
```bash
~/.claude/remote-exec.sh "type C:\Users\berke\Desktop\nsneuron1\bitirme2\logs\innate_rollout_200.log | more +0"
~/.claude/remote-exec.sh "schtasks /query /tn INNATE_ROLLOUT /fo LIST | findstr /i \"Status Last\""
```

**Tamamlanma kontrolü:**
```bash
~/.claude/remote-exec.sh "dir C:\Users\berke\Desktop\nsneuron1\bitirme2\data\sim_states\innate_rollout_200.npz"
~/.claude/remote-exec.sh "findstr /C:\"BITTI\" /C:\"Saved\" /C:\"NaN\" /C:\"Traceback\" C:\Users\berke\Desktop\nsneuron1\bitirme2\logs\innate_rollout_200.log"
```

**Beklenen NPZ alanları:** `slice_y_mid`, `slice_z_mid`, `full_snaps` (~60 snapshot, 5 alan × 96×160×64), `metrics_t` (TKE, Nu, theta_rms vs.), `t` (zaman vektörü 0→200).

---

### Adım 2 — Sayısal Karşılaştırma Figürleri (Mac veya 4090, ~5 dk)

| Alan | Değer |
|------|-------|
| Ne | LES vs INNATE (t∈[0,200]): TKE(t), Nu(t), θ_rms(t), E(k) spektrum, hata zarfı |
| Nerede | Mac (matplotlib yeterli, network transfer 300 MB rollout npz indirilir) |
| Süre | 5-10 dk |
| Çıktı | `tez_figurleri/fig_05_innate_vs_les_metrikler.png`, `fig_06_spektrum_karsilastirma.png`, `fig_07_hata_zarfi.png` |
| Bağımlılık | Adım 1 |

**Önce npz'yi Mac'e indir:**
```bash
scp 4090:C:/Users/berke/Desktop/nsneuron1/bitirme2/data/sim_states/innate_rollout_200.npz /Users/apple/Desktop/nsneuron1/bitirme2/data/sim_states/
```

**Yeni script gerekli:** `viz_pipeline/05_compare_metrics.py` (developer agent yazacak, ~150 satır)

Ana içerik:
- LES npz ilk 10000 step'i al → t ∈ [0,200], dt=0.02
- INNATE npz: tüm 2000 INNATE-step → t ∈ [0,200], dt=0.1
- 4 panel figür: TKE(t), Nu(t), θ_rms(t), E(k) (t=200'de)
- Logaritmik spektrum, -5/3 ve -3 referans çizgileri
- TR etiketler, font: DejaVu Sans

**Komut (script hazırlandıktan sonra, Mac):**
```bash
cd /Users/apple/Desktop/nsneuron1/bitirme2
python3 viz_pipeline/05_compare_metrics.py \
  --les data/sim_states/les_real_60k.npz \
  --innate data/sim_states/innate_rollout_200.npz \
  --t-max 200.0 \
  --output-dir tez_figurleri
```

---

### Adım 3 — LES 3D 4K@60fps Video (4090, ~2-4 saat)

| Alan | Değer |
|------|-------|
| Ne | LES tüm 60 snapshot (t∈[20,1200]) rotation, volume+Q-iso+streamline 4K@60fps |
| Nerede | 4090, Schtasks (background, GPU encode nvenc) |
| Süre | Precompute ~30 dk (8 worker) + render ~60-120 dk + encode ~10 dk |
| Çıktı | `data\renders\les_3d_4k_60fps.mp4` (~1-2 GB) |
| Bağımlılık | Yok (les_real_60k.npz mevcut) |

**Komut:**
```bash
~/.claude/remote-exec.sh "mkdir C:\Users\berke\Desktop\nsneuron1\bitirme2\data\renders 2>nul"

~/.claude/remote-exec.sh "schtasks /create /tn LES_3D_RENDER /sc once /st 00:00 /tr \"cmd /c cd /d C:\Users\berke\Desktop\nsneuron1\bitirme2 && set PYTHONUNBUFFERED=1 && C:\Users\berke\anaconda3\python.exe -u viz_pipeline\04_render_3d.py --input data\sim_states\les_real_60k.npz --output data\renders\les_3d_4k_60fps.mp4 --label \\\"LES Referansi (Smagorinsky SGS, Cs=0.17)\\\" --duration 60 --fps 60 --bitrate 40M --workers 8 > logs\les_3d_render.log 2>&1\" /f /ru SYSTEM"

~/.claude/remote-exec.sh "schtasks /run /tn LES_3D_RENDER"
```

**İzleme:**
```bash
~/.claude/remote-exec.sh "type C:\Users\berke\Desktop\nsneuron1\bitirme2\logs\les_3d_render.log | findstr /C:\"PRECOMPUTE\" /C:\"frame\" /C:\"fps\" /C:\"saved\" /C:\"Traceback\""
```

**Tek frame önce test (60 dk koşmadan önce ZORUNLU):**
```bash
~/.claude/remote-exec.sh "cd /d C:\Users\berke\Desktop\nsneuron1\bitirme2 && C:\Users\berke\anaconda3\python.exe viz_pipeline\04_render_3d.py --input data\sim_states\les_real_60k.npz --output test_les_45.png --single-frame 45"
```
→ PNG çıkıyorsa schtasks başlat.

---

### Adım 4 — INNATE 2D Slice Video (4090, ~30-45 dk)

| Alan | Değer |
|------|-------|
| Ne | INNATE rollout slice frame'lerinden 4K@60fps karşılaştırma (LES yarı / INNATE yarı) |
| Nerede | 4090 |
| Süre | 30-45 dk (matplotlib only, GPU encode) |
| Çıktı | `data\renders\karsilastirma_2d_4k_60fps.mp4` (~500 MB) |
| Bağımlılık | Adım 1 |

`03_render_ansys.py` mevcut, LES+INNATE iki bölümlü video üretir (28 sn LES + 2 sn geçiş + 28 sn INNATE).

**Komut:**
```bash
~/.claude/remote-exec.sh "schtasks /create /tn INNATE_2D_RENDER /sc once /st 00:00 /tr \"cmd /c cd /d C:\Users\berke\Desktop\nsneuron1\bitirme2 && set PYTHONUNBUFFERED=1 && C:\Users\berke\anaconda3\python.exe -u viz_pipeline\03_render_ansys.py --les data\sim_states\les_real_60k.npz --innate data\sim_states\innate_rollout_200.npz --output data\renders\karsilastirma_2d_4k_60fps.mp4 --fps 60 --les-duration 28 --innate-duration 28 --transition-duration 2 --bitrate 20M --les-tail 10000 > logs\innate_2d_render.log 2>&1\" /f /ru SYSTEM"

~/.claude/remote-exec.sh "schtasks /run /tn INNATE_2D_RENDER"
```

> `--les-tail 10000` ile LES'in ilk 10000 step'i alınır (eğitim mesafesi = 200 birim, dt=0.02). Argüman yoksa script tüm LES'i kullanır.

---

### Adım 5 — INNATE 3D 4K@60fps Video (4090, ~2-3 saat)

| Alan | Değer |
|------|-------|
| Ne | INNATE rollout full_snaps (~60 snapshot) rotation, aynı format LES gibi |
| Nerede | 4090 |
| Süre | ~2-3 saat |
| Çıktı | `data\renders\innate_3d_4k_60fps.mp4` (~1-2 GB) |
| Bağımlılık | Adım 1, Adım 3 (önce LES 3D testi geçsin format için) |

**Tek frame test ZORUNLU (innate npz formatı 04_render_3d uyumlu mu doğrula):**
```bash
~/.claude/remote-exec.sh "cd /d C:\Users\berke\Desktop\nsneuron1\bitirme2 && C:\Users\berke\anaconda3\python.exe viz_pipeline\04_render_3d.py --input data\sim_states\innate_rollout_200.npz --output test_innate_30.png --single-frame 30"
```

> Eğer hata verirse: `02_innate_rollout.py` çıktı npz formatı `full_snaps` array shape'i farklı olabilir. Düzeltici küçük adapter script gerekebilir (`viz_pipeline/_adapter_innate_to_les_format.py`).

**Komut (test geçtikten sonra):**
```bash
~/.claude/remote-exec.sh "schtasks /create /tn INNATE_3D_RENDER /sc once /st 00:00 /tr \"cmd /c cd /d C:\Users\berke\Desktop\nsneuron1\bitirme2 && set PYTHONUNBUFFERED=1 && C:\Users\berke\anaconda3\python.exe -u viz_pipeline\04_render_3d.py --input data\sim_states\innate_rollout_200.npz --output data\renders\innate_3d_4k_60fps.mp4 --label \\\"INNATE Spectral-Cs (9905 param, saf-fizik)\\\" --duration 60 --fps 60 --bitrate 40M --workers 8 > logs\innate_3d_render.log 2>&1\" /f /ru SYSTEM"

~/.claude/remote-exec.sh "schtasks /run /tn INNATE_3D_RENDER"
```

---

### Adım 6 — Karşılaştırma Yan Yana 3D (4090, ~3-5 saat) — OPSİYONEL

| Alan | Değer |
|------|-------|
| Ne | Tek video: sol yarı LES 3D, sağ yarı INNATE 3D, senkron zaman (eğitim mesafesi=200 birim) |
| Nerede | 4090 |
| Süre | 3-5 saat |
| Çıktı | `data\renders\sidebyside_3d_4k_60fps.mp4` |
| Bağımlılık | Adım 3, 5 tamamlandıktan sonra ffmpeg ile birleştirme |

**Pratik yaklaşım: ffmpeg hstack ile mevcut iki videoyu yan yana koy**
```bash
~/.claude/remote-exec.sh "ffmpeg -i data\renders\les_3d_4k_60fps.mp4 -i data\renders\innate_3d_4k_60fps.mp4 -filter_complex \"[0:v]scale=1920:2160[l];[1:v]scale=1920:2160[r];[l][r]hstack=inputs=2\" -c:v h264_nvenc -b:v 60M -preset p5 data\renders\sidebyside_3d_4k_60fps.mp4"
```

> Senkron için iki video aynı `duration` ve `fps` ile üretilmeli (60s, 60fps). Adım 3 ve 5'te bu sağlanıyor.

---

### Adım 7 — Tez Şekilleri (Mac, ~3-4 saat insan + AI)

| Alan | Değer |
|------|-------|
| Ne | bitirme1 örnek alarak ~16 LaTeX şekli üret |
| Nerede | Mac, TikZ + matplotlib + PNG snapshot |
| Süre | 3-4 saat (developer agent yardımı) |
| Çıktı | `tez_figurleri/` altında ek PNG/PDF dosyaları |
| Bağımlılık | Adım 1 (snapshot için), Adım 3/5 (3D PNG frame'ler için) |

**Hedef şekiller listesi:**

| # | İsim | Tip | Kaynak | Komut/Yöntem |
|---|------|-----|--------|--------------|
| 01 | LES metrikleri | Hazır | `fig_01_les_metrikleri.png` | ✓ |
| 02 | LES spektrum | Hazır | `fig_02_les_spektrum.png` | ✓ |
| 03 | INNATE eğitim eğrileri | Hazır | `fig_03_innate_egitim.png` | ✓ |
| 04 | Eval karşılaştırma | Hazır | `fig_04_eval_karsilastirma.png` | ✓ |
| 05 | INNATE vs LES metrikler (TKE/Nu/θ_rms) | Yeni | Adım 2 |
| 06 | Spektrum karşılaştırma (E(k) LES vs INNATE) | Yeni | Adım 2 |
| 07 | Hata zarfı (rollout drift) | Yeni | Adım 2 |
| 08 | INNATE mimari diyagramı | TikZ | Yeni, 20-layer fractional-step blok şeması | `tez/figs/fig_08_mimari.tex` |
| 09 | Spectral-Cs modulator şeması | TikZ | Fourier mode coefficient + IFFT-back grafiği | `tez/figs/fig_09_spectral.tex` |
| 10 | Curriculum şeması (A→D) | matplotlib bar | Tier 1+2+3+4 epoch aralıkları + loss ağırlıkları | yeni script |
| 11 | 2D slice 4-zaman snapshot (LES vs INNATE) | matplotlib | t∈{50,100,150,200} XZ slice | Adım 1 + yeni script `06_slice_snapshots.py` |
| 12 | 3D snapshot 4-zaman (LES, render) | PyVista PNG | 4090 single-frame mode × 4 | `04_render_3d.py --single-frame {12,24,36,48}` |
| 13 | 3D snapshot 4-zaman (INNATE) | PyVista PNG | 4090 single-frame mode × 4 | aynı, innate npz |
| 14 | Grid bağımsızlık / LES skor tablosu | TikZ veya tablo | grid_analysis.py çıktısı | TR tablo |
| 15 | Loss curve TR/training tier'lar | matplotlib | `train_spectral.log` parse | yeni script |
| 16 | Domain ve BC şeması | TikZ | Hot/cold wall, periodic, gravity vector | `tez/figs/fig_16_domain.tex` |

**Türkçe LaTeX kuralları:**
- `\usepackage[utf8]{inputenc}`, `\usepackage[T1]{fontenc}`, `\usepackage[turkish]{babel}`
- Tüm metin/etiket TR (ç,ğ,ı,İ,ö,ş,ü doğru)
- TikZ etiketleri Türkçe

**Yöntem:**
1. `architect` agent ile şekil 08, 09, 16 (TikZ) tasarımı yap
2. `developer` agent ile şekil 10, 11, 15 (matplotlib script) yaz
3. `04_render_3d.py --single-frame` ile şekil 12, 13 PNG frame al
4. Şekil 14 mevcut `grid_analysis.py` çıktısından TR tablo

---

## 4. Tahmini Toplam Süre

| Faz | İnsan + AI | Hesaplama |
|-----|-----------|-----------|
| Adım 0 doğrulama | 5 dk | - |
| Adım 1 rollout | 5 dk komut | 10-15 dk (4090) |
| Adım 2 metrik fig | 30 dk script | 5 dk run |
| Adım 3 LES 3D video | 10 dk komut | 2-4 saat |
| Adım 4 INNATE 2D video | 5 dk komut | 30-45 dk |
| Adım 5 INNATE 3D video | 10 dk komut | 2-3 saat |
| Adım 6 side-by-side | 10 dk komut | 30 dk ffmpeg |
| Adım 7 şekiller (12 yeni) | 3-4 saat | 1-2 saat run |
| **Toplam** | **~5 saat aktif** | **~8-10 saat hesap (paralelleştirilebilir)** |

**Paralelleştirme:** Adım 3 (LES 3D) ve Adım 1 (rollout) eş zamanlı 4090'da çalıştırılabilir (GPU paylaşımı sıkı olabilir, sıralı önerilir). Adım 2 metrik figürleri Adım 1 biter bitmez Mac'te paralel başlar.

**Önerilen sıra:**
```
[Adım 0]
    ↓
[Adım 1: INNATE rollout, 4090] ────┐
    ↓                              │
[Adım 2: metrik figürleri, Mac]    │ (paralel)
    ↓                              ↓
[Adım 4: 2D karşılaştırma video, 4090]
    ↓
[Adım 3: LES 3D video, 4090]
    ↓ (LES test geçince)
[Adım 5: INNATE 3D video, 4090]
    ↓
[Adım 6: side-by-side, 4090]
    ↓
[Adım 7: tez şekilleri, Mac] (Adım 1 sonrası başlanabilir)
```

---

## 5. Beklenmedik Durumlar / Kontrol Noktaları

| Risk | Belirti | Çözüm |
|------|---------|-------|
| NaN rollout step 200+ | log'da `NaN detected` | dt'yi 0.005 → 0.003'e düşür, --n-steps 66667 |
| Spectral ckpt formatı uyumsuz | `Missing keys: spectral.kx_max` | `--use-spectral-cs --spectral-kx 5 --spectral-ky 8 --spectral-kz 6` flag eksik mi kontrol |
| 04_render_3d.py innate npz okuyamıyor | KeyError: `full_snaps` | innate npz key isimlerini LES formatına map et (adapter script) |
| Schtasks "Access denied" | SYSTEM olarak run gerek | `/ru SYSTEM` yerine `/ru %USERNAME% /rp <pass>` veya elevated PowerShell |
| nvenc encoder yok | ffmpeg "Unknown encoder h264_nvenc" | `--no-nvenc` flag → libx264 (5× yavaş ama çalışır) |
| Disk dolu | 4090'da `Insufficient space` | Render mp4 + npz toplam ~5-6 GB. `data\renders\` ayrı disk'e taşı |
| LES tail mismatch | INNATE 200 birim, LES tüm 1200 | `--les-tail 10000` LES'i 0-200 aralığına kısıtlar |
| Türkçe karakter cp1254 hatası | `UnicodeEncodeError` | Script başı `sys.stdout.reconfigure(encoding="utf-8")` doğrulanmış ✓ |
| SSH oturum koparsa | Schtasks kullanılmadıysa iş ölür | **HEPSI** Schtasks olmalı, Start-Process yasak |
| 4090 GPU başka iş çalıştırıyor | nvidia-smi'de %100 | `~/.claude/remote-exec.sh --status` ile kontrol, gerekirse beklet |

**Kontrol noktaları (her adım bitiminde):**
- Adım 1: `findstr /C:"BITTI"` log'da görünüyor mu? npz boyutu >100 MB mı?
- Adım 2: 3 PNG figür oluştu mu? TKE eğrisi makul mü (drift varsa rapor et)?
- Adım 3/5: video dosyası >500 MB mı? `ffprobe` ile süre 60s mi?
- Adım 7: 16 şekil listesi tam mı? LaTeX'te `\includegraphics` ile derlenebiliyor mu?

---

## 6. Mac'e İndirilecek Final Dosyalar

Tüm pipeline bittikten sonra 4090'dan Mac'e çekilecekler:

```bash
# Veri dosyaları (analiz için)
scp 4090:C:/Users/berke/Desktop/nsneuron1/bitirme2/data/sim_states/innate_rollout_200.npz \
    /Users/apple/Desktop/nsneuron1/bitirme2/data/sim_states/

# Videolar (~5-6 GB toplam)
mkdir -p /Users/apple/Desktop/nsneuron1/bitirme2/data/renders
scp 4090:C:/Users/berke/Desktop/nsneuron1/bitirme2/data/renders/les_3d_4k_60fps.mp4 \
    /Users/apple/Desktop/nsneuron1/bitirme2/data/renders/
scp 4090:C:/Users/berke/Desktop/nsneuron1/bitirme2/data/renders/innate_3d_4k_60fps.mp4 \
    /Users/apple/Desktop/nsneuron1/bitirme2/data/renders/
scp 4090:C:/Users/berke/Desktop/nsneuron1/bitirme2/data/renders/karsilastirma_2d_4k_60fps.mp4 \
    /Users/apple/Desktop/nsneuron1/bitirme2/data/renders/
scp 4090:C:/Users/berke/Desktop/nsneuron1/bitirme2/data/renders/sidebyside_3d_4k_60fps.mp4 \
    /Users/apple/Desktop/nsneuron1/bitirme2/data/renders/

# Loglar (debug/dokümantasyon için)
mkdir -p /Users/apple/Desktop/nsneuron1/bitirme2/logs
scp 4090:C:/Users/berke/Desktop/nsneuron1/bitirme2/logs/*.log \
    /Users/apple/Desktop/nsneuron1/bitirme2/logs/

# 3D snapshot PNG'ler (tez şekilleri 12, 13)
scp 4090:C:/Users/berke/Desktop/nsneuron1/bitirme2/tez_figurleri/*.png \
    /Users/apple/Desktop/nsneuron1/bitirme2/tez_figurleri/
```

**Final teslim klasör yapısı (Mac):**
```
bitirme2/
├── data/
│   ├── sim_states/
│   │   ├── les_real_60k.npz          (2.78 GB, 4090'dan inmedi, opsiyonel)
│   │   ├── innate_rollout_200.npz    (~300 MB) ← İNDİR
│   │   └── shared_ic_seed42.npz      (19 MB)
│   └── renders/
│       ├── les_3d_4k_60fps.mp4           ← İNDİR
│       ├── innate_3d_4k_60fps.mp4        ← İNDİR
│       ├── karsilastirma_2d_4k_60fps.mp4 ← İNDİR
│       └── sidebyside_3d_4k_60fps.mp4    ← İNDİR
├── tez_figurleri/
│   ├── fig_01..04 (hazır)
│   ├── fig_05_innate_vs_les_metrikler.png
│   ├── fig_06_spektrum_karsilastirma.png
│   ├── fig_07_hata_zarfi.png
│   ├── fig_10_curriculum.png
│   ├── fig_11_slice_4zaman.png
│   ├── fig_12_les_3d_4zaman.png  (4 PNG birleşik)
│   ├── fig_13_innate_3d_4zaman.png
│   └── fig_15_loss_curves.png
├── tez/
│   ├── section5_thermal_failure.tex (mevcut)
│   └── figs/
│       ├── fig_08_mimari.tex (TikZ)
│       ├── fig_09_spectral.tex (TikZ)
│       └── fig_16_domain.tex (TikZ)
└── PIPELINE.md (bu dosya)
```

---

## 7. Onay Kontrolleri (Berke için)

Bu pipeline'ı onaylamadan önce kararlaştırılması gereken sorular:

1. **Spectral ckpt mevcut mu?** Adım 0'da netleştirilecek. Eğer yoksa standart 9905 param ckpt kullanılır.
2. **Eğitim mesafesi (200 birim) vs LES tam süre (1200 birim):** Tez metni karşılaştırma penceresini hangisi olarak sunsun?
   - Önerilen: **Eğitim mesafesi (200 birim)** — INNATE'in nominal performansı + extrapolation analizi için 200-400 birim ek "out-of-training" bölgesi raporlanır.
3. **Negatif bulgu raporlanma şekli:** Memory'de yazılı — eval iyileşmedi, under-actuation çürütüldü. Bu pipeline figürleri **dürüst negatif sonuç** sunacak şekilde tasarlandı (hata zarfı, drift gösterimi).
4. **Side-by-side video (Adım 6)** zorunlu mu? Tez savunması için **çok güçlü** ama opsiyonel.
5. **Tez şekli 12 ve 13 için kaç zaman noktası?** Önerilen: t∈{50, 100, 150, 200} (4 noktada hem LES hem INNATE)

---

## 8. Onaydan Sonra İlk Komut

```bash
# Adım 0 doğrulama (paralel)
~/.claude/remote-exec.sh "if exist C:\Users\berke\Desktop\nsneuron1\bitirme2\results_v2\4090_sync\checkpoint_epoch000099_spectral.pt (echo SPECTRAL_VAR) else (echo SPECTRAL_YOK)"
~/.claude/remote-exec.sh --status
~/.claude/remote-exec.sh "dir C:\Users\berke\Desktop\nsneuron1\bitirme2\data\sim_states"
```

Çıktıya göre ckpt seçilir, Adım 1 başlatılır.

---

*Hazırlayan: architect agent — 2026-05-11*
*Bağlam: Saf-INNATE Spectral v3, eğitim 100 ep × 2000 INNATE-step × dt_layer=0.005 = 200 zaman birim, eval negatif (under-actuation çürütüldü, TBPTT distance gap asıl darboğaz)*
