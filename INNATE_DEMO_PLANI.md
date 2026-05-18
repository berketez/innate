# INNATE Demo Planı

## Hedef
Tek bir Jupyter Notebook: `INNATE_demo.ipynb`
Açan kişi "Run All" yapınca 5 dakikada çalışsın, sonuçları görsün.
Mülakatta, GitHub'da, bitirme sunumunda gösterilebilir.

---

## Demo Ne İçermeli (5 Bölüm)

### Bölüm 1: "INNATE Nedir?" (Markdown hücre, kod yok)
- 1 paragraf: "285 parametrelik neural operator ile 3D Navier-Stokes çözüyoruz"
- 1 şema: Mimari diyagram (pipeline: input → FNO layers → Leray projector → output)
- Karşılaştırma tablosu: INNATE vs klasik CFD vs diğer neural operator'lar

### Bölüm 2: Model Yükleme + TGV3D Demo (Çalışan kod)
- Eğitilmiş checkpoint'u yükle (`bitirme/results_innate_tgv3d/innate_tgv_epoch_5000.pth`)
- Bu basit, çalışan versiyon — bitirme2'deki sorunlu olan değil
- 1 satır: model yükle
- 1 satır: initial condition oluştur (Taylor-Green Vortex)
- 1 satır: 100 adım simüle et
- Çıktı: 3D velocity field

### Bölüm 3: Görselleştirme (En etkileyici kısım)
- Velocity magnitude slice (2D kesit, renkli)
- Enerji decay grafiği (INNATE vs analitik çözüm)
- Vorticity isosurface (3D render — zaten `visualize/renderer3d.py` var)
- Eğer `room_simulation.gif` çalışıyorsa, onu da göster

### Bölüm 4: Fizik Doğrulama (Bilimsel güvenilirlik)
- Energy conservation: toplam kinetik enerji zamanla nasıl değişiyor
- Divergence-free: ∇·u ≈ 0 olduğunu göster (Leray projector çalışıyor)
- Enstrophy decay: beklenen davranışla karşılaştır
- Tablo: "Metrik | INNATE | Analitik | Hata %"

### Bölüm 5: Neden Önemli (Markdown hücre, kod yok)
- "285 parametre vs standart FNO'ların 500K+ parametresi"
- "Apple Silicon'da eğitilebilir — GPU cluster gerektirmez"
- "Klasik CFD solver'dan X kat hızlı inference"
- Gelecek: farklı Re sayıları, farklı geometriler

---

## Teknik Notlar

### Hangi checkpoint'u kullan?
`bitirme/` dizinindeki basit TGV3D modeli (5000 epoch).
NEDEN: bitirme2 hala sorunlu (termal collapse, mode collapse).
Demo için ÇALIŞAN şey lazım, en iyi şey değil.

### Gerekli dosyalar:
- `innate.py` (ana kütüphane)
- `bitirme/model.py` (INNATE3D_TGV sınıfı)
- `bitirme/results_innate_tgv3d/innate_tgv_epoch_5000.pth` (checkpoint)
- Notebook kendisi

### Bağımlılıklar:
```
torch
numpy
matplotlib
```

### Çalışma süresi hedefi:
- Model yükleme: <1 saniye
- 100 adım simülasyon: <30 saniye (MPS)
- Görselleştirme: <10 saniye
- TOPLAM: <1 dakika

---

## Yapılacaklar Sırası

1. [ ] Notebook oluştur, Bölüm 1'i yaz (10 dk)
2. [ ] Bölüm 2: checkpoint yükle, çalıştığını doğrula (30 dk)
3. [ ] Bölüm 3: matplotlib ile slice + enerji grafiği (30 dk)
4. [ ] Bölüm 4: divergence + enstrophy hesapla (20 dk)
5. [ ] Bölüm 5: sonuç yazısı (5 dk)
6. [ ] README.md yaz (15 dk)
7. [ ] GitHub'a pushla (5 dk)

**Toplam: ~2 saat**

---

## Demo Bittikten Sonra

- GitHub repo'suna `README.md` ekle (proje açıklaması + demo linki)
- LinkedIn'e 1 post: "285 parametreyle 3D Navier-Stokes çözdüm" + notebook screenshot
- arXiv pre-print (opsiyonel ama güçlü): 4-6 sayfa, INNATE mimarisi + TGV sonuçları
