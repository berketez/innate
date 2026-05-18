# Bitirme2 Ders Notu — REVİZE PLAN (2026-05-09)

## Hedef
Berke'nin projesini A'dan Z'ye anlatan kapsamlı ders notu — bir dönemlik ders gibi.
**Sadece teoriyi değil, yapılanları, çalışmayanları, neden çalışmadıklarını da içerir.**
LaTeX (saf siyah arka plan), tahmini 200 sayfa.

## Tarih Notu
- 8 Mart 2026 versiyonu sadece teori + plan içeriyordu (12 bölüm).
- Mart-Mayıs arası proje çok evrim geçirdi: MLP-SGS hibrit → Tier 1+2+3 freeze → Saf-INNATE Spectral.
- 9 Mayıs 2026 revizyonu: yapılanlar + başarısızlıklar + nihai mimari kapsamı eklendi.

## YENİ BÖLÜM YAPISI (16 bölüm + appendix)

### KISIM I — TEORİK ALTYAPI (Bölüm 1-7, ~80 sayfa)
Mevcut bölümler büyük oranda korunur, küçük güncellemeler.

1. Giriş ve Motivasyon (7 sayfa)
2. Akışkanlar Dinamiği Temelleri (12)
3. Boyutsuz Analiz (10)
4. Boussinesq Konveksiyon (10)
5. Spectral Methods (12) — rfftn, dealiasing detayı eklenir
6. Türbülans ve LES (12) — dynamic Smag, scale-similarity eklenir
7. Neural Operators (10)

### KISIM II — PROBLEM VE MİMARİ (Bölüm 8-9, ~30 sayfa)
8. Bizim Problemimiz — 3D Mixed Convection (15)
9. INNATE Mimarisi (15) — Tier 1+2+3 felsefesi yeni

### KISIM III — DENEMELER VE BAŞARISIZLIKLAR (Bölüm 10-13, ~60 sayfa) ← YENİ
**Bu kısım lisans tezinin asıl katkısı**

10. Eğitim Stratejisi Evrimi (15) — v1 hibrit → v2 MLP-SGS → v3 Spectral
11. Karşılaşılan Buglar ve Çözümleri (20) ← EN UZUN, EN KRİTİK
    - 11.1 Amplitude pumping (Mart)
    - 11.2 MLP fc2 split → Reynolds analojisi kırılması (Nisan)
    - 11.3 Phase A→B grad spike
    - 11.4 Phase C divergence (5 alt-fix)
    - 11.5 torch.compile dynamo hang (5-6 saat)
    - 11.6 dt CFL violation
    - 11.7 Eval rollout-time drift
    - 11.8 Kapasite eksikliği — under-actuation
12. Multi-Agent Debug Pipeline (10) — Codex + Cursor + 4 ajan
13. Saf-INNATE Spectral Mimari (15) — Fourier mode coefficient

### KISIM IV — SONUÇLAR (Bölüm 14-16, ~35 sayfa)
14. DNS/LES Referans ve Validasyon (10)
15. Sonuçlar — 503 vs 9905 Karşılaştırma (15)
16. Tartışma ve Future Work (10)

### APPENDIX (~5 sayfa)
- A: Analitik çözümler (mevcut)
- B: Parametre tablosu (Tier 1 listesi + Spectral truncation eklenir)

## YAZIM YOL HARİTASI

### Faz 1 — ŞİMDİ (eğitim koşarken paralel)
- [x] Arka plan siyah yapıldı (#000000)
- [x] PLAN revize edildi
- [ ] Bölüm 11 codex ile yazılır (~3 saat) ← öncelik
- [ ] Bölüm 13 architect ile yazılır (~2 saat)
- [ ] Bölüm 12 codex ile yazılır (~2 saat)

### Faz 2 — EĞİTİM BİTİNCE
- [ ] Bölüm 15 sayısal sonuçlar doldurulur
- [ ] Görseller üretilir (spectrum, time series, vs.)
- [ ] Bölüm 16 tartışma yazılır

### Faz 3 — RAPOR HAZIRLIK (4-5 gün)
- [ ] Mevcut Bölüm 1-9 gözden geçir, küçük güncellemeler
- [ ] reviewer ajanı tüm bölümleri kontrol eder
- [ ] LaTeX derleme, düzeltme, son rötuş
- [ ] PDF teslim

## TAHMİNİ SAYFA SAYISI
- Kısım I (teori): 80
- Kısım II (problem+mimari): 30
- Kısım III (denemeler+buglar): 60
- Kısım IV (sonuçlar): 35
- Appendix: 5
- **TOPLAM: ~210 sayfa**

## STRATEJİ
- Mevcut chapters/01-12.tex'leri silmiyoruz, gerekli yerlere update yapıyoruz.
- Yeni chapters: 11_buglar.tex, 12_multi_agent.tex, 13_spectral_innate.tex, 14_dns_les.tex, 15_sonuclar.tex, 16_tartisma.tex
- Ajanlar paralel yazsın, sonra reviewer cross-check yapar.
