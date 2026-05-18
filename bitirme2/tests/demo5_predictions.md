# Demo 5: Engineering Problems -- Analitik/Yari-Analitik Tahminler

**Tarih:** 2026-03-12
**Hazirlayan:** Codex Consultant (Teknik Danismanlik)
**Amac:** INNATE demo 5 case'leri icin bagimsiz fiziksel buyukluk tahminleri

---

## KALIBRASYON ANCHOR NOKTASI

INNATE LES referans verileri (96x160x64 grid, Kolmogorov forcing, Boussinesq):

| Parametre | Re=7000, Ra=1e5 | Re=10000, Ra=1e5 |
|-----------|-----------------|-------------------|
| Ri | 0.00287 | 0.00141 |
| TKE | 0.00848 | 0.00731 |
| Nu | 39.3 | 70.5 |
| spectrum_slope | -1.739 | -1.777 |
| theta_rms | 0.0240 | 0.0268 |
| max_velocity | 0.345 | 0.299 |
| CFL | 0.106 | 0.095 |
| mean_nu_t | 1.04e-4 | 1.08e-4 |
| enstrophy | 0.501 | 0.538 |
| dissipation_total | 2.47e-4 | 2.24e-4 |
| forcing_power | 3.00e-4 | 2.36e-4 |
| dt | 0.02 | 0.02 |

**Onemli gozlemler:**
- Re artinca TKE AZALIYOR (0.00848 -> 0.00731): grid-relative normalizasyon etkisi
- Re artinca Nu ARTIYOR (39 -> 70): daha fazla konvektif karisim
- Slope Re ile steepening (-1.74 -> -1.78): daha genis inertial range
- theta_rms artisi (0.024 -> 0.027): daha guclu termal fluktuasyonlar
- P_in/eps ~ 1.0-1.2: yaklasik denge durumunda (quasi-stationary)
- Her iki durumda da Ri << 1: neredeyse saf forced convection

**Normalizasyon:**
- Hizlar: U_ref = forcing_amplitude * Re_tau benzeri bir olcek (Kolmogorov forcing)
- Uzunluklar: domain boyutlari (Lx=6, Ly=10, Lz=4)
- TKE: = 0.5 * <u'_i u'_i> / U_ref^2 (boyutsuz)
- Nu: = 1 + (Re * Pr * <v'*theta'>) / (kappa * dT/Ly) seklinde boyutsuz

---

## 5a: RUZGAR TURBINI IZ BOLGESI (Wind Turbine Wake)

### Parametreler
- Re = 100,000 | Ra ~ 0 (1e-10) | Pr = 0.71
- Ri = Ra/(Re^2 * Pr) = 1e-10 / (1e10 * 0.71) ~ 1.4e-21 (etkin sifir)
- nu = 1/Re = 1e-5
- kappa = 1/(Re*Pr) = 1.41e-5
- Forcing: uniform (Kolmogorov degil!)
- Post-step: Actuator disk, C_T=0.75, sigma=0.3, x_disk=Lx/4=1.5

### Akis Rejimi
- **Saf forced convection** (Ri ~ 0, buoyancy tamamen kapali)
- Re = 100,000: egitimden 10-14x buyuk
- Uniform forcing: egitimden farkli (Kolmogorov yerine)
- Kolmogorov microscale: eta = (nu^3/eps)^(1/4)

### Actuator Disk Teorisi (Betz/Momentum Theory)

**1D Actuator Disk (Betz):**
Thrust coefficient ve induction factor iliskisi:
  C_T = 4a(1-a)

C_T = 0.75 icin:
  4a - 4a^2 = 0.75
  a^2 - a + 0.1875 = 0
  a = (1 - sqrt(1 - 4*0.1875)) / 2 = (1 - sqrt(0.25)) / 2 = (1 - 0.5) / 2 = 0.25

Disk arkasi hiz:
  u_wake/U_inf = 1 - 2a = 1 - 0.5 = 0.5

**Wake deficit = 0.50** (near-wake, ideal momentum theory)

**Jensen Wake Model (far-wake decay):**
  Delta_u(x) / U_inf = (1 - sqrt(1 - C_T)) / (1 + k*x/r_0)^2

k ~ 0.075 (offshore), r_0 ~ D/2 (turbin yarican)
Periyodik domain'de D tanimli degil, ama drag disk sigma=0.3 -> etkin D ~ 2*sigma ~ 0.6

Far-wake decay (x = 3 boyutsuz birim = 5D downstream):
  (1-sqrt(0.25)) / (1 + 0.075*3/0.3)^2 = 0.5 / (1.75)^2 = 0.5/3.06 = 0.163

### TKE Tahmini

LES referansindan scaling:
- Re=10K'de TKE=0.00731 (Kolmogorov forcing)
- Re=100K'de uniform forcing ile:

TKE uniform vs Kolmogorov: Uniform forcing daha homojen, Kolmogorov daha lokalize.
Genel olarak uniform forcing'de TKE biraz dusuk olabilir (shear uretimi az).

Re scaling (izotropik turbulansta TKE ~ Re^0 veya zayif Re bagimli):
- Ama bu periodic box'ta TKE ~ forcing_power/dissipation dengesine bagli
- Dissipation orani eps ~ nu * enstrophy + nu_t * |S|^2
- Re artinca molecular dissipation azalir, SGS artar
- Uniform forcing'de power input farkli

Actuator disk TKE artisi:
- Disk arkasinda shear kaynak terimi: TKE_wake ~ 0.5 * (Delta_U)^2 * turbulence_intensity
- Near-wake TI ~ 15-25%, far-wake TI ~ 5-10%
- TKE_wake_peak ~ 0.5 * (0.5*U_inf)^2 * 0.20^2 ~ 0.5 * 0.25 * 0.04 = 0.005

Volume-averaged TKE (wake + freestream):
- Wake bolgesi domain'in ~ 20-30%'u
- TKE_avg ~ 0.7 * TKE_freestream + 0.3 * TKE_wake

**TKE tahmini: 0.003 - 0.012**
- Alt sinir: uniform forcing dusuk enerji girdisi, yuksek Re dissipation
- Ust sinir: actuator disk shear'i TKE'yi artirir
- En olasilik: 0.005 - 0.008

### Nu Tahmini
- Ra ~ 0 -> buoyancy kapali -> termal konveksiyon yok
- Uniform forcing sicaklik gradyani olusturmaz
- **Nu ~ 1.0 (molekuler difuzyon)** veya tanimsiz
- theta alaninda onemli bir dinamik beklenmez

Ancak INNATE'in IC'si T_hot=20, T_cold=0:
- Baslangicta termal gradyan var
- Turbulansin mekanik karisimi theta'yi homojenlestirecek
- Baslangicta Nu > 1, sonra azalarak ~1'e yaklasir (uniform T'ye dogru)
- **Nu: 1.0 - 5.0** (gecici rejim, azalan trend)

### Spectral Slope
- Re=100K'de inertial range cok genis
- Grid 96x160x64 -> k_max ~ 32-80 arasi
- Kolmogorov microscale: eta ~ (nu^3/eps)^(1/4)
- eps ~ 2e-4 (referanstan ekstrapolasyon)
- eta ~ ((1e-5)^3 / 2e-4)^(1/4) = (1e-15/2e-4)^0.25 = (5e-12)^0.25 ~ 1.5e-3
- k_eta = 1/eta ~ 670 (grid k_max ~ 48 << k_eta)
- Grid cozunurlugu yetersiz -> LES agirlikli SGS baskisi
- **Slope: -1.6 ile -2.0 arasi**
- Yorum: Genis inertial range ama grid-cutoff etkisi slope'u diklestirebilir
- En olasilik: -1.7 ile -1.9

### theta_rms
- Buoyancy kapali, uniform forcing
- Turbulansin theta'yi karistirmasi: theta_rms once yukselir sonra duser
- Denge durumunda theta_rms ~ kappa * dT / (u_rms * L) benzeri bir scaling
- kappa = 1.41e-5, dT = 20, u_rms ~ 0.3, L ~ 10
- theta_rms ~ 1.41e-5 * 20 / (0.3 * 10) ~ 9.4e-5 (cok kucuk, denge)
- Gecici rejimde daha yuksek: ~ 0.01-0.03
- **theta_rms: 0.005 - 0.03** (gecisi rejime bagli)

### Stabilite
- Re = 100K: egitimden 10x buyuk -> model disi regime
- nu = 1e-5: molecular dissipation cok dusuk
- Actuator disk post-step: ek bir kuvvet -> momentum dengesi bozulabilir
- CFL: max_velocity ~ 0.3-0.5, dt=0.025, dx_min=0.0625
  - CFL ~ 0.5*0.025/0.0625 = 0.2 (makul)
  - Ama Re=100K'de hiz artarsa CFL yukselir
- Periyodik BC + wake recirculation: enerji birikimi riski

**NaN riski: ORTA-YUKSEK**
- Model bu Re'de egitimedigi icin SGS noronlari yetersiz kalabilir
- Actuator disk enerji cekiyor ama periyodik BC ile geri donuyor
- 200-300 adim stabil kalma sansi %50-60
- 500 adim stabil kalma sansi %30-40

### Muhendislik Yorumu
Gercek ruzgar turbini wake'leri:
- Re ~ 1e7 (atmosferik)
- Turbin D ~ 100-150m, hub height ~ 100m
- Wake recovery ~ 10-15D downstream
- TI ~ 10-15% ambient, 25-40% near-wake
- Spectral gap fenomeni (meso-micro gap)

INNATE'in 100K Re'deki temsili:
- Wake deficit fizigi dogru (momentum theory valid)
- Turbulansin anizotropisi kismen dogru (periodic BC limiti)
- Recovery mesafesi periyodiklik nedeniyle anlamsiz
- C_T=0.75 gercekci (modern turbinler 0.7-0.9)
- **SONUC:** Fiziksel mekanizma dogru, kanttatif tahmin %20-30 hata ile mumkun

---

## 5b: BINA ETRAFINDA RUZGAR AKISI (Urban Wind)

### Parametreler
- Re = 50,000 | Ra = 1e8 | Pr = 0.71
- Ri = 1e8 / (50000^2 * 0.71) = 1e8 / (1.775e9) = **0.0563**
- nu = 1/Re = 2e-5
- kappa = 1/(Re*Pr) = 2.82e-5
- Forcing: uniform
- Post-step: Brinkman penalization, building 1x2x1, alpha=10000

### Akis Rejimi
- **Forced convection baskIn** (Ri = 0.056 < 1)
- Buoyancy katkilari var ama ruzgar dominant
- Ri ~ 0.056: egitim referanslarinin araliginda (Re7K: Ri=0.0029, Re10K: Ri=0.0014)
  - Aslinda bu Ri, egitim Ri'larindan 20-40x buyuk!
  - Ra = 1e8 vs egitim Ra = 1e5: 1000x buyuk
- Bluff body akisi: karakteristik ayrisma, recirculation, vortex shedding

### Bluff Body Aerodinamigi

**Recirculation Length (Lr):**
Roshko (1961) ve ESDU korelasyonlari, kare kesitli engel icin:
- Re > 20,000: tamamen turbulansi ayrisma
- Lr/D ~ 1.0-1.5 (turbulansi rejim, kare kesit)
- D = bina genisligi = 1.0 (x-yonu)
- **Lr ~ 1.0 - 1.5** (bina genisligi biriminde)
- Periyodik domain'de: Lx - x_bina_arka = 6 - (3+0.5) = 2.5 >> Lr, yeterli

Ancak INNATE periodic box icin: "bina" Brinkman penalizasyon, gercek no-slip degil.
Brinkman alpha=10000 -> etkin slip length ~ 1/sqrt(alpha) ~ 0.01 << dx
Bu yeterince sert: no-slip benzeri etki.

Fakat periyodik BC: bina tekrari! Aslinda sonsuz bina dizisi (urban canopy).
Bina arasi mesafe: Lx - 1.0 = 5.0 (x-yonu), Lz - 1.0 = 3.0 (z-yonu)
-> Blockage ratio = (1*2*1)/(6*10*4) = 2/240 = 0.83% (dusuk, etkilesim az)

**Wake TKE:**
Bluff body arkasinda turbulansi kinetik enerji:
- TI_wake ~ 0.1 * (U_inf/U_ref)^2 (kare kesit, ESDU)
- Volume-averaged TKE: daha yuksek (bluff body shear layer TKE uretimi)

Referans scaling:
- Re=10K'de TKE = 0.00731 (uniform forcing yok, Kolmogorov)
- Re=50K'de: molecular dissipation 5x azalir
- Uniform forcing etkisi + Brinkman drag -> enerji dengesi degisir

Bluff body'nin TKE artisi:
- Ayrisma noktasinda shear layer: TKE_local ~ 0.5 * (U_inf * 0.3)^2 ~ 0.045 * U_inf^2
- Volume-averaged: bluff body wake ~ domain'in %5-10'u
- **TKE tahmini: 0.005 - 0.015**
- En olasilik: 0.007 - 0.012

### Nu Tahmini

Mixed convection (Ri = 0.056, forced dominant):

**Aicher & Martin korelasyonu (mixed convection, Re > 2000):**
Nu_mixed^3 = Nu_forced^3 + Nu_natural^3

Forced convection (Dittus-Boelter benzeri, kanal):
  Nu_forced = 0.023 * Re^0.8 * Pr^0.4
  = 0.023 * 50000^0.8 * 0.71^0.4
  = 0.023 * 6310 * 0.872
  = 126.5

Ancak bu standart pipe/channel korelasyonu. INNATE periodic box farkli.

INNATE LES scaling'inden:
- Re=7K: Nu=39.3, Re=10K: Nu=70.5
- Nu ~ Re^alpha -> alpha = log(70.5/39.3)/log(10000/7000) = log(1.793)/log(1.429) = 0.584/0.357 = 1.64
- Bu cok dik bir scaling! Muhtemelen Ra=1e5 sabit ve Re degisince Ri degisiyor.
- Daha dogrusu: Nu = f(Re, Ra, Pr) karmasik

Ekstrapolasyon (dikkatli, buyuk belirsizlik):
- Egitim: Kolmogorov forcing, Ra=1e5
- Simdi: Uniform forcing, Ra=1e8

Natural convection (Ra=1e8):
Globe-Dropkin: Nu_nat = 0.069 * Ra^(1/3) * Pr^0.074
  = 0.069 * (1e8)^(1/3) * 0.71^0.074
  = 0.069 * 464.2 * 0.975
  = 31.2

Churchill-Chu (vertical plate, Ra=1e8):
  Nu = [0.825 + 0.387*Ra^(1/6) / (1+(0.492/Pr)^(9/16))^(8/27)]^2
  Ra^(1/6) = (1e8)^(1/6) = 21.54
  [1+(0.492/0.71)^(9/16)]^(8/27) = [1+0.814]^(8/27) = 1.814^(0.296) = 1.193
  Nu = [0.825 + 0.387*21.54/1.193]^2 = [0.825 + 6.99]^2 = [7.81]^2 = 61.0

Forced contribution (INNATE scaling):
- Re=50K, Ra=1e8, uniform forcing
- Kolmogorov->uniform faktor: ~0.7x (daha az lokal shear)
- Re=50K scaling: LES referanstan ekstrapolasyon uygulanmaz (cok uzak)

**Nu tahmini: 30 - 120**
- Alt sinir: natural convection alone (Ra=1e8) ~ 30-65
- Ust sinir: mixed + bluff body enhancement ~ 80-120
- En olasilik: 50 - 90

### Spectral Slope
- Re=50K: genis inertial range
- Bluff body shear layer: enerji injeksiyonu orta-yuksek k'lerde
- Grid cozunurlugu: LES skoru ~ 60-70 (Re=50K, 96x160x64)
- **Slope: -1.6 ile -2.0**
- Bluff body etkisi: daha dik slope mumkun (enerji cascade hizlanir)
- En olasilik: -1.7 ile -1.9

### theta_rms
- Ra=1e8: guclu termal gradyanlar
- Buoyancy-driven plume'lar + ruzgar tasinimi
- Ri=0.056: forced dominant ama termal aktivite var
- LES referans (Re=10K, Ra=1e5): theta_rms = 0.027
- Ra 1000x buyuk -> termal fluktuasyonlar artacak
- **theta_rms: 0.05 - 0.20**
- En olasilik: 0.08 - 0.15

### Stabilite
- Re = 50K: egitimden 5-7x
- Ra = 1e8: egitimden 1000x
- Ri = 0.056: makul (O(0.01) seviye)
- Brinkman penalizasyonu: implicit form (stabil)
- Re-projection sonrasi divergence-free garanti

**NaN riski: ORTA**
- Ra=1e8 agresif ama Ri makul
- Brinkman implicit -> bu taraf stabil
- Asil risk: Re=50K'de SGS yetersizligi
- 200-300 adim stabil kalma sansi %60-70
- 500 adim stabil kalma sansi %40-50

### Muhendislik Yorumu
Gercek sehir ruzgari:
- Re ~ 1e6-1e7 (bina olcegi, atmosferik)
- Recirculation length: 5-7 bina genisligi (kaynaklarda)
  - DIKKAT: Gorevde "5-7 building widths" denilmis, ama turbulansi rejimde
    kare kesit icin Lr/D ~ 1.0-1.5 daha gercekci (Roshko 1961, ESDU).
  - 5-7D degeri muhtemelen dusuk Re laminer-gecis rejimi icin.
  - Turbulansi rejimde (Re>20K) recirculation kisalir.
- Urban heat island: asfalt -> buoyancy (Ra ~ 1e10-1e12 gercekte)
- Pedestrian wind comfort: mean + gust = mean + 3*sigma

INNATE'in 50K Re'deki temsili:
- Brinkman penalizasyon: no-slip yaklasimi makul (alpha=10000 yeterli)
- Periyodik BC: sonsuz bina dizisi (urban canopy array)
- Bu aslinda gercekci bir yaklasim! Urban canopy modelleri boyle calisir.
- Ra=1e8 gercek ABL'den dusuk ama etki gorulecek kadar buyuk
- **SONUC:** En gercekci muhendislik case'i. Urban canopy benzeri fizik.

---

## 5c: DERE/NEHIR AKISI (Environmental Flow)

### Parametreler
- Re = 20,000 | Ra = 1e7 | Pr = 0.71
- Ri = 1e7 / (20000^2 * 0.71) = 1e7 / (2.84e8) = **0.0352**
- nu = 1/Re = 5e-5
- kappa = 1/(Re*Pr) = 7.04e-5
- Forcing: uniform
- Post-step: YOK (standart mixed convection)

### Akis Rejimi
- **Forced convection baskin** (Ri = 0.035 < 1)
- Ri degeri egitim Ri'larindan ~ 10-25x buyuk (Re7K: 0.003, Re10K: 0.001)
- Ama hala Ri << 1: ruzgar/akim dominant
- Unstable stratification (T_hot altta): buoyancy-driven mixing

### Nu Tahmini

**Interpolasyon ve ekstrapolasyon:**

LES referanstan:
- Re=7K, Ra=1e5: Nu=39.3
- Re=10K, Ra=1e5: Nu=70.5

Re=20K, Ra=1e7: iki farklilik var:
1. Re artisi (20K vs 7-10K): Nu artar
2. Ra artisi (1e7 vs 1e5): Buoyancy katkisi artar

Natural convection katkisi (Ra=1e7):
Globe-Dropkin: Nu_nat = 0.069 * (1e7)^(1/3) * 0.71^0.074
  = 0.069 * 215.4 * 0.975 = 14.5

Hollands et al. (bottom-heated, horizontal):
  Nu = 1 + 1.44*(1 - 1708/Ra)^+ + [(Ra/5830)^(1/3) - 1]^+
  Ra=1e7 >> 1708 ve >> 5830:
  Nu ~ 1 + 1.44 + (1e7/5830)^(1/3) - 1 = 1.44 + (1716)^(1/3) - 1 = 1.44 + 12.0 - 1 = 12.4
  (Bu korelasyon Ra ~ 1e7'de accuracy sinirinda)

Forced convection katkisi:
- Re scaling'den: Nu_forced ~ (20K/10K)^1.64 * 70.5 (aggressive extrapolation!)
  = 2^1.64 * 70.5 = 3.12 * 70.5 = 220 (bu cok yuksek, korelasyon bozuldu)
- Daha muhafazakar: Nu ~ Re^0.8 scaling ile
  Nu_forced ~ (20K/10K)^0.8 * 70.5 = 1.74 * 70.5 = 123

Ancak forcing modu degisiyor (Kolmogorov -> uniform).
Uniform forcing'de shear kaynak terimi farkli. Genel olarak ~0.5-0.7x.

**Nu tahmini: 40 - 120**
- Alt sinir: uniform forcing + daha az shear -> ~40
- Ust sinir: Re=20K forced + Ra=1e7 natural -> ~120
- En olasilik: 60 - 100
- Bu case egitim rejimine en yakin -> tahmin en guvenilir

### TKE Tahmini

LES scaling:
- Re=10K: TKE = 0.00731
- Re=20K: TKE ~ Re^0 (dissipation artisi ile dengeli) veya hafif azalma
- Uniform vs Kolmogorov: ~0.7-1.0x
- Ra=1e7 buoyancy: TKE'ye ek kaynak (buoyancy production)

Buoyancy TKE uretimi: P_b = Ri * <v'*theta'>
- Ri = 0.035, <v'*theta'> ~ 0.001 (referanstan)
- P_b ~ 0.035 * 0.001 = 3.5e-5 (TKE referansinin ~%5'i)

**TKE tahmini: 0.005 - 0.012**
- En olasilik: 0.006 - 0.009

### Spectral Slope
- Re=20K: makul inertial range
- Grid cozunurlugu: LES skoru ~ 70 (Re=20K, 96x160x64)
- **Slope: -1.6 ile -1.9**
- En olasilik: -1.7 ile -1.8
- Egitim rejimine yakin -> model en iyi burada yapar

### theta_rms
- Ra=1e7: orta duzey termal aktivite
- LES referans (Re=10K, Ra=1e5): theta_rms = 0.027
- Ra 100x buyuk -> theta_rms ~ (Ra_new/Ra_ref)^(1/3) * theta_rms_ref
  = 100^(1/3) * 0.027 = 4.64 * 0.027 = 0.125 (ust sinir)
- Daha muhafazakar: theta_rms ~ Ra^(1/4) scaling
  = 100^(1/4) * 0.027 = 3.16 * 0.027 = 0.085
- **theta_rms: 0.04 - 0.15**
- En olasilik: 0.06 - 0.10

### Stabilite

**NaN riski: DUSUK**
- Re = 20K: egitimden 2-3x (makul ekstrapolasyon)
- Ra = 1e7: egitimden 100x (ama Ri hala << 1)
- Custom physics YOK: standart INNATE pipeline
- nu = 5e-5: yeterli molecular dissipation
- CFL tahmini: max_v ~ 0.3, dt=0.025, dx=0.0625 -> CFL ~ 0.12 (iyi)

- 500 adim stabil kalma sansi: **%80-90**
- Bu en guvenli case

### Muhendislik Yorumu
Gercek nehir/gol stratifikasyonu:
- Re ~ 1e4-1e6 (nehir olcegi)
- Ra ~ 1e7-1e10 (gunluk isinma dongusu)
- Ri ~ 0.01-1.0 (forced-mixed gecis bolgesi)
- Termal stratifikasyon yikilmasi: sonbahar overturn
- Ekman spiral, Coriolis etkileri (burada yok)

INNATE'in 20K Re'deki temsili:
- Fizik tamamen ayni: mixed convection + stratifikasyon
- Ri = 0.035 gercekci (gunduz, ruzgarli kosul)
- Periyodik BC: sonsuz nehir/kanal yaklasimi (dogru!)
- **SONUC:** En guvenilir case. Egitim rejimine yakin. Kantitatif sonuclar beklenir.

---

## 5d: VERI MERKEZI SOGUTMA (Data Center Cooling)

### Parametreler
- Re = 5,000 | Ra = 1e9 | Pr = 0.71
- Ri = 1e9 / (5000^2 * 0.71) = 1e9 / (1.775e7) = **56.3**
- nu = 1/Re = 2e-4
- kappa = 1/(Re*Pr) = 2.82e-4
- Forcing: uniform
- Post-step: YOK

### Akis Rejimi
- **Buoyancy TAMAMEN baskin** (Ri = 56.3 >> 1)
- Bu INNATE'in hic gormedigi bir rejim!
- Egitim Ri: 0.001-0.003, simdi Ri=56.3 -> **20,000-56,000x buyuk!**
- Re = 5000: egitim rejimine yakin (iyi)
- Ra = 1e9: egitimden 10,000x buyuk (cok kotu)

### Nu Tahmini (Ra-dominant korelasyonlar)

Ra=1e9 >> Ra_crit. Buoyancy-driven turbulansi:

**Globe-Dropkin (enclosure, turbulansi):**
  Nu = 0.069 * Ra^(1/3) * Pr^0.074
  = 0.069 * (1e9)^(1/3) * 0.71^0.074
  = 0.069 * 1000 * 0.975
  = 67.3

**Churchill-Chu (vertical plate, tum Ra):**
  Nu = [0.825 + 0.387*(1e9)^(1/6) / (1+(0.492/0.71)^(9/16))^(8/27)]^2
  (1e9)^(1/6) = 31.62
  [1+0.814]^(8/27) = 1.193
  Nu = [0.825 + 0.387*31.62/1.193]^2 = [0.825 + 10.26]^2 = [11.09]^2 = 122.9

**Mixed convection (Aicher-Martin):**
  Nu^3 = Nu_forced^3 + Nu_natural^3
  Forced (Re=5000): Nu_f ~ 39 (LES referanstan, Re=7K)
    Daha dogrusu Re=5K: Nu_f ~ (5K/7K)^1.64 * 39.3 = 0.576 * 39.3 = 22.6
  Natural (Ra=1e9): Nu_n ~ 67-130

  Nu_mixed = (22.6^3 + 100^3)^(1/3) ~ 100 (natural dominate eder)

Ama INNATE periodic box farkli:
- Bottom-heated Rayleigh-Benard benzeri
- Horizontal layer: Hollands korelasyonu uygun degil (Ra cok buyuk)
- Nu ~ 0.1 * Ra^(2/7) (Niemela et al., turbulansi RB):
  = 0.1 * (1e9)^(2/7) = 0.1 * 1e9^0.286 = 0.1 * 549 = 54.9

**Nu tahmini: 50 - 150**
- Alt sinir: periodic box, bottom-heated, turbulansi RB ~ 55
- Ust sinir: vertical plate + mixed ~ 130
- En olasilik: 60 - 100
- DIKKAT: INNATE bunu dogru hesaplayabilir mi ciddi soru (Ri=56 hic gorulmedi)

### TKE Tahmini (Buoyancy-driven)

Buoyancy-driven TKE:
- Free-fall velocity scaling: U_ff = sqrt(g*beta*dT*H) (boyutsuz: sqrt(Ri))
- U_ff = sqrt(56.3) ~ 7.5 (boyutsuz!) -- bu cok buyuk!
- Ama bu scale hiz, TKE bunun karesine oranli olmaz (normalize edilir)

Rayleigh-Benard TKE scaling:
- TKE ~ (Ra/Ra_crit)^(2/3) * baseline
- Ra_crit ~ 1708 (horizontal layer)
- (1e9/1708)^(2/3) ~ (5.86e5)^(2/3) ~ 6900

Bu anlamsiz buyuklukte. Asil scaling:
- u_rms / (nu/H * Ra^(1/2)) ~ O(0.1) (Grossmann-Lohse theory)
- u_rms ~ 0.1 * (nu * Ra^(1/2)) / H
- nu=2e-4, Ra=1e9, H=Ly=10:
  u_rms ~ 0.1 * 2e-4 * sqrt(1e9) / 10 = 0.1 * 2e-4 * 31623 / 10 = 0.0633
- TKE ~ 0.5 * 3 * u_rms^2 = 1.5 * 0.004 = 0.006

Ama Ri=56.3: momentum denklemindeki buoyancy terimi Ri*theta*e_y:
- Ri*theta ~ 56.3 * 10 (ortalama T ~ 10) = 563 gibi bir ivme!
- Bu fiziksel olarak anlamsiz (boyutsuz formulasyonda)

INNATE'in gercek davranisi:
- Model Ri=56'yi hic gormedi -> buoyancy terimi patlayabilir
- Ya da model theta'yi cok hizli homojenlestirir (theta->0 -> buoyancy kaybeder)

**TKE tahmini: 0.003 - 0.05 (BUYUK BELIRSIZLIK)**
- Eger model stabil kalirsa: 0.005 - 0.02
- Buoyancy-driven turbulansi: TKE genelde daha yuksek
- En olasilik: 0.008 - 0.015 (eger stabil kalirsa)

### Spectral Slope
- Ra=1e9 buoyancy-driven turbulansi
- Bolgert-Obukhov: k^(-5/3) (momentum), k^(-5/3) veya k^(-1) (sicaklik)
- Re=5K: grid LES skoru ~ 75 (makul)
- **Slope: -1.5 ile -2.0**
- Buoyancy-driven akislarda slope daha dik olabilir
- En olasilik: -1.6 ile -1.9

### theta_rms
- Ra=1e9: COKK guclu termal fluktuasyonlar
- Rayleigh-Benard scaling: theta_rms/dT ~ Ra^(-1/7) (Grossmann-Lohse)
  = (1e9)^(-1/7) = 1/(1e9)^(0.143) = 1/26.8 = 0.037
  theta_rms ~ 0.037 * 20 = 0.75 (boyutsuz theta_rms/theta_ref)
- Daha pratik: theta_rms ~ dT * (Nu*kappa/(u_rms*H))^(1/2) benzeri
- **theta_rms: 0.1 - 1.0 (BUYUK BELIRSIZLIK)**
- INNATE'in boyutsuzlastirmasinda: muhtemelen 0.1 - 0.5
- Cok yuksek theta_rms -> buoyancy terimi Ri*theta_rms = 56*0.3 ~ 17 (devasa)

### Stabilite

**NaN riski: COK YUKSEK**
- Ri = 56.3: modelin hic gormedigi rejim
- Ra = 1e9: egitimden 10,000x
- Buoyancy terimi: Ri * theta ~ 56 * O(1) ~ O(56) -> devasa kaynak terimi
- SGS noronlari bu buyuklukteki buoyancy'yi handle edemeyebilir
- dt_base = 0.025: buoyancy terimi icin cok buyuk olabilir
  - Gerekli dt ~ 1/(Ri * U_conv) ~ 1/(56 * 0.1) ~ 0.18 (belki ok?)
  - Ama theta gradyanlari keskinlesirse: dt ihtiyaci drastik azalir

- 50 adim stabil kalma sansi: %40-50
- 200 adim stabil kalma sansi: %20-30
- 500 adim stabil kalma sansi: **%10-20**

Muhtemel basarisizlik senaryosu:
1. Buoyancy terimi buyuk theta gradyani olusturur
2. Theta overshoot -> daha buyuk buoyancy -> pozitif feedback
3. 50-150 adimda NaN

### Muhendislik Yorumu
Gercek data center:
- Rack power: 5-30 kW/rack
- Supply air: 18-24 C, return air: 30-40 C (dT ~ 10-20 C)
- Airflow: 0.5-2 m/s (dusuk Re, ~5000-20000)
- Rayleigh (oda icinde): ~ 1e8-1e10
- Ri tipik: 1-50 (buoyancy dominant, dogru!)

INNATE'in temsili:
- Re=5000 gercekci
- Ra=1e9 gercekci (ama model icin extreme)
- Ri=56 gercekci (ama modelin ogrenme araligi disi)
- Uniform forcing (fan etkisi) dogru yaklasim
- **SONUC:** Fiziksel olarak en gercekci parametreler ama model icin en zor.
  NaN beklenir. Stabil kalirsa buyuk basari.

---

## 5e: ATMOSFERIK SINIR TABAKASI (ABL) -- EXTREME

### Parametreler
- Re = 500,000 | Ra = 1e10 | Pr = 0.71
- Ri = 1e10 / (500000^2 * 0.71) = 1e10 / (1.775e11) = **0.0563**
- nu = 1/Re = 2e-6
- kappa = 1/(Re*Pr) = 2.82e-6
- Forcing: uniform
- Post-step: YOK

### Akis Rejimi
- **Forced convection baskin** (Ri = 0.056 < 1, ayni urban wind!)
- Ilginc: Ri = 0.056 = Re=50K/Ra=1e8 ile AYNI!
  (Cunku Ra/Re^2 oranli ve her ikisi de 10x artmis)
- Re = 500K: egitimden **50-70x** buyuk
- Ra = 1e10: egitimden **100,000x** buyuk
- nu = 2e-6: molecular dissipation neredeyse SIFIR

### ABL Benzerligi Analizi

Gercek ABL parametreleri:
- ABL yuksekligi: H ~ 1-2 km
- Geostrophic ruzgar: U_g ~ 10-15 m/s
- Yer yuzeyi sicakligi: dT ~ 5-15 K
- Kinematic viscosity: nu_air = 1.5e-5 m^2/s
- Re_ABL = U_g * H / nu ~ 10*1000/1.5e-5 = 6.7e8
- Ra_ABL ~ g*beta*dT*H^3/(nu*kappa) ~ 10*3.3e-3*10*1e9/(1.5e-5*2.1e-5) ~ 1e15

INNATE vs Gercek ABL:
- Re: 500K vs 6.7e8 -> INNATE 1000x dusuk (ama yine de cok buyuk)
- Ra: 1e10 vs 1e15 -> INNATE 100,000x dusuk
- Ri: 0.056 vs ~0.01-1 (gercekci aralik!) -> INNATE **gercekci Ri!**
- Pr: 0.71 = 0.71 -> DOGRU

### Nu Tahmini

Bu extreme Re/Ra icin:

**Monin-Obukhov Similarity Theory (MOST):**
ABL'de Nusselt:
  Nu ~ (kappa_T / kappa) * (u_*/kappa_von) * L_MO

Ama periodic box'ta MOST direkt uygulanmaz (yer yuzeyi yok).

Standard korelasyonlardan:
- Globe-Dropkin (Ra=1e10):
  Nu = 0.069 * (1e10)^(1/3) * 0.71^0.074
  = 0.069 * 2154 * 0.975 = 144.8

- Niemela et al. (turbulansi RB, Ra=1e10):
  Nu ~ 0.1 * Ra^(2/7) = 0.1 * (1e10)^(0.286) = 0.1 * 1326 = 132.6

Mixed (Ri=0.056, forced dominant):
  Forced katkisi Re=500K'de cok buyuk olacak
  Nu_forced ~ Re^0.8 * Pr^0.4 * C (cok buyuk sayi)
  = 0.023 * 500000^0.8 * 0.71^0.4 = 0.023 * 36240 * 0.872 = 727

**Nu tahmini: 100 - 500 (DEVASA BELIRSIZLIK)**
- Alt sinir: natural convection ~ 130-145
- Ust sinir: mixed + forced ~ 500+
- Ama INNATE bunu hesaplayamayabilir (model limiti)
- Model stabil kalirsa gercek Nu: muhtemelen 100-300

### TKE Tahmini

Log-law profili mumkun mu?
- Periyodik domain'de duvar yok -> klasik log-law gecerli degil
- Ama uniform forcing + dissipation dengesi bir boundary layer benzeri akis olusturabilir
- Re=500K'de inertial subrange cok genis: k^(-5/3) beklenir

TKE scaling:
- u_rms ~ forcing_amplitude * Re^(alpha) benzeri bir iliski
- Ama Re=500K'de molecular dissipation ~ 0 -> tum dissipation SGS'de
- SGS noronlari bu Re'yi gormedigi icin: ya cok az dissipe eder (enerji birikir -> NaN)
  ya da asiri dissipe eder (TKE -> 0)

INNATE'in muhtemel davranisi:
- EddyViscosity3D'nin ogrendigi nu_t ~ (Cs*Delta)^2*|S|
- Cs ogrenmeden farkli Re'de: Cs ~ 0.1-0.2 sabit kalirsa nu_t yetebilir
- nu_t ~ (0.17 * 0.0625)^2 * |S| ~ 1.13e-4 * |S|
- Re=500K'de |S| artacak -> nu_t artacak -> SGS kendi kendini ayarlayabilir

**TKE tahmini: 0.003 - 0.020 (eger stabil kalirsa)**
- En olasilik: 0.005 - 0.012
- Smagorinsky benzeri SGS: Re-independent TKE mumkun (nu_t ~ Re^0 old SGS'de)

### Spectral Slope
- Re=500K: muazzam inertial range (gercek turbulansi!)
- Grid 96x160x64 -> k_max ~ 32-80
- Kolmogorov scale: eta ~ (nu^3/eps)^(1/4) ~ ((2e-6)^3/1e-4)^(1/4) ~ 2.7e-4
- k_eta ~ 3700 >> k_max (grid COKK kaba)
- Tum grid inertial range'de: k^(-5/3) beklenir (eger model dogru SGS yaparsa)
- **Slope: -1.5 ile -2.0**
- En olasilik: -1.65 ile -1.85
- Gercek -5/3 = -1.667'ye en yakin slope bu case'den gelebilir
  (cunku tum grid inertial range'de ve dissipation tamamen SGS'de)

### theta_rms
- Ra=1e10: cok guclu termal aktivite
- Grossmann-Lohse: theta_rms/dT ~ Ra^(-1/7) = (1e10)^(-0.143) = 1/38.3 = 0.026
  theta_rms ~ 0.026 * 20 = 0.52
- Ama Ri=0.056: forced dominant -> termal homojenlesme daha hizli
- **theta_rms: 0.05 - 0.50**
- En olasilik: 0.1 - 0.3

### Stabilite

**NaN riski: COK YUKSEK**
- Re = 500K: egitimden 50-70x (extreme)
- Ra = 1e10: egitimden 100,000x (extreme)
- nu = 2e-6: molecular dissipation neredeyse sifir
- SGS noronlari: bu Re'de yeterli dissipation saglayabilir mi?
- CFL: dt=0.025, dx=0.0625, max_v ~ 0.5 (tahmin)
  CFL ~ 0.5 * 0.025 / 0.0625 = 0.2 (ok gorünuyor)
  AMA: Re=500K'de hizlar cok artarsa CFL patlayabilir

- 50 adim stabil kalma sansi: %30-40
- 200 adim stabil kalma sansi: %15-25
- 500 adim stabil kalma sansi: **%5-15**

Muhtemel basarisizlik senaryolari:
1. SGS yetersiz -> enerji birikmesi -> velocity patlama -> NaN (10-50 adim)
2. Termal gradyan keskinlesmesi -> theta NaN (50-200 adim)
3. CFL limit asimi -> numerik kararsizlik (herhangi bir adimda)

Demo kodundaki beklenti ile uyumlu:
- 50 adim stabil: KOTU (ben: %30-40 sans)
- 200 adim stabil: MAKUL (ben: %15-25 sans)
- 500 adim stabil: IYI, beklenmiyor (ben: %5-15 sans)

### Muhendislik Yorumu
Gercek ABL:
- Konvektif BL (gunduz): zi ~ 1-2 km, w* ~ 1-2 m/s
- Stabil BL (gece): zi ~ 100-300 m, turbulansi zayif
- Surface layer: z < 0.1*zi, logaritmik profil
- Ekman spiral: Coriolis + friction balance
- ABL modelleri: MOST + LES (tipik Re_eff ~ 1e6-1e8)

INNATE'in temsili:
- Re=500K << Re_ABL (1000x dusuk) ama yine de cok buyuk
- Ri=0.056 GERCEKCI (near-neutral ABL)
- Periyodik BC: homogeneous horizontal (yatay homojen ABL yaklasimi, dogru!)
- Coriolis yok, yer yuzeyi pururuzlugu yok
- Bu daha cok "sonsuz yatay domain'de turbulansi karisim" testi
- **SONUC:** Extreme stres testi. Model patlarsa beklenen. Stabil kalirsa etkileyici
  ama kantitatif guvenirliligi dusuk.

---

## OZET TABLOSU

| Case | Re | Ra | Ri | Rejim | Nu (aralik) | TKE (aralik) | Slope | theta_rms | NaN Riski |
|------|-----|-----|-----|-------|-------------|--------------|-------|-----------|-----------|
| 5a Wind Turbine | 100K | ~0 | ~0 | Forced | 1-5 | 0.003-0.012 | -1.7/-2.0 | 0.005-0.03 | ORTA-YUKSEK |
| 5b Urban Wind | 50K | 1e8 | 0.056 | Forced (mixed) | 30-120 | 0.005-0.015 | -1.6/-2.0 | 0.05-0.20 | ORTA |
| 5c River Flow | 20K | 1e7 | 0.035 | Forced (mixed) | 40-120 | 0.005-0.012 | -1.6/-1.9 | 0.04-0.15 | DUSUK |
| 5d Data Center | 5K | 1e9 | 56.3 | Buoyancy | 50-150 | 0.003-0.05 | -1.5/-2.0 | 0.1-1.0 | COK YUKSEK |
| 5e ABL | 500K | 1e10 | 0.056 | Forced (extreme) | 100-500 | 0.003-0.020 | -1.5/-2.0 | 0.05-0.50 | COK YUKSEK |

### Stabil Kalma Olasiliklari (500 adim)

| Case | %50 adim | %200 adim | %500 adim |
|------|----------|-----------|-----------|
| 5a Wind Turbine | 70% | 50% | 35% |
| 5b Urban Wind | 80% | 60% | 45% |
| 5c River Flow | 95% | 90% | 85% |
| 5d Data Center | 45% | 25% | 15% |
| 5e ABL | 35% | 20% | 10% |

### Beklenen Siralama (En basarilidan en zora)

1. **5c River Flow** (DUSUK risk) -- Egitim rejimine en yakin. En guvenilir sonuclar.
2. **5b Urban Wind** (ORTA risk) -- Brinkman stabil. Ra=1e8 agresif ama Ri makul.
3. **5a Wind Turbine** (ORTA-YUKSEK) -- Re=100K agresif. Actuator disk ek belirsizlik.
4. **5e ABL** (COK YUKSEK) -- Re=500K extreme. SGS kritik.
5. **5d Data Center** (COK YUKSEK) -- Ri=56.3 modelin hic gormedigi rejim.

Not: 5d ve 5e siralama tartismali. 5d'de Re egitim icinde AMA Ri extreme.
5e'de Ri makul AMA Re extreme. Ikisi de buyuk olasilikla NaN verecek.
Farki: 5d fiziksel olarak daha zor (buoyancy dominant, tamamiyla farkli rejim),
5e'de ise Ri ayni ama sadece Re olcek sorunu (SGS yeterliligi).

---

## METODOLOJI NOTLARI

### Kullanilan Korelasyonlar
1. **Actuator Disk:** Betz momentum theory (Manwell et al. 2009)
2. **Jensen Wake:** Jensen (1983), k=0.075 offshore default
3. **Bluff Body:** Roshko (1961), ESDU 80025
4. **Globe-Dropkin:** Globe & Dropkin (1959), enclosure natural convection
5. **Churchill-Chu:** Churchill & Chu (1975), vertical plate
6. **Hollands:** Hollands et al. (1976), horizontal bottom-heated
7. **Niemela:** Niemela et al. (2000), high-Ra RB convection
8. **Grossmann-Lohse:** Grossmann & Lohse (2000, 2001), RB scaling theory
9. **Monin-Obukhov:** Monin & Obukhov (1954), ABL similarity
10. **Aicher-Martin:** Aicher & Martin (1997), mixed convection

### Belirsizlik Kaynaklari
1. **Forcing modu degisimi** (Kolmogorov -> uniform): Kantitatif etki bilinmiyor
2. **Egitim disi ekstrapolasyon**: Model davranisi tahmin edilemez
3. **Periyodik BC**: Gercek geometri yok, wake/recirculation periyodik tekrar
4. **Brinkman yaklasimi**: Gercek no-slip degil (ama alpha=10000 yeterli)
5. **Boussinesq siniri**: Ra=1e9-1e10'da Boussinesq bozulabilir (dT/T_ref ~ 7%)
6. **Grid cozunurlugu**: Re > 50K'de LES skoru < 50 (yetersiz)
7. **SGS model transferi**: Ogrenilmis SGS parametreleri farkli Re'de gecerli mi?

### Kalibrasyon Guclu/Zayif Yonleri
- **Guclu:** LES referans 2 noktada mevcut (Re=7K, Re=10K)
- **Guclu:** Grid ve domain ayni -> numerik artefaktlar tutarli
- **Zayif:** Referans Kolmogorov forcing, demo'lar uniform
- **Zayif:** Referans Ra=1e5, demo'lar 1e7-1e10 (1000-100000x buyuk)
- **Zayif:** Sadece 2 kalibrasyon noktasi, ekstrapolasyon cok uzak

---

## SONUC

Bu 5 demo case'i INNATE'in muhendislik uygulamalarindaki potansiyelini ve sinirlarini test ediyor.
Kritik soru: **SGS noronlarinin (EddyViscosity3D) egitim rejimi disinda ne kadar genelledigi.**

- 5c (River) basarili olmali (egitim icinde)
- 5b (Urban) Brinkman ile ilginc sonuclar verebilir
- 5a (Wind) actuator disk fizigi dogru ama Re zorlu
- 5d ve 5e buyuk olasilikla NaN verecek (Ri>>1 veya Re>>Re_train)

En degerli bilgi: NaN adimi ve son stabil metrikler.
Model 100 adim bile stabil kalirsa, o anki metrikler fiziksel anlam tasir.
