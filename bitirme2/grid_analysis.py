#!/usr/bin/env python3
"""
Grid / Mesh Bagimsizlik Analizi
===============================
Kolmogorov akisi + Rayleigh-Benard konveksiyonu icin
DNS ve LES cozunurluk gereksinimlerini hesaplar.

Domain: Lx=6, Ly=10, Lz=4 (boyutsuz)
Forcing: F = A * sin(k_f * 2*pi*y/Ly), A=1.0, k_f=1
Pr = 0.71

Sadece math modulu kullanir, hicbir sey simule etmez.
Calistirmak icin: python3 grid_analysis.py
"""

import math


# ============================================================================
# SABITLER
# ============================================================================

Lx = 6.0    # boyutsuz domain boyutlari
Ly = 10.0
Lz = 4.0

A = 1.0     # forcing genlik
k_f = 1     # forcing dalga sayisi
Pr = 0.71   # Prandtl sayisi

N_FIELDS = 9  # u, v, w, p, omega_x, omega_y, omega_z, t(zaman?), theta(sicaklik)
BYTES_PER_FLOAT = 4  # float32

RE_VALUES = [100, 500, 1000, 2000, 3000, 5000, 7000, 10000]

# Bizim grid'ler
OUR_GRIDS = {
    "kucuk": (32, 48, 24),
    "buyuk": (96, 160, 64),
}


# ============================================================================
# YARDIMCI FONKSIYONLAR
# ============================================================================

def compute_dissipation_rate():
    """
    Self-consistent epsilon hesabi.

    Turbulanslı denge:
        epsilon = P_forcing
        P_forcing ~ A * U_rms / 2
        U_rms ~ (epsilon * L_f)^(1/3)   (Kolmogorov olceklenmesi)

    L_f = Ly / k_f = 10 (forcing uzunluk olcegi)

    Iki denklemi birlestir:
        epsilon = A * (epsilon * L_f)^(1/3) / 2
        epsilon = A / 2 * epsilon^(1/3) * L_f^(1/3)
        epsilon^(2/3) = A * L_f^(1/3) / 2
        epsilon = (A * L_f^(1/3) / 2)^(3/2)
    """
    L_f = Ly / k_f  # = 10.0
    eps = (A * L_f ** (1.0 / 3.0) / 2.0) ** (3.0 / 2.0)
    return eps


def compute_urms(eps, L_f):
    """U_rms ~ (epsilon * L_f)^(1/3)"""
    return (eps * L_f) ** (1.0 / 3.0)


def compute_kolmogorov_scale(nu, eps):
    """eta = (nu^3 / epsilon)^(1/4)"""
    return (nu ** 3 / eps) ** 0.25


def compute_taylor_microscale(nu, E_kin, eps):
    """
    Taylor microscale: lambda = sqrt(10 * nu * E / epsilon)
    E = U_rms^2 / 2 (kinetik enerji yogunlugu)
    """
    return math.sqrt(10.0 * nu * E_kin / eps)


def grid_for_spacing(dx):
    """Verilen dx icin gerekli grid noktasi sayisi (her yonde)."""
    nx = math.ceil(Lx / dx)
    ny = math.ceil(Ly / dx)
    nz = math.ceil(Lz / dx)
    return nx, ny, nz


def total_points(nx, ny, nz):
    return nx * ny * nz


def memory_estimate_mb(n_total):
    """Bellek tahmini MB (float32, N_FIELDS alan)."""
    return n_total * N_FIELDS * BYTES_PER_FLOAT / (1024 ** 2)


def time_per_step_s(n_total):
    """
    FFT-bazli pseudo-spektral cozucu icin tahmini sure/step.
    ~ N * log2(N) * 1e-9 saniye (tek GPU, optimistik)
    Gercekte 3D FFT = 3 * N * log2(N) islem, ama sabit faktor belirsiz.
    """
    if n_total <= 0:
        return 0.0
    return n_total * math.log2(n_total) * 1e-9


def spacing_from_grid(nx, ny, nz):
    """Grid'den efektif dx hesapla (her yonde ayri, en kucugunu al)."""
    dx = Lx / nx
    dy = Ly / ny
    dz = Lz / nz
    return dx, dy, dz


# ============================================================================
# ANA HESAPLAMA
# ============================================================================

def analyze_re(Re):
    """Bir Re degeri icin tum turetilmis miktarlari hesapla."""
    nu = 1.0 / Re
    eps = compute_dissipation_rate()
    L_f = Ly / k_f
    U_rms = compute_urms(eps, L_f)
    E_kin = U_rms ** 2 / 2.0

    # Turbulanslı olcekler
    eta = compute_kolmogorov_scale(nu, eps)
    lam = compute_taylor_microscale(nu, E_kin, eps)

    # Re_lambda (Taylor Reynolds sayisi)
    Re_lambda = U_rms * lam / nu

    # Integral olcek tahmini
    L_int = U_rms ** 3 / eps  # L ~ u'^3 / epsilon

    # DNS grid: dx = pi * eta (Pope kriteri)
    dx_dns = math.pi * eta
    nx_dns, ny_dns, nz_dns = grid_for_spacing(dx_dns)
    n_dns = total_points(nx_dns, ny_dns, nz_dns)

    # LES grid: dx = lambda (Taylor olcegi)
    dx_les = lam
    nx_les, ny_les, nz_les = grid_for_spacing(dx_les)
    n_les = total_points(nx_les, ny_les, nz_les)

    return {
        "Re": Re,
        "nu": nu,
        "eps": eps,
        "U_rms": U_rms,
        "E_kin": E_kin,
        "eta": eta,
        "lambda": lam,
        "Re_lambda": Re_lambda,
        "L_int": L_int,
        # DNS
        "dx_dns": dx_dns,
        "dns_grid": (nx_dns, ny_dns, nz_dns),
        "n_dns": n_dns,
        "mem_dns_mb": memory_estimate_mb(n_dns),
        "t_dns_s": time_per_step_s(n_dns),
        # LES
        "dx_les": dx_les,
        "les_grid": (nx_les, ny_les, nz_les),
        "n_les": n_les,
        "mem_les_mb": memory_estimate_mb(n_les),
        "t_les_s": time_per_step_s(n_les),
    }


def find_equivalent_re(nx, ny, nz, results):
    """
    Verilen grid hangi Re degerine en yakin?
    En kucuk dx'i (en ince cozunurluk) baz alir.
    DNS ve LES olarak ayri ayri karsilastirir.
    """
    dx, dy, dz = spacing_from_grid(nx, ny, nz)
    dx_eff = min(dx, dy, dz)  # en ince cozunurluk

    # DNS: dx_dns = pi * eta, yani eta_eff = dx_eff / pi
    # LES: dx_les = lambda

    best_dns_re = None
    best_dns_err = float("inf")
    best_les_re = None
    best_les_err = float("inf")

    for r in results:
        # DNS karsilastirmasi
        err_dns = abs(dx_eff - r["dx_dns"]) / r["dx_dns"]
        if err_dns < best_dns_err:
            best_dns_err = err_dns
            best_dns_re = r["Re"]

        # LES karsilastirmasi
        err_les = abs(dx_eff - r["dx_les"]) / r["dx_les"]
        if err_les < best_les_err:
            best_les_err = err_les
            best_les_re = r["Re"]

    return {
        "grid": (nx, ny, nz),
        "dx_eff": dx_eff,
        "dns_equiv_Re": best_dns_re,
        "dns_match_err": best_dns_err,
        "les_equiv_Re": best_les_re,
        "les_match_err": best_les_err,
    }


def accuracy_estimate(nx, ny, nz, r):
    """
    Verilen grid ve Re icin dogruluk tahmini (0-100).

    Kriter:
    - DNS dogrulugu: dx_eff / (pi*eta).  Eger <= 1 ise %100, ustunde azalir.
    - LES dogrulugu: dx_eff / lambda.     Eger <= 1 ise %80 (LES tavanı), ustunde azalir.
    - Ek penalti: anizotropi (dx/dy/dz oranları 1'den uzaksa)

    Sonuc: "bu grid bu Re'de ne kadar guvenilir" sorusuna kaba bir cevap.
    """
    dx, dy, dz = spacing_from_grid(nx, ny, nz)
    dx_eff = min(dx, dy, dz)
    dx_max = max(dx, dy, dz)

    # Anizotropi penaltisi: ideal = 1.0 (izotropik)
    aniso_ratio = dx_max / dx_eff
    aniso_penalty = max(0, (aniso_ratio - 1.0) * 5.0)  # %5 per unit aniso

    # DNS skoru
    ratio_dns = dx_eff / (math.pi * r["eta"])
    if ratio_dns <= 1.0:
        dns_score = 100.0
    elif ratio_dns <= 2.0:
        dns_score = 100.0 - (ratio_dns - 1.0) * 30.0  # 70-100 arasi
    elif ratio_dns <= 5.0:
        dns_score = 70.0 - (ratio_dns - 2.0) * 10.0   # 40-70 arasi
    else:
        dns_score = max(0, 40.0 - (ratio_dns - 5.0) * 4.0)

    # LES skoru (tavan %80 cunku LES zaten model hatasi tasiyor)
    ratio_les = dx_eff / r["lambda"]
    if ratio_les <= 1.0:
        les_score = 80.0
    elif ratio_les <= 2.0:
        les_score = 80.0 - (ratio_les - 1.0) * 20.0
    elif ratio_les <= 5.0:
        les_score = 60.0 - (ratio_les - 2.0) * 10.0
    else:
        les_score = max(0, 30.0 - (ratio_les - 5.0) * 3.0)

    # En iyi skoru sec (DNS veya LES modundayiz)
    best = max(dns_score, les_score)
    final = max(0, best - aniso_penalty)
    return final, dns_score, les_score, ratio_dns, ratio_les, aniso_ratio


def sweet_spot_search(results):
    """
    Monte Carlo benzeri sweet spot arama.
    Farkli grid boyutlarini tara, cost/accuracy trade-off'u goster.
    Cost = total_points * N_FIELDS (bellek + islem maliyeti)
    """
    # Taranacak grid boyutlari (her eksen icin)
    candidates_x = [16, 24, 32, 48, 64, 96, 128, 192, 256]
    candidates_y = [24, 32, 48, 64, 96, 128, 160, 192, 256, 320]
    candidates_z = [16, 24, 32, 48, 64, 96, 128, 192]

    # Aspect ratio'yu koru: Nx/Ny/Nz ~ Lx/Ly/Lz = 6/10/4 = 3/5/2
    # Yani Ny ~ (5/3)*Nx, Nz ~ (2/3)*Nx
    # Sadece makul aspect ratio'lu grid'leri dene
    target_Re = 5000  # birincil hedef Re
    r_target = None
    for r in results:
        if r["Re"] == target_Re:
            r_target = r
            break
    if r_target is None:
        r_target = results[-1]

    sweet_spots = []
    base_sizes = [16, 24, 32, 48, 64, 80, 96, 128, 160, 192, 256]

    for base in base_sizes:
        # Aspect ratio'ya gore grid boyutlari
        nx = base
        ny = max(1, round(base * Ly / Lx))   # *10/6 = *5/3
        nz = max(1, round(base * Lz / Lx))   # *4/6 = *2/3

        n_total = total_points(nx, ny, nz)
        cost = n_total * N_FIELDS  # toplam float sayisi
        mem_mb = memory_estimate_mb(n_total)
        t_step = time_per_step_s(n_total)

        acc, dns_sc, les_sc, r_dns, r_les, aniso = accuracy_estimate(
            nx, ny, nz, r_target
        )

        sweet_spots.append({
            "grid": (nx, ny, nz),
            "n_total": n_total,
            "mem_mb": mem_mb,
            "t_step_ms": t_step * 1000,
            "accuracy": acc,
            "dns_score": dns_sc,
            "les_score": les_sc,
            "ratio_dns": r_dns,
            "ratio_les": r_les,
            "aniso": aniso,
            "efficiency": acc / max(1, mem_mb),  # accuracy per MB
        })

    return sweet_spots


# ============================================================================
# GORSEL CIKTI
# ============================================================================

def print_separator(char="=", width=120):
    print(char * width)


def print_header(title):
    print()
    print_separator()
    print(f"  {title}")
    print_separator()
    print()


def print_results_table(results):
    """Ana sonuc tablosu."""
    print_header("MESH BAGIMSIZLIK ANALIZI - KOLMOGOROV AKISI + RAYLEIGH-BENARD")

    print(f"  Domain: Lx={Lx}, Ly={Ly}, Lz={Lz} (boyutsuz)")
    print(f"  Forcing: F = {A} * sin({k_f} * 2*pi*y/{Ly})")
    print(f"  Pr = {Pr}")
    print()

    # Dissipation (Re'den bagimsiz)
    eps = compute_dissipation_rate()
    L_f = Ly / k_f
    U_rms = compute_urms(eps, L_f)
    print(f"  Self-consistent cozum:")
    print(f"    L_f = Ly/k_f = {L_f:.1f}")
    print(f"    epsilon = (A * L_f^(1/3) / 2)^(3/2) = {eps:.6f}")
    print(f"    U_rms = (epsilon * L_f)^(1/3) = {U_rms:.6f}")
    print()

    # Tablo basligi
    hdr = (
        f"{'Re':>6s}  {'nu':>10s}  {'eta':>10s}  {'lambda':>10s}  "
        f"{'Re_lam':>7s}  {'L_int':>7s}  "
        f"{'DNS grid':>16s}  {'DNS pts':>12s}  {'DNS MB':>8s}  {'DNS ms/step':>11s}  "
        f"{'LES grid':>16s}  {'LES pts':>12s}  {'LES MB':>8s}  {'LES ms/step':>11s}"
    )
    print(hdr)
    print("-" * len(hdr))

    for r in results:
        dns_g = "{}x{}x{}".format(*r["dns_grid"])
        les_g = "{}x{}x{}".format(*r["les_grid"])
        line = (
            f"{r['Re']:6d}  {r['nu']:10.2e}  {r['eta']:10.6f}  {r['lambda']:10.6f}  "
            f"{r['Re_lambda']:7.1f}  {r['L_int']:7.3f}  "
            f"{dns_g:>16s}  {r['n_dns']:12,d}  {r['mem_dns_mb']:8.1f}  {r['t_dns_s']*1000:11.3f}  "
            f"{les_g:>16s}  {r['n_les']:12,d}  {r['mem_les_mb']:8.1f}  {r['t_les_s']*1000:11.3f}"
        )
        print(line)
    print()


def print_grid_comparison(results):
    """Bizim grid'leri karsilastir."""
    print_header("BIZIM GRID'LER - HANGI Re'YE KARSILIK GELIYOR?")

    for name, (nx, ny, nz) in OUR_GRIDS.items():
        eq = find_equivalent_re(nx, ny, nz, results)
        dx, dy, dz = spacing_from_grid(nx, ny, nz)

        print(f"  Grid: {name} = {nx}x{ny}x{nz}")
        print(f"    dx={dx:.4f}, dy={dy:.4f}, dz={dz:.4f}")
        print(f"    dx_eff (min) = {eq['dx_eff']:.6f}")
        print()

        # Her Re icin skor goster
        print(f"    {'Re':>6s}  {'dx_dns':>10s}  {'dx_les':>10s}  "
              f"{'dx/dx_dns':>10s}  {'dx/dx_les':>10s}  "
              f"{'DNS skor':>10s}  {'LES skor':>10s}  {'Genel':>10s}")
        print(f"    {'-'*84}")

        for r in results:
            acc, dns_sc, les_sc, r_dns, r_les, aniso = accuracy_estimate(
                nx, ny, nz, r
            )
            print(
                f"    {r['Re']:6d}  {r['dx_dns']:10.6f}  {r['dx_les']:10.6f}  "
                f"{r_dns:10.2f}  {r_les:10.2f}  "
                f"{dns_sc:10.1f}  {les_sc:10.1f}  {acc:10.1f}"
            )

        print()
        print(f"    >> DNS olarak en yakin: Re = {eq['dns_equiv_Re']} "
              f"(hata: {eq['dns_match_err']*100:.1f}%)")
        print(f"    >> LES olarak en yakin: Re = {eq['les_equiv_Re']} "
              f"(hata: {eq['les_match_err']*100:.1f}%)")
        print()
        print_separator("-", 90)
        print()


def print_sweet_spot(sweet_spots):
    """Sweet spot tablosu."""
    print_header("SWEET SPOT ANALIZI (Re=5000 icin)")

    print(f"  {'Grid':>16s}  {'Toplam':>12s}  {'Bellek':>8s}  "
          f"{'ms/step':>8s}  {'DNS/eta':>8s}  {'LES/lam':>8s}  "
          f"{'Aniso':>6s}  {'DNS':>6s}  {'LES':>6s}  {'Skor':>6s}  "
          f"{'Verim':>8s}")
    print(f"  {'-'*110}")

    for s in sweet_spots:
        g = "{}x{}x{}".format(*s["grid"])
        marker = ""
        # En verimli noktayi isaretle
        if s["accuracy"] >= 60 and s["mem_mb"] < 500:
            marker = " <<<"
        print(
            f"  {g:>16s}  {s['n_total']:12,d}  {s['mem_mb']:7.1f}M  "
            f"{s['t_step_ms']:8.3f}  {s['ratio_dns']:8.2f}  {s['ratio_les']:8.2f}  "
            f"{s['aniso']:6.2f}  {s['dns_score']:6.1f}  {s['les_score']:6.1f}  "
            f"{s['accuracy']:6.1f}  {s['efficiency']:8.3f}{marker}"
        )
    print()

    # En iyi adaylar
    viable = [s for s in sweet_spots if s["accuracy"] >= 40]
    if viable:
        best_eff = max(viable, key=lambda s: s["efficiency"])
        best_acc = max(viable, key=lambda s: s["accuracy"])
        print(f"  En verimli (accuracy/MB): {best_eff['grid']} "
              f"(skor={best_eff['accuracy']:.1f}, "
              f"mem={best_eff['mem_mb']:.1f}MB)")
        print(f"  En dogruluklu:            {best_acc['grid']} "
              f"(skor={best_acc['accuracy']:.1f}, "
              f"mem={best_acc['mem_mb']:.1f}MB)")
    print()


def print_summary(results):
    """Turkce sonuc ozeti."""
    print_header("SONUC OZETI")

    eps = compute_dissipation_rate()
    L_f = Ly / k_f
    U_rms = compute_urms(eps, L_f)

    print("  1. FIZIKSEL OLCEKLER")
    print(f"     Dissipation rate epsilon = {eps:.6f} (Re'den bagimsiz, forcing'e bagli)")
    print(f"     RMS hiz U_rms = {U_rms:.4f}")
    print()

    r100 = results[0]   # Re=100
    r5000 = None
    r10000 = results[-1]
    for r in results:
        if r["Re"] == 5000:
            r5000 = r

    print("  2. KOLMOGOROV OLCEGI (eta)")
    print(f"     Re=100:   eta = {r100['eta']:.6f}   (DNS grid: {'x'.join(map(str,r100['dns_grid']))})")
    if r5000:
        print(f"     Re=5000:  eta = {r5000['eta']:.6f}  (DNS grid: {'x'.join(map(str,r5000['dns_grid']))})")
    print(f"     Re=10000: eta = {r10000['eta']:.6f}  (DNS grid: {'x'.join(map(str,r10000['dns_grid']))})")
    print()

    print("  3. BIZIM GRID'LER")
    for name, (nx, ny, nz) in OUR_GRIDS.items():
        dx, dy, dz = spacing_from_grid(nx, ny, nz)
        dx_eff = min(dx, dy, dz)

        if r5000:
            ratio_dns = dx_eff / (math.pi * r5000["eta"])
            ratio_les = dx_eff / r5000["lambda"]
            acc, _, _, _, _, _ = accuracy_estimate(nx, ny, nz, r5000)

            if ratio_dns <= 1.0:
                dns_verdict = "DNS YETERLI"
            elif ratio_dns <= 2.0:
                dns_verdict = "DNS SINIRDA"
            elif ratio_dns <= 5.0:
                dns_verdict = "DNS YETERSIZ, LES uygun"
            else:
                dns_verdict = "DNS ve LES YETERSIZ"

            if ratio_les <= 1.0:
                les_verdict = "LES YETERLI"
            elif ratio_les <= 2.0:
                les_verdict = "LES SINIRDA"
            else:
                les_verdict = "LES YETERSIZ"

            print(f"     {name:6s} ({nx}x{ny}x{nz}):  "
                  f"dx_eff={dx_eff:.4f}  "
                  f"dx/eta={ratio_dns:.1f}x  "
                  f"dx/lambda={ratio_les:.1f}x  "
                  f"skor={acc:.0f}  "
                  f">> {dns_verdict}, {les_verdict}")

    print()
    print("  4. ONERILER")
    print()

    # Buyuk grid hangi Re'lere uygun?
    kucuk_nx, kucuk_ny, kucuk_nz = OUR_GRIDS["kucuk"]
    buyuk_nx, buyuk_ny, buyuk_nz = OUR_GRIDS["buyuk"]

    print("     KUCUK GRID (32x48x24):")
    for r in results:
        acc, dns_sc, les_sc, r_dns, r_les, aniso = accuracy_estimate(
            kucuk_nx, kucuk_ny, kucuk_nz, r
        )
        if acc >= 70:
            print(f"       Re={r['Re']:>5d}: DNS olarak KULLANILABILIR (skor={acc:.0f})")
        elif acc >= 50:
            print(f"       Re={r['Re']:>5d}: LES olarak KULLANILABILIR (skor={acc:.0f})")

    print()
    print("     BUYUK GRID (96x160x64):")
    for r in results:
        acc, dns_sc, les_sc, r_dns, r_les, aniso = accuracy_estimate(
            buyuk_nx, buyuk_ny, buyuk_nz, r
        )
        if acc >= 70:
            print(f"       Re={r['Re']:>5d}: DNS olarak KULLANILABILIR (skor={acc:.0f})")
        elif acc >= 50:
            print(f"       Re={r['Re']:>5d}: LES olarak KULLANILABILIR (skor={acc:.0f})")

    print()
    print("  5. GENEL DEGERLENDIRME")
    print()

    if r5000:
        les_grid = r5000["les_grid"]
        dns_grid = r5000["dns_grid"]
        print(f"     Re=5000 icin ideal DNS grid: {'x'.join(map(str, dns_grid))} "
              f"({r5000['n_dns']:,d} nokta, {r5000['mem_dns_mb']:.0f} MB)")
        print(f"     Re=5000 icin ideal LES grid: {'x'.join(map(str, les_grid))} "
              f"({r5000['n_les']:,d} nokta, {r5000['mem_les_mb']:.1f} MB)")
        print()

        b_acc, _, _, _, _, _ = accuracy_estimate(buyuk_nx, buyuk_ny, buyuk_nz, r5000)
        print(f"     Buyuk grid (96x160x64) Re=5000'deki skoru: {b_acc:.0f}/100")
        if b_acc >= 60:
            print("     >> Bu grid LES cozucu icin MAKUL bir secim.")
        elif b_acc >= 40:
            print("     >> Bu grid LES icin SINIRDA. Sonuclara dikkatli yaklasılmali.")
        else:
            print("     >> Bu grid yetersiz. Daha ince grid veya daha dusuk Re kullanin.")

    print()
    print("  6. NEURAL OPERATOR ICIN NOT")
    print()
    print("     Neural operator (INNATE) fizigi ogrendigi icin, tam DNS cozunurlugu")
    print("     gerekmez. Ancak training verisi yeterli olcekleri icermeli.")
    print("     Oneri: Training'i kucuk grid (32x48x24) + dusuk Re (100-1000) ile baslat,")
    print("     sonra buyuk grid (96x160x64) + yuksek Re'ye transfer et.")
    print()
    print_separator()


# ============================================================================
# MAIN
# ============================================================================

def main():
    # Tum Re degerleri icin hesapla
    results = [analyze_re(Re) for Re in RE_VALUES]

    # Tablolari yazdir
    print_results_table(results)
    print_grid_comparison(results)

    sweet_spots = sweet_spot_search(results)
    print_sweet_spot(sweet_spots)

    print_summary(results)


if __name__ == "__main__":
    main()
