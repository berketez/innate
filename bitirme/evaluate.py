#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
INNATE3D Evaluation & DNS Comparison for Taylor-Green Vortex 3D

Değerlendirme metrikleri:
1. Fiziksel büyüklükler: Kinetik enerji, Enstrophy, Helicity
2. Sayısal kalite: Divergence, CFL
3. DNS karşılaştırması (varsa)

Not: 3D TGV için t>0 anında analitik çözüm YOK!
     Sadece DNS referans verisi ile karşılaştırma yapılabilir.

Author: INNATE TGV3D Evaluation (2024)
"""

import os
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# Try to import matplotlib
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not found. Plotting disabled.")

# Model import
from model import INNATE3D_TGV, create_model, FluidState3D


# =============================================================================
# PHYSICAL DIAGNOSTICS
# =============================================================================

def compute_kinetic_energy(u: np.ndarray, v: np.ndarray, w: np.ndarray) -> float:
    """Kinetik enerji: E = 0.5 * ⟨u²⟩"""
    return 0.5 * np.mean(u**2 + v**2 + w**2)


def compute_enstrophy(u: np.ndarray, v: np.ndarray, w: np.ndarray, dx: float) -> float:
    """
    Enstrophy: Z = 0.5 * ⟨ω²⟩

    ω = ∇×u (vorticity)
    """
    # Periodic gradient
    def pgrad(f, axis):
        return (np.roll(f, -1, axis=axis) - np.roll(f, 1, axis=axis)) / (2 * dx)

    # Vorticity components
    omega_x = pgrad(w, 1) - pgrad(v, 2)
    omega_y = pgrad(u, 2) - pgrad(w, 0)
    omega_z = pgrad(v, 0) - pgrad(u, 1)

    return 0.5 * np.mean(omega_x**2 + omega_y**2 + omega_z**2)


def compute_divergence(u: np.ndarray, v: np.ndarray, w: np.ndarray, dx: float) -> Tuple[float, float]:
    """
    Divergence: ∇·u (sıfır olmalı)

    Returns:
        (mean_div, max_div)
    """
    def pgrad(f, axis):
        return (np.roll(f, -1, axis=axis) - np.roll(f, 1, axis=axis)) / (2 * dx)

    div = pgrad(u, 0) + pgrad(v, 1) + pgrad(w, 2)

    return np.mean(np.abs(div)), np.max(np.abs(div))


def compute_helicity(u: np.ndarray, v: np.ndarray, w: np.ndarray, dx: float) -> float:
    """
    Helicity: H = ⟨u·ω⟩

    TGV için sıfıra yakın olmalı.
    """
    def pgrad(f, axis):
        return (np.roll(f, -1, axis=axis) - np.roll(f, 1, axis=axis)) / (2 * dx)

    omega_x = pgrad(w, 1) - pgrad(v, 2)
    omega_y = pgrad(u, 2) - pgrad(w, 0)
    omega_z = pgrad(v, 0) - pgrad(u, 1)

    return np.mean(u * omega_x + v * omega_y + w * omega_z)


# =============================================================================
# MODEL EVALUATION
# =============================================================================

def evaluate_model(
    model: INNATE3D_TGV,
    device: torch.device,
    T_final: float = 0.8,
    dt: float = 0.005,
    num_snapshots: int = 20
) -> Dict:
    """
    Model değerlendirmesi.

    IC'den T_final'e kadar evolve et ve metrikleri kaydet.
    """
    model.eval()
    resolution = model.resolution
    domain_size = 2 * math.pi
    dx = domain_size / resolution

    # Storage
    times = []
    energies = []
    enstrophies = []
    divergences_mean = []
    divergences_max = []
    helicities = []
    snapshots = []

    with torch.no_grad():
        # IC
        state = model.tgv_initial_condition(batch_size=1)
        state = FluidState3D(
            u=state.u.to(device),
            v=state.v.to(device),
            w=state.w.to(device),
            p=state.p.to(device),
            omega_x=state.omega_x.to(device),
            omega_y=state.omega_y.to(device),
            omega_z=state.omega_z.to(device),
            t=state.t.to(device)
        )

        num_steps = int(T_final / dt)
        snapshot_interval = max(1, num_steps // num_snapshots)

        for step in range(num_steps + 1):
            if step % snapshot_interval == 0 or step == num_steps:
                # Extract numpy arrays
                u = state.u.squeeze(0).cpu().numpy()
                v = state.v.squeeze(0).cpu().numpy()
                w = state.w.squeeze(0).cpu().numpy()
                t = state.t.item()

                # Compute metrics
                times.append(t)
                energies.append(compute_kinetic_energy(u, v, w))
                enstrophies.append(compute_enstrophy(u, v, w, dx))
                div_mean, div_max = compute_divergence(u, v, w, dx)
                divergences_mean.append(div_mean)
                divergences_max.append(div_max)
                helicities.append(compute_helicity(u, v, w, dx))

                # Store snapshot
                snapshots.append({
                    'time': t,
                    'u': u.copy(),
                    'v': v.copy(),
                    'w': w.copy()
                })

            if step < num_steps:
                state = model.step(state, dt)

    return {
        'times': np.array(times),
        'energies': np.array(energies),
        'enstrophies': np.array(enstrophies),
        'divergences_mean': np.array(divergences_mean),
        'divergences_max': np.array(divergences_max),
        'helicities': np.array(helicities),
        'snapshots': snapshots,
        'resolution': resolution,
        'T_final': T_final,
        'dt': dt
    }


# =============================================================================
# DNS COMPARISON
# =============================================================================

def load_dns_data(filepath: str) -> Optional[Dict]:
    """
    DNS referans verisini yükle (NPZ formatı).

    Beklenen NPZ içeriği:
        - 'times': zaman dizisi
        - 'energies': kinetik enerji
        - 'enstrophies': enstrophy
        - 'resolution': çözünürlük (opsiyonel)
        - 'nu': viskozite (opsiyonel)

    Alternatif olarak velocity fields:
        - 'u', 'v', 'w': hız alanları (time, x, y, z)
    """
    if not os.path.exists(filepath):
        print(f"DNS data not found: {filepath}")
        return None

    try:
        # NPZ dosyasını yükle
        data = np.load(filepath, allow_pickle=True)

        result = {}

        # Doğrudan times/energies varsa kullan
        # Önce history verilerini kontrol et (daha detaylı)
        if 'time_history' in data and 'ke_history' in data:
            result['times'] = data['time_history']
            result['energies'] = data['ke_history']
            if 'enstrophy_history' in data:
                result['enstrophies'] = data['enstrophy_history']
        else:
            # Fallback to snapshot times
            if 'times' in data:
                result['times'] = data['times']
            elif 't' in data:
                result['times'] = data['t']

            if 'energies' in data:
                result['energies'] = data['energies']
            elif 'kinetic_energy' in data:
                result['energies'] = data['kinetic_energy']
            elif 'energy' in data:
                result['energies'] = data['energy']

            if 'enstrophies' in data:
                result['enstrophies'] = data['enstrophies']
            elif 'enstrophy' in data:
                result['enstrophies'] = data['enstrophy']

        # Velocity fields varsa kaydet
        if 'u' in data:
            result['u'] = data['u']
            result['v'] = data['v']
            result['w'] = data['w']
            if 'p' in data:
                result['p'] = data['p']
            if 'times' in data:
                result['snapshot_times'] = data['times']

            # Enerji yoksa hesapla
            if 'energies' not in result:
                u = data['u']
                v = data['v']
                w = data['w']
                if u.ndim == 4:  # (time, x, y, z)
                    result['energies'] = 0.5 * np.mean(u**2 + v**2 + w**2, axis=(1, 2, 3))
                else:  # (x, y, z) tek snapshot
                    result['energies'] = np.array([0.5 * np.mean(u**2 + v**2 + w**2)])

        # Divergence
        if 'divergences' in data:
            result['divergences'] = data['divergences']
        elif 'divergence_history' in data:
            result['divergences'] = data['divergence_history']

        # Resolution ve nu
        if 'resolution' in data:
            result['resolution'] = int(data['resolution'])
        if 'nu' in data:
            result['nu'] = float(data['nu'])

        print(f"DNS data loaded (NPZ): {filepath}")
        print(f"  Keys: {list(result.keys())}")
        if 'times' in result:
            print(f"  Time range: [{result['times'][0]:.3f}, {result['times'][-1]:.3f}]")

        return result

    except Exception as e:
        print(f"Error loading DNS data: {e}")
        return None


def compare_with_dns(
    model_results: Dict,
    dns_data: Dict
) -> Dict:
    """
    Model sonuçlarını DNS ile karşılaştır.
    """
    # Interpolate to common time points
    model_times = model_results['times']
    model_energies = model_results['energies']
    model_enstrophies = model_results['enstrophies']

    dns_times = np.array(dns_data['times'])
    dns_energies = np.array(dns_data['energies'])
    dns_enstrophies = np.array(dns_data['enstrophies'])

    # Find common time range
    t_max = min(model_times[-1], dns_times[-1])
    t_common = np.linspace(0, t_max, 50)

    # Interpolate
    model_E_interp = np.interp(t_common, model_times, model_energies)
    model_Z_interp = np.interp(t_common, model_times, model_enstrophies)
    dns_E_interp = np.interp(t_common, dns_times, dns_energies)
    dns_Z_interp = np.interp(t_common, dns_times, dns_enstrophies)

    # Compute errors
    E_error = np.abs(model_E_interp - dns_E_interp)
    Z_error = np.abs(model_Z_interp - dns_Z_interp)

    # Normalize by initial values
    E0 = dns_energies[0]
    Z0 = dns_enstrophies[0] if dns_enstrophies[0] > 0 else 1.0

    return {
        't_common': t_common,
        'model_energies': model_E_interp,
        'model_enstrophies': model_Z_interp,
        'dns_energies': dns_E_interp,
        'dns_enstrophies': dns_Z_interp,
        'energy_error': E_error,
        'enstrophy_error': Z_error,
        'energy_L2_error': np.sqrt(np.mean((E_error / E0)**2)),
        'enstrophy_L2_error': np.sqrt(np.mean((Z_error / Z0)**2)),
        'energy_max_error': np.max(E_error / E0),
        'enstrophy_max_error': np.max(Z_error / Z0)
    }


def compare_velocity_fields(
    model_snapshots: List[Dict],
    dns_data: Dict,
    model_resolution: int = 32
) -> Dict:
    """
    Velocity field karşılaştırması - DNS'i model resolution'a downsample ederek.

    Returns:
        Dict with L2 errors, max errors, correlation for u, v, w at each time
    """
    if 'u' not in dns_data or 'snapshot_times' not in dns_data:
        print("DNS velocity snapshots not available")
        return None

    dns_u = dns_data['u']  # (n_times, nx, ny, nz)
    dns_v = dns_data['v']
    dns_w = dns_data['w']
    dns_times = dns_data['snapshot_times']
    dns_resolution = dns_u.shape[1]

    # Downsample factor
    factor = dns_resolution // model_resolution

    results = {
        'times': [],
        'u_L2_error': [],
        'v_L2_error': [],
        'w_L2_error': [],
        'u_max_error': [],
        'v_max_error': [],
        'w_max_error': [],
        'u_correlation': [],
        'v_correlation': [],
        'w_correlation': [],
        'velocity_L2_error': [],  # Combined
    }

    for snap in model_snapshots:
        t = snap.get('t', snap.get('time', 0))

        # Find closest DNS time
        dns_idx = np.argmin(np.abs(dns_times - t))
        if np.abs(dns_times[dns_idx] - t) > 0.05:
            continue  # Skip if no close DNS snapshot

        # Downsample DNS to model resolution
        dns_u_down = dns_u[dns_idx, ::factor, ::factor, ::factor]
        dns_v_down = dns_v[dns_idx, ::factor, ::factor, ::factor]
        dns_w_down = dns_w[dns_idx, ::factor, ::factor, ::factor]

        # Model velocity (numpy)
        model_u = snap['u'].cpu().numpy() if hasattr(snap['u'], 'cpu') else snap['u']
        model_v = snap['v'].cpu().numpy() if hasattr(snap['v'], 'cpu') else snap['v']
        model_w = snap['w'].cpu().numpy() if hasattr(snap['w'], 'cpu') else snap['w']

        # Remove batch dimension if present
        if model_u.ndim == 4:
            model_u = model_u[0]
            model_v = model_v[0]
            model_w = model_w[0]

        # L2 errors (relative)
        def rel_l2_error(pred, true):
            return np.linalg.norm(pred - true) / (np.linalg.norm(true) + 1e-10)

        def max_error(pred, true):
            return np.max(np.abs(pred - true))

        def correlation(pred, true):
            p_flat = pred.flatten() - pred.mean()
            t_flat = true.flatten() - true.mean()
            return np.dot(p_flat, t_flat) / (np.linalg.norm(p_flat) * np.linalg.norm(t_flat) + 1e-10)

        results['times'].append(t)
        results['u_L2_error'].append(rel_l2_error(model_u, dns_u_down) * 100)  # %
        results['v_L2_error'].append(rel_l2_error(model_v, dns_v_down) * 100)
        results['w_L2_error'].append(rel_l2_error(model_w, dns_w_down) * 100)
        results['u_max_error'].append(max_error(model_u, dns_u_down))
        results['v_max_error'].append(max_error(model_v, dns_v_down))
        results['w_max_error'].append(max_error(model_w, dns_w_down))
        results['u_correlation'].append(correlation(model_u, dns_u_down))
        results['v_correlation'].append(correlation(model_v, dns_v_down))
        results['w_correlation'].append(correlation(model_w, dns_w_down))

        # Combined velocity error
        vel_model = np.sqrt(model_u**2 + model_v**2 + model_w**2)
        vel_dns = np.sqrt(dns_u_down**2 + dns_v_down**2 + dns_w_down**2)
        results['velocity_L2_error'].append(rel_l2_error(vel_model, vel_dns) * 100)

    # Summary statistics
    if results['times']:
        results['summary'] = {
            'mean_u_L2': np.mean(results['u_L2_error']),
            'mean_v_L2': np.mean(results['v_L2_error']),
            'mean_w_L2': np.mean(results['w_L2_error']),
            'mean_velocity_L2': np.mean(results['velocity_L2_error']),
            'mean_u_corr': np.mean(results['u_correlation']),
            'mean_v_corr': np.mean(results['v_correlation']),
            'mean_w_corr': np.mean(results['w_correlation']),
            'max_u_error': np.max(results['u_max_error']),
            'max_v_error': np.max(results['v_max_error']),
            'max_w_error': np.max(results['w_max_error']),
        }

    return results


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_evolution(
    results: Dict,
    dns_data: Optional[Dict] = None,
    save_path: Optional[str] = None
):
    """
    Fiziksel büyüklüklerin evrimini çiz.
    """
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available for plotting")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    times = results['times']
    energies = results['energies']
    enstrophies = results['enstrophies']
    div_mean = results['divergences_mean']
    helicities = results['helicities']

    # 1. Kinetic Energy
    ax = axes[0, 0]
    ax.plot(times, energies, 'b-', linewidth=2, label='INNATE3D')
    if dns_data:
        ax.plot(dns_data['times'], dns_data['energies'], 'k--', linewidth=2, label='DNS')
    ax.set_xlabel('Time')
    ax.set_ylabel('Kinetic Energy')
    ax.set_title('Kinetic Energy Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Enstrophy
    ax = axes[0, 1]
    ax.plot(times, enstrophies, 'r-', linewidth=2, label='INNATE3D')
    if dns_data and 'enstrophies' in dns_data:
        ax.plot(dns_data['times'], dns_data['enstrophies'], 'k--', linewidth=2, label='DNS')
    ax.set_xlabel('Time')
    ax.set_ylabel('Enstrophy')
    ax.set_title('Enstrophy Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Divergence
    ax = axes[1, 0]
    ax.semilogy(times, div_mean, 'g-', linewidth=2)
    ax.axhline(y=1e-6, color='r', linestyle='--', label='Target (1e-6)')
    ax.set_xlabel('Time')
    ax.set_ylabel('|∇·u| (mean)')
    ax.set_title('Divergence (should be ~0)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Helicity
    ax = axes[1, 1]
    ax.plot(times, helicities, 'm-', linewidth=2)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time')
    ax.set_ylabel('Helicity')
    ax.set_title('Helicity (should be ~0 for TGV)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_velocity_comparison(
    model_snapshot: Dict,
    dns_data: Dict,
    t: float,
    model_resolution: int = 32,
    save_path: str = None
):
    """
    INNATE vs DNS velocity field karşılaştırması - yan yana.
    """
    if not HAS_MATPLOTLIB:
        return

    dns_times = dns_data['snapshot_times']
    dns_idx = np.argmin(np.abs(dns_times - t))

    factor = dns_data['u'].shape[1] // model_resolution

    # Downsample DNS
    dns_u = dns_data['u'][dns_idx, ::factor, ::factor, ::factor]
    dns_v = dns_data['v'][dns_idx, ::factor, ::factor, ::factor]
    dns_w = dns_data['w'][dns_idx, ::factor, ::factor, ::factor]

    # Model data
    model_u = model_snapshot['u']
    model_v = model_snapshot['v']
    model_w = model_snapshot['w']

    if model_u.ndim == 4:
        model_u = model_u[0]
        model_v = model_v[0]
        model_w = model_w[0]

    # Mid slice
    z_idx = model_resolution // 2

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))

    vmin, vmax = -1, 1
    cmap = 'RdBu_r'

    # Row 1: u component
    im = axes[0, 0].imshow(model_u[:, :, z_idx].T, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0, 0].set_title('INNATE: u', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('x'); axes[0, 0].set_ylabel('y')

    axes[0, 1].imshow(dns_u[:, :, z_idx].T, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0, 1].set_title('DNS: u', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('x'); axes[0, 1].set_ylabel('y')

    diff_u = model_u[:, :, z_idx] - dns_u[:, :, z_idx]
    im_diff = axes[0, 2].imshow(diff_u.T, origin='lower', cmap='coolwarm', vmin=-0.1, vmax=0.1)
    axes[0, 2].set_title(f'Difference (max={np.abs(diff_u).max():.4f})', fontsize=12)
    axes[0, 2].set_xlabel('x'); axes[0, 2].set_ylabel('y')
    plt.colorbar(im_diff, ax=axes[0, 2])

    # Row 2: v component
    axes[1, 0].imshow(model_v[:, :, z_idx].T, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1, 0].set_title('INNATE: v', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('x'); axes[1, 0].set_ylabel('y')

    axes[1, 1].imshow(dns_v[:, :, z_idx].T, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1, 1].set_title('DNS: v', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('x'); axes[1, 1].set_ylabel('y')

    diff_v = model_v[:, :, z_idx] - dns_v[:, :, z_idx]
    im_diff = axes[1, 2].imshow(diff_v.T, origin='lower', cmap='coolwarm', vmin=-0.1, vmax=0.1)
    axes[1, 2].set_title(f'Difference (max={np.abs(diff_v).max():.4f})', fontsize=12)
    axes[1, 2].set_xlabel('x'); axes[1, 2].set_ylabel('y')
    plt.colorbar(im_diff, ax=axes[1, 2])

    # Row 3: w component
    axes[2, 0].imshow(model_w[:, :, z_idx].T, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
    axes[2, 0].set_title('INNATE: w', fontsize=12, fontweight='bold')
    axes[2, 0].set_xlabel('x'); axes[2, 0].set_ylabel('y')

    axes[2, 1].imshow(dns_w[:, :, z_idx].T, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
    axes[2, 1].set_title('DNS: w', fontsize=12, fontweight='bold')
    axes[2, 1].set_xlabel('x'); axes[2, 1].set_ylabel('y')

    diff_w = model_w[:, :, z_idx] - dns_w[:, :, z_idx]
    im_diff = axes[2, 2].imshow(diff_w.T, origin='lower', cmap='coolwarm', vmin=-0.1, vmax=0.1)
    axes[2, 2].set_title(f'Difference (max={np.abs(diff_w).max():.4f})', fontsize=12)
    axes[2, 2].set_xlabel('x'); axes[2, 2].set_ylabel('y')
    plt.colorbar(im_diff, ax=axes[2, 2])

    plt.suptitle(f'Velocity Field Comparison at t={t:.3f} (z-slice at mid-plane)', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison plot saved: {save_path}")
    plt.close()


def plot_velocity_slice(
    snapshot: Dict,
    plane: str = 'xy',
    slice_idx: int = None,
    save_path: Optional[str] = None
):
    """
    Hız alanının 2D kesitini çiz.
    """
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available for plotting")
        return

    u = snapshot['u']
    v = snapshot['v']
    w = snapshot['w']
    t = snapshot['time']

    N = u.shape[0]
    if slice_idx is None:
        slice_idx = N // 2

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    if plane == 'xy':
        u_slice = u[:, :, slice_idx]
        v_slice = v[:, :, slice_idx]
        w_slice = w[:, :, slice_idx]
        xlabel, ylabel = 'x', 'y'
    elif plane == 'xz':
        u_slice = u[:, slice_idx, :]
        v_slice = v[:, slice_idx, :]
        w_slice = w[:, slice_idx, :]
        xlabel, ylabel = 'x', 'z'
    else:  # yz
        u_slice = u[slice_idx, :, :]
        v_slice = v[slice_idx, :, :]
        w_slice = w[slice_idx, :, :]
        xlabel, ylabel = 'y', 'z'

    vmax = max(np.abs(u_slice).max(), np.abs(v_slice).max(), np.abs(w_slice).max())

    im0 = axes[0].imshow(u_slice.T, origin='lower', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[0].set_title(f'u (t={t:.3f})')
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel)
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(v_slice.T, origin='lower', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[1].set_title(f'v (t={t:.3f})')
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel(ylabel)
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(w_slice.T, origin='lower', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[2].set_title(f'w (t={t:.3f})')
    axes[2].set_xlabel(xlabel)
    axes[2].set_ylabel(ylabel)
    plt.colorbar(im2, ax=axes[2])

    plt.suptitle(f'{plane.upper()} plane at index {slice_idx}')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_vorticity_magnitude(
    snapshot: Dict,
    dx: float,
    plane: str = 'xy',
    slice_idx: int = None,
    save_path: Optional[str] = None
):
    """
    Vortisite büyüklüğünü çiz.
    """
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available for plotting")
        return

    u = snapshot['u']
    v = snapshot['v']
    w = snapshot['w']
    t = snapshot['time']

    N = u.shape[0]
    if slice_idx is None:
        slice_idx = N // 2

    # Compute vorticity
    def pgrad(f, axis):
        return (np.roll(f, -1, axis=axis) - np.roll(f, 1, axis=axis)) / (2 * dx)

    omega_x = pgrad(w, 1) - pgrad(v, 2)
    omega_y = pgrad(u, 2) - pgrad(w, 0)
    omega_z = pgrad(v, 0) - pgrad(u, 1)

    omega_mag = np.sqrt(omega_x**2 + omega_y**2 + omega_z**2)

    if plane == 'xy':
        omega_slice = omega_mag[:, :, slice_idx]
        xlabel, ylabel = 'x', 'y'
    elif plane == 'xz':
        omega_slice = omega_mag[:, slice_idx, :]
        xlabel, ylabel = 'x', 'z'
    else:  # yz
        omega_slice = omega_mag[slice_idx, :, :]
        xlabel, ylabel = 'y', 'z'

    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(omega_slice.T, origin='lower', cmap='hot')
    ax.set_title(f'Vorticity Magnitude |ω| (t={t:.3f})')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.colorbar(im, ax=ax, label='|ω|')

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved: {save_path}")
    else:
        plt.show()

    plt.close()


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_report(
    results: Dict,
    comparison: Optional[Dict] = None,
    save_path: Optional[str] = None
) -> str:
    """
    Değerlendirme raporu oluştur.
    """
    lines = []
    lines.append("=" * 60)
    lines.append("INNATE3D TGV Evaluation Report")
    lines.append("=" * 60)
    lines.append("")

    # Model info
    lines.append("Model Configuration:")
    lines.append(f"  Resolution: {results['resolution']}³")
    lines.append(f"  T_final: {results['T_final']}")
    lines.append(f"  dt: {results['dt']}")
    lines.append("")

    # Physical metrics
    lines.append("Physical Metrics:")
    lines.append(f"  Initial Energy: {results['energies'][0]:.6f}")
    lines.append(f"  Final Energy: {results['energies'][-1]:.6f}")
    lines.append(f"  Energy Decay: {(results['energies'][0] - results['energies'][-1]) / results['energies'][0] * 100:.2f}%")
    lines.append("")
    lines.append(f"  Peak Enstrophy: {np.max(results['enstrophies']):.6f}")
    lines.append(f"  Peak Time: {results['times'][np.argmax(results['enstrophies'])]:.3f}")
    lines.append("")

    # Numerical quality
    lines.append("Numerical Quality:")
    lines.append(f"  Mean Divergence: {np.mean(results['divergences_mean']):.2e}")
    lines.append(f"  Max Divergence: {np.max(results['divergences_max']):.2e}")
    lines.append(f"  Mean |Helicity|: {np.mean(np.abs(results['helicities'])):.2e}")
    lines.append("")

    # DNS comparison
    if comparison:
        lines.append("DNS Comparison:")
        lines.append(f"  Energy L2 Error: {comparison['energy_L2_error']:.4f}")
        lines.append(f"  Energy Max Error: {comparison['energy_max_error']:.4f}")
        lines.append(f"  Enstrophy L2 Error: {comparison['enstrophy_L2_error']:.4f}")
        lines.append(f"  Enstrophy Max Error: {comparison['enstrophy_max_error']:.4f}")
        lines.append("")

    lines.append("=" * 60)

    report = "\n".join(lines)

    if save_path:
        with open(save_path, 'w') as f:
            f.write(report)
        print(f"Report saved: {save_path}")

    return report


# =============================================================================
# MAIN
# =============================================================================

def main(checkpoint_path: str, dns_path: Optional[str] = None, output_dir: str = "eval_results"):
    """
    Ana değerlendirme fonksiyonu.
    """
    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")

    # Output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    print(f"Loading model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    config = checkpoint.get('config', {})
    model = create_model(
        resolution=config.get('resolution', 32),
        nu=config.get('nu', 0.001),
        num_layers=config.get('num_layers', 6),
        neurons_per_layer=config.get('neurons_per_layer', 800),
        target_params=config.get('target_params', 10000)
    )
    # Load only trainable parameters, skip buffers (kx, ky, kz, X, Y, Z)
    state_dict = checkpoint['model_state_dict']

    # Filter: only load learnable parameters (skip all buffers)
    filtered_state = {}
    for k, v in state_dict.items():
        # Skip all spectral ops buffers and grid coordinates
        skip_patterns = ['kx', 'ky', 'kz', 'k2', 'dealias', 'X', 'Y', 'Z']
        if any(pattern in k for pattern in skip_patterns):
            continue
        try:
            filtered_state[k] = v.clone() if hasattr(v, 'clone') else v
        except:
            continue

    model.load_state_dict(filtered_state, strict=False)
    model = model.to(device)
    model.eval()

    # Evaluate
    print("Evaluating model...")
    # Force T_final=0.8 to match DNS data range
    results = evaluate_model(
        model, device,
        T_final=0.8,
        dt=config.get('dt', 0.005)
    )

    # DNS comparison
    comparison = None
    velocity_comparison = None
    if dns_path:
        dns_data = load_dns_data(dns_path)
        if dns_data:
            comparison = compare_with_dns(results, dns_data)

            # Velocity field comparison
            print("Comparing velocity fields with DNS...")
            velocity_comparison = compare_velocity_fields(
                results['snapshots'],
                dns_data,
                model_resolution=config.get('resolution', 32)
            )

    # Generate report
    report = generate_report(
        results, comparison,
        save_path=os.path.join(output_dir, "report.txt")
    )
    print("\n" + report)

    # Print velocity comparison if available
    if velocity_comparison and 'summary' in velocity_comparison:
        s = velocity_comparison['summary']
        print("\n" + "="*60)
        print("VELOCITY FIELD COMPARISON (DNS downsampled to model resolution)")
        print("="*60)
        print(f"\nL2 Errors (%):")
        print(f"  u: {s['mean_u_L2']:.2f}%")
        print(f"  v: {s['mean_v_L2']:.2f}%")
        print(f"  w: {s['mean_w_L2']:.2f}%")
        print(f"  Combined velocity: {s['mean_velocity_L2']:.2f}%")
        print(f"\nCorrelation:")
        print(f"  u: {s['mean_u_corr']:.4f}")
        print(f"  v: {s['mean_v_corr']:.4f}")
        print(f"  w: {s['mean_w_corr']:.4f}")
        print(f"\nMax Absolute Errors:")
        print(f"  u: {s['max_u_error']:.4f}")
        print(f"  v: {s['max_v_error']:.4f}")
        print(f"  w: {s['max_w_error']:.4f}")
        print("="*60)

        # Save velocity comparison to file
        with open(os.path.join(output_dir, "velocity_comparison.txt"), 'w') as f:
            f.write("VELOCITY FIELD COMPARISON\n")
            f.write("="*60 + "\n\n")
            f.write("L2 Errors (%):\n")
            f.write(f"  u: {s['mean_u_L2']:.2f}%\n")
            f.write(f"  v: {s['mean_v_L2']:.2f}%\n")
            f.write(f"  w: {s['mean_w_L2']:.2f}%\n")
            f.write(f"  Combined: {s['mean_velocity_L2']:.2f}%\n\n")
            f.write("Correlation:\n")
            f.write(f"  u: {s['mean_u_corr']:.4f}\n")
            f.write(f"  v: {s['mean_v_corr']:.4f}\n")
            f.write(f"  w: {s['mean_w_corr']:.4f}\n\n")
            f.write("Time-series:\n")
            for i, t in enumerate(velocity_comparison['times']):
                f.write(f"  t={t:.3f}: u_err={velocity_comparison['u_L2_error'][i]:.2f}%, ")
                f.write(f"v_err={velocity_comparison['v_L2_error'][i]:.2f}%, ")
                f.write(f"w_err={velocity_comparison['w_L2_error'][i]:.2f}%\n")
        print(f"Velocity comparison saved: {output_dir}/velocity_comparison.txt")

    # Plots
    if HAS_MATPLOTLIB:
        dns_data = load_dns_data(dns_path) if dns_path else None

        plot_evolution(
            results, dns_data,
            save_path=os.path.join(output_dir, "evolution.png")
        )

        # Plot snapshots
        for i, snap in enumerate(results['snapshots'][::max(1, len(results['snapshots'])//4)]):
            plot_velocity_slice(
                snap, plane='xy',
                save_path=os.path.join(output_dir, f"velocity_t{snap['time']:.3f}.png")
            )

        # INNATE vs DNS comparison plots
        if dns_data and 'u' in dns_data:
            print("Generating INNATE vs DNS comparison plots...")
            for snap in results['snapshots'][::max(1, len(results['snapshots'])//4)]:
                t = snap.get('time', 0)
                plot_velocity_comparison(
                    snap, dns_data, t,
                    model_resolution=config.get('resolution', 32),
                    save_path=os.path.join(output_dir, f"comparison_t{t:.3f}.png")
                )

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate INNATE3D TGV model")
    parser.add_argument("checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--dns", type=str, default=None, help="Path to DNS data (NPZ format)")
    parser.add_argument("--output", type=str, default="eval_results", help="Output directory")
    args = parser.parse_args()

    main(args.checkpoint, args.dns, args.output)
