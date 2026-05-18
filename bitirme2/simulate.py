"""
simulate.py - Egitimsiz saf fizik simulasyonu (no training, just forward pass)

Modeli egitmeden sadece forward step'ler atarak fizigi gozlemle.
Phase 1 (Boussinesq) veya Phase 2 (Non-Boussinesq) secelebilir.

Kullanim:
  # Faz 1: Boussinesq (untrained)
  python simulate.py --phase 1 --steps 200

  # Faz 2: Non-Boussinesq (untrained)
  python simulate.py --phase 2 --steps 200

  # Checkpoint'tan yukle
  python simulate.py --phase 1 --steps 500 --checkpoint results/checkpoints/checkpoint_epoch003000.pt

  # Kucuk grid (hizli test)
  python simulate.py --phase 1 --steps 100 --small

  # Ozel Re/Ra
  python simulate.py --phase 1 --steps 200 --Re 2000 --Ra 1e7
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_this_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_this_dir.parent))
sys.path.insert(0, str(_this_dir))

import torch
from config import Config
from model import INNATE3D_MixedConvection, ThermalFluidState


def simulate(
    phase: int = 1,
    steps: int = 200,
    checkpoint: str = None,
    small: bool = False,
    Re: float = None,
    Ra: float = None,
    device_override: str = None,
    log_interval: int = 10,
):
    """Egitimsiz forward simulasyon."""

    cfg = Config()

    # Grid
    if small:
        cfg.domain.Nx, cfg.domain.Ny, cfg.domain.Nz = 32, 48, 24
    # else: default 96x160x64

    # Phase
    cfg.physics.non_boussinesq = (phase == 2)
    if phase == 2:
        cfg.physics.T_cold = -30.0  # dT=50K

    # Re/Ra override
    if Re is not None:
        cfg.physics.Re = Re
    if Ra is not None:
        cfg.physics.Ra = Ra

    # Device
    if device_override:
        cfg.device = device_override

    device = cfg.device
    dom = cfg.domain
    phys = cfg.physics

    print("=" * 70)
    print(f"SIMULATE — Phase {phase} ({'Non-Boussinesq' if phase == 2 else 'Boussinesq'})")
    print("=" * 70)
    print(f"Grid:    {dom.Nx}x{dom.Ny}x{dom.Nz} = {dom.Nx*dom.Ny*dom.Nz:,} points")
    print(f"Domain:  {dom.Lx}x{dom.Ly}x{dom.Lz}")
    print(f"Re={phys.Re:.0f}  Ra={phys.Ra:.0e}  Ri={phys.Ri:.4f}  Pr={phys.Pr}")
    print(f"nu={phys.nu:.2e}  kappa={phys.kappa:.2e}  dt={phys.dt}")
    print(f"dT={phys.dT:.0f}K  ({phys.T_hot}C -> {phys.T_cold}C)")
    print(f"Steps:   {steps}")
    print(f"Device:  {device}")
    print()

    # Model
    model = INNATE3D_MixedConvection(cfg).to(device)
    print(f"Parameters: {model.count_parameters():,}")

    if checkpoint:
        print(f"Loading: {checkpoint}")
        ckpt = torch.load(checkpoint, weights_only=False, map_location=device)
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        if missing:
            print(f"  Missing keys: {len(missing)}")
        epoch = ckpt.get("epoch", "?")
        print(f"  Loaded epoch {epoch}")
    else:
        print("No checkpoint — untrained model (random init)")

    print()
    model.eval()

    # IC
    state = model.create_initial_condition(batch_size=1, device=device)

    # Header
    header = f"{'Step':>6s}  {'E':>12s}  {'Z':>12s}  {'max|div|':>12s}  {'Nu':>8s}  {'T_min':>8s}  {'T_max':>8s}"
    if phase == 2:
        header += f"  {'rho_min':>8s}  {'rho_max':>8s}  {'<rho>':>8s}"
    print(header)
    print("-" * len(header))

    # Simulate
    t0 = time.time()
    nan_step = -1

    with torch.no_grad():
        for step in range(steps):
            state = model(state)

            # NaN check
            has_nan = (
                torch.isnan(state.u).any()
                or torch.isnan(state.v).any()
                or torch.isnan(state.w).any()
                or torch.isnan(state.theta).any()
                or (state.rho is not None and torch.isnan(state.rho).any())
            )
            if has_nan:
                nan_step = step
                print(f"\n*** NaN detected at step {step}! ***")
                break

            if step % log_interval == 0 or step == steps - 1:
                E = state.kinetic_energy().mean().item()
                ox, oy, oz = model.ops.curl(state.u, state.v, state.w)
                Z = (ox**2 + oy**2 + oz**2).mean().item()
                div = model.ops.divergence(state.u, state.v, state.w).abs().max().item()
                vtheta = (state.v * state.theta).mean().item()
                Nu = 1.0 + vtheta / (phys.kappa * (1.0 / dom.Ly) + 1e-30)

                # T_total bounds
                y = torch.linspace(0, dom.Ly, dom.Ny + 1, device=device)[:-1].view(1, 1, dom.Ny, 1)
                T_base = phys.T_hot - (phys.dT / dom.Ly) * y
                T_total = T_base + state.theta
                T_min = T_total.min().item()
                T_max = T_total.max().item()

                line = f"{step:6d}  {E:12.6f}  {Z:12.6f}  {div:12.2e}  {Nu:8.4f}  {T_min:8.2f}  {T_max:8.2f}"

                if phase == 2 and state.rho is not None:
                    rho_min = state.rho.min().item()
                    rho_max = state.rho.max().item()
                    rho_mean = state.rho.mean().item()
                    line += f"  {rho_min:8.4f}  {rho_max:8.4f}  {rho_mean:8.4f}"

                print(line)

    elapsed = time.time() - t0
    print()
    print(f"Completed: {steps} steps in {elapsed:.1f}s ({elapsed/steps:.3f}s/step)")
    if nan_step >= 0:
        print(f"*** UNSTABLE: NaN at step {nan_step} ***")
    else:
        print("Stable: no NaN detected")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="INNATE 3D Mixed Convection — Physics Simulation (no training)")
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2], help="1=Boussinesq, 2=Non-Boussinesq")
    parser.add_argument("--steps", type=int, default=200, help="Forward steps")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path")
    parser.add_argument("--small", action="store_true", help="Use small grid (32x48x24)")
    parser.add_argument("--Re", type=float, default=None)
    parser.add_argument("--Ra", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--log-interval", type=int, default=10)
    args = parser.parse_args()

    simulate(
        phase=args.phase,
        steps=args.steps,
        checkpoint=args.checkpoint,
        small=args.small,
        Re=args.Re,
        Ra=args.Ra,
        device_override=args.device,
        log_interval=args.log_interval,
    )
