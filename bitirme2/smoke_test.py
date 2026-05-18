#!/usr/bin/env python3
"""
INNATE v2 3D Mixed Convection - Comprehensive Smoke Test

Tests:
1. Import test - all modules load without errors
2. Forward pass - config, model init, forward, NaN check
3. Training pipeline - PhysicsLoss, CurriculumScheduler phases
4. Phase weights - A/B/C/D weight validation
5. Multi-step forward - energy stability over 3 steps
6. Parameter count - report & sanity check (v2 ~500-600 range)
7. Gradient flow - buoyancy, forcing, eddy, MLP SGS
8. MLP SGS - shared MLP produces valid Cs/kappa
9. IMEX integration - implicit molecular diffusion check
10. Re sweep - set_physics works for all 5 Re points
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import torch
import traceback
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────────
# Test utilities
# ──────────────────────────────────────────────────────────────────────────────

class TestResult:
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.message = ""
        self.details = {}

    def pass_test(self, message: str = "", **details):
        self.passed = True
        self.message = message
        self.details = details

    def fail_test(self, message: str, **details):
        self.passed = False
        self.message = message
        self.details = details

    def print_result(self):
        status = "PASS" if self.passed else "FAIL"
        print(f"{status} | {self.name}")
        if self.message:
            print(f"      {self.message}")
        for key, val in self.details.items():
            print(f"      {key}: {val}")
        print()


def run_test(name: str, func):
    result = TestResult(name)
    try:
        func(result)
        if not result.passed and not result.message:
            result.fail_test("Test did not call pass_test or fail_test")
    except Exception as e:
        result.fail_test(f"Exception: {e}", traceback=traceback.format_exc())
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Helper: small-grid config
# ──────────────────────────────────────────────────────────────────────────────

def _make_config():
    from config import Config
    cfg = Config()
    cfg.domain.Nx = 16
    cfg.domain.Ny = 24
    cfg.domain.Nz = 8
    cfg._device_override = "cpu"
    return cfg


# ──────────────────────────────────────────────────────────────────────────────
# Test 1: Import Test
# ──────────────────────────────────────────────────────────────────────────────

def test_imports(result: TestResult):
    try:
        from config import Config, DomainConfig, PhysicsConfig, ModelConfig, TrainingConfig
        from model import INNATE3D_MixedConvection, ThermalFluidState
        from train import PhysicsLoss, CurriculumScheduler
        result.pass_test("All modules imported successfully")
    except Exception as e:
        result.fail_test(f"Import failed: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# Test 2: Forward Pass
# ──────────────────────────────────────────────────────────────────────────────

def test_forward_pass(result: TestResult):
    from model import INNATE3D_MixedConvection

    cfg = _make_config()
    model = INNATE3D_MixedConvection(cfg)
    model.eval()

    state = model.create_initial_condition()

    with torch.no_grad():
        out = model(state)

    has_nan = any(
        torch.isnan(x).any().item()
        for x in [out.u, out.v, out.w, out.theta]
    )

    if has_nan:
        result.fail_test("NaN detected in forward pass outputs")
        return

    result.pass_test(
        "Forward pass OK",
        grid=f"{cfg.domain.Nx}x{cfg.domain.Ny}x{cfg.domain.Nz}",
        nan_detected="No"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Test 3: Training Pipeline (Phase Boundaries)
# ──────────────────────────────────────────────────────────────────────────────

def test_training_pipeline(result: TestResult):
    from train import CurriculumScheduler
    from config import Config

    cfg = Config()
    scheduler = CurriculumScheduler(cfg)

    # v2 phase boundaries: A(0-300), B(300-600), C(600-1000), D(1000-1500)
    phases_to_test = {
        "A": 0,
        "A_mid": 150,
        "B": 300,
        "B_mid": 450,
        "C": 600,
        "C_mid": 800,
        "D": 1000,
        "D_mid": 1250,
        "D_beyond": 2000,
    }

    expected = {
        "A": "A", "A_mid": "A",
        "B": "B", "B_mid": "B",
        "C": "C", "C_mid": "C",
        "D": "D", "D_mid": "D", "D_beyond": "D",
    }

    for label, epoch in phases_to_test.items():
        phase = scheduler.get_phase(epoch)
        exp = expected[label]
        if phase != exp:
            result.fail_test(
                f"Phase mismatch at epoch {epoch} ({label})",
                expected=exp, got=phase
            )
            return

    result.pass_test("All phase boundaries correct (A:0-300, B:300-600, C:600-1000, D:1000-1500)")


# ──────────────────────────────────────────────────────────────────────────────
# Test 4: Phase Weights Validation
# ──────────────────────────────────────────────────────────────────────────────

def test_phase_weights(result: TestResult):
    from config import Config
    from train import CurriculumScheduler

    cfg = Config()
    scheduler = CurriculumScheduler(cfg)

    # 2026-04-26 TUNING v2: Phase A'da L_nu_phys=0.5 (zayif Nu sinyali,
    # cfd-expert onerisi). Eski 0 -> termal kuplaj donuktu.
    w_a = scheduler.get_weights(100)
    if not (0.0 <= w_a.get("L_nu_phys", -1) <= 1.0):
        result.fail_test("Phase A: L_nu_phys [0.0, 1.0] araliginda olmali",
                         got=w_a.get("L_nu_phys"))
        return
    if w_a.get("L_germano", -1) != 0.0:
        result.fail_test("Phase A: L_germano should be 0.0", got=w_a.get("L_germano"))
        return

    # Phase B midpoint: interpolated between A and C
    w_b = scheduler.get_weights(450)
    # L_nu_phys ramps 0 -> 5.0, at midpoint ~2.5
    nu_phys_b = w_b.get("L_nu_phys", -1)
    if not (1.0 < nu_phys_b < 4.0):
        result.fail_test(f"Phase B mid: L_nu_phys should be ~2.5, got {nu_phys_b}")
        return

    # Phase C: L_nu_phys=5.0 (active), L_germano=0 (still disabled)
    w_c = scheduler.get_weights(800)
    if w_c.get("L_nu_phys", -1) != 5.0:
        result.fail_test("Phase C: L_nu_phys should be 5.0", got=w_c.get("L_nu_phys"))
        return
    if w_c.get("L_germano", -1) != 0.0:
        result.fail_test("Phase C: L_germano should be 0.0", got=w_c.get("L_germano"))
        return

    # Phase D end: L_germano=8.0 (fully ramped)
    w_d = scheduler.get_weights(1499)
    germano_d = w_d.get("L_germano", -1)
    if not (7.0 < germano_d <= 8.0):
        result.fail_test(f"Phase D end: L_germano should be ~8.0, got {germano_d}")
        return

    # L_divergence always 10.0
    for epoch, label in [(100, "A"), (450, "B"), (800, "C"), (1250, "D")]:
        w = scheduler.get_weights(epoch)
        div_w = w.get("L_divergence", -1)
        if div_w != 10.0:
            result.fail_test(f"Phase {label}: L_divergence should be 10.0, got {div_w}")
            return

    result.pass_test(
        "Phase weights correct",
        A_nu_phys=f"{w_a['L_nu_phys']:.1f}",
        C_nu_phys=f"{w_c['L_nu_phys']:.1f}",
        D_germano=f"{germano_d:.2f}",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Test 5: Multi-Step Forward
# ──────────────────────────────────────────────────────────────────────────────

def test_multi_step_forward(result: TestResult):
    from model import INNATE3D_MixedConvection

    cfg = _make_config()
    model = INNATE3D_MixedConvection(cfg)
    model.eval()

    state = model.create_initial_condition()
    energies = []

    with torch.no_grad():
        for step in range(3):
            energy = (state.u**2 + state.v**2 + state.w**2).mean().item()
            energies.append(energy)

            if not torch.isfinite(torch.tensor(energy)):
                result.fail_test(f"Energy became non-finite at step {step}", energy=energy)
                return

            state = model(state)

    result.pass_test(
        "Multi-step forward successful",
        steps=3,
        energies=[f"{e:.6e}" for e in energies]
    )


# ──────────────────────────────────────────────────────────────────────────────
# Test 6: Parameter Count
# ──────────────────────────────────────────────────────────────────────────────

def test_parameter_count(result: TestResult):
    from model import INNATE3D_MixedConvection

    cfg = _make_config()
    model = INNATE3D_MixedConvection(cfg)

    total = model.count_parameters()
    summary = model.parameter_summary()

    # v2 expected range: ~500-600 params
    if total < 200 or total > 2000:
        result.fail_test(
            f"Parameter count {total} outside expected v2 range (200-2000)",
            total=total
        )
        return

    # MLP SGS should contribute significant params
    mlp_params = sum(v for k, v in summary.items() if 'mlp' in k.lower())

    details = {"total": total, "mlp_sgs": mlp_params}
    for key, val in sorted(summary.items()):
        details[key] = val

    result.pass_test(f"Parameter count: {total}", **details)


# ──────────────────────────────────────────────────────────────────────────────
# Test 7: Gradient Flow
# ──────────────────────────────────────────────────────────────────────────────

def test_gradient_flow(result: TestResult):
    from model import INNATE3D_MixedConvection

    cfg = _make_config()
    model = INNATE3D_MixedConvection(cfg)
    model.train()

    state = model.create_initial_condition()
    out = model(state)

    loss = out.u.pow(2).mean() + out.theta.pow(2).mean()
    loss.backward()

    # Buoyancy gradient
    has_buoy_grad = False
    if hasattr(model, 'buoyancies'):
        for b in model.buoyancies:
            if b.buoyancy_strength.grad is not None and b.buoyancy_strength.grad.abs().sum().item() > 0:
                has_buoy_grad = True
                break
    elif hasattr(model, 'buoyancy'):
        p = model.buoyancy.buoyancy_strength
        has_buoy_grad = p.grad is not None and p.grad.abs().sum().item() > 0

    # Forcing gradient
    force_param = model.forcing.amplitude
    has_force_grad = force_param.grad is not None and force_param.grad.abs().sum().item() > 0

    # EddyViscosity gradient
    has_eddy_grad = False
    if len(model.eddy_viscosities) > 0:
        for name, p in model.eddy_viscosities[0].named_parameters():
            if p.grad is not None and p.grad.abs().sum().item() > 0:
                has_eddy_grad = True
                break

    # MLP SGS gradient
    has_mlp_grad = False
    if model.mlp_sgs is not None:
        for name, p in model.mlp_sgs.named_parameters():
            if p.grad is not None and p.grad.abs().sum().item() > 0:
                has_mlp_grad = True
                break

    if not has_buoy_grad:
        result.fail_test("No gradient flowing through buoyancy")
        return

    if not has_mlp_grad and model.mlp_sgs is not None:
        result.fail_test("No gradient flowing through MLP SGS")
        return

    result.pass_test(
        "Gradient flow working",
        buoyancy="OK" if has_buoy_grad else "ZERO",
        forcing="OK" if has_force_grad else "ZERO",
        eddy="OK" if has_eddy_grad else "ZERO",
        mlp_sgs="OK" if has_mlp_grad else "ZERO/N/A",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Test 8: MLP SGS Validation
# ──────────────────────────────────────────────────────────────────────────────

def test_mlp_sgs(result: TestResult):
    from model import INNATE3D_MixedConvection

    cfg = _make_config()
    model = INNATE3D_MixedConvection(cfg)

    if model.mlp_sgs is None:
        result.fail_test("MLP SGS not initialized (use_mlp_sgs=False?)")
        return

    mlp = model.mlp_sgs
    mlp_params = sum(p.numel() for p in mlp.parameters())

    # Test MLP forward with realistic dummy inputs matching actual API:
    # forward(strain_mag, omega_mag, Ri_g, Re_normalized, layer_idx, delta)
    B, Nx, Ny, Nz = 1, cfg.domain.Nx, cfg.domain.Ny, cfg.domain.Nz
    strain_mag = torch.rand(B, Nx, Ny, Nz) * 10.0  # |S| ~ O(10)
    omega_mag = torch.rand(B, Nx, Ny, Nz) * 10.0   # |Omega| ~ O(10)
    Ri_g = torch.rand(B, Nx, Ny, Nz) * 0.01         # Ri_g ~ O(0.003)
    Re_normalized = 0.5   # Re=10000 / 20000
    layer_idx = 10
    delta = 0.0625  # dx for 96x160x64 grid

    with torch.no_grad():
        Cs, Pr_t = mlp(strain_mag, omega_mag, Ri_g, Re_normalized, layer_idx, delta)

    # Cs should be in [0.05, 0.25], Pr_t in [0.3, 1.5]
    cs_min, cs_max = Cs.min().item(), Cs.max().item()
    Pr_t_val = Pr_t.item() if Pr_t.ndim == 0 else Pr_t.mean().item()

    cs_ok = 0.04 <= cs_min and cs_max <= 0.26
    Pr_t_ok = 0.29 <= Pr_t_val <= 1.51

    if not cs_ok:
        result.fail_test(f"Cs range [{cs_min:.4f}, {cs_max:.4f}] outside [0.05, 0.25]")
        return
    if not Pr_t_ok:
        result.fail_test(f"Pr_t = {Pr_t_val:.4f} outside [0.3, 1.5]")
        return

    result.pass_test(
        f"MLP SGS OK ({mlp_params} params)",
        Cs_range=f"[{cs_min:.4f}, {cs_max:.4f}]",
        Pr_t=f"{Pr_t_val:.4f}",
        output_shape=str(list(Cs.shape)),
        mlp_params=mlp_params,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Test 9: IMEX Integration
# ──────────────────────────────────────────────────────────────────────────────

def test_imex(result: TestResult):
    from model import INNATE3D_MixedConvection

    cfg = _make_config()
    model = INNATE3D_MixedConvection(cfg)

    # Check IMEX flag
    if not model.use_imex:
        result.fail_test("IMEX integration disabled (use_imex=False)")
        return

    # Run forward and check no NaN (IMEX should be unconditionally stable for mol. diffusion)
    model.eval()
    state = model.create_initial_condition()

    with torch.no_grad():
        out = model(state)

    has_nan = any(torch.isnan(x).any().item() for x in [out.u, out.v, out.w, out.theta])

    if has_nan:
        result.fail_test("IMEX forward produced NaN")
        return

    # Energy should not blow up (IMEX stability)
    e_in = (state.u**2 + state.v**2 + state.w**2).mean().item()
    e_out = (out.u**2 + out.v**2 + out.w**2).mean().item()

    # 20 layers, energy shouldn't grow more than 100x
    if e_out > 0 and e_in > 0 and e_out / (e_in + 1e-12) > 100:
        result.fail_test(f"Energy blowup: {e_in:.4e} -> {e_out:.4e}")
        return

    result.pass_test(
        "IMEX integration stable",
        use_imex=True,
        E_in=f"{e_in:.4e}",
        E_out=f"{e_out:.4e}",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Test 10: Re Sweep (set_physics)
# ──────────────────────────────────────────────────────────────────────────────

def test_re_sweep(result: TestResult):
    from model import INNATE3D_MixedConvection

    cfg = _make_config()
    model = INNATE3D_MixedConvection(cfg)
    model.eval()

    re_values = [5000, 7000, 10000, 15000, 20000]
    Ra = 1e5
    Pr = 0.71

    for Re in re_values:
        model.set_physics(Re=Re, Ra=Ra, Pr=Pr)

        # Verify physics updated
        if abs(model.nu - 1.0/Re) > 1e-10:
            result.fail_test(f"Re={Re}: nu mismatch", expected=1.0/Re, got=model.nu)
            return
        if abs(model.Re - Re) > 1e-10:
            result.fail_test(f"Re={Re}: Re not set", got=model.Re)
            return

        # Quick forward pass
        state = model.create_initial_condition()
        with torch.no_grad():
            out = model(state)

        has_nan = any(torch.isnan(x).any().item() for x in [out.u, out.v, out.w, out.theta])
        if has_nan:
            result.fail_test(f"Re={Re}: NaN in forward pass")
            return

    result.pass_test(
        f"set_physics works for all {len(re_values)} Re values",
        re_values=str(re_values),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("INNATE v2 3D Mixed Convection - Smoke Test")
    print("=" * 80)
    print()

    tests = [
        ("1.  Import Test", test_imports),
        ("2.  Forward Pass", test_forward_pass),
        ("3.  Training Pipeline (Phases)", test_training_pipeline),
        ("4.  Phase Weights Validation", test_phase_weights),
        ("5.  Multi-Step Forward", test_multi_step_forward),
        ("6.  Parameter Count", test_parameter_count),
        ("7.  Gradient Flow", test_gradient_flow),
        ("8.  MLP SGS Validation", test_mlp_sgs),
        ("9.  IMEX Integration", test_imex),
        ("10. Re Sweep (set_physics)", test_re_sweep),
    ]

    results = []
    for name, func in tests:
        print(f"Running: {name}")
        res = run_test(name, func)
        res.print_result()
        results.append(res)

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed

    print(f"Total: {len(results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print()

    if failed > 0:
        print("Failed tests:")
        for r in results:
            if not r.passed:
                print(f"  - {r.name}: {r.message}")
        print()
        sys.exit(1)
    else:
        print("All tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
