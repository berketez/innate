"""
Lid-Driven Cavity Flow Test - INNATE Benchmark

Klasik CFD benchmark problemi:
- Kare kavite, no-slip duvarlar
- Üst kapak sabit hızla hareket eder (u=1, v=0)
- Re = 100, 400, 1000 için Ghia et al. 1982 referans verisi

Kullanım:
    python -m tests.cavity.lid_driven_cavity --Re 100 --resolution 64
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import sys

# Scipy sparse solver
try:
    from scipy.sparse import diags, lil_matrix
    from scipy.sparse.linalg import spsolve
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not found, using slow Jacobi solver")

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from innate import DEVICE


# =============================================================================
# GHIA ET AL. 1982 REFERENCE DATA
# =============================================================================

GHIA_Y = np.array([0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719,
                   0.2813, 0.4531, 0.5000, 0.6172, 0.7344, 0.8516,
                   0.9531, 0.9609, 0.9688, 0.9766, 1.0000])

GHIA_U_RE100 = np.array([0.00000, -0.03717, -0.04192, -0.04775, -0.06434, -0.10150,
                          -0.15662, -0.21090, -0.20581, -0.13641, 0.00332, 0.23151,
                          0.68717, 0.73722, 0.78871, 0.84123, 1.00000])

GHIA_U_RE400 = np.array([0.00000, -0.08186, -0.09266, -0.10338, -0.14612, -0.24299,
                          -0.32726, -0.17119, -0.11477, 0.02135, 0.16256, 0.29093,
                          0.55892, 0.61756, 0.68439, 0.75837, 1.00000])

GHIA_U_RE1000 = np.array([0.00000, -0.18109, -0.20196, -0.22220, -0.29730, -0.38289,
                           -0.27805, -0.10648, -0.06080, 0.05702, 0.18719, 0.33304,
                           0.46604, 0.51117, 0.57492, 0.65928, 1.00000])

GHIA_X = np.array([0.0000, 0.0625, 0.0703, 0.0781, 0.0938, 0.1563,
                   0.2266, 0.2344, 0.5000, 0.8047, 0.8594, 0.9063,
                   0.9453, 0.9531, 0.9609, 0.9688, 1.0000])

GHIA_V_RE100 = np.array([0.00000, 0.09233, 0.10091, 0.10890, 0.12317, 0.16077,
                          0.17507, 0.17527, 0.05454, -0.24533, -0.22445, -0.16914,
                          -0.10313, -0.08864, -0.07391, -0.05906, 0.00000])

GHIA_V_RE400 = np.array([0.00000, 0.18360, 0.19713, 0.20920, 0.22965, 0.28124,
                          0.30203, 0.30174, 0.05186, -0.38598, -0.44993, -0.23827,
                          -0.22847, -0.19254, -0.15663, -0.12146, 0.00000])

GHIA_V_RE1000 = np.array([0.00000, 0.27485, 0.29012, 0.30353, 0.32627, 0.37095,
                           0.33075, 0.32235, 0.02526, -0.31966, -0.42665, -0.51550,
                           -0.39188, -0.33714, -0.27669, -0.21388, 0.00000])


def get_ghia_data(Re: float):
    """Reynolds sayısına göre Ghia verisi döndür"""
    if Re == 100:
        return GHIA_Y, GHIA_U_RE100, GHIA_X, GHIA_V_RE100
    elif Re == 400:
        return GHIA_Y, GHIA_U_RE400, GHIA_X, GHIA_V_RE400
    elif Re == 1000:
        return GHIA_Y, GHIA_U_RE1000, GHIA_X, GHIA_V_RE1000
    else:
        return GHIA_Y, GHIA_U_RE100, GHIA_X, GHIA_V_RE100


# =============================================================================
# SIMPLE FINITE DIFFERENCE CAVITY SOLVER
# =============================================================================

class CavitySolver:
    """
    Simple finite difference solver for lid-driven cavity.
    Uses explicit time stepping with projection method.
    """
    def __init__(self, resolution: int, Re: float, device=DEVICE):
        self.N = resolution
        self.Re = Re
        self.nu = 1.0 / Re
        self.device = device

        # Grid
        self.dx = 1.0 / (resolution - 1)
        self.dy = self.dx

        # Initialize fields
        self.u = torch.zeros(resolution, resolution, device=device)
        self.v = torch.zeros(resolution, resolution, device=device)
        self.p = torch.zeros(resolution, resolution, device=device)

        # Set lid velocity
        self.u[0, :] = 1.0  # Top lid moves with u=1

    def apply_velocity_bc(self):
        """Apply velocity boundary conditions"""
        # Top lid: u=1, v=0
        self.u[0, :] = 1.0
        self.v[0, :] = 0.0

        # Bottom wall: u=0, v=0
        self.u[-1, :] = 0.0
        self.v[-1, :] = 0.0

        # Left wall: u=0, v=0
        self.u[:, 0] = 0.0
        self.v[:, 0] = 0.0

        # Right wall: u=0, v=0
        self.u[:, -1] = 0.0
        self.v[:, -1] = 0.0

    def compute_divergence(self):
        """Compute divergence of velocity field"""
        div = torch.zeros_like(self.u)
        # Central difference for interior
        div[1:-1, 1:-1] = (
            (self.u[1:-1, 2:] - self.u[1:-1, :-2]) / (2 * self.dx) +
            (self.v[2:, 1:-1] - self.v[:-2, 1:-1]) / (2 * self.dy)
        )
        return div

    def build_laplacian_matrix(self):
        """Build sparse Laplacian matrix for Poisson equation with Neumann BC"""
        N = self.N
        dx2 = self.dx ** 2
        n_interior = (N - 2) ** 2

        # Build sparse matrix for interior points
        A = lil_matrix((n_interior, n_interior))

        def idx(i, j):
            """Convert 2D interior index to 1D"""
            return (i - 1) * (N - 2) + (j - 1)

        for i in range(1, N - 1):
            for j in range(1, N - 1):
                k = idx(i, j)
                A[k, k] = -4.0 / dx2

                # Left neighbor
                if j > 1:
                    A[k, idx(i, j - 1)] = 1.0 / dx2
                else:
                    A[k, k] += 1.0 / dx2  # Neumann BC

                # Right neighbor
                if j < N - 2:
                    A[k, idx(i, j + 1)] = 1.0 / dx2
                else:
                    A[k, k] += 1.0 / dx2  # Neumann BC

                # Top neighbor
                if i > 1:
                    A[k, idx(i - 1, j)] = 1.0 / dx2
                else:
                    A[k, k] += 1.0 / dx2  # Neumann BC

                # Bottom neighbor
                if i < N - 2:
                    A[k, idx(i + 1, j)] = 1.0 / dx2
                else:
                    A[k, k] += 1.0 / dx2  # Neumann BC

        # Fix one point to remove null space (pressure gauge)
        A[0, :] = 0
        A[0, 0] = 1.0

        return A.tocsr()

    def pressure_poisson_sparse(self, div):
        """Solve Poisson equation using sparse direct solver"""
        N = self.N

        # Build matrix if not cached
        if not hasattr(self, '_laplacian_matrix'):
            self._laplacian_matrix = self.build_laplacian_matrix()

        # Extract interior RHS and flatten
        rhs = div[1:-1, 1:-1].cpu().numpy().flatten()
        rhs[0] = 0.0  # Fix pressure at one point

        # Solve
        p_interior = spsolve(self._laplacian_matrix, rhs)

        # Reshape and pad with boundary values
        p_np = np.zeros((N, N))
        p_np[1:-1, 1:-1] = p_interior.reshape(N - 2, N - 2)

        # Apply Neumann BC
        p_np[0, :] = p_np[1, :]
        p_np[-1, :] = p_np[-2, :]
        p_np[:, 0] = p_np[:, 1]
        p_np[:, -1] = p_np[:, -2]

        # Remove mean
        p_np = p_np - p_np.mean()

        p = torch.from_numpy(p_np).float().to(self.device)
        self.p = p
        return p

    def pressure_poisson_jacobi(self, div, max_iter=500):
        """Jacobi iteration for Poisson equation with Neumann BC"""
        p = self.p.clone()  # Use previous pressure as initial guess
        dx2 = self.dx ** 2

        for _ in range(max_iter):
            p_new = 0.25 * (
                p[1:-1, 2:] + p[1:-1, :-2] +
                p[2:, 1:-1] + p[:-2, 1:-1] -
                dx2 * div[1:-1, 1:-1]
            )
            p[1:-1, 1:-1] = p_new

            # Neumann BC
            p[0, :] = p[1, :]
            p[-1, :] = p[-2, :]
            p[:, 0] = p[:, 1]
            p[:, -1] = p[:, -2]
            p = p - p.mean()

        self.p = p
        return p

    def pressure_poisson(self, div):
        """Main Poisson solver - uses sparse direct solver if available"""
        if HAS_SCIPY:
            return self.pressure_poisson_sparse(div)
        else:
            return self.pressure_poisson_jacobi(div)

    def project_velocity(self, dt):
        """Make velocity divergence-free using projection"""
        # Compute divergence
        div = self.compute_divergence()

        # Solve: ∇²p = div/dt
        # Then: u = u - dt * ∇p
        # Combined: ∇²φ = div, u = u - ∇φ (where φ = dt*p)
        phi = self.pressure_poisson(div)

        # Correct velocity: u = u - ∇φ
        self.u[1:-1, 1:-1] -= (phi[1:-1, 2:] - phi[1:-1, :-2]) / (2 * self.dx)
        self.v[1:-1, 1:-1] -= (phi[2:, 1:-1] - phi[:-2, 1:-1]) / (2 * self.dy)

        # Re-apply BCs
        self.apply_velocity_bc()

    def compute_rhs(self):
        """Compute RHS of momentum equation: -advection + diffusion"""
        u, v = self.u, self.v
        dx, dy = self.dx, self.dy
        nu = self.nu

        # Allocate
        du_dt = torch.zeros_like(u)
        dv_dt = torch.zeros_like(v)

        # Interior points only
        # Advection: -(u*du/dx + v*du/dy) for u
        #            -(u*dv/dx + v*dv/dy) for v
        u_c = u[1:-1, 1:-1]
        v_c = v[1:-1, 1:-1]

        # du/dx, du/dy
        du_dx = (u[1:-1, 2:] - u[1:-1, :-2]) / (2 * dx)
        du_dy = (u[2:, 1:-1] - u[:-2, 1:-1]) / (2 * dy)

        # dv/dx, dv/dy
        dv_dx = (v[1:-1, 2:] - v[1:-1, :-2]) / (2 * dx)
        dv_dy = (v[2:, 1:-1] - v[:-2, 1:-1]) / (2 * dy)

        # Advection
        adv_u = u_c * du_dx + v_c * du_dy
        adv_v = u_c * dv_dx + v_c * dv_dy

        # Diffusion: nu * (d2u/dx2 + d2u/dy2)
        d2u_dx2 = (u[1:-1, 2:] - 2*u[1:-1, 1:-1] + u[1:-1, :-2]) / dx**2
        d2u_dy2 = (u[2:, 1:-1] - 2*u[1:-1, 1:-1] + u[:-2, 1:-1]) / dy**2

        d2v_dx2 = (v[1:-1, 2:] - 2*v[1:-1, 1:-1] + v[1:-1, :-2]) / dx**2
        d2v_dy2 = (v[2:, 1:-1] - 2*v[1:-1, 1:-1] + v[:-2, 1:-1]) / dy**2

        diff_u = nu * (d2u_dx2 + d2u_dy2)
        diff_v = nu * (d2v_dx2 + d2v_dy2)

        # RHS
        du_dt[1:-1, 1:-1] = -adv_u + diff_u
        dv_dt[1:-1, 1:-1] = -adv_v + diff_v

        return du_dt, dv_dt

    def step(self, dt):
        """Advance one time step using Forward Euler + projection"""
        # Compute RHS: du/dt = -advection + diffusion
        du_dt, dv_dt = self.compute_rhs()

        # Forward Euler update (interior only)
        self.u[1:-1, 1:-1] += dt * du_dt[1:-1, 1:-1]
        self.v[1:-1, 1:-1] += dt * dv_dt[1:-1, 1:-1]
        self.apply_velocity_bc()

        # Projection to make divergence-free
        self.project_velocity(dt)

    def simulate(self, num_steps: int, dt: float, print_every: int = 100):
        """Run simulation"""
        for step in range(num_steps):
            self.step(dt)

            if step % print_every == 0:
                div = self.compute_divergence()
                max_div = div[1:-1, 1:-1].abs().max().item()
                max_u = self.u[1:-1, 1:-1].abs().max().item()
                print(f"Step {step}/{num_steps}, max|div|={max_div:.2e}, max|u|={max_u:.4f}")

                # Check for NaN
                if torch.isnan(self.u).any() or torch.isnan(self.v).any():
                    print("ERROR: NaN detected!")
                    break

        return self.u, self.v, self.p


def extract_centerline_profiles(u, v, resolution):
    """Extract centerline velocity profiles"""
    u_np = u.cpu().numpy()
    v_np = v.cpu().numpy()

    # u along vertical centerline (x=0.5)
    mid_x = resolution // 2
    u_centerline = u_np[:, mid_x]
    y_coords = np.linspace(1, 0, resolution)  # Top to bottom

    # v along horizontal centerline (y=0.5)
    mid_y = resolution // 2
    v_centerline = v_np[mid_y, :]
    x_coords = np.linspace(0, 1, resolution)

    return y_coords, u_centerline, x_coords, v_centerline


def compute_error(y_sim, u_sim, y_ref, u_ref):
    """Compute RMSE against reference data"""
    u_interp = np.interp(y_ref, y_sim[::-1], u_sim[::-1])
    error = np.sqrt(np.mean((u_interp - u_ref)**2))
    return error


def plot_results(y_sim, u_sim, x_sim, v_sim, Re, save_path=None):
    """Plot results and compare with Ghia data"""
    y_ref, u_ref, x_ref, v_ref = get_ghia_data(Re)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(u_sim, y_sim, 'b-', linewidth=2, label='INNATE')
    axes[0].plot(u_ref, y_ref, 'ro', markersize=8, label='Ghia et al. 1982')
    axes[0].set_xlabel('u-velocity')
    axes[0].set_ylabel('y')
    axes[0].set_title(f'u along vertical centerline (Re={Re})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x_sim, v_sim, 'b-', linewidth=2, label='INNATE')
    axes[1].plot(x_ref, v_ref, 'ro', markersize=8, label='Ghia et al. 1982')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('v-velocity')
    axes[1].set_title(f'v along horizontal centerline (Re={Re})')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to {save_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Lid-Driven Cavity Flow Test')
    parser.add_argument('--Re', type=float, default=100, help='Reynolds number')
    parser.add_argument('--resolution', type=int, default=64, help='Grid resolution')
    parser.add_argument('--steps', type=int, default=10000, help='Number of time steps')
    parser.add_argument('--dt', type=float, default=0.001, help='Time step size')
    args = parser.parse_args()

    print("=" * 60)
    print("LID-DRIVEN CAVITY - INNATE BENCHMARK")
    print("=" * 60)
    print(f"Re = {args.Re}")
    print(f"Resolution = {args.resolution}")
    print(f"Steps = {args.steps}")
    print(f"dt = {args.dt}")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    # Create solver
    solver = CavitySolver(args.resolution, args.Re, DEVICE)

    # Run simulation
    print("\nRunning simulation...")
    u, v, p = solver.simulate(args.steps, args.dt, print_every=args.steps // 10)

    # Extract profiles
    y_sim, u_centerline, x_sim, v_centerline = extract_centerline_profiles(
        u, v, args.resolution
    )

    # Compute errors
    y_ref, u_ref, x_ref, v_ref = get_ghia_data(args.Re)
    u_error = compute_error(y_sim, u_centerline, y_ref, u_ref)
    v_error = compute_error(x_sim, v_centerline, x_ref, v_ref)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"u-velocity RMSE vs Ghia: {u_error:.4f}")
    print(f"v-velocity RMSE vs Ghia: {v_error:.4f}")
    print(f"Average RMSE: {(u_error + v_error) / 2:.4f}")

    # Plot
    save_dir = Path(__file__).parent / "results"
    save_dir.mkdir(exist_ok=True)
    save_path = save_dir / f"cavity_Re{int(args.Re)}_N{args.resolution}.png"

    plot_results(y_sim, u_centerline, x_sim, v_centerline, args.Re, save_path)

    # Pass/Fail
    threshold = 0.15
    avg_error = (u_error + v_error) / 2
    if avg_error < threshold:
        print(f"\n[PASS] Average error {avg_error:.4f} < {threshold}")
    else:
        print(f"\n[FAIL] Average error {avg_error:.4f} >= {threshold}")


if __name__ == "__main__":
    main()
