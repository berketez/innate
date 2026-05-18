# Literature Survey: Recent Advances in Numerical Solvers for Turbulent Flow Simulations (2025-2026)

**Author:** CFD Expert Agent
**Date:** 2026-02-19
**Focus:** Methods relevant for 3D mixed convection (thermal + wind forcing), Re=5000-10000, single GPU (RTX 4090 level)

---

## Executive Summary

This survey covers six categories of recent advances in computational turbulence. The key finding is that **differentiable CFD solvers on GPU** have matured significantly in 2024-2025, with JAX and PyTorch frameworks now offering production-quality NS solvers. For Berke's target problem (3D mixed convection at Re~5000-10000 on a single RTX 4090), the most promising paths are:

1. A **pseudo-spectral solver in PyTorch/JAX** using cuFFT for periodic directions + Chebyshev for wall-normal (if bounded)
2. **IMEX time integration** (BDF3 or RK-IMEX) for handling the stiff diffusion + buoyancy coupling
3. **Online-trained ML subgrid models** via differentiable solvers for LES at higher Re

The survey identified 25+ relevant papers and 10+ open-source projects.

---

## 1. Spectral Methods and Pseudo-Spectral Solvers for Turbulent Convection

### 1.1 GPU Pseudo-Spectral DNS for Multiphase Turbulence
- **Paper:** "A GPU-ready pseudo-spectral method for direct numerical simulations of multiphase turbulence"
- **Authors:** Alessio Roccon (2024)
- **Link:** https://arxiv.org/abs/2406.01322
- **Method:** GPU-ported pseudo-spectral solver for interface-resolved DNS of drop/bubble-laden turbulent flows. Navier-Stokes + phase-field method. Uses cuFFT for spectral transforms.
- **Problem:** Large-scale DNS of multiphase turbulence on GPUs.
- **Performance:** Demonstrates efficient GPU porting of a production pseudo-spectral code. cuFFT-based 3D FFT is the dominant cost (~60-70% of total). Single-GPU performance comparable to ~100 CPU cores.
- **Open Source:** Not explicitly stated, but the methodology is well documented.
- **Relevance for Berke:** HIGH. Demonstrates the feasibility of pseudo-spectral DNS on a single GPU for 3D turbulence. The FFT-based approach maps directly to periodic boundary conditions in mixed convection.

### 1.2 NekRS: GPU-Accelerated Spectral Element NS Solver
- **Paper:** "NekRS, a GPU-Accelerated Spectral Element Navier-Stokes Solver"
- **Authors:** Paul Fischer, Stefan Kerkemeier, Misun Min, Yu-Hsiang Lan et al. (2021, but heavily updated through 2025)
- **Link:** https://arxiv.org/abs/2104.05829
- **Method:** Spectral element method (SEM) on GPUs using OCCA for portability. High-order (p=7-15) elements. SIMPLE/PISO-like pressure-velocity coupling.
- **Problem:** Thermal-fluids simulation at scale. Designed for reactor thermal-hydraulics but applicable to general convection problems.
- **Performance:** Achieves 80% of roofline on NVIDIA GPUs. Scales to 27,648 V100 GPUs. For single-GPU, handles O(10^6) spectral elements.
- **Open Source:** YES - https://github.com/Nek5000/nekRS (active development, Argonne National Lab)
- **Relevance for Berke:** MEDIUM. Excellent for complex geometry mixed convection but overkill for periodic/simple domains. The spectral element approach is more general than pseudo-spectral but slower per DOF.

### 1.3 Dedalus Project (Spectral PDE Framework)
- **Paper:** Burns, Vasil, Oishi, Lecoanet, Brown (2020, with ongoing v3 development through 2025-2026)
- **Link:** https://dedalus-project.org/
- **Method:** Sparse spectral methods for PDEs. Chebyshev (bounded) + Fourier (periodic) bases. Automatic equation parsing. Tau method for boundary conditions.
- **Problem:** General PDE solving with emphasis on fluid dynamics, convection, MHD. Heavily used for Rayleigh-Benard convection studies.
- **Performance:** MPI-parallel, CPU-based. Not natively GPU, but v3 has experimental GPU support. Used for Ra up to 10^12 on large clusters.
- **Open Source:** YES - https://github.com/DedalusProject/dedalus (MIT license)
- **Relevance for Berke:** MEDIUM-HIGH. The gold standard for spectral convection simulations. However, CPU-focused. The mathematical framework (Chebyshev + Fourier) is exactly what INNATE should reproduce.

### 1.4 Mixed Convection DNS Studies
- **Paper:** "Velocity dip in turbulent mixed convection of an open Poiseuille-Rayleigh-Benard channel"
- **Authors:** Ben-Rui Xu, Ao Xu, Heng-Dong Xi (2025)
- **Link:** https://arxiv.org/abs/2512.20977
- **Method:** 3D DNS for Poiseuille-Rayleigh-Benard (PRB) flow. Ra = 10^5 to 10^8, Pr = 0.71, Re_b = 2850.
- **Problem:** Mixed convection in open channels with free-slip upper boundary. Transition from shear-dominated to buoyancy-dominated regimes.
- **Performance:** Standard DNS on ~256^3 grids (estimated).
- **Relevance for Berke:** VERY HIGH. This is exactly the mixed convection regime Berke wants to simulate. Provides benchmark data for Re_b = 2850, which is close to Berke's target range. The velocity-dip phenomenon is a physically rich validation case.

### 1.5 Data-Driven Discovery from Convection DNS
- **Paper:** "Data-driven discovery of the equations of turbulent convection"
- **Authors:** Wareing, Roy, Golden, Grigoriev (2025)
- **Link:** https://arxiv.org/abs/2505.10109
- **Method:** SINDy and SPIDER algorithms applied to DNS data of 2D and 3D Rayleigh-Benard convection.
- **Problem:** Recovering governing equations and boundary conditions from DNS data.
- **Relevance for Berke:** MEDIUM. Demonstrates that DNS data of turbulent convection can be used to discover/validate PDE structure. Could validate INNATE's physics embedding.

---

## 2. GPU-Accelerated CFD Solvers (JAX/PyTorch/CUDA)

### 2.1 Diff-FlowFSI: JAX-Based Differentiable CFD Platform
- **Paper:** "Diff-FlowFSI: A GPU-Optimized Differentiable CFD Platform for High-Fidelity Turbulence and FSI Simulations"
- **Authors:** Xiantao Fan, Xin-Yang Liu, Meng Wang, Jian-Xun Wang (2025)
- **Link:** https://arxiv.org/abs/2505.23940
- **Method:** JAX-based, fully differentiable, GPU-accelerated FVM solver. Vectorized finite volume + immersed boundary method. Supports AD for gradient-based inverse problems.
- **Problem:** High-fidelity turbulence and fluid-structure interaction simulations. Designed for hybrid neural-CFD.
- **Performance:** GPU-native, supports forward simulation acceleration and gradient computation through the entire simulation pipeline.
- **Open Source:** Not confirmed from abstract.
- **Relevance for Berke:** HIGH. The differentiable FVM approach could be combined with INNATE-style physics embedding. The JAX framework enables end-to-end training.

### 2.2 JAX-Fluids 2.0: HPC Differentiable CFD
- **Paper:** "JAX-Fluids 2.0: Towards HPC for Differentiable CFD of Compressible Two-phase Flows"
- **Authors:** Deniz A. Bezgin, Aaron B. Buhendwa, Nikolaus A. Adams (2024)
- **Link:** https://arxiv.org/abs/2402.05193
- **Method:** Python/JAX fully-differentiable CFD solver for compressible single/two-phase flows. Parallel via JAX primitive operations.
- **Problem:** Compressible flows with machine learning integration. Supports automatic differentiation through the entire solver.
- **Performance:** Scales to 512 NVIDIA A100 GPUs. Single GPU performance excellent for moderate resolution. ~5x speedup from original JAX-Fluids.
- **Open Source:** YES - https://github.com/tumaer/JAXFLUIDS (likely available)
- **Relevance for Berke:** MEDIUM. Compressible solver (Berke needs incompressible), but the JAX differentiable framework and parallel strategy are directly transferable. The AD-through-solver capability is key for INNATE training.

### 2.3 JAX-Shock: Differentiable Shock-Capturing Solver
- **Paper:** "JAX-Shock: A Differentiable, GPU-Accelerated, Shock-Capturing Neural Solver for Compressible Flow Simulation"
- **Authors:** Bo Zhang (2026)
- **Link:** https://arxiv.org/abs/2601.04400
- **Method:** Fully differentiable, GPU-accelerated, high-order shock-capturing solver in JAX for compressible Navier-Stokes. Supports AD for optimization and parameter inference.
- **Problem:** Compressible flow with shock-solid interactions.
- **Performance:** End-to-end differentiable, GPU-native.
- **Open Source:** Not confirmed.
- **Relevance for Berke:** LOW-MEDIUM. Compressible flow focus, but demonstrates the JAX+AD+GPU paradigm that is applicable to incompressible solvers.

### 2.4 PICT: Differentiable PISO Solver in PyTorch
- **Paper:** "PICT -- A Differentiable, GPU-Accelerated Multi-Block PISO Solver for Simulation-Coupled Learning Tasks in Fluid Dynamics"
- **Authors:** Aleksandra Franz, Hao Wei, Luca Guastoni, Nils Thuerey (2025)
- **Link:** https://arxiv.org/abs/2505.16992
- **Method:** PyTorch-based differentiable pressure-implicit (PISO) solver. Multi-block structured grids. GPU-accelerated.
- **Problem:** Simulation-coupled deep learning for fluid dynamics. Enables end-to-end training through the solver.
- **Performance:** GPU-accelerated, verified against OpenFOAM benchmarks. Supports backpropagation through 100+ timesteps.
- **Open Source:** Code availability stated in paper.
- **Relevance for Berke:** VERY HIGH. This is a PyTorch-native NS solver with GPU support. The PISO algorithm handles pressure-velocity coupling (exactly what incompressible mixed convection needs). Being in PyTorch, it could interface directly with INNATE.

### 2.5 WaterLily.jl: Differentiable Fluid Solver in Julia
- **Paper:** "WaterLily.jl: A differentiable fluid simulator in Julia with fast heterogeneous execution"
- **Authors:** Gabriel D. Weymouth, Bernat Font (2023, updated 2025)
- **Link:** https://arxiv.org/abs/2304.08159
- **Method:** Julia-based incompressible viscous flow solver. Multi-dimensional, multi-platform (CPU, threaded, GPU). Backend-agnostic. Differentiable via AD.
- **Problem:** Incompressible viscous flow with optimization and ML integration.
- **Performance:** GPU execution via CUDA.jl. Compact codebase (~500 lines core). Differentiable for gradient-based optimization.
- **Open Source:** YES - https://github.com/WaterLily-jl/WaterLily.jl
- **Relevance for Berke:** LOW-MEDIUM. Julia-based, not Python. But the design philosophy (minimal, differentiable, GPU) is instructive.

### 2.6 Google JAX-CFD (Kochkov et al.)
- **Paper:** "Machine learning-accelerated computational fluid dynamics" (PNAS 2021, but the JAX-CFD codebase has been updated through 2025)
- **Authors:** Kochkov, Smith, Alieva, Wang, Brenner, Hoyer (Google Research)
- **Link:** https://github.com/google/jax-cfd
- **Method:** JAX-based pseudo-spectral solver for 2D/3D incompressible NS. Supports learned corrections (ML-augmented LES). Uses spectral methods for spatial discretization.
- **Problem:** Demonstrates ML-accelerated CFD achieving 8-10x finer resolution equivalent accuracy with 40-80x speedup.
- **Performance:** Single GPU, 2D and 3D periodic domains. Pseudo-spectral with cuFFT.
- **Open Source:** YES - https://github.com/google/jax-cfd (Apache 2.0)
- **Relevance for Berke:** VERY HIGH. This is the canonical JAX pseudo-spectral incompressible NS solver. The 3D version supports exactly the kind of turbulence simulations Berke needs. The ML-augmented LES approach is directly relevant to INNATE.

---

## 3. Lattice Boltzmann Methods for Thermal Convection

### 3.1 Fluctuating LBM with Central Moments
- **Paper:** "A fluctuating lattice Boltzmann formulation based on orthogonal central moments"
- **Authors:** Alessandro De Rosis, Yang Zhou (2026)
- **Link:** https://arxiv.org/abs/2602.16486
- **Method:** LBM with stochastic forcing in central moment space. Mode-dependent relaxation. Consistent with fluctuation-dissipation theorem.
- **Problem:** Mesoscale fluid dynamics with thermal fluctuations. Relevant for nanoscale convection and stochastic flows.
- **Performance:** Not yet benchmarked on GPU explicitly.
- **Relevance for Berke:** LOW. Mesoscale focus, not the right scale for Re=5000-10000 macroscopic mixed convection.

### 3.2 Rayleigh-Benard Convection in Emulsions via LBM
- **Paper:** "Rayleigh-Benard thermal convection in emulsions: a short review"
- **Authors:** Pelusi, Scagliarini, Sbragaglia, Bernaschi (2025)
- **Link:** https://arxiv.org/abs/2512.16830
- **Method:** LBM for thermal Rayleigh-Benard convection in emulsions. Multicomponent thermal LBM with GPU acceleration.
- **Problem:** Buoyancy-driven convection with complex rheology (emulsions). Concentration-dependent rheology from Newtonian to yield-stress.
- **Performance:** GPU-accelerated via in-house code. Handles complex fluid rheology.
- **Open Source:** Not confirmed.
- **Relevance for Berke:** LOW-MEDIUM. Demonstrates LBM can handle thermal convection, but the emulsion focus is different.

### 3.3 XLB: JAX-Based LBM Framework (Mentioned in Community)
- **Note:** The XLB (Accelerated Lattice Boltzmann) project is a JAX-based LBM framework that has gained traction in 2025. While I did not find a formal arxiv paper from the search, it is known in the community:
- **Link:** https://github.com/Autodesk/XLB
- **Method:** JAX-based 2D/3D LBM solver with GPU acceleration. Supports thermal models (D3Q19 + energy equation).
- **Performance:** Competitive with CUDA implementations. Differentiable via JAX AD.
- **Open Source:** YES - Apache 2.0 license
- **Relevance for Berke:** MEDIUM. An alternative to pseudo-spectral for thermal flows. LBM is inherently parallel and GPU-friendly. However, for Re=5000-10000 with spectral accuracy, pseudo-spectral remains superior.

### Assessment of LBM vs. Pseudo-Spectral for Berke's Problem

For 3D mixed convection at Re=5000-10000:
- **LBM Advantages:** Naturally parallel, simple boundary conditions, handles complex geometry, good GPU utilization.
- **LBM Disadvantages:** Low formal accuracy (2nd order), requires many more grid points than spectral methods for equivalent accuracy, memory-intensive (multiple distribution functions per node).
- **Verdict:** Pseudo-spectral is strongly preferred for Berke's periodic-domain mixed convection. LBM would only be worth considering for complex geometries.

---

## 4. Physics-Informed / Hybrid Solvers (Solver + ML)

### 4.1 Online Learning of SGS Models via Differentiable Solvers
- **Paper:** "Online learning of subgrid-scale models for quasi-geostrophic turbulence in planetary interiors"
- **Authors:** Hugo Frezat, T. Gastine, Alexandre Fournier (2025)
- **Link:** https://arxiv.org/abs/2511.14581
- **Method:** Online end-to-end training of neural SGS models using a pseudo-spectral differentiable solver. The dynamical solver participates in training.
- **Problem:** SGS modeling for turbulence in bounded domains (planetary interiors). Quasi-geostrophic equations.
- **Performance:** SGS model trained on ~1 turnover time remains stable and accurate for 100+ turnover times. Reproduces slow processes (jet drift) on timescales far exceeding training duration.
- **Open Source:** Not confirmed, but uses Dedalus-like framework.
- **Relevance for Berke:** VERY HIGH. This is the state-of-the-art for differentiable solver + ML hybrid. The approach is: (1) build a differentiable pseudo-spectral solver, (2) train a neural SGS model end-to-end through the solver. Directly applicable to INNATE's philosophy of physics-embedded ML.

### 4.2 ML Wall Model for LES (Rough Surfaces)
- **Paper:** "Machine-learning wall model of large-eddy simulation for low- and high-speed flows over rough surfaces"
- **Authors:** Rong Ma, Adrian Lozano-Duran (2026)
- **Link:** https://arxiv.org/abs/2601.20786
- **Method:** ANN-based wall model trained on DNS database of turbulent channel flows. 372 training cases spanning various roughness topographies.
- **Problem:** Wall-modeled LES for flows over rough surfaces. Transitional to fully rough regimes.
- **Performance:** Demonstrated accuracy across Gaussian and Weibull roughness distributions.
- **Open Source:** Not confirmed.
- **Relevance for Berke:** LOW-MEDIUM. Wall-bounded, not periodic convection. But the methodology of training on DNS data for LES closure is relevant.

### 4.3 Numerically Consistent SGS Models via Data Assimilation
- **Paper:** "Numerically Consistent Non-Boussinesq Subgrid-scale Stress Model with Enhanced Convergence"
- **Authors:** Yuenong Ling, Adrian Lozano-Duran (2026)
- **Link:** https://arxiv.org/abs/2601.20265
- **Method:** ML-based SGS stress models that are consistent with the numerical scheme. Non-Boussinesq formulation. Data assimilation approach.
- **Problem:** SGS closure for LES in turbulent boundary layers under adverse pressure gradients.
- **Performance:** Overcomes limitations of linear eddy-viscosity closures.
- **Relevance for Berke:** MEDIUM. The "numerically consistent" idea is important -- the ML model should know about the numerical discretization.

### 4.4 Neural Network SGS Models: Recursive and S4ND Approaches
- **Paper:** "A recursive neural-network-based subgrid-scale model for large eddy simulation: application to homogeneous isotropic turbulence"
- **Authors:** Cho, Park, Choi (2023, updated 2025)
- **Link:** https://arxiv.org/abs/2312.15143
- **Method:** Recursive training: (1) Train NN-SGS on filtered DNS at low Re, (2) Apply to LES at higher Re, (3) Recursively extend to even higher Re.
- **Problem:** Scaling SGS models to high Reynolds numbers without expensive DNS at each Re.
- **Performance:** Successfully extends from moderate to high Re through recursive self-training.
- **Relevance for Berke:** HIGH. The recursive approach could let INNATE be trained at Re=1000 and extended to Re=10000 through self-consistent recursion.

- **Paper:** "Subgrid Stress Modelling with Multi-dimensional State Space Sequence Models"
- **Authors:** Andy Wu, Sanjiva K. Lele (2025)
- **Link:** https://arxiv.org/abs/2511.10910
- **Method:** S4ND (Structured State Space for ND data) U-Net architecture for SGS stress prediction. Continuous representation facilitates grid-dependent modeling.
- **Problem:** SGS stress prediction with principled grid dependence.
- **Performance:** S4ND enables learning continuous representations from discrete data.
- **Relevance for Berke:** MEDIUM. Novel architecture for SGS modeling with natural grid-size adaptation.

### 4.5 Rotationally Invariant CNN SGS Models
- **Paper:** "A Priori Assessment of Rotational Invariance in Multiscale Convolutional Neural Network-Based Subgrid-Scale Model for Wall-Bounded Turbulent Flows"
- **Authors:** Bahrul Jalaali, Kie Okabayashi (2025)
- **Link:** https://arxiv.org/abs/2511.16995
- **Method:** Modified CNN architecture satisfying material objectivity (rotational invariance). Removes bias terms and batch normalization, adds spatial transformer network.
- **Problem:** Ensuring physical symmetries in learned SGS models. Previous NN-SGS models violated rotational invariance.
- **Performance:** Improved generalization compared to non-invariant models.
- **Relevance for Berke:** MEDIUM-HIGH. Enforcing physical symmetries (like Galilean and rotational invariance) in neural CFD models is directly relevant to INNATE's philosophy of embedding physics into the architecture rather than the loss function.

### 4.6 FNO+Diffusion for 3D Turbulence Prediction
- **Paper:** "Integrating Fourier Neural Operator with Diffusion Model for Autoregressive Predictions of Three-dimensional Turbulence"
- **Authors:** Jiang, Wang, Yang, Wang (2025)
- **Link:** https://arxiv.org/abs/2512.12628
- **Method:** DiAFNO = Implicit Adaptive FNO + Diffusion Model. Autoregressive 3D turbulence prediction.
- **Problem:** Reliable long-term autoregressive prediction of 3D turbulence -- previous methods suffered from error accumulation and energy spectral distortion.
- **Performance:** Improved spectral accuracy over pure FNO, more stable autoregressive rollout.
- **Relevance for Berke:** MEDIUM. Not a solver-ML hybrid per se, but shows that combining structured architectures (FNO) with generative models (diffusion) can improve 3D turbulence predictions.

### 4.7 NORi: ML-Augmented Ocean Boundary Layer Parameterization
- **Paper:** "NORi: An ML-Augmented Ocean Boundary Layer Parameterization"
- **Authors:** Xin Kai Lee, Ali Ramadhan, Andre Souza, Gregory LeClaire Wagner (2025)
- **Link:** https://arxiv.org/abs/2512.04452
- **Method:** Neural ODEs for Richardson number-dependent diffusivity/viscosity. Physical closure augmented with NODEs for entrainment through boundary layer base.
- **Problem:** Ocean boundary layer turbulence parameterization. Physics-based structure with ML corrections.
- **Performance:** Stable long-term integration. Generalizes to unseen conditions.
- **Relevance for Berke:** MEDIUM. The "physics-based structure + ML correction" approach is exactly INNATE's philosophy. The Richardson number controls the buoyancy-shear balance in mixed convection.

---

## 5. Novel Time Integration Schemes for Turbulent Flows

### 5.1 IMEX BDF3 for Incompressible NS
- **Paper:** "Long-time stability and convergence analysis of an IMEX BDF3 scheme for 2-D incompressible Navier-Stokes equation"
- **Authors:** Kelong Cheng, Jingwei Sun, Hong Zhang (2025)
- **Link:** https://arxiv.org/abs/2512.17182
- **Method:** Third-order BDF3 time stepping with Adams-Bashforth extrapolation for the nonlinear convection term. Fourier pseudo-spectral spatial discretization. IMEX splitting: implicit for diffusion (linear, stiff), explicit for convection (nonlinear).
- **Problem:** Long-time stability of high-order IMEX schemes for incompressible NS. Proves unconditional energy stability.
- **Performance:** Third-order temporal accuracy with unconditional stability (no CFL constraint on diffusion). Pseudo-spectral spatial accuracy.
- **Open Source:** Not confirmed.
- **Relevance for Berke:** VERY HIGH. This is exactly the time integration scheme needed for mixed convection:
  - BDF3 implicit for diffusion (allows large dt for viscous/thermal terms)
  - Adams-Bashforth explicit for convection (simple, no nonlinear solves)
  - Fourier pseudo-spectral spatial discretization (matches INNATE's spectral approach)
  - Proven long-time stability (crucial for statistical turbulence)

### 5.2 High-Order IMEX Methods Assessment
- **Paper:** "Assessment of high-order IMEX methods for incompressible flow"
- **Authors:** Montadhar Guesmi, Martina Grotteschi, Jorg Stiller (2021, relevant review)
- **Link:** https://arxiv.org/abs/2112.04167
- **Method:** Comparison of semi-implicit Runge-Kutta (RK) and spectral deferred correction (SDC) up to 6th order. Partitioned implicit-explicit RK scheme.
- **Problem:** Systematic comparison of IMEX methods for incompressible NS with high-order DG spatial discretization.
- **Performance:** RK-IMEX schemes up to order 4 are practical. SDC can reach order 6 but with higher cost. Order 3-4 is the sweet spot for turbulence.
- **Relevance for Berke:** HIGH. Provides guidance on choosing the right IMEX order. For Re=5000-10000, IMEX order 3 (BDF3 or RK3-IMEX) is optimal.

### 5.3 Efficient Exponential Integrators for Compressible Viscous Flow (Reference)
- **Paper:** "Efficient Time Integration of IMEX Type using Exponential Integrators for Compressible, Viscous Flow Simulation"
- **Authors:** (2016, but foundational for current methods)
- **Link:** Referenced in Semantic Scholar results
- **Method:** Exponential integrators combined with IMEX splitting. The linear part (diffusion) is solved exactly via matrix exponential.
- **Performance:** Allows much larger time steps than standard explicit methods for stiff problems.
- **Relevance for Berke:** MEDIUM. The exponential integrator concept is theoretically appealing for thermal convection (where the diffusion operator is linear and stiff), but practical implementations for 3D pseudo-spectral codes are still uncommon. IMEX-BDF is more practical.

### Time Integration Recommendation for Berke's Problem

For 3D mixed convection (Boussinesq NS + energy equation) with pseudo-spectral spatial discretization:

**Best choice: IMEX-BDF3 or IMEX-RK3 (Ascher-Ruuth-Spiteri)**

The equations are:
```
du/dt = -u.grad(u) - grad(p) + nu*laplacian(u) + Ra*Pr*T*e_z  [momentum]
dT/dt = -u.grad(T) + kappa*laplacian(T)                        [energy]
div(u) = 0                                                       [continuity]
```

IMEX splitting:
- **Implicit (stiff):** nu*laplacian(u), kappa*laplacian(T) -- these become diagonal in Fourier space, so "implicit" = trivially exact
- **Explicit (nonlinear):** -u.grad(u), -u.grad(T), buoyancy term

With Fourier pseudo-spectral, the implicit part is diagonal (each wavenumber independent), so the "implicit solve" is just a division -- essentially free. This makes IMEX spectral methods extremely efficient.

CFL constraint: Only from the explicit convection term.
```
dt < C * dx / U_max, where C depends on the explicit scheme (C~1.7 for AB3, C~1.0 for RK3)
```

---

## 6. Differentiable CFD Solvers for Optimization and Inverse Problems

### 6.1 Diff-FlowFSI (Already covered in Section 2.1)
- JAX-based, fully differentiable, FVM + immersed boundary. Best for complex geometry problems.

### 6.2 JAX-Fluids 2.0 (Already covered in Section 2.2)
- JAX-based, fully differentiable, compressible flows. Best for compressible flow optimization.

### 6.3 PICT (Already covered in Section 2.4)
- PyTorch-based, PISO solver, differentiable. Best for incompressible flow learning tasks.

### 6.4 Adjoint Method on Flow Maps
- **Paper:** "An Adjoint Method for Differentiable Fluid Simulation on Flow Maps"
- **Authors:** Zhiqi Li, Jinjin He, Barnabas Borcsok, Taiyuan Zhang (2025)
- **Link:** https://arxiv.org/abs/2511.01259
- **Method:** Novel adjoint solver using bidirectional flow maps. Forward pass: transport impulse variables from initial to current frame. Backward pass: same flow map propagates adjoint variables backward.
- **Problem:** Efficient gradient computation through fluid simulations for optimization/control.
- **Performance:** More memory-efficient than standard checkpointing-based adjoint methods.
- **Relevance for Berke:** MEDIUM. The flow-map adjoint approach is elegant but specialized.

### 6.5 Property-Preserving Operator Learning for Incompressible Flows
- **Paper:** "Fluids You Can Trust: Property-Preserving Operator Learning for Incompressible Flows"
- **Authors:** Ramansh Sharma, Matthew Lowery, Houman Owhadi, Varun Shankar (2026)
- **Link:** https://arxiv.org/abs/2602.15472
- **Method:** Kernel-based operator learning that exactly enforces incompressibility, periodicity, and turbulence properties. Maps inputs to expansion coefficients of divergence-free basis functions.
- **Problem:** Existing neural operators fail to exactly enforce physical properties. This method guarantees incompressibility by construction.
- **Performance:** Exact divergence-free outputs. Comparable accuracy to unconstrained methods on benchmarks.
- **Relevance for Berke:** HIGH. The "exactly enforce physics by construction" approach is philosophically identical to INNATE. Instead of physics loss, the architecture guarantees the constraint. Could be combined with INNATE's approach.

### 6.6 Geometric Neural Operators via Lie Groups
- **Paper:** "Geometric Neural Operators via Lie Group-Constrained Latent Dynamics"
- **Authors:** Jiaquan Zhang et al. (2026)
- **Link:** https://arxiv.org/abs/2602.16209
- **Method:** Constrain neural operator latent space to Lie group manifolds. Low-rank Lie algebra parameterization. Preserves conservation laws and geometric structure.
- **Problem:** Instability in multi-layer iteration and long-horizon rollout due to unconstrained latent updates.
- **Performance:** Improved stability for long-horizon predictions.
- **Relevance for Berke:** MEDIUM-HIGH. The Lie group constraint on latent dynamics is a mathematically rigorous way to embed conservation laws. Could inform INNATE's architecture design.

---

## Summary: Open-Source Project Landscape

| Project | Language | Method | GPU | Differentiable | Best For | Link |
|---------|----------|--------|-----|----------------|----------|------|
| **JAX-CFD** | Python/JAX | Pseudo-spectral FDM | Yes | Yes | 2D/3D incompressible periodic | github.com/google/jax-cfd |
| **JAX-Fluids** | Python/JAX | FVM | Yes | Yes | Compressible two-phase | github.com/tumaer/JAXFLUIDS |
| **PICT** | Python/PyTorch | FVM PISO | Yes | Yes | Incompressible with learning | (paper link) |
| **NekRS** | C++/OCCA | Spectral element | Yes | No | Complex geometry thermal-fluids | github.com/Nek5000/nekRS |
| **Dedalus** | Python | Spectral (Chebyshev+Fourier) | Partial | No | RB convection, MHD | dedalus-project.org |
| **WaterLily.jl** | Julia | FDM | Yes | Yes | Incompressible viscous | github.com/WaterLily-jl |
| **XLB** | Python/JAX | Lattice Boltzmann | Yes | Yes | LBM thermal flows | github.com/Autodesk/XLB |
| **PhiFlow** | Python (TF/PyTorch/JAX) | FDM | Yes | Yes | Differentiable physics | github.com/tum-pbs/PhiFlow |
| **SpectralDNS** | Python | Pseudo-spectral | Partial | No | Homogeneous turbulence | github.com/spectralDNS |

---

## Recommendations for Berke's 3D Mixed Convection Project

### Priority 1: Reference Solver (for generating training data and validation)
**Build a pseudo-spectral DNS solver in PyTorch** using:
- Fourier basis (periodic x,y directions) + Fourier or Chebyshev (z direction depending on BC)
- IMEX-BDF3 or IMEX-RK3 time integration (implicit diffusion, explicit convection)
- Boussinesq approximation for buoyancy
- `torch.fft.fftn` / `torch.fft.ifftn` for spectral transforms (maps to cuFFT on GPU)
- De-aliasing via 2/3 rule

Grid requirement estimate for Re=5000, single GPU (RTX 4090, 24GB VRAM):
- Kolmogorov scale: eta ~ L * Re^(-3/4) ~ L * 5000^(-0.75) ~ L/594
- DNS grid: N ~ (L/eta) ~ 594, so ~600^3 ~ 216M points
- Memory for 6 fields (u,v,w,p,T + workspace): ~216M * 6 * 8 bytes * ~4 (spectral+physical) ~ 41 GB -- too large for single GPU
- Practical approach: N=256^3 at Re=2000-3000, or N=384^3 at Re=5000 with reduced precision (fp32)

### Priority 2: Look at JAX-CFD as Reference Architecture
Google's JAX-CFD is the closest existing code to what Berke needs. Study its:
- Pseudo-spectral implementation
- Time stepping
- How it integrates ML corrections

### Priority 3: INNATE Integration with Differentiable Solver
The Online SGS Learning paper (Frezat et al., 2025) is the clearest blueprint:
1. Build differentiable pseudo-spectral solver (Forward)
2. Run coarse LES through it
3. INNATE predicts SGS correction
4. Loss = deviation from filtered DNS
5. Backprop through solver to train INNATE

### Priority 4: For Higher Re (5000-10000), Use LES + INNATE
Full DNS at Re=10000 on a single RTX 4090 is infeasible (needs ~1000^3 grid, ~8B DOF).
Instead:
- DNS at Re=1000-2000 for training data (256^3, feasible)
- LES at Re=5000-10000 with INNATE-learned SGS model (128^3, very feasible)
- Validate via statistics (energy spectrum, Nusselt number, velocity profiles)

---

## Key Takeaways

1. **The field has converged on JAX and PyTorch as the dominant frameworks** for differentiable CFD. CUDA-only solvers are being replaced.

2. **IMEX-BDF3 with pseudo-spectral spatial discretization** is the emerging standard for time integration in spectral turbulence codes. The implicit part is trivially diagonal in Fourier space.

3. **Online training of SGS models through differentiable solvers** is the most impactful recent development. The Frezat et al. (2025) paper shows this works even in bounded domains.

4. **Physics-embedded architectures** (like INNATE) are gaining traction over physics-loss approaches. The "Fluids You Can Trust" paper (2026) demonstrates exact constraint enforcement through architecture design.

5. **Single-GPU DNS is feasible up to Re~3000** on RTX 4090 for 3D periodic domains. Beyond that, LES with learned closures is the practical path.

6. **Mixed convection DNS data exists** for validation: Xu et al. (2025) provide data for PRB flow at Ra up to 10^8 and Re_b=2850.
