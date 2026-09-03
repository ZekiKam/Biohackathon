# Acoustic 3D GBM Tumour-Culture Simulation Pipeline

## System Overview

A spherical array of acoustic phase transducers generates a configurable pressure field inside a fluid-filled chamber. GBM spheroids, each encapsulated in a water droplet, are extruded through the sphere wall. Cymatics (standing-wave cymatics) directs the droplets to a common trap point, where their aqueous shells merge by droplet coalescence, assembling the component spheroids into a single tumoroid. This replicates the microgravity-like, 3D, dynamic environment of the human body far more faithfully than 2D monolayers or static 3D gels.

---

## Revised & Optimised Pipeline

The original proposal is retained but reordered and tightly integrated so the output of each stage feeds the next as binary NumPy arrays, and **all stages run from Python** (Windows-native) except Stage 4 (coalescence), which uses Basilisk on WSL 2.

```
Stage 1 → Stage 2 → Stage 3 ───→ Stage 4 (Basilisk / WSL2)
  (phases)  (field)  (tracks)         (coalescence)
     │          │        │                  │
     └──────────┴────────┴──────────────────┴── Stage 5 (visualisation)
```

| Stage | Task | Tool | Platform |
|-------|------|------|----------|
| 1 | Per-transducer phase & amplitude (trap computation) | Python / GS-PAT | Windows |
| 2 | Full 3-D acoustic pressure field | k-wave-python *or* analytical | Windows |
| 3 | Gorkov potential, radiation force, droplet trajectories | Python / SciPy | Windows |
| 4 | Droplet coalescence after convergence | Basilisk C | WSL 2 |
| 5 | Publication-quality visualisation | ParaView + matplotlib | Windows |

**Key changes from the original proposal:**
- Stage 1 (phase computation) now comes *before* Stage 2 (field simulation) — you must know the phase pattern before you can compute the field.
- GSPAT is replaced with a pure-Python GS-PAT implementation; no C++ compilation required on Windows.
- Basilisk is kept for Stage 4 (it is genuinely the best free tool for droplet coalescence) but a Python SciPy fallback is provided for rapid prototyping.
- All data exchange uses `.npz` files in `output/` so any stage can be re-run independently.

---

## Physics Summary

### Stage 1 — Trap Computation (GS-PAT)

GS-PAT (Gerchberg–Saxton Phased Array Transducers) iteratively solves for per-transducer complex activations $q_i = A_i e^{i\phi_i}$ by cycling between the transducer domain and a set of target field points.

Propagation from transducer $i$ at position $\mathbf{r}_i$ to field point $\mathbf{x}$ (monopole free-space Green's function):

$$H_{ij} = \frac{e^{ik|\mathbf{x}_j - \mathbf{r}_i|}}{4\pi|\mathbf{x}_j - \mathbf{r}_i|}$$

where $k = 2\pi f/c_0$.

The algorithm alternately applies the amplitude constraint at the field points and re-normalises the transducer activations to unit amplitude. For a **node trap** at position $\mathbf{x}_0$ (the type needed to stably levitate water droplets in air), the target pressure at a ring of surrounding evaluation points is set to a high value while the target at $\mathbf{x}_0$ itself is zero — this forces a pressure minimum (node) at the trap.

### Stage 2 — Acoustic Pressure Field

For free-space propagation (valid here since the sphere contains air and the wavelength $\lambda = c_0/f \approx 8.6\,\text{mm}$ at 40 kHz is well resolved):

$$p(\mathbf{x}) = \sum_{i=1}^{N} q_i \frac{e^{ik|\mathbf{x}-\mathbf{r}_i|}}{4\pi|\mathbf{x}-\mathbf{r}_i|}$$

k-Wave solves the full linear wave equation including boundary effects; the analytical superposition is used as the fast default.

### Stage 3 — Gorkov Potential & Radiation Force

For a spherical droplet of volume $V_p = \tfrac{4}{3}\pi a^3$, density $\rho_p$, and sound speed $c_p$ immersed in a medium ($\rho_0, c_0$), the Gorkov radiation-pressure potential (Gorkov 1962; Bruus 2012) is:

$$U_\text{rad}(\mathbf{x}) = V_p \left[\frac{f_1}{4\rho_0 c_0^2}|\hat{p}|^2 - \frac{3f_2}{4\omega^2\rho_0}|\nabla\hat{p}|^2\right]$$

where $\hat{p}$ is the complex pressure amplitude, $\omega = 2\pi f$, and the contrast factors are:

$$f_1 = 1 - \frac{\rho_0 c_0^2}{\rho_p c_p^2}, \qquad f_2 = \frac{2(\rho_p - \rho_0)}{2\rho_p + \rho_0}$$

For a water droplet ($\rho_p = 1000\,\text{kg/m}^3$, $c_p = 1484\,\text{m/s}$) in air ($\rho_0 = 1.204\,\text{kg/m}^3$, $c_0 = 343\,\text{m/s}$): $f_1 \approx +1$, $f_2 \approx +1$.

**Important:** water droplets in air migrate to **pressure nodes** (minima of $U_\text{rad}$). A simple in-phase focus creates a pressure antinode (maximum) at the centre, which repels dense droplets. The GS-PAT node-trap phase pattern is therefore essential.

The acoustic radiation force is:
$$\mathbf{F}_\text{rad} = -\nabla U_\text{rad}$$

Droplet equations of motion (Stokes drag regime, $Re \ll 1$):
$$m\ddot{\mathbf{x}} = \mathbf{F}_\text{rad} - 6\pi\eta a\,\dot{\mathbf{x}}$$

where $\eta = 1.81\times10^{-5}\,\text{Pa·s}$ is the dynamic viscosity of air and $a$ is the droplet radius.

### Stage 4 — Droplet Coalescence (Basilisk)

Basilisk solves the two-phase Navier–Stokes equations with surface tension (Volume-of-Fluid method). The simulation starts from the impact conditions exported by Stage 3: droplet radii, relative impact velocity $v_\text{impact}$, and impact parameter $b$.

The Weber number $We = \rho_p v_\text{impact}^2 D / \sigma$ (where $D = 2a$ and $\sigma = 0.072\,\text{N/m}$ for water–air) determines the outcome:
- $We < 5$: gentle coalescence (expected in our system)
- $5 < We < 20$: stretching separation
- $We > 20$: reflexive separation / shattering

At 40 kHz, typical levitation velocities are $\mathcal{O}(1\,\text{mm/s})$, giving $We \ll 1$ — clean coalescence is expected.

### Stage 5 — Visualisation

- Pressure field: volumetric rendering in ParaView (VTK export) or 2-D cross-section heatmaps in matplotlib
- Gorkov potential: isosurface of $U_\text{rad}$ showing the trap geometry
- Droplet trajectories: 3-D animated line plots (matplotlib + FuncAnimation)
- Coalescence: ParaView animation from Basilisk VTK output

---

## Directory Structure

```
acoustic_3D_culture/
├── pipeline.md               ← this file
├── requirements.txt
├── setup.ps1                 ← Windows environment setup
├── run_pipeline.py           ← runs all stages in sequence
├── src/
│   ├── config.py             ← shared physical constants & geometry
│   ├── 01_phase_computation.py
│   ├── 02_acoustic_field.py
│   ├── 03_gorkov_tracking.py
│   ├── 04_coalescence/
│   │   ├── droplet_coalescence.c   ← Basilisk simulation
│   │   ├── Makefile
│   │   ├── postprocess.py
│   │   └── README_wsl.md
│   └── 05_visualize.py
└── output/
    ├── transducer_config.npz
    ├── pressure_field.npz
    ├── gorkov_potential.npz
    ├── trajectories.npz
    ├── coalescence/          ← Basilisk VTK output
    └── figures/
```

---

## How to Run

### 1. Environment Setup (Windows PowerShell)

```powershell
.\setup.ps1
```

This creates a virtual environment, installs Python dependencies, and downloads k-Wave binaries.

### 2. Run the Full Pipeline

```powershell
python run_pipeline.py
```

Or run individual stages:

```powershell
python src/01_phase_computation.py   # ~seconds
python src/02_acoustic_field.py      # ~minutes (analytical); ~30 min (k-Wave 3D)
python src/03_gorkov_tracking.py     # ~minutes
python src/05_visualize.py           # after stages 1–3
```

### 3. Coalescence (Basilisk, requires WSL 2)

```powershell
wsl bash src/04_coalescence/build_and_run.sh
```

Results are written to `output/coalescence/` and can be visualised in Windows via ParaView.

---

## Tuning Parameters

All key parameters are in `src/config.py`. The most important ones to adjust for your experimental rig:

| Parameter | Default | Notes |
|-----------|---------|-------|
| `FREQ` | 40 000 Hz | Transducer frequency |
| `R_SPHERE` | 0.10 m | Sphere radius (10 cm) |
| `N_TRANS` | 256 | Number of transducers |
| `DROPLET_RADIUS` | 1.0 mm | Water droplet radius |
| `TRAP_POS` | (0,0,0) | Desired trap location |
| `N_DROPLETS` | 6 | Simulated droplets entering the sphere |
| `GS_ITERATIONS` | 300 | GS-PAT convergence iterations |

---

## Software References

| Tool | Version | Reference |
|------|---------|-----------|
| k-wave-python | ≥ 0.3 | Treeby & Cox, J. Biomed. Eng. 2010; Simson 2023 |
| Basilisk | current | Popinet, J. Comput. Phys. 2009 |
| GS-PAT algorithm | — | Plasencia et al., ACM Trans. Graph. 2020 |
| Gorkov potential | — | Gorkov, Sov. Phys. Doklady 1962; Bruus, Lab Chip 2012 |
| ParaView | ≥ 5.11 | Ahrens et al., Visualization Handbook 2005 |
