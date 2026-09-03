# 3D Cell Assembly Simulation with Acoustic Forces

## Overview

Simulates how **ultrasonic transducer arrays** drive **cell assembly into
tissue-like structures** using analytical acoustic radiation forces computed by
[AcousTools](https://github.com/BristolMyersSquibb/acoustools).

Uses analytical Gor'kov forces via `compute_force()` + sticky-particle merge model.

## Quick Start

```bash
cd cell_assembly_sim

# Quick test (~30 s)
python run_acous.py --quick

# Full run (~90 s, 500 steps × 4 experiments)
python run_acous.py
```

Output images → `output_v2/`

## Experiments

- Assembly: percentage of cells that joined a cluster with at least one other cell
- Largest Cluster: size of the biggest merged group
- Rod score: measures how elongated the particles are using PCA (1.0: particles are arranged almost along one straight line)
- Sphere score: measures how similarly distant particles are from the arrangement’s center (1.0: particles lie at nearly the same radius, like a spherical shell)

| # | Configuration | Assembly | Largest Cluster | Rod Score | Sphere Score |
|---|---|---|---|---|---|
| 1 | Standing wave (all transducers in phase) | 24 % | 3 / 50 | 0.427 |0.636 |
| 2 | Single focus + levitation signature | 40 % | 8 / 50 | 0.589 | 0.590 |
| 3 | 3-point focus | 20 % | 4 / 50 | 0.495 | 0.632 |
| 4 | Twin trap (two z-axis foci) | 40 % | 6 / 50 | 0.595 | 0.594 |
| 5 | Rod trap | 78 % | 11 / 50 | 0.577 | 0.356 |
| 6 | Sphere trap | 70 % | 7 / 50 | 0.578 | 0.528 |

## Physics

### Acoustic radiation force

The Gor'kov potential for a small sphere in an acoustic field:

    U = V [ K₁ ⟨p²⟩ − K₂ ⟨|v|²⟩ ]

    F = −∇U

AcousTools' `compute_force()` evaluates this analytically (including full
second derivatives of the pressure field via PyTorch autograd) — no grid
interpolation or finite differences.

### Cell-cell interaction (sticky particle model)

When two cells (or any members of two different clusters) come within the
merge distance (~4 mm for 1 mm beads), they permanently fuse into a rigid
cluster. The cluster moves as a single body under the sum of its members'
acoustic forces. This models secondary Bjerknes forces + contact adhesion.

### Overdamped dynamics

    v = F_total / γ

Drag γ is auto-calibrated at step 0:

    γ = |F|_max × Δt / Δx_max

This keeps motion stable regardless of force magnitude.

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `n_cells` | 50 | Number of cell-particles |
| `init_spread` | 8 mm | Std dev of random initial positions |
| `merge_distance` | 4 mm | Cluster merge threshold |
| `max_step` | 1 mm | Max displacement per time step |
| `dt` | 0.001 | Time step (s) |
| `bounds` | 40 mm | Half-extent of simulation box |
| `n_steps` | 500 | Number of simulation steps |

## Output Files

| File | Description |
|---|---|
| `*_spheres.png` | vedo 3D sphere render of final state |
| `*_timelapse.png` | Matplotlib time-evolution panels |
| `*_animation.gif` | 16-second animated playback of the recorded snapshots |
| `assembly_metrics.png` | Cluster count, assembly %, largest cluster vs step |
| `final_comparison.png` | 2×2 comparison grid of all experiments |

## Dependencies

- AcousTools (+ PyTorch)
- NumPy, SciPy
- vedo (+ VTK)
- Matplotlib, Pillow
