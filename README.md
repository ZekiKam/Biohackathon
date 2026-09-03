# Glioblastoma Acoustic Assembly Simulator

## Project Purpose

Glioblastoma (GBM) is an extremely aggressive brain cancer with very limited
treatment options, principally surgery and chemotherapy. The supplied project
brief describes GBM as affecting approximately 3.19 people per 100,000
worldwide, representing around 32% of primary brain tumours, with an average
mortality period of approximately 12-18 months. These figures should be
checked against the specific epidemiological source used in any publication
or presentation.

Drug discovery is hindered by the gap between laboratory models and patient
tumours. Standard 2D cultures grow GBM cells flat on plastic. That geometry
can silence or alter the drug-resistance and invasion programmes that make GBM
lethal, so a drug that appears effective in a dish may fail repeatedly in
patients.

## Why 3D culture and microgravity matter

Three-dimensional culture allows cells to form structures more like those in
the brain. It can restore more complex cell organization, allow a hypoxic
core to develop, and prevent cells from simply adhering to a flat plate. These
properties can increase tumour-like behaviour, tumourigenicity, and invasion,
which helps explain why results from 2D drug testing often translate poorly to
clinical trials.

Ground-based 3D cultures still experience gravity. Spheroids can settle,
disaggregate, and develop inconsistent morphologies. The project concept is
that microgravity removes this gravity-driven settling artefact, potentially
supporting more uniform and reproducible organoids, larger and more complex
spheroids, and a more faithful spatial arrangement of tumour cells. The
intended model is one that better preserves tumour architecture, hypoxia
gradients, cell-cell interactions, immune context, and aggressive molecular
signatures. These are scientific motivations for the design; this repository's
simulation does not itself create a microgravity environment or validate gene
expression and immune fidelity.

## Why cymatics and acoustic manipulation matter

Cymatics uses sound fields to create organized pressure and radiation-force
patterns. A transducer phase array is an electronically controlled collection
of sound emitters. By changing the phase and amplitude of the emitters, the
array can create acoustic traps at chosen positions.

This makes acoustic manipulation precise and contactless: particles can be
moved without a mechanical probe. In a future biological rig, the pressure
field could be designed to control the location, shape, and cell-cell pressure
of a tumouroid. In this repository, those ideas are represented by simulated
particles, acoustic focal points, and simplified cell-cell interaction rules.

## Proposed application and rig concept

The broader Wav.io concept is a brain-cancer microgravity modeller: a
purpose-fit functional test rig combining a 3D GBM culture environment with
acoustic positioning. A possible workflow is to form tumour spheroids in a
hydrogel, transport the culture to a microgravity environment, introduce a
tailored growth medium, and use acoustic pressure gradients to aggregate or
position the floating spheroids into a tumouroid that can be observed and
tested.

The proposed rig concept includes a spherical transducer arrangement, optical
access for imaging and treatment tracking, fluid or hydrogel injection ports,
and sealed pneumatic dosing. A feedback controller could use real-time
observations to optimize the acoustic parameters. The exact rig dimensions,
frequency, number of transducers, and control algorithm are design parameters
for the hardware and future application; they are not all implemented by this
particle simulation.

The intended impact is a more uniform and reproducible GBM model, improved
drug-efficacy evaluation, and faster therapeutic screening. Microgravity and
acoustic levitation are complementary: microgravity supplies the biological
context, while acoustic forces provide controllable positioning and assembly.

## Overview

Simulates how ultrasonic transducer arrays drive cell assembly into
tissue-like structures using analytical acoustic radiation forces computed by
[AcousTools](https://github.com/BristolMyersSquibb/acoustools)

**It models:**
- Particles/cells moving in a 3D volume
- Acoustic radiation forces calculated with AcousTools’ Gor’kov-force model
- Overdamped motion, where particles move step-by-step in the direction of the net force
- Short-range cell interaction, using optional Lennard-Jones attraction and repulsion
- Permanent fusion, where particles closer than merge_distance become one cluster
- Shape and assembly measurements, including cluster count, largest cluster, assembly fraction, rod score, and sphere score
- Visualization, through static 3D plots, timelapses, and 16-second GIF animations.
  
**Disclaimer:** 

This simulator models acoustic forces only, with simplified particle interactions. It does not currently model microgravity. 
The particles move according to:
- Acoustic radiation forces from AcousTools
- Optional Lennard-Jones attraction/repulsion
- Cluster merging when particles are close enough
Microgravity is currently part of the project’s scientific motivation and proposed application, but not part of the implemented simulation physics. The current code effectively assumes gravity is absent or ignored, rather than explicitly simulating microgravity.



It is a simplified computational model, not a complete biological tumor model. It does not simulate actual GBM biology, cell growth, gene expression, immune cells, fluid dynamics, drug response, or real microgravity. Its main purpose is to test whether designed acoustic fields could position and assemble particles into useful 3D geometries such as clusters, rods, or spherical structures.

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

---

# What the Code Does

The code models particles as cell or droplet-like points in metres. AcousTools
provides the acoustic radiation force generated by a transducer board. The
simulation moves particles according to that force, optionally adds a
simplified Lennard-Jones adhesion/repulsion force, and permanently merges
particles that come within a specified distance. It then saves plots and
animations showing the resulting assembly.

The model is a computational demonstration rather than a complete biological
or microgravity model. In particular, it does not model cell growth, gene
expression, fluid flow, immune cells, hydrogel phase change, or experimentally
calibrated microgravity conditions.

## `acous_simulation.py`

This module contains the reusable simulation class and plotting functions. It
does not run the six experiments by itself.

### `CellAssemblySimulation.__init__()`

Stores the acoustic activation pattern and transducer board, sets the
simulation parameters, seeds NumPy's random generator, creates initial cell
positions, and gives every cell its own cluster ID.

### `_init_positions()`

Creates the initial 3D positions. `random` uses a Gaussian distribution,
`settled` places particles near the bottom of the simulation box, and `seed`
starts them in a tighter organoid-like group. Positions are clipped to the
simulation bounds.

### `_get_acoustic_forces()`

Converts the current particle positions into a PyTorch tensor and calls
AcousTools `compute_force()`. It returns one acoustic force vector per
particle.

### `_auto_calibrate_drag()`

Estimates a drag value from the largest initial acoustic force and the desired
maximum step. When Lennard-Jones forces are enabled without a specified energy,
it also estimates the Lennard-Jones well depth from the acoustic force scale.

### `_compute_lj_forces()`

Computes pairwise Lennard-Jones forces between cluster centres. Particles closer
than the equilibrium scale repel, nearby particles attract, and pairs beyond
the cutoff are ignored. The cluster forces are broadcast back to member
particles.

### `_merge_clusters()`

Computes all pairwise particle distances and permanently gives two clusters the
same cluster ID when any members are closer than `merge_distance`. Repeated
passes allow connected groups to become one cluster.

### `_move_clusters()`

Sums acoustic forces within each cluster, adds optional Lennard-Jones forces,
compresses the force range using `power_alpha`, moves all members of a cluster
together, and clips positions to the simulation box.

### `_metrics()`

Calculates the number of clusters, largest cluster, and assembly fraction. It
also uses PCA to calculate `rod_score` and compares the variation in particle
radius from the centre to calculate `sphere_score`.

### `run()`

Runs the requested number of steps. It calculates the initial forces, records
the initial state, repeatedly computes forces and moves clusters, enables
merging after the warm-up period, and stores snapshots and metrics at the
requested interval. It returns `metrics_history`.

### `_log()`

Prints the current step, number of clusters, assembly fraction, largest cluster,
rod score, and sphere score.

### `render_spheres_vedo()`

Renders the final particle positions as 3D spheres using Vedo and saves a PNG.
Cluster IDs determine the sphere colours.

### `render_timelapse_vedo()`

Renders selected snapshots as separate Vedo images, combines them into one
matplotlib image, and removes temporary files. The current runner uses the
matplotlib timelapse instead.

### `plot_assembly_mpl()`

Creates a static matplotlib 3D scatter plot of one assembly state.

### `plot_timelapse_mpl()`

Creates a multi-panel static figure showing selected snapshots from the
simulation.

### `save_animation_mpl()`

Creates a 3D animated GIF from the stored snapshots. It interpolates particle
positions between recorded snapshots and writes 320 frames at 20 frames per
second by default, giving a 16-second playback. This changes presentation
speed only; it does not add physical simulation steps.

### `plot_metrics()`

Plots cluster count, assembly percentage, and largest-cluster size for several
experiments on one comparison figure.

### `_draw_clusters_mpl()`

Draws coloured particle clusters on a matplotlib 3D axis and applies the
millimetre scale, bounds, labels, and tick formatting.

## `run_acous.py`

This is the executable experiment driver. It imports the simulation and
visualization functions above, constructs six acoustic activation patterns,
runs each experiment, and saves the resulting figures and animations.

### `fibonacci_sphere()`

Generates evenly distributed points on the surface of a sphere using a
Fibonacci or golden-ratio lattice. The points become the target locations for
the sphere-trap experiment.

### `run_experiment()`

Creates a `CellAssemblySimulation` with the supplied parameters, runs it, logs
the elapsed time, and returns both the simulation object and its metrics.

### `main()`

Parses `--quick` and `--output-dir`, loads the AcousTools transducer board and
solvers, defines common parameters, runs all six experiments, generates final
plots and 16-second animations, and prints the final metrics summary.

## The six acoustic configurations

All six experiments use the same random seed and common base settings in the
current runner: 50 particles, `init_spread=0.015` m (15 mm),
`merge_distance=0.004` m (4 mm), `bounds=0.04` m (40 mm half-extent),
`max_step=0.001` m (1 mm), `dt=0.001` s, and 600 steps. The `--quick` option
uses 200 steps. The normal warm-up is 300 steps; particles move during the
warm-up but cannot merge until it ends. Experiments 5 and 6 use settled
initialization and enable Lennard-Jones interaction.

### 1. Standing Wave

All transducers receive the same complex activation, so they are driven in
phase. The resulting standing-wave field creates repeated acoustic force
regions rather than one explicitly specified focus. It is useful as a baseline
for observing how a global standing wave organizes particles and where
pressure nodes naturally collect them.

### 2. Single Focus plus Levitation Signature

Weighted Gerchberg-Saxton (`wgs`) creates one target at the origin. The optional
`add_lev_sig` call adds the AcousTools levitation signature when available. This
configuration is the most direct test of concentrating particles into one
central acoustic trap and is the best candidate for maximizing a single large
cluster.

### 3. Three-Point Focus

`wgs` creates three target points at approximately `(-10, 10, 0)`,
`(0, 0, 0)`, and `(10, -10, 0)` mm. It is useful for testing whether the field
can create and maintain multiple separate assembly sites, for example to form
several tissue regions or compare different local environments.

### 4. Twin Trap

Two WGS targets are placed on the Z-axis at approximately `+10` and `-10` mm,
with the levitation signature attempted as well. This tests a two-site
configuration and is useful for studying separated assemblies, symmetric
organization, and controlled spacing along one axis.

### 5. Rod Trap

Seven WGS targets are placed along the Z-axis from approximately `-15` to
`+15` mm. Particles start in the settled mode, and Lennard-Jones forces provide
short-range adhesion and repulsion. This configuration is designed to form an
elongated, tubular, or rod-like tissue structure and should be evaluated with
both largest-cluster results and `rod_score`.

### 6. Sphere Trap

Sixteen target points are generated on a sphere of radius 12 mm using
`fibonacci_sphere()`. WGS creates the acoustic activation for these targets.
Settled initialization and Lennard-Jones interaction are enabled. The purpose
is to test a spherical shell or spheroid-like arrangement and evaluate it with
`sphere_score`, while recognizing that the simulation's score is a geometric
proxy rather than a biological validation.

## How to interpret the outputs

`assembly_fraction` is the fraction of particles that belong to a cluster with
at least two particles. It is not the fraction contained in the single largest
cluster. `largest_cluster` is the number of particles in the biggest cluster.
`rod_score` measures elongation using PCA, while `sphere_score` measures how
similar the particles' distances from the centre are. A high assembly fraction
and high largest cluster indicate aggregation; rod and sphere scores indicate
the shape of the overall particle arrangement.

The normal runner writes to `cell_assembly_sim/output_v2/`. For each experiment
it produces a final-state PNG, a static timelapse PNG, and a 16-second GIF such
as `1_standing_wave_animation.gif`. It also writes `assembly_metrics.png` and
`final_comparison.png`. If Vedo cannot render, the final-state image falls back
to matplotlib.

# Tools Used

## Required Python packages

- **PyTorch**: Stores transducer activations and positions as tensors and is
    used by AcousTools for differentiable acoustic-field calculations.
- **AcousTools**: Supplies the transducer board, weighted Gerchberg-Saxton
    (`wgs`) hologram solver, levitation signature helper, and analytical
    Gor'kov-force function `compute_force()`.
- **NumPy**: Generates initial positions and Fibonacci-sphere targets, stores
    simulation arrays, computes distances, forces, statistics, and metrics.
- **SciPy**: Provides `pdist` and `squareform` for cluster-distance checks.
- **Matplotlib**: Creates static 3D plots, metric comparisons, fallback
    renderings, and the animation frames.
- **Pillow**: Provides the GIF writer used by matplotlib's
    `PillowWriter`.
- **Vedo**: Creates higher-quality off-screen 3D sphere renderings of final
    particle states when available.
- **VTK**: Backend used by Vedo for 3D geometry and off-screen rendering.

## Python standard-library modules used

- **`os`**: Builds paths, creates output directories, and removes temporary
    rendering files.
- **`sys`**: Adds the script directory to the import path.
- **`argparse`**: Implements `--quick` and `--output-dir` command-line options.
- **`time`**: Measures and reports experiment duration.
- **`pathlib.Path`**: Imported for path-oriented filesystem handling.

The project does not require a separate database or web server. The generated
PNG and GIF files are local outputs, and the simulation inputs are created in
code from the transducer board and acoustic target definitions.
