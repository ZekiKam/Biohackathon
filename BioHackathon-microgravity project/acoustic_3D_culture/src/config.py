"""
config.py — shared physical constants and geometry parameters.

Edit this file to match your experimental rig.  Every other script imports from here.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Acoustic medium: air at 20 °C, 1 atm
# ---------------------------------------------------------------------------
C0       = 343.0           # speed of sound in air  [m/s]
RHO0     = 1.204           # density of air          [kg/m³]
ETA_AIR  = 1.81e-5         # dynamic viscosity       [Pa·s]

# ---------------------------------------------------------------------------
# Droplet: water spheroid encapsulating a GBM spheroid
# ---------------------------------------------------------------------------
RHO_DROP = 1000.0          # water density           [kg/m³]
C_DROP   = 1484.0          # speed of sound in water [m/s]
SIGMA    = 0.072           # surface tension (water–air) [N/m]
A_DROP   = 1.0e-3          # droplet radius          [m]  (1 mm)
V_DROP   = (4/3) * np.pi * A_DROP**3   # droplet volume [m³]
M_DROP   = RHO_DROP * V_DROP            # droplet mass   [kg]

# Gorkov contrast factors (water droplet in air)
KAPPA0 = 1.0 / (RHO0   * C0**2)        # air compressibility  [Pa⁻¹]
KAPPA1 = 1.0 / (RHO_DROP * C_DROP**2)  # water compressibility [Pa⁻¹]
F1     = 1.0 - KAPPA1 / KAPPA0         # monopole (compressibility) contrast ≈ +1
F2     = 2.0*(RHO_DROP - RHO0) / (2.0*RHO_DROP + RHO0)  # dipole (density) contrast ≈ +1

# Stokes drag coefficient  6πηa
STOKES_COEFF = 6.0 * np.pi * ETA_AIR * A_DROP   # [N·s/m]

# ---------------------------------------------------------------------------
# Transducer array geometry
# ---------------------------------------------------------------------------
FREQ       = 40_000.0                   # operating frequency   [Hz]
OMEGA      = 2.0 * np.pi * FREQ        # angular frequency     [rad/s]
WAVELENGTH = C0 / FREQ                  # ≈ 8.575 mm
K          = 2.0 * np.pi / WAVELENGTH  # wave number           [rad/m]

R_SPHERE   = 0.10          # sphere radius           [m]   (10 cm)
N_TRANS    = 256           # number of transducers
P0_TRANS   = 10.0          # transducer source pressure amplitude  [Pa]

# Desired trap position (world coordinates)
TRAP_POS   = np.array([0.0, 0.0, 0.0])  # centroid of the sphere

# ---------------------------------------------------------------------------
# GS-PAT phase computation
# ---------------------------------------------------------------------------
GS_ITERATIONS    = 300    # Gerchberg–Saxton iterations
NODE_RING_RADIUS = WAVELENGTH * 0.6   # radius of the pressure-antinode target ring
NODE_RING_N      = 24                  # evaluation points on the ring
NODE_RING_LAYERS = 3                   # axial layers of ring points

# ---------------------------------------------------------------------------
# Droplet simulation (Stage 3)
# ---------------------------------------------------------------------------
N_DROPLETS   = 6           # number of droplets entering the sphere
T_END        = 5.0         # simulation duration [s]
DT_OUTPUT    = 1e-3        # output time interval [s]
ENTRY_RADIUS = R_SPHERE * 0.5   # droplets start at this radius from centre
                                # (0.5× R avoids near-wall near-field artefacts)

# ---------------------------------------------------------------------------
# Pressure field evaluation grid (Stage 2)
# ---------------------------------------------------------------------------
GRID_N = 150        # points per axis for 3-D evaluation grid
GRID_LIM = R_SPHERE * 1.05   # half-extent of evaluated volume [m]

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
import pathlib
OUTPUT_DIR  = pathlib.Path(__file__).parent.parent / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"
