"""
Stage 1 — Per-transducer Phase & Amplitude Computation (GS-PAT)
================================================================
Computes the complex activation  q_i = A_i exp(i phi_i)  for each of the
N_TRANS transducers so that the resulting acoustic field creates a
NODE TRAP (pressure minimum) at TRAP_POS.

Algorithm
---------
We use a version of the Gerchberg–Saxton Phased Array Transducers (GS-PAT)
algorithm (Plasencia et al., ACM Trans. Graph. 2020).  The key idea for a
*node* trap is:

  1. Define a set of "ring" evaluation points *around* the trap (not at it).
  2. Target high pressure at those ring points -> the trap is the local minimum
     in the resulting standing-wave interference pattern.
  3. Iteratively alternate:
       a. Propagate transducer activations to field points.
       b. Replace field-point amplitudes with target amplitudes (keep phases).
       c. Back-propagate to transducers.
       d. Normalise transducer activations to unit amplitude.

Physics note
------------
For water droplets in air (f1 ≈ +1, f2 ≈ +1) the Gorkov potential is
minimised at *pressure nodes*.  A plain focus (all transducers in phase)
creates a pressure antinode at the target — which *repels* the droplets.
The node-trap pattern returned by this script creates a pressure minimum
surrounded by a ring of antinodes, providing the restoring 3-D gradient
force needed for stable levitation.

Output
------
output/transducer_config.npz  ->  positions, normals, phases, amplitudes
output/figures/transducer_phases.png
"""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import trange

import config as cfg

# ---------------------------------------------------------------------------
# Helper: Fibonacci-sphere transducer layout
# ---------------------------------------------------------------------------

def fibonacci_sphere(n: int, radius: float):
    """
    Place n points uniformly on a sphere of given radius using the
    Fibonacci (golden-angle) method.  Returns positions [n×3] and
    inward-pointing unit normals [n×3].
    """
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    idx = np.arange(n, dtype=float)
    y   = 1.0 - (idx / (n - 1)) * 2.0          # y in [-1, 1]
    r   = np.sqrt(np.clip(1.0 - y**2, 0.0, None))
    theta = golden_angle * idx

    x   = np.cos(theta) * r
    z   = np.sin(theta) * r
    pos = radius * np.column_stack([x, y, z])
    normals = -pos / radius                       # pointing inward
    return pos, normals


# ---------------------------------------------------------------------------
# Helper: Green's-function propagation matrix
# ---------------------------------------------------------------------------

def build_propagation_matrix(trans_pos: np.ndarray,
                             field_pts: np.ndarray,
                             k: float) -> np.ndarray:
    """
    H[i, j]  = monopole Green's function from transducer i to field point j.

    p(x_j) = Σ_i q_i * H[i,j]

    Shape: (n_trans, n_field)
    """
    n_t = trans_pos.shape[0]
    n_f = field_pts.shape[0]
    H   = np.zeros((n_t, n_f), dtype=complex)

    for i in range(n_t):
        diff = field_pts - trans_pos[i]          # (n_f, 3)
        r    = np.linalg.norm(diff, axis=1)       # (n_f,)
        r    = np.maximum(r, 1e-9)
        H[i] = np.exp(1j * k * r) / (4.0 * np.pi * r)

    return H


# ---------------------------------------------------------------------------
# Helper: build a ring of evaluation points around the trap
# ---------------------------------------------------------------------------

def build_ring_evaluation_points(trap_pos: np.ndarray,
                                  ring_radius: float,
                                  ring_n: int,
                                  ring_layers: int,
                                  wavelength: float) -> np.ndarray:
    """
    Return a set of evaluation points that surround the trap_pos.

    These are arranged in concentric rings in the XY plane displaced axially
    by multiples of λ/4.  Targeting high pressure at these points with GS-PAT
    forces a pressure minimum at trap_pos.
    """
    pts = []
    axial_offsets = np.linspace(-wavelength/4, wavelength/4, ring_layers)
    phi_vals      = np.linspace(0, 2*np.pi, ring_n, endpoint=False)

    for dz in axial_offsets:
        for phi in phi_vals:
            pt = trap_pos + np.array([
                ring_radius * np.cos(phi),
                ring_radius * np.sin(phi),
                dz
            ])
            pts.append(pt)

    return np.array(pts)


# ---------------------------------------------------------------------------
# Core GS-PAT iterator
# ---------------------------------------------------------------------------

def gs_pat_node_trap(trans_pos: np.ndarray,
                     ring_pts: np.ndarray,
                     k: float,
                     n_iter: int = 300,
                     verbose: bool = True) -> np.ndarray:
    """
    Run GS-PAT to create a node trap.

    Parameters
    ----------
    trans_pos  : (N_trans, 3) transducer positions
    ring_pts   : (N_ring,  3) evaluation points that should have HIGH pressure
                               (surrounding the trap -> creates a node at trap)
    k          : wave number [rad/m]
    n_iter     : number of Gerchberg–Saxton iterations

    Returns
    -------
    q : (N_trans,) complex transducer activations, |q_i| = 1 (unit amplitude)
    """
    n_t = trans_pos.shape[0]
    n_f = ring_pts.shape[0]

    # Build propagation matrix  H : (n_trans, n_ring)
    if verbose:
        print("  Building propagation matrix …")
    H = build_propagation_matrix(trans_pos, ring_pts, k)   # (n_t, n_f)

    # Target: uniform amplitude at every ring point, random initial phase
    rng           = np.random.default_rng(seed=42)
    target_amp    = np.ones(n_f)                             # desired |p|
    target_phase  = rng.uniform(0, 2*np.pi, n_f)
    p_target      = target_amp * np.exp(1j * target_phase)  # (n_f,)

    # Initial transducer activations (unit amplitude, random phase)
    phi_init = rng.uniform(0, 2*np.pi, n_t)
    q        = np.exp(1j * phi_init)                        # (n_t,)

    iter_range = trange(n_iter, desc="  GS-PAT") if verbose else range(n_iter)

    for _ in iter_range:
        # Forward: pressure at ring evaluation points
        p = H.T @ q                                         # (n_f,)

        # Apply amplitude constraint, keep phases
        p_amp       = np.abs(p)
        valid       = p_amp > 1e-15
        p_new       = np.where(valid, target_amp * p / p_amp, p_target)

        # Backward: update transducer activations
        q = H.conj() @ p_new                               # (n_t,)

        # Normalise to unit amplitude
        q_amp = np.abs(q)
        q     = np.where(q_amp > 1e-15, q / q_amp, np.exp(1j * rng.uniform(0, 2*np.pi, n_t)))

    return q


# ---------------------------------------------------------------------------
# Verification: compute pressure at trap and ring
# ---------------------------------------------------------------------------

def verify_trap(q, trans_pos, trap_pos, ring_pts, k):
    """
    Print pressure magnitude at trap and at ring points to verify the node.
    Returns pressure at trap (complex scalar).
    """
    H_trap = build_propagation_matrix(trans_pos, trap_pos.reshape(1, 3), k)
    p_trap = float(np.abs(H_trap.T @ q)[0])

    H_ring  = build_propagation_matrix(trans_pos, ring_pts, k)
    p_ring  = np.abs(H_ring.T @ q)

    print(f"\n  Trap pressure magnitude : {p_trap:.4f}  (should be low / near 0)")
    print(f"  Ring pressure magnitude : mean={p_ring.mean():.4f}, "
          f"max={p_ring.max():.4f}  (should be high)")
    contrast = p_ring.mean() / max(p_trap, 1e-9)
    print(f"  Trap-to-ring contrast   : 1 : {contrast:.1f}")
    return p_trap


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(save=True):
    print("=" * 60)
    print("Stage 1 — GS-PAT Phase Computation")
    print("=" * 60)
    print(f"  Frequency  : {cfg.FREQ/1e3:.1f} kHz")
    print(f"  Wavelength : {cfg.WAVELENGTH*1e3:.2f} mm")
    print(f"  Sphere R   : {cfg.R_SPHERE*1e2:.0f} cm")
    print(f"  Transducers: {cfg.N_TRANS}")
    print(f"  Trap pos   : {cfg.TRAP_POS}")

    # --- Array geometry ---
    trans_pos, trans_normals = fibonacci_sphere(cfg.N_TRANS, cfg.R_SPHERE)

    # --- Ring evaluation points that GS-PAT will target ---
    ring_pts = build_ring_evaluation_points(
        cfg.TRAP_POS,
        cfg.NODE_RING_RADIUS,
        cfg.NODE_RING_N,
        cfg.NODE_RING_LAYERS,
        cfg.WAVELENGTH,
    )
    print(f"\n  Ring evaluation points : {len(ring_pts)}")

    # --- Run GS-PAT ---
    print("\nRunning GS-PAT …")
    q = gs_pat_node_trap(trans_pos, ring_pts, cfg.K, n_iter=cfg.GS_ITERATIONS)

    phases     = np.angle(q)      # (N_trans,)
    amplitudes = np.abs(q)        # all ≈ 1

    # --- Verify ---
    verify_trap(q, trans_pos, cfg.TRAP_POS, ring_pts, cfg.K)

    # --- Save ---
    if save:
        cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cfg.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(
            cfg.OUTPUT_DIR / "transducer_config.npz",
            positions  = trans_pos,
            normals    = trans_normals,
            phases     = phases,
            amplitudes = amplitudes,
            q_complex  = q,
        )
        print(f"\n  Saved -> {cfg.OUTPUT_DIR / 'transducer_config.npz'}")

    # --- Plot ---
    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection="3d")
    sc  = ax.scatter(
        trans_pos[:, 0]*1e2, trans_pos[:, 1]*1e2, trans_pos[:, 2]*1e2,
        c=np.degrees(phases), cmap="hsv", s=35, vmin=-180, vmax=180,
    )
    cb = plt.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
    cb.set_label("Phase (°)")
    ax.scatter(*cfg.TRAP_POS*1e2, c="red", s=250, marker="*",
               zorder=10, label="Trap (node)")
    ax.scatter(ring_pts[:, 0]*1e2, ring_pts[:, 1]*1e2, ring_pts[:, 2]*1e2,
               c="lime", s=20, alpha=0.5, label="Ring targets")
    ax.set_xlabel("X [cm]"); ax.set_ylabel("Y [cm]"); ax.set_zlabel("Z [cm]")
    ax.set_title(f"GS-PAT Node Trap  —  {cfg.N_TRANS} transducers, {cfg.FREQ/1e3:.0f} kHz")
    ax.legend(loc="upper left")

    if save:
        fig.savefig(cfg.FIGURES_DIR / "transducer_phases.png", dpi=150,
                    bbox_inches="tight")
        print(f"  Saved -> {cfg.FIGURES_DIR / 'transducer_phases.png'}")
    plt.close(fig)

    print("\nStage 1 complete.")
    return trans_pos, phases, amplitudes, q


if __name__ == "__main__":
    main()
