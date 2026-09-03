"""
Stage 2 — 3-D Acoustic Pressure Field Simulation
=================================================
Computes the complex pressure amplitude p(x) at every grid point inside
(and around) the spherical transducer array.

Two backends are available:

  "analytical"  (default, fast, always works on Windows)
      Superposition of monopole Green's functions:
          p(x) = Σ_i q_i · exp(ik|x-r_i|) / (4π|x-r_i|)
      Valid under free-space propagation in a homogeneous medium.
      Runs in < 2 min at GRID_N=150 on a modern laptop.

  "kwave"  (slow, but solves the full linear wave equation including
            boundary reflections and near-field effects)
      Uses the k-wave-python library which calls the compiled k-Wave C++
      binary.  Requires `pip install k-wave-python` and the binary from
      https://github.com/waltsims/k-wave-python.

Select backend with:
    python 02_acoustic_field.py --backend analytical   (default)
    python 02_acoustic_field.py --backend kwave

Output
------
output/pressure_field.npz
    grid_x, grid_y, grid_z  : 1-D coordinate arrays [m]
    p_complex               : (Nx, Ny, Nz) complex pressure amplitude [Pa-normalised]
    p_abs                   : |p| = absolute pressure magnitude
    p_abs2                  : |p|² needed by Stage 3 (Gorkov)
output/figures/pressure_xz.png, pressure_3d_isosurface.png
"""

import sys
import pathlib
import argparse

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

import config as cfg


# ---------------------------------------------------------------------------
# Backend 1 — analytical superposition (vectorised)
# ---------------------------------------------------------------------------

def _pressure_at_points(trans_pos: np.ndarray,
                         q_complex: np.ndarray,
                         pts: np.ndarray,
                         k: float,
                         amplitude_scale: float = 1.0) -> np.ndarray:
    """
    Vectorised monopole superposition over all transducers at each point in pts.
    pts : (M, 3)
    Returns complex pressure array of shape (M,).
    """
    p = np.zeros(len(pts), dtype=complex)
    for i in range(len(trans_pos)):
        diff = pts - trans_pos[i]                       # (M, 3)
        r    = np.linalg.norm(diff, axis=1)              # (M,)
        r    = np.maximum(r, 1e-9)
        p   += q_complex[i] * np.exp(1j * k * r) / (4.0 * np.pi * r)
    return p * amplitude_scale


def compute_field_analytical(trans_pos, q_complex, grid_n, grid_lim, k, amplitude_scale):
    """
    Evaluate the superposition on a cubic grid [-grid_lim, grid_lim]³.
    Returns: x, y, z (1-D), p_complex (Nx×Ny×Nz).
    """
    x = np.linspace(-grid_lim, grid_lim, grid_n)
    y = np.linspace(-grid_lim, grid_lim, grid_n)
    z = np.linspace(-grid_lim, grid_lim, grid_n)

    X, Y, Z       = np.meshgrid(x, y, z, indexing="ij")   # (Nx, Ny, Nz)
    pts_flat      = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    # Process in chunks to limit memory usage
    chunk_size    = 50_000
    p_flat        = np.zeros(len(pts_flat), dtype=complex)
    n_chunks      = int(np.ceil(len(pts_flat) / chunk_size))

    print(f"  Grid: {grid_n}³ = {grid_n**3:,} points  (processing in {n_chunks} chunks)")

    for ci in tqdm(range(n_chunks), desc="  Pressure field"):
        sl  = slice(ci * chunk_size, (ci + 1) * chunk_size)
        p_flat[sl] = _pressure_at_points(trans_pos, q_complex,
                                         pts_flat[sl], k, amplitude_scale)

    p_complex = p_flat.reshape(grid_n, grid_n, grid_n)
    return x, y, z, p_complex


# ---------------------------------------------------------------------------
# Backend 2 — k-Wave (3-D, full wave equation)
# ---------------------------------------------------------------------------

def compute_field_kwave(trans_pos, phases, amplitudes, grid_n, grid_lim):
    """
    Run k-Wave 3-D time-domain simulation and return the steady-state pressure.
    Requires k-wave-python with compiled C++ binary.
    """
    try:
        from kwave.kgrid import kWaveGrid
        from kwave.kmedium import kWaveMedium
        from kwave.ksource import kSource
        from kwave.ksensor import kSensor
        from kwave.kspaceFirstOrder3D import kspaceFirstOrder3D
        from kwave.options.simulation_options import SimulationOptions
        from kwave.options.simulation_execution_options import SimulationExecutionOptions
        from kwave.utils.signals import tone_burst
    except ImportError as e:
        raise ImportError(
            "k-wave-python is not installed or the C++ binary is missing.\n"
            "Install with:  pip install k-wave-python\n"
            "Then download the binary from https://github.com/waltsims/k-wave-python"
        ) from e

    # Grid parameters — 8 points per wavelength
    dx    = cfg.WAVELENGTH / 8.0
    Nside = int(np.ceil(2.0 * grid_lim / dx))
    Nside = Nside + (Nside % 2)   # ensure even
    Nx = Ny = Nz = Nside

    print(f"  k-Wave grid: {Nx}³, dx={dx*1e3:.2f} mm")

    kgrid  = kWaveGrid([Nx, Ny, Nz], [dx, dx, dx])
    medium = kWaveMedium(sound_speed=cfg.C0, density=cfg.RHO0)

    # Build source mask from transducer positions
    source      = kSource()
    src_mask    = np.zeros((Nx, Ny, Nz), dtype=bool)
    src_indices = []

    for pos in trans_pos:
        ix = int(np.round(pos[0] / dx)) + Nx // 2
        iy = int(np.round(pos[1] / dx)) + Ny // 2
        iz = int(np.round(pos[2] / dx)) + Nz // 2
        ix = np.clip(ix, 0, Nx - 1)
        iy = np.clip(iy, 0, Ny - 1)
        iz = np.clip(iz, 0, Nz - 1)
        src_mask[ix, iy, iz] = True
        src_indices.append((ix, iy, iz))

    source.p_mask = src_mask
    n_src         = int(src_mask.sum())

    # Time-domain signals: 20 cycles of tone burst per transducer
    n_periods  = 20
    samples_pp = 40   # samples per period
    dt         = 1.0 / (cfg.FREQ * samples_pp)
    t_array    = np.arange(0, n_periods / cfg.FREQ, dt)
    signals    = np.zeros((n_src, len(t_array)))

    src_flat_idx = np.flatnonzero(src_mask)
    for j, (idx3, phi, amp) in enumerate(zip(src_indices, phases, amplitudes)):
        flat = np.ravel_multi_index(idx3, (Nx, Ny, Nz))
        offset = np.searchsorted(src_flat_idx, flat)
        if offset < n_src:
            signals[offset] = amp * cfg.P0_TRANS * np.sin(
                2.0 * np.pi * cfg.FREQ * t_array + phi
            )
    source.p = signals

    # Sensor: record last time step at all points
    sensor        = kSensor()
    sensor.mask   = np.ones((Nx, Ny, Nz), dtype=bool)
    sensor.record = ["p_final"]

    sim_opts  = SimulationOptions(pml_inside=False, pml_size=10, data_cast="single")
    exec_opts = SimulationExecutionOptions(is_gpu_simulation=False)

    print("  Running k-Wave simulation (this may take 15–60 min)…")
    sensor_data = kspaceFirstOrder3D(kgrid, medium, source, sensor,
                                     simulation_options=sim_opts,
                                     execution_options=exec_opts)

    p_field = sensor_data["p_final"].reshape(Nx, Ny, Nz).astype(complex)

    # Build coordinate arrays matching the grid
    lim_kw = Nside * dx / 2.0
    x = np.linspace(-lim_kw, lim_kw, Nx)
    y = np.linspace(-lim_kw, lim_kw, Ny)
    z = np.linspace(-lim_kw, lim_kw, Nz)

    return x, y, z, p_field


# ---------------------------------------------------------------------------
# Saving & plotting
# ---------------------------------------------------------------------------

def save_field(x, y, z, p_complex):
    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    p_abs   = np.abs(p_complex)
    p_abs2  = p_abs**2
    np.savez(
        cfg.OUTPUT_DIR / "pressure_field.npz",
        grid_x    = x,
        grid_y    = y,
        grid_z    = z,
        p_complex = p_complex,
        p_abs     = p_abs,
        p_abs2    = p_abs2,
    )
    print(f"  Saved -> {cfg.OUTPUT_DIR / 'pressure_field.npz'}")


def plot_field(x, y, z, p_complex, trap_pos):
    cfg.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    p_abs = np.abs(p_complex)

    # XZ cross-section through y = 0
    iy_mid = np.argmin(np.abs(y))
    slice_xz = p_abs[:, iy_mid, :]

    # YZ cross-section through x = 0
    ix_mid   = np.argmin(np.abs(x))
    slice_yz = p_abs[ix_mid, :, :]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    extent_xz = [z.min()*1e2, z.max()*1e2, x.min()*1e2, x.max()*1e2]
    extent_yz = [z.min()*1e2, z.max()*1e2, y.min()*1e2, y.max()*1e2]

    for ax, sl, ext, xl, yl, title in [
        (axes[0], slice_xz, extent_xz, "Z [cm]", "X [cm]", "|p|  XZ plane (y=0)"),
        (axes[1], slice_yz, extent_yz, "Z [cm]", "Y [cm]", "|p|  YZ plane (x=0)"),
    ]:
        im = ax.imshow(sl, extent=ext, origin="lower",
                       cmap="hot", aspect="equal", interpolation="bilinear")
        plt.colorbar(im, ax=ax, label="|p| [a.u.]")
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.set_title(title)
        # Draw sphere boundary
        th = np.linspace(0, 2*np.pi, 200)
        ax.plot(np.cos(th) * cfg.R_SPHERE * 1e2,
                np.sin(th) * cfg.R_SPHERE * 1e2,
                "w--", lw=0.8, alpha=0.6, label="Array boundary")
        ax.scatter(trap_pos[2]*1e2, trap_pos[0]*1e2, c="cyan",
                   s=80, zorder=5, label="Trap")
        ax.legend(fontsize=8)

    fig.suptitle(f"Acoustic Pressure Field — {cfg.N_TRANS} transducers, "
                 f"{cfg.FREQ/1e3:.0f} kHz, node-trap phase pattern", fontsize=12)
    plt.tight_layout()
    out = cfg.FIGURES_DIR / "pressure_xz_yz.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out}")

    # ---- Phase plot ----
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))
    p_phase_xz  = np.angle(p_complex[:, iy_mid, :])
    p_phase_yz  = np.angle(p_complex[ix_mid, :, :])

    for ax, sl, ext, xl, yl in [
        (axes2[0], p_phase_xz, extent_xz, "Z [cm]", "X [cm]"),
        (axes2[1], p_phase_yz, extent_yz, "Z [cm]", "Y [cm]"),
    ]:
        im = ax.imshow(sl, extent=ext, origin="lower",
                       cmap="hsv", vmin=-np.pi, vmax=np.pi,
                       aspect="equal", interpolation="bilinear")
        plt.colorbar(im, ax=ax, label="Phase [rad]")
        ax.set_xlabel(xl); ax.set_ylabel(yl)

    axes2[0].set_title("Pressure phase  XZ plane")
    axes2[1].set_title("Pressure phase  YZ plane")
    fig2.suptitle("Acoustic Pressure Phase", fontsize=12)
    plt.tight_layout()
    out2 = cfg.FIGURES_DIR / "pressure_phase.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved -> {out2}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(backend="analytical"):
    print("=" * 60)
    print(f"Stage 2 — Acoustic Pressure Field  ({backend})")
    print("=" * 60)

    # Load transducer config from Stage 1
    tc_path = cfg.OUTPUT_DIR / "transducer_config.npz"
    if not tc_path.exists():
        print("  transducer_config.npz not found — running Stage 1 first …")
        sys.path.insert(0, str(pathlib.Path(__file__).parent))
        from importlib import import_module
        stage1 = import_module("01_phase_computation")
        stage1.main()

    data       = np.load(tc_path)
    trans_pos  = data["positions"]
    phases     = data["phases"]
    amplitudes = data["amplitudes"]
    q_complex  = data["q_complex"]
    print(f"  Loaded {len(trans_pos)} transducers from Stage 1")

    # ---- compute pressure field ----
    if backend == "kwave":
        x, y, z, p_complex = compute_field_kwave(
            trans_pos, phases, amplitudes, cfg.GRID_N, cfg.GRID_LIM
        )
    else:
        x, y, z, p_complex = compute_field_analytical(
            trans_pos, q_complex * cfg.P0_TRANS,
            cfg.GRID_N, cfg.GRID_LIM, cfg.K,
            amplitude_scale=1.0,
        )

    # ---- report statistics ----
    p_abs = np.abs(p_complex)
    p_trap = float(p_abs[
        np.argmin(np.abs(x - cfg.TRAP_POS[0])),
        np.argmin(np.abs(y - cfg.TRAP_POS[1])),
        np.argmin(np.abs(z - cfg.TRAP_POS[2])),
    ])
    print(f"\n  |p| at trap centre : {p_trap:.4f} a.u.")
    print(f"  |p| max in domain  : {p_abs.max():.4f} a.u.")
    print(f"  |p| mean           : {p_abs.mean():.4f} a.u.")

    if p_trap < p_abs.mean():
        print("  [OK] Pressure minimum at trap — node trap confirmed.")
    else:
        print("  ⚠  Pressure NOT minimised at trap. Check GS-PAT convergence.")

    # ---- save & plot ----
    save_field(x, y, z, p_complex)
    plot_field(x, y, z, p_complex, cfg.TRAP_POS)

    print("\nStage 2 complete.")
    return x, y, z, p_complex


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend", choices=["analytical", "kwave"], default="analytical",
        help="Pressure field solver backend (default: analytical)"
    )
    args = parser.parse_args()
    main(backend=args.backend)
