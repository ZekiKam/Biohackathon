"""
Stage 5 — Comprehensive Visualisation
======================================
Reads all outputs from Stages 1–4 and produces publication-quality figures
and a ParaView-ready VTK scene file.

Figures produced
----------------
  figures/01_transducer_phases.png       — already created by Stage 1
  figures/02_pressure_xz_yz.png          — already created by Stage 2
  figures/03_gorkov_potential_xz.png     — already created by Stage 3
  figures/03_trajectories_3d.png         — already created by Stage 3
  figures/03_trajectories_animation.gif  — already created by Stage 3
  figures/05_dashboard.png               — summary 4-panel dashboard (NEW)
  figures/05_pressure_isosurface.png     — 3-D isosurface (pyvista, NEW)
  paraview/scene.pvsm                    — ParaView state file (NEW)

Run
---
    python src/05_visualize.py

Requires: matplotlib, numpy, pyvista (optional — only for 3D isosurface)
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import config as cfg

OUTPUT_DIR  = cfg.OUTPUT_DIR
FIGURES_DIR = cfg.FIGURES_DIR


# ---------------------------------------------------------------------------
# Helper: load npz safely
# ---------------------------------------------------------------------------

def _load(name):
    path = OUTPUT_DIR / name
    if not path.exists():
        print(f"  Warning: {path} not found — skipping.")
        return None
    return np.load(path)


# ---------------------------------------------------------------------------
# 1. Four-panel summary dashboard
# ---------------------------------------------------------------------------

def plot_dashboard():
    tc = _load("transducer_config.npz")
    pf = _load("pressure_field.npz")
    gp = _load("gorkov_potential.npz")
    tr = _load("trajectories.npz")

    if all(d is None for d in [tc, pf, gp, tr]):
        print("  No data found — run Stages 1–3 first.")
        return

    fig = plt.figure(figsize=(18, 14))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    # --- Panel A: transducer phase map ---
    if tc is not None:
        ax = fig.add_subplot(gs[0, 0], projection="3d")
        pos    = tc["positions"]
        phases = tc["phases"]
        sc = ax.scatter(pos[:, 0]*1e2, pos[:, 1]*1e2, pos[:, 2]*1e2,
                        c=np.degrees(phases), cmap="hsv",
                        s=25, vmin=-180, vmax=180)
        fig.colorbar(sc, ax=ax, shrink=0.55, pad=0.1, label="Phase (°)")
        ax.scatter(*cfg.TRAP_POS*1e2, c="red", s=200, marker="*", zorder=10)
        ax.set_title("A — Transducer Phase Pattern (GS-PAT)", fontsize=11)
        ax.set_xlabel("X [cm]"); ax.set_ylabel("Y [cm]"); ax.set_zlabel("Z [cm]")
        ax.tick_params(labelsize=7)

    # --- Panel B: pressure field XZ ---
    if pf is not None:
        ax = fig.add_subplot(gs[0, 1])
        x, y, z   = pf["grid_x"], pf["grid_y"], pf["grid_z"]
        iy_mid    = np.argmin(np.abs(y))
        p_abs_xz  = pf["p_abs"][:, iy_mid, :]
        ext       = [z.min()*1e2, z.max()*1e2, x.min()*1e2, x.max()*1e2]
        im = ax.imshow(p_abs_xz, extent=ext, origin="lower",
                       cmap="hot", aspect="equal", interpolation="bilinear")
        fig.colorbar(im, ax=ax, label="|p| [a.u.]")
        th = np.linspace(0, 2*np.pi, 200)
        ax.plot(np.cos(th)*cfg.R_SPHERE*1e2, np.sin(th)*cfg.R_SPHERE*1e2,
                "w--", lw=0.8, alpha=0.5)
        ax.scatter(cfg.TRAP_POS[2]*1e2, cfg.TRAP_POS[0]*1e2,
                   c="cyan", s=80, zorder=5, marker="*")
        ax.set_xlabel("Z [cm]"); ax.set_ylabel("X [cm]")
        ax.set_title("B — Acoustic Pressure Field  |p|  (XZ plane)", fontsize=11)

    # --- Panel C: Gorkov potential XZ ---
    if gp is not None:
        ax = fig.add_subplot(gs[1, 0])
        x, y, z = gp["grid_x"], gp["grid_y"], gp["grid_z"]
        iy_mid  = np.argmin(np.abs(y))
        U_xz    = gp["U_rad"][:, iy_mid, :]
        vmax    = float(np.percentile(np.abs(U_xz), 99))
        ext     = [z.min()*1e2, z.max()*1e2, x.min()*1e2, x.max()*1e2]
        im = ax.imshow(U_xz, extent=ext, origin="lower",
                       cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                       aspect="equal", interpolation="bilinear")
        fig.colorbar(im, ax=ax, label="U_rad [normalised J]")
        th = np.linspace(0, 2*np.pi, 200)
        ax.plot(np.cos(th)*cfg.R_SPHERE*1e2, np.sin(th)*cfg.R_SPHERE*1e2,
                "w--", lw=0.8, alpha=0.5)
        ax.scatter(cfg.TRAP_POS[2]*1e2, cfg.TRAP_POS[0]*1e2,
                   c="cyan", s=120, zorder=5, marker="*")
        ax.set_xlabel("Z [cm]"); ax.set_ylabel("X [cm]")
        ax.set_title("C — Gorkov Radiation Potential  U_rad  (XZ plane)", fontsize=11)

    # --- Panel D: droplet trajectories (XZ projection) ---
    if tr is not None:
        ax   = fig.add_subplot(gs[1, 1])
        traj = tr["positions"]      # (N, Nt, 3)
        t    = tr["t"]
        N    = traj.shape[0]
        colors = plt.cm.tab10(np.linspace(0, 1, N))
        for i, col in enumerate(colors):
            ax.plot(traj[i, :, 2]*1e2, traj[i, :, 0]*1e2,
                    color=col, lw=1.2, label=f"Droplet {i+1}")
            ax.scatter(traj[i, 0, 2]*1e2, traj[i, 0, 0]*1e2,
                       color=col, s=40, marker="o", zorder=5)
            ax.scatter(traj[i, -1, 2]*1e2, traj[i, -1, 0]*1e2,
                       color=col, s=40, marker="X", zorder=5)
        th = np.linspace(0, 2*np.pi, 200)
        ax.plot(np.cos(th)*cfg.R_SPHERE*1e2, np.sin(th)*cfg.R_SPHERE*1e2,
                "k--", lw=0.8, alpha=0.4)
        ax.scatter(cfg.TRAP_POS[2]*1e2, cfg.TRAP_POS[0]*1e2,
                   c="red", s=150, marker="*", zorder=10, label="Trap")
        ax.set_xlabel("Z [cm]"); ax.set_ylabel("X [cm]")
        ax.set_title("D — Droplet Trajectories (XZ projection)", fontsize=11)
        ax.set_aspect("equal")
        ax.legend(fontsize=7, loc="upper right")

    fig.suptitle(
        "GBM Acoustic 3D Culture — Simulation Summary\n"
        f"f = {cfg.FREQ/1e3:.0f} kHz  |  R = {cfg.R_SPHERE*1e2:.0f} cm  |  "
        f"{cfg.N_TRANS} transducers  |  {cfg.N_DROPLETS} droplets  |  "
        f"a = {cfg.A_DROP*1e3:.1f} mm",
        fontsize=13, y=0.98,
    )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "05_dashboard.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out}")


# ---------------------------------------------------------------------------
# 2. 3-D isosurface visualisation with pyvista
# ---------------------------------------------------------------------------

def plot_isosurface_pyvista():
    try:
        import pyvista as pv
    except ImportError:
        print("  pyvista not installed — skipping 3D isosurface.")
        print("  Install with: pip install pyvista")
        return

    pf = _load("pressure_field.npz")
    gp = _load("gorkov_potential.npz")
    if pf is None or gp is None:
        return

    x, y, z = pf["grid_x"], pf["grid_y"], pf["grid_z"]
    p_abs    = pf["p_abs"]
    U_rad    = gp["U_rad"]

    # Build a pyvista ImageData (uniform grid)
    grid = pv.ImageData()
    grid.dimensions = np.array(p_abs.shape) + 1
    grid.origin     = (x.min(), y.min(), z.min())
    grid.spacing    = (x[1]-x[0], y[1]-y[0], z[1]-z[0])

    # Cell data (one cell per voxel)
    grid.cell_data["pressure"]     = p_abs.ravel(order="F")
    grid.cell_data["gorkov_U"]     = U_rad.ravel(order="F")

    # Off-screen rendering
    pv.start_xvfb() if hasattr(pv, "start_xvfb") else None
    pl = pv.Plotter(off_screen=True)

    # Isosurface of pressure at 50th percentile
    p50  = float(np.percentile(p_abs, 50))
    iso  = grid.contour([p50], scalars="pressure")
    pl.add_mesh(iso, scalars="pressure", cmap="hot", opacity=0.5,
                show_scalar_bar=True, scalar_bar_args={"title": "|p| [a.u.]"})

    # Add gorkov potential minimum as a sphere
    trap_sphere = pv.Sphere(radius=cfg.A_DROP*1e2, center=cfg.TRAP_POS*1e2)
    pl.add_mesh(trap_sphere, color="cyan", opacity=0.9, label="Trap")

    pl.set_background("black")
    pl.camera_position = "iso"
    pl.add_axes()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = str(FIGURES_DIR / "05_pressure_isosurface.png")
    pl.screenshot(out, window_size=(1200, 900))
    pl.close()
    print(f"  Saved -> {out}")


# ---------------------------------------------------------------------------
# 3. Export VTK files for ParaView
# ---------------------------------------------------------------------------

def export_vtk_for_paraview():
    """
    Export pressure field and Gorkov potential as a single VTK file that
    ParaView can open on Windows without needing Basilisk or Python.
    """
    try:
        import pyvista as pv
    except ImportError:
        print("  pyvista not installed — skipping VTK export.")
        return

    pf = _load("pressure_field.npz")
    gp = _load("gorkov_potential.npz")
    tr = _load("trajectories.npz")
    if pf is None:
        print("  Pressure field not found — skipping VTK export.")
        return

    x, y, z = pf["grid_x"], pf["grid_y"], pf["grid_z"]

    grid = pv.ImageData()
    grid.dimensions = np.array(pf["p_abs"].shape) + 1
    grid.origin     = (x.min(), y.min(), z.min())
    grid.spacing    = (x[1]-x[0], y[1]-y[0], z[1]-z[0])
    grid.cell_data["pressure_abs"]   = pf["p_abs"].ravel(order="F")
    grid.cell_data["pressure_abs2"]  = pf["p_abs2"].ravel(order="F")
    if gp is not None:
        grid.cell_data["gorkov_U"]   = gp["U_rad"].ravel(order="F")
        grid.cell_data["force_x"]    = gp["Fx"].ravel(order="F")
        grid.cell_data["force_y"]    = gp["Fy"].ravel(order="F")
        grid.cell_data["force_z"]    = gp["Fz"].ravel(order="F")

    pv_dir = OUTPUT_DIR / "paraview"
    pv_dir.mkdir(parents=True, exist_ok=True)

    vtk_path = str(pv_dir / "acoustic_field.vti")
    grid.save(vtk_path)
    print(f"  Saved -> {vtk_path}")

    # Export trajectories as poly lines
    if tr is not None:
        pos  = tr["positions"]    # (N, Nt, 3)
        N    = pos.shape[0]
        polylines = []
        for i in range(N):
            pts  = pos[i]    # (Nt, 3)
            line = pv.lines_from_points(pts)
            polylines.append(line)
        combined = pv.merge(polylines) if len(polylines) > 1 else polylines[0]
        traj_path = str(pv_dir / "droplet_trajectories.vtp")
        combined.save(traj_path)
        print(f"  Saved -> {traj_path}")

    print("\n  Open these files in ParaView (Windows):")
    print(f"    {vtk_path}")
    if tr is not None:
        print(f"    {str(pv_dir / 'droplet_trajectories.vtp')}")
    print("\n  Suggested ParaView workflow:")
    print("    1. Open acoustic_field.vti -> Apply")
    print("    2. Add Contour filter on 'gorkov_U' to show trap isosurface")
    print("    3. Open droplet_trajectories.vtp -> Tube filter for visibility")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Stage 5 — Comprehensive Visualisation")
    print("=" * 60)

    print("\n[1] Generating summary dashboard …")
    plot_dashboard()

    print("\n[2] Generating 3D isosurface (pyvista) …")
    plot_isosurface_pyvista()

    print("\n[3] Exporting VTK files for ParaView …")
    export_vtk_for_paraview()

    print("\nStage 5 complete.")
    print(f"\nAll figures saved to:  {FIGURES_DIR}")
    print(f"ParaView VTK files  :  {OUTPUT_DIR / 'paraview'}")


if __name__ == "__main__":
    main()
