"""
Run Cell Assembly Experiments with AcousTools
=============================================
Uses analytical Gor'kov forces via compute_force() and vedo sphere rendering.

Experiments:
  1. Standing Wave           — all transducers in phase
  2. Single Focus + Lev Sig  — focused trap at origin
  3. 3-Point Focus           — three simultaneous traps
  4. Twin Trap               — two traps along z-axis

Usage:
    python run_acous.py
    python run_acous.py --quick
"""

import torch
import numpy as np
import os
import sys
import argparse
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from acous_simulation import (
    CellAssemblySimulation,
    render_spheres_vedo,
    plot_assembly_mpl,
    plot_timelapse_mpl,
    plot_metrics,
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Helper: Fibonacci sphere for acoustic holograms ─────────────────

def fibonacci_sphere(n_pts, radius):
    """
    Generate n_pts evenly distributed on a sphere of given radius (m).

    Uses the Fibonacci lattice / golden-ratio method.
    Applied in acoustic holography by Marzo et al. 2015 (Nature Comms)
    and Melde et al. 2016 (Nature) to create spherical acoustic traps.
    """
    golden = (1.0 + np.sqrt(5.0)) / 2.0
    i      = np.arange(n_pts, dtype=float)
    theta  = np.arccos(1.0 - 2.0 * (i + 0.5) / n_pts)
    phi    = 2.0 * np.pi * i / golden
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)
    return np.column_stack([x, y, z])   # (n_pts, 3)

# ══════════════════════════════════════════════════════════════════════

def run_experiment(name, activations, board, **kw):
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")
    t0 = time.time()
    sim = CellAssemblySimulation(
        activations=activations, board=board,
        n_cells=kw['n_cells'],
        init_spread=kw['init_spread'],
        merge_distance=kw['merge_distance'],
        dt=kw['dt'],
        bounds=kw['bounds'],
        max_step=kw['max_step'],
        power_alpha=kw.get('power_alpha', 1.0),
        seed=kw['seed'],
        enable_lj=kw.get('enable_lj', False),
        lj_acoustic_ratio=kw.get('lj_acoustic_ratio', 0.5),
        cell_radius=kw.get('cell_radius', 0.0005),
        init_mode=kw.get('init_mode', 'random'),
    )
    metrics = sim.run(
        n_steps=kw['n_steps'],
        snapshot_every=kw['snapshot_every'],
        warmup_steps=kw.get('warmup_steps', 0),
    )
    print(f"  Done ({time.time() - t0:.1f}s)")
    return sim, metrics


# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="3D Cell Assembly (AcousTools)")
    parser.add_argument('--quick', action='store_true',
                        help='Quick test run (fewer steps)')
    args = parser.parse_args()

    n_steps = 200 if args.quick else 600
    warmup = 100 if args.quick else 300
    snapshot_every = 25

    output_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "output_v2"
    )
    os.makedirs(output_dir, exist_ok=True)

    # ── AcousTools setup ──────────────────────────────────────────
    from acoustools.Utilities import TRANSDUCERS, add_lev_sig
    from acoustools.Solvers import wgs

    board = TRANSDUCERS
    n_trans = board.shape[0]

    kw = dict(
        n_cells=60,
        init_spread=0.020,      # 20 mm std — spread across field
        merge_distance=0.003,   # 3 mm merge threshold
        dt=0.001,
        bounds=0.04,            # 40 mm half-extent
        max_step=0.001,         # 1 mm max displacement/step
        power_alpha=0.3,        # compress force dynamic range
        n_steps=n_steps,
        warmup_steps=warmup,    # acoustic sorting before merging starts
        snapshot_every=snapshot_every,
        seed=42,
    )

    results = []   # (short_name, title, sim, metrics)

    # ══════════════════════════════════════════════════════════════
    # Experiment 1: Standing Wave (all transducers in phase)
    # ══════════════════════════════════════════════════════════════
    act = torch.ones(1, n_trans, 1, dtype=torch.complex64)
    sim, m = run_experiment("Exp 1: Standing Wave", act, board, **kw)
    results.append(("1_standing_wave", "Standing Wave", sim, m))

    # ══════════════════════════════════════════════════════════════
    # Experiment 2: Single Focus + Levitation Signature
    # ══════════════════════════════════════════════════════════════
    pt = torch.zeros(1, 3, 1)
    act = wgs(pt)
    try:
        act = add_lev_sig(act, pt)
    except Exception:
        pass
    sim, m = run_experiment("Exp 2: Focus + Lev Sig", act, board, **kw)
    results.append(("2_focus_lev", "Focus + Lev Sig", sim, m))

    # ══════════════════════════════════════════════════════════════
    # Experiment 3: Multi-Point Focus (3 targets)
    # ══════════════════════════════════════════════════════════════
    pts = torch.tensor([
        [[-0.01,  0.01,  0.0]],
        [[ 0.00,  0.00,  0.0]],
        [[ 0.01, -0.01,  0.0]],
    ]).permute(1, 2, 0)   # (1, 3, 3)
    act = wgs(pts)
    sim, m = run_experiment("Exp 3: 3-Point Focus", act, board, **kw)
    results.append(("3_multipoint", "3-Point Focus", sim, m))

    # ══════════════════════════════════════════════════════════════
    # Experiment 4: Twin Trap (two foci along z-axis)
    # ══════════════════════════════════════════════════════════════
    pts2 = torch.tensor([
        [[0.0, 0.0,  0.01]],
        [[0.0, 0.0, -0.01]],
    ]).permute(1, 2, 0)   # (1, 3, 2)
    act = wgs(pts2)
    try:
        act = add_lev_sig(act, pts2)
    except Exception:
        pass
    sim, m = run_experiment("Exp 4: Twin Trap", act, board, **kw)
    results.append(("4_twin_trap", "Twin Trap", sim, m))

    # ══════════════════════════════════════════════════════════════
    # Experiment 5: Rod Trap (linear multi-point array along Z)
    # ──────────────────────────────────────────────────────────────
    # 7 WGS focal points spaced along the Z-axis (±15 mm, ~5 mm apart).
    # Cells driven to each node form a column -> rod/tubular tissue.
    # Ref: Yanagawa et al. 2011 (FASEB J); Compton et al. 2014 (Biomaterials)
    # LJ forces provide short-range adhesion once cells reach their nodes.
    # Init 'settled': cells start at chamber bottom (sedimentation model,
    #   Bouyer et al. 2016).
    # ══════════════════════════════════════════════════════════════
    n_rod   = 7
    z_rod   = np.linspace(-0.015, 0.015, n_rod)   # −15 → +15 mm
    rod_xyz = np.vstack([np.zeros((2, n_rod)), z_rod.reshape(1, -1)])  # (3, 7)
    pts_rod = torch.tensor(rod_xyz, dtype=torch.float32).unsqueeze(0)  # (1,3,7)
    act_rod = wgs(pts_rod)
    kw_rod  = dict(kw,
                   enable_lj=True,
                   lj_acoustic_ratio=0.5,
                   init_mode='settled',
                   merge_distance=0.004)
    sim, m = run_experiment("Exp 5: Rod Trap", act_rod, board, **kw_rod)
    results.append(("5_rod_trap", "Rod Trap", sim, m))

    # ══════════════════════════════════════════════════════════════
    # Experiment 6: Sphere Trap (Fibonacci-sphere hologram)
    # ──────────────────────────────────────────────────────────────
    # 16 WGS focal points on a 12-mm-radius sphere (Fibonacci lattice).
    # Cells accumulate at the surface nodes → spherical shell / spheroid.
    # Ref: Marzo et al. 2015 (Nature Comms); Melde et al. 2016 (Nature);
    #      Bouyer et al. 2016 – levitated spheroid formation.
    # ══════════════════════════════════════════════════════════════
    sph_pts = fibonacci_sphere(n_pts=16, radius=0.012)       # (16, 3)  in metres
    pts_sph = torch.tensor(sph_pts.T, dtype=torch.float32).unsqueeze(0)  # (1,3,16)
    act_sph = wgs(pts_sph)
    kw_sph  = dict(kw,
                   enable_lj=True,
                   lj_acoustic_ratio=0.5,
                   init_mode='settled',
                   merge_distance=0.004)
    sim, m = run_experiment("Exp 6: Sphere Trap", act_sph, board, **kw_sph)
    results.append(("6_sphere_trap", "Sphere Trap", sim, m))

    # ══════════════════════════════════════════════════════════════
    # GENERATE VISUALIZATIONS
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}")
    print("  Generating visualizations...")
    print(f"{'=' * 60}")

    all_metrics = []
    all_titles = []
    vedo_ok = True    # track whether vedo rendering works

    for short, title, sim, metrics in results:
        all_metrics.append(metrics)
        all_titles.append(title)

        # ── vedo sphere rendering (final state) ──
        _, final_pos, final_cids = sim.snapshots[-1]
        vedo_path = os.path.join(output_dir, f"{short}_spheres.png")
        if vedo_ok:
            try:
                render_spheres_vedo(
                    final_pos, final_cids, vedo_path,
                    radius=0.001, bounds=kw['bounds'],
                    title=f"{title} (Final)",
                )
                print(f"  Saved {vedo_path}  [vedo]")
            except Exception as e:
                print(f"  vedo failed ({e}), switching to matplotlib")
                vedo_ok = False

        if not vedo_ok:
            # matplotlib fallback
            vedo_path = vedo_path.replace('_spheres.png', '_final_mpl.png')
            plot_assembly_mpl(
                final_pos, final_cids,
                title=f"{title} (Final)", save_path=vedo_path,
                bounds=kw['bounds'],
            )
            print(f"  Saved {vedo_path}  [matplotlib]")

        # ── matplotlib timelapse ──
        p = os.path.join(output_dir, f"{short}_timelapse.png")
        plot_timelapse_mpl(
            sim.snapshots, title=title, save_path=p,
            bounds=kw['bounds'],
        )
        print(f"  Saved {p}")

    # ── Assembly metrics comparison ──
    mp = os.path.join(output_dir, "assembly_metrics.png")
    plot_metrics(all_metrics, all_titles, save_path=mp)
    print(f"  Saved {mp}")

    # ── 2×3 final-state comparison (matplotlib) ──
    fig = plt.figure(figsize=(21, 14))
    for i, (short, title, sim, metrics) in enumerate(results):
        ax = fig.add_subplot(2, 3, i + 1, projection='3d')
        _, fp, cids = sim.snapshots[-1]
        unique = np.unique(cids)
        cmap = plt.cm.get_cmap('tab20', max(len(unique), 2))
        for cidx, cid in enumerate(unique):
            mask = cids == cid
            n_m = int(np.sum(mask))
            pos = fp[mask] * 1000
            ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2],
                       c=[cmap(cidx % 20)] * n_m,
                       s=max(30, 12 * n_m), alpha=0.85,
                       edgecolors='k', linewidth=0.2)
        b = kw['bounds'] * 1000
        ax.set_xlim(-b, b)
        ax.set_ylim(-b, b)
        ax.set_zlim(-b, b)
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        final_m = metrics[-1][1]
        score_str = (f"rod={final_m.get('rod_score', 0):.2f} "
                     f"sph={final_m.get('sphere_score', 0):.2f}")
        ax.set_title(f"{title}\n{score_str}", fontsize=10)
    fig.suptitle('Final Cell Assembly — Experiment Comparison',
                 fontsize=14, y=0.97)
    plt.tight_layout()
    cp = os.path.join(output_dir, "final_comparison.png")
    fig.savefig(cp, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {cp}")

    # ══════════════════════════════════════════════════════════════
    # RESULTS SUMMARY
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}")
    print("  RESULTS SUMMARY")
    print(f"{'=' * 60}")

    for short, title, sim, metrics in results:
        final = metrics[-1][1]
        print(f"\n  {title}:")
        print(f"    Clusters:     {final['n_clusters']}")
        print(f"    Largest:      {final['largest_cluster']} / {kw['n_cells']}")
        print(f"    Assembly:     {final['assembly_fraction']:.0%}")
        print(f"    Rod  score:   {final.get('rod_score', 0):.3f}  "
              f"(1.0=perfect rod,   0.33=isotropic)")
        print(f"    Sph  score:   {final.get('sphere_score', 0):.3f}  "
              f"(1.0=perfect shell, low=scattered)")

    print(f"\n  All output saved to: {output_dir}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
