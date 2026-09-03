"""
3D Cell Assembly Simulation — AcousTools Edition
=================================================
Uses AcousTools' analytical compute_force() for proper Gor'kov radiation forces.
Cell-cell interaction: simplified "sticky particle" merge model —
cells that come within merge_distance permanently fuse into rigid clusters.

Visualization: vedo for proper 3D sphere rendering, matplotlib as fallback.
"""

import torch
import numpy as np
from scipy.spatial.distance import pdist, squareform
from acoustools.Force import compute_force

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ══════════════════════════════════════════════════════════════════════
# SIMULATION
# ══════════════════════════════════════════════════════════════════════

class CellAssemblySimulation:
    """
    Particle-based simulation of acoustic cell assembly.

    Acoustic forces  : analytical Gor'kov model via AcousTools compute_force().
    Cell interactions: Lennard-Jones soft potential (adhesion + volume exclusion)
                       combined with optional hard-merge for permanent fusing.
    Dynamics         : overdamped with power-law force compression.

    Research basis
    --------------
    * Init 'settled': cells sedimented at chamber bottom then lifted by acoustics
                       — Bouyer et al. 2016 (Biomaterials)
    * LJ adhesion   : simplified secondary Bjerknes + cadherin adhesion model
                       — Compton et al. 2014, Melde et al. 2016 (Nature)
    * Rod trap      : multi-point linear array for tubular tissue
                       — Yanagawa et al. 2011, Compton et al. 2014
    * Sphere trap   : Fibonacci-sphere hologram for spheroid assembly
                       — Marzo et al. 2015 (Nature Comms), Bouyer et al. 2016
    """

    def __init__(self, activations, board, n_cells=50,
                 init_spread=0.015, merge_distance=0.003,
                 dt=0.001, bounds=0.04, max_step=0.0005,
                 power_alpha=1.0, seed=42,
                 enable_lj=False, lj_epsilon=None,
                 lj_acoustic_ratio=0.5, cell_radius=0.0005,
                 init_mode='random'):
        """
        Parameters
        ----------
        activations       : torch.Tensor — transducer hologram (1, M, 1)
        board             : torch.Tensor — transducer positions
        n_cells           : int          — number of cells
        init_spread       : float        — std dev of random initial positions (m)
        merge_distance    : float        — cluster hard-merge threshold (m)
        dt                : float        — time step (s)
        bounds            : float        — half-extent of simulation box (m)
        max_step          : float        — max displacement per step (m)
        power_alpha       : float        — force compression exponent (<1 boosts
                                          weak-force cells; 0.3 is a good default)
        seed              : int          — random seed
        enable_lj         : bool         — Lennard-Jones inter-particle forces
        lj_epsilon        : float|None   — LJ well depth (J); None = auto from
                                          acoustic force scale at step 0
        lj_acoustic_ratio : float        — when lj_epsilon=None: LJ attractive
                                          force at 1.5σ as fraction of max acoustic
        cell_radius       : float        — cell radius for LJ σ = 2×radius (m)
        init_mode         : str          — 'random' (Gaussian spread),
                                          'settled' (bottom of chamber, per
                                          Bouyer 2016), 'seed' (tight organoid)
        """
        self.activations      = activations
        self.board            = board
        self.n_cells          = n_cells
        self.merge_distance   = merge_distance
        self.dt               = dt
        self.bounds           = bounds
        self.max_step         = max_step
        self.power_alpha      = power_alpha
        self.enable_lj        = enable_lj
        self.lj_acoustic_ratio = lj_acoustic_ratio
        self.cell_radius      = cell_radius
        self._lj_epsilon_auto = (lj_epsilon is None)
        self.lj_epsilon       = lj_epsilon if lj_epsilon is not None else 1e-14

        np.random.seed(seed)
        self.positions = self._init_positions(n_cells, init_spread, bounds, init_mode)

        # Each cell starts as its own cluster
        self.cluster_ids = np.arange(n_cells)

        self.drag = None  # auto-calibrated at step 0
        self.snapshots = []
        self.metrics_history = []

    # ── Initial positions ─────────────────────────────────────────

    def _init_positions(self, n_cells, init_spread, bounds, init_mode):
        """
        Research-informed initial cell placement.

        'random'  — Gaussian spread; standard in acoustic manipulation papers.
        'settled' — Cells sedimented to chamber bottom before acoustic activation
                    (Bouyer et al. 2016, Compton et al. 2014).
        'seed'    — Tight organoid seed; mimics spheroid seeding
                    (Kang et al. 2021, Lancaster & Knoblich 2014).
        """
        if init_mode == 'settled':
            # XY: Gaussian; Z: uniformly near the bottom of the chamber
            xy  = np.random.randn(n_cells, 2) * init_spread
            z   = np.random.uniform(-bounds * 0.9, -bounds * 0.4, n_cells)
            pos = np.column_stack([xy, z])
        elif init_mode == 'seed':
            pos = np.random.randn(n_cells, 3) * (init_spread * 0.15)
        else:   # 'random'
            pos = np.random.randn(n_cells, 3) * init_spread
        return np.clip(pos, -bounds * 0.9, bounds * 0.9)

    # ── Force computation ─────────────────────────────────────────

    def _get_acoustic_forces(self):
        """Compute analytical Gor'kov radiation force at each cell position."""
        points = torch.tensor(
            self.positions.T, dtype=torch.float32
        ).unsqueeze(0)   # (1, 3, N)

        forces = compute_force(
            self.activations, points, board=self.board
        )  # (1, N, 3)

        return forces[0].detach().numpy()   # (N, 3)

    def _auto_calibrate_drag(self, forces):
        """Set drag so maximum single-particle displacement ≈ max_step.
        When lj_epsilon is auto, scale it relative to the acoustic force."""
        max_f = np.max(np.linalg.norm(forces, axis=1))
        if max_f > 0:
            self.drag = max_f * self.dt / self.max_step
            if self.enable_lj and self._lj_epsilon_auto:
                # Want F_LJ(1.5σ) ≈ lj_acoustic_ratio × max_acoustic_force
                # F(1.5σ) ≈ 24ε/(1.5σ)² × 0.073
                # → ε = F_target × (1.5σ)² / (24 × 0.073)
                sigma   = 2.0 * self.cell_radius
                r_ref   = 1.5 * sigma
                self.lj_epsilon = (self.lj_acoustic_ratio * max_f
                                   * r_ref**2 / (24.0 * 0.073))
        else:
            self.drag = 1.0

    # ── Lennard-Jones inter-particle forces ──────────────────────

    def _compute_lj_forces(self):
        """
        Lennard-Jones forces between cluster centers.

        σ = 2 × cell_radius   (equilibrium: two cells touching)
        r_cut = 2.5 σ          (beyond this LJ ≈ 0)

        Provides:
          r < σ       → strong repulsion (volume exclusion)
          σ < r < r_cut → attraction (adhesion, secondary Bjerknes)

        Reference: simplified Gor'kov + adhesion model following
        Compton et al. 2014 (Biomaterials) and Melde et al. 2016 (Nature).
        """
        unique  = np.unique(self.cluster_ids)
        n_c     = len(unique)
        sigma   = 2.0 * self.cell_radius
        r_cut   = 2.5 * sigma

        centers = np.array([
            self.positions[self.cluster_ids == cid].mean(axis=0)
            for cid in unique
        ])   # (n_c, 3)

        cluster_forces = np.zeros((n_c, 3))
        for i in range(n_c):
            for j in range(i + 1, n_c):
                dr    = centers[j] - centers[i]
                r     = np.linalg.norm(dr)
                if r < 1e-10 or r >= r_cut:
                    continue
                r_eff = max(r, 0.5 * sigma)   # soft floor — avoids singularity
                sr6   = (sigma / r_eff) ** 6
                # F_i = (24ε/r²)[−2(σ/r)¹² + (σ/r)⁶] × dr
                f_scale = (24.0 * self.lj_epsilon / r_eff**2
                           * (-(2.0 * sr6**2) + sr6))
                f_vec   = f_scale * dr
                cluster_forces[i] += f_vec
                cluster_forces[j] -= f_vec    # Newton's 3rd law

        # Broadcast: each cell gets its cluster's force vector
        cell_forces = np.zeros_like(self.positions)
        for k, cid in enumerate(unique):
            cell_forces[self.cluster_ids == cid] = cluster_forces[k]
        return cell_forces

    # ── Cluster merging ───────────────────────────────────────────

    def _merge_clusters(self):
        """Fuse clusters whose members are within merge_distance."""
        dists = squareform(pdist(self.positions))
        changed = True
        while changed:
            changed = False
            for i in range(self.n_cells):
                for j in range(i + 1, self.n_cells):
                    if self.cluster_ids[i] != self.cluster_ids[j]:
                        if dists[i, j] < self.merge_distance:
                            old = self.cluster_ids[j]
                            new = self.cluster_ids[i]
                            self.cluster_ids[self.cluster_ids == old] = new
                            changed = True

    # ── Dynamics ──────────────────────────────────────────────────

    def _move_clusters(self, forces):
        """Move each cluster under combined acoustic + Lennard-Jones forces."""
        unique = np.unique(self.cluster_ids)

        # Acoustic net force per cluster (vector sum over member cells)
        cluster_acoustic = {
            cid: forces[self.cluster_ids == cid].sum(axis=0) for cid in unique
        }

        # LJ inter-cluster force (same vector for all cells in a cluster)
        if self.enable_lj:
            lj_all    = self._compute_lj_forces()   # (N_cells, 3)
            cluster_lj = {
                cid: lj_all[self.cluster_ids == cid][0] for cid in unique
            }
        else:
            cluster_lj = {cid: np.zeros(3) for cid in unique}

        # Combined net force and global max for power-law normalization
        cluster_net = {
            cid: cluster_acoustic[cid] + cluster_lj[cid] for cid in unique
        }
        mags  = np.array([np.linalg.norm(f) for f in cluster_net.values()])
        max_f = np.max(mags) if np.max(mags) > 1e-20 else 1.0

        for cid in unique:
            net_f = cluster_net[cid]
            mag_f = np.linalg.norm(net_f)
            if mag_f < 1e-20:
                continue
            direction = net_f / mag_f
            # Power-law compression: alpha<1 boosts weak-force clusters so
            # distant cells still migrate (0.3 maps 1% force → 20% speed)
            rel   = min(mag_f / max_f, 1.0)
            speed = self.max_step * (rel ** self.power_alpha)
            self.positions[self.cluster_ids == cid] += direction * speed

        # Enforce simulation bounds
        self.positions = np.clip(self.positions, -self.bounds, self.bounds)

    # ── Metrics ───────────────────────────────────────────────────

    def _metrics(self):
        unique  = np.unique(self.cluster_ids)
        sizes   = [int(np.sum(self.cluster_ids == c)) for c in unique]
        singles = sum(1 for s in sizes if s == 1)
        d = {
            'n_clusters':       len(unique),
            'largest_cluster':  max(sizes),
            'assembly_fraction': 1.0 - singles / self.n_cells,
        }

        # ── Shape analysis via PCA ────────────────────────────────
        pos = self.positions
        if len(pos) > 3:
            centered = pos - pos.mean(axis=0)
            cov      = centered.T @ centered / len(pos)
            eigvals  = np.sort(np.linalg.eigvalsh(cov))[::-1]   # descending
            total    = eigvals.sum()
            # Rod score: 1.0 = perfect line, 0.33 = isotropic sphere
            d['rod_score']    = float(eigvals[0] / total) if total > 1e-20 else 0.333
            # Sphere score: 1.0 = perfect shell, 0 = radially scattered
            radii             = np.linalg.norm(centered, axis=1)
            mean_r            = radii.mean()
            d['sphere_score'] = (float(1.0 - radii.std() / mean_r)
                                 if mean_r > 1e-10 else 0.0)
        else:
            d['rod_score']    = 0.333
            d['sphere_score'] = 0.0
        return d

    # ── Main loop ─────────────────────────────────────────────────

    def run(self, n_steps=300, snapshot_every=30, warmup_steps=0,
             verbose=True):
        """
        Parameters
        ----------
        n_steps      : total simulation steps (warm-up + assembly)
        snapshot_every : record a snapshot every N steps
        warmup_steps : steps where cells move under acoustic force but
                       do NOT merge — lets the field sort them spatially
                       before adhesion kicks in.
        """
        total = n_steps
        # Step 0: record initial state and auto-calibrate drag
        forces = self._get_acoustic_forces()
        self._auto_calibrate_drag(forces)
        if verbose:
            print(f"    Drag auto-calibrated: {self.drag:.4e}")
            if warmup_steps:
                print(f"    Warm-up phase: {warmup_steps} steps (no merging)")

        self.snapshots.append((0, self.positions.copy(), self.cluster_ids.copy()))
        m = self._metrics()
        self.metrics_history.append((0, m))
        if verbose:
            self._log(0, total, m)

        for s in range(1, total + 1):
            forces = self._get_acoustic_forces()
            self._move_clusters(forces)

            # Only merge after warm-up phase
            if s > warmup_steps:
                self._merge_clusters()

            if s % snapshot_every == 0 or s == total:
                m = self._metrics()
                self.metrics_history.append((s, m))
                self.snapshots.append(
                    (s, self.positions.copy(), self.cluster_ids.copy())
                )
                if verbose:
                    phase = "warm" if s <= warmup_steps else "asm "
                    self._log(s, total, m, phase)

        return self.metrics_history

    def _log(self, step, total, m, phase=""):
        print(f"    [{phase}] Step {step:5d}/{total}: "
              f"clusters={m['n_clusters']}, "
              f"assembly={m['assembly_fraction']:.0%}, "
              f"largest={m['largest_cluster']}, "
              f"rod={m.get('rod_score', 0):.2f}, "
              f"sphere={m.get('sphere_score', 0):.2f}")


# ══════════════════════════════════════════════════════════════════════
# VISUALIZATION — vedo (3D spheres)
# ══════════════════════════════════════════════════════════════════════

def render_spheres_vedo(positions, cluster_ids, save_path,
                        radius=0.001, bounds=0.04, title=""):
    """Render cells as proper 3D spheres using vedo."""
    import vedo

    unique = np.unique(cluster_ids)
    # Build a colour map: one colour per cluster
    cmap = plt.cm.get_cmap('tab20', max(len(unique), 2))

    spheres = []
    for idx, cid in enumerate(unique):
        mask = cluster_ids == cid
        n_m = int(np.sum(mask))
        rgb = cmap(idx % 20)[:3]
        # Slightly enlarge spheres for big clusters to make them visible
        r = radius * (1 + 0.15 * min(n_m, 10))
        for pos in positions[mask]:
            s = vedo.Sphere(pos=pos, r=r, res=16)
            s.color(rgb).alpha(0.85)
            spheres.append(s)

    # Bounding box
    b = bounds
    box = vedo.Box(pos=(0, 0, 0), length=2 * b, width=2 * b, height=2 * b)
    box.wireframe().alpha(0.15).color('grey')

    plotter = vedo.Plotter(offscreen=True, size=(1200, 1000), title=title)
    plotter.show(
        *spheres, box,
        axes=dict(
            xtitle='X (m)', ytitle='Y (m)', ztitle='Z (m)',
            xrange=(-b, b), yrange=(-b, b), zrange=(-b, b),
        ),
    )
    plotter.screenshot(save_path)
    plotter.close()


def render_timelapse_vedo(snapshots, save_path,
                          radius=0.001, bounds=0.04, title="", n_show=5):
    """Render a multi-panel timelapse using vedo."""
    import vedo

    n = len(snapshots)
    indices = np.linspace(0, n - 1, min(n_show, n), dtype=int)

    # vedo doesn't have a native multi-panel layout like matplotlib,
    # so we render each frame to a temporary file and compose with matplotlib
    tmp_paths = []
    for i, idx in enumerate(indices):
        step, pos, cids = snapshots[idx]
        tmp = save_path.replace('.png', f'_tmp_{i}.png')
        render_spheres_vedo(
            pos, cids, tmp, radius=radius, bounds=bounds,
            title=f'Step {step}'
        )
        tmp_paths.append(tmp)

    # Compose into one image
    import os
    from PIL import Image
    images = [Image.open(p) for p in tmp_paths]
    widths = [im.width for im in images]
    max_h = max(im.height for im in images)
    total_w = sum(widths)

    composite = Image.new('RGB', (total_w, max_h), (255, 255, 255))
    x_off = 0
    for im in images:
        composite.paste(im, (x_off, 0))
        x_off += im.width
    composite.save(save_path)

    # Clean up temp files
    for p in tmp_paths:
        try:
            os.remove(p)
        except OSError:
            pass


# ══════════════════════════════════════════════════════════════════════
# VISUALIZATION — matplotlib (fallback)
# ══════════════════════════════════════════════════════════════════════

def plot_assembly_mpl(positions, cluster_ids, title="",
                      save_path=None, bounds=0.04):
    """Plot cells as coloured dots in 3D (matplotlib fallback)."""
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    _draw_clusters_mpl(ax, positions, cluster_ids, bounds)
    ax.set_title(title, fontsize=12)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)


def plot_timelapse_mpl(snapshots, title="", save_path=None,
                       bounds=0.04, n_show=5):
    n = len(snapshots)
    indices = np.linspace(0, n - 1, min(n_show, n), dtype=int)
    fig = plt.figure(figsize=(5 * len(indices), 5))
    for i, idx in enumerate(indices):
        step, pos, cids = snapshots[idx]
        ax = fig.add_subplot(1, len(indices), i + 1, projection='3d')
        _draw_clusters_mpl(ax, pos, cids, bounds, fs=6)
        ax.set_title(f'Step {step}', fontsize=10)
    fig.suptitle(title, fontsize=13, y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)


def plot_metrics(all_metrics, labels, save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for metrics, label in zip(all_metrics, labels):
        steps = [m[0] for m in metrics]
        axes[0].plot(steps, [m[1]['n_clusters'] for m in metrics],
                     '-o', label=label, ms=3)
        axes[1].plot(steps, [m[1]['assembly_fraction'] * 100 for m in metrics],
                     '-o', label=label, ms=3)
        axes[2].plot(steps, [m[1]['largest_cluster'] for m in metrics],
                     '-o', label=label, ms=3)
    axes[0].set_title('Clusters')
    axes[0].set_xlabel('Step')
    axes[0].legend(fontsize=7)
    axes[1].set_title('Assembly (%)')
    axes[1].set_xlabel('Step')
    axes[2].set_title('Largest Cluster')
    axes[2].set_xlabel('Step')
    for ax in axes:
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)


def _draw_clusters_mpl(ax, positions, cluster_ids, bounds, fs=8):
    unique = np.unique(cluster_ids)
    cmap = plt.cm.get_cmap('tab20', max(len(unique), 2))
    for idx, cid in enumerate(unique):
        mask = cluster_ids == cid
        n_m = int(np.sum(mask))
        p = positions[mask] * 1000   # → mm
        sz = max(30, 12 * n_m)
        ax.scatter(p[:, 0], p[:, 1], p[:, 2],
                   c=[cmap(idx % 20)] * n_m,
                   s=sz, alpha=0.85, edgecolors='k', linewidth=0.3)
    b = bounds * 1000
    ax.set_xlim(-b, b)
    ax.set_ylim(-b, b)
    ax.set_zlim(-b, b)
    ax.set_xlabel('X (mm)', fontsize=fs)
    ax.set_ylabel('Y (mm)', fontsize=fs)
    ax.set_zlabel('Z (mm)', fontsize=fs)
    ax.tick_params(labelsize=max(5, fs - 2))
