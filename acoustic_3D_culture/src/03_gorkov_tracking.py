"""
Stage 3 — Gorkov Radiation-Force Potential & Droplet Trajectory Simulation
==========================================================================
Uses the 3-D pressure field from Stage 2 to:

  1. Compute the Gorkov radiation-pressure potential U_rad(x).
  2. Compute the acoustic radiation force  F_rad = -∇U_rad.
  3. Integrate Newton's equations of motion for N_DROPLETS water droplets
     entering from randomised positions near the sphere wall, subject to:
       • Acoustic radiation force (F_rad)
       • Stokes drag in air (F_drag = -6πηa · v)
  4. Detect when any two droplets come within coalescence distance and
     record the impact conditions (positions, velocities) for Stage 4.

Physics (Gorkov 1962 / Bruus 2012)
-----------------------------------
For a spherical droplet (volume V, density ρ_p, speed of sound c_p)
in a medium (ρ_0, c_0) with monochromatic pressure field p̂(x):

    U_rad(x) = V · [ f1 · |p̂|² / (4ρ_0 c_0²)
                   - f2 · 3|∇p̂|² / (4ω²ρ_0) ]

    f1 = 1 − ρ_0 c_0² / (ρ_p c_p²)   ≈ +1  for water-in-air
    f2 = 2(ρ_p − ρ_0) / (2ρ_p + ρ_0) ≈ +1  for water-in-air

Both contrast factors are positive for water in air: droplets are pulled
toward pressure NODES (U_rad minimised where |p̂|² is minimised and
|∇p̂|² is maximised — numerically the dominant term is usually |∇p̂|²
which is zero at a pressure node, so U is minimised there).

The gradient ∇p̂ is computed with 2nd-order central differences on the grid.

Equations of motion (overdamped regime)
----------------------------------------
    m ẍ = F_rad(x) − ζ ẋ
    ζ = 6π η a   (Stokes drag coefficient)

Integrated with scipy RK45 adaptive solver.

Output
------
output/gorkov_potential.npz  ->  x, y, z grids, U_rad, F_rad (x/y/z components)
output/trajectories.npz      ->  time, positions, velocities per droplet
output/impact_conditions.npz ->  impact pair positions & velocities (-> Stage 4)
output/figures/gorkov_potential_xz.png
output/figures/trajectories_3d.png
output/figures/trajectories_animation.gif
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator
from tqdm import trange

import config as cfg


# ---------------------------------------------------------------------------
# 1. Gorkov potential from pressure field
# ---------------------------------------------------------------------------

def compute_gorkov_potential(x, y, z, p_complex: np.ndarray):
    """
    Compute Gorkov potential on the same grid as p_complex.

    Parameters
    ----------
    x, y, z    : 1-D coordinate arrays [m]
    p_complex  : (Nx, Ny, Nz) complex pressure amplitude [Pa or normalised]

    Returns
    -------
    U  : (Nx, Ny, Nz) Gorkov potential  [J or normalised]
    Fx : (Nx, Ny, Nz) x-component of radiation force
    Fy, Fz : similarly
    """
    print("  Computing |p|² and |∇p|² …")
    p_abs2 = np.abs(p_complex) ** 2     # |p̂|²

    # Gradient of p̂ using 2nd-order central finite differences
    # We need |∇p̂|² = |∂p/∂x|² + |∂p/∂y|² + |∂p/∂z|²
    dx_spacing = x[1] - x[0]
    dy_spacing = y[1] - y[0]
    dz_spacing = z[1] - z[0]

    # np.gradient returns the gradient of a REAL array; for complex we
    # compute gradient of real and imaginary parts separately.
    def _grad_complex(arr, spacing, axis):
        return (  np.gradient(arr.real, spacing, axis=axis)
                + 1j * np.gradient(arr.imag, spacing, axis=axis))

    print("  Computing ∇p̂ …")
    grad_px = _grad_complex(p_complex, dx_spacing, axis=0)
    grad_py = _grad_complex(p_complex, dy_spacing, axis=1)
    grad_pz = _grad_complex(p_complex, dz_spacing, axis=2)

    grad_p_abs2 = (np.abs(grad_px)**2 +
                   np.abs(grad_py)**2 +
                   np.abs(grad_pz)**2)    # |∇p̂|²

    # ---- Gorkov potential ----
    # U = V [f1 |p̂|² / (4 ρ0 c0²) − 3f2 |∇p̂|² / (4 ω² ρ0)]
    term_pressure  = cfg.F1 * p_abs2       / (4.0 * cfg.RHO0 * cfg.C0**2)
    term_velocity  = cfg.F2 * 3.0 * grad_p_abs2 / (4.0 * cfg.OMEGA**2 * cfg.RHO0)
    U = cfg.V_DROP * (term_pressure - term_velocity)   # (Nx, Ny, Nz)

    # ---- Radiation force = −∇U ----
    print("  Computing F_rad = −∇U …")
    Fx = -np.gradient(U, dx_spacing, axis=0)
    Fy = -np.gradient(U, dy_spacing, axis=1)
    Fz = -np.gradient(U, dz_spacing, axis=2)

    U_min  = float(U.min())
    U_max  = float(U.max())
    F_max  = float(np.sqrt(Fx**2 + Fy**2 + Fz**2).max())
    print(f"  U_rad range : [{U_min:.3e}, {U_max:.3e}] J (normalised)")
    print(f"  |F_rad| max : {F_max:.3e} N (normalised)")

    return U, Fx, Fy, Fz


def build_force_interpolators(x, y, z, Fx, Fy, Fz):
    """
    Build tri-linear RegularGridInterpolators so the ODE solver can query
    force at arbitrary positions.
    """
    kw = dict(method="linear", bounds_error=False, fill_value=0.0)
    interp_fx = RegularGridInterpolator((x, y, z), Fx, **kw)
    interp_fy = RegularGridInterpolator((x, y, z), Fy, **kw)
    interp_fz = RegularGridInterpolator((x, y, z), Fz, **kw)
    return interp_fx, interp_fy, interp_fz


# ---------------------------------------------------------------------------
# 2. Droplet trajectory simulation
# ---------------------------------------------------------------------------

def make_ode_rhs(interp_fx, interp_fy, interp_fz, mass, stokes):
    """
    Return the ODE right-hand side for a single droplet:
        state = [x, y, z, vx, vy, vz]
        d/dt(state) = [vx, vy, vz, ax, ay, az]
    """
    def rhs(t, state):
        pos = np.array([[state[0], state[1], state[2]]])
        fx  = float(interp_fx(pos))
        fy  = float(interp_fy(pos))
        fz  = float(interp_fz(pos))
        vx, vy, vz = state[3], state[4], state[5]
        ax  = (fx - stokes * vx) / mass
        ay  = (fy - stokes * vy) / mass
        az  = (fz - stokes * vz) / mass
        return [vx, vy, vz, ax, ay, az]

    return rhs


def random_entry_position(r_entry: float, rng: np.random.Generator) -> np.ndarray:
    """
    Sample a uniformly-random point on the sphere of radius r_entry.
    Uses Marsaglia's method (two standard normals -> normalise).
    """
    v = rng.standard_normal(3)
    return r_entry * v / np.linalg.norm(v)


def simulate_droplets(interp_fx, interp_fy, interp_fz,
                      n_droplets: int,
                      t_end: float,
                      dt_out: float,
                      trap_pos: np.ndarray,
                      entry_radius: float,
                      mass: float,
                      stokes: float):
    """
    Simulate N independent droplets entering the sphere from random positions.

    Returns
    -------
    t_arr   : (Nt,) time array
    pos_all : (N, Nt, 3) positions
    vel_all : (N, Nt, 3) velocities
    """
    rng    = np.random.default_rng(seed=0)
    t_eval = np.arange(0.0, t_end + dt_out, dt_out)
    Nt     = len(t_eval)

    pos_all = np.zeros((n_droplets, Nt, 3))
    vel_all = np.zeros((n_droplets, Nt, 3))

    rhs = make_ode_rhs(interp_fx, interp_fy, interp_fz, mass, stokes)

    for i in trange(n_droplets, desc="  Integrating droplets"):
        x0 = random_entry_position(entry_radius, rng)
        # Initial velocity: small push toward trap
        v_dir = trap_pos - x0
        v_dir = v_dir / np.linalg.norm(v_dir)
        v0    = v_dir * 1e-4      # 0.1 mm/s initial nudge

        state0 = np.concatenate([x0, v0])

        # Terminate early if droplet reaches the trap
        def reached_trap(t, state, tol=cfg.A_DROP * 2.0):
            return np.linalg.norm(state[:3] - trap_pos) - tol
        reached_trap.terminal  = True
        reached_trap.direction = -1.0

        sol = solve_ivp(
            rhs, [0.0, t_end], state0,
            method="RK45",
            t_eval=t_eval,
            events=reached_trap,
            dense_output=False,
            rtol=1e-6, atol=1e-9,
            max_step=dt_out,
        )

        Nt_sol  = sol.y.shape[1]
        pos_all[i, :Nt_sol, :] = sol.y[:3].T
        vel_all[i, :Nt_sol, :] = sol.y[3:].T

        # Freeze at last position after termination
        if Nt_sol < Nt:
            pos_all[i, Nt_sol:, :] = pos_all[i, Nt_sol - 1, :]

    return t_eval, pos_all, vel_all


# ---------------------------------------------------------------------------
# 3. Detect coalescence impact conditions
# ---------------------------------------------------------------------------

def detect_impacts(t_arr, pos_all, vel_all, coalescence_distance: float):
    """
    Find the first time step when any pair of droplets is within
    coalescence_distance of each other.

    Returns a list of dicts with keys: t, i, j, pos_i, pos_j, vel_i, vel_j,
    v_impact (relative speed), Weber_number.
    """
    impacts     = []
    n_droplets  = pos_all.shape[0]
    coal_dist2  = coalescence_distance**2

    for ti, t in enumerate(t_arr):
        for i in range(n_droplets):
            for j in range(i + 1, n_droplets):
                dp  = pos_all[i, ti] - pos_all[j, ti]
                d2  = float(np.dot(dp, dp))
                if d2 < coal_dist2:
                    dv      = vel_all[i, ti] - vel_all[j, ti]
                    v_imp   = float(np.linalg.norm(dv))
                    We      = (cfg.RHO_DROP * v_imp**2 * 2.0 * cfg.A_DROP) / cfg.SIGMA
                    impacts.append(dict(
                        t=t, i=i, j=j,
                        pos_i=pos_all[i, ti].copy(),
                        pos_j=pos_all[j, ti].copy(),
                        vel_i=vel_all[i, ti].copy(),
                        vel_j=vel_all[j, ti].copy(),
                        v_impact=v_imp,
                        Weber=We,
                    ))
                    # record only the first event for each pair
    # Deduplicate: keep first event per (i, j) pair
    seen   = set()
    unique = []
    for ev in impacts:
        key = (ev["i"], ev["j"])
        if key not in seen:
            seen.add(key)
            unique.append(ev)
    return unique


# ---------------------------------------------------------------------------
# 4. Saving
# ---------------------------------------------------------------------------

def save_results(x, y, z, U, Fx, Fy, Fz, t_arr, pos_all, vel_all, impacts):
    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    np.savez(
        cfg.OUTPUT_DIR / "gorkov_potential.npz",
        grid_x=x, grid_y=y, grid_z=z,
        U_rad=U, Fx=Fx, Fy=Fy, Fz=Fz,
    )
    print(f"  Saved -> {cfg.OUTPUT_DIR / 'gorkov_potential.npz'}")

    np.savez(
        cfg.OUTPUT_DIR / "trajectories.npz",
        t=t_arr, positions=pos_all, velocities=vel_all,
    )
    print(f"  Saved -> {cfg.OUTPUT_DIR / 'trajectories.npz'}")

    if impacts:
        # Flatten for npz
        imp_data = {k: np.array([ev[k] for ev in impacts]) for k in impacts[0]}
        np.savez(cfg.OUTPUT_DIR / "impact_conditions.npz", **imp_data)
        print(f"  Saved {len(impacts)} impact event(s) -> "
              f"{cfg.OUTPUT_DIR / 'impact_conditions.npz'}")
    else:
        print("  No coalescence events detected (droplets may not have converged yet).")
        print("  Consider increasing T_END in config.py.")


# ---------------------------------------------------------------------------
# 5. Plotting
# ---------------------------------------------------------------------------

def plot_gorkov(x, y, z, U, Fx, Fz, trap_pos):
    cfg.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    iy_mid = np.argmin(np.abs(y))
    U_xz   = U[:, iy_mid, :]
    Fx_xz  = Fx[:, iy_mid, :]
    Fz_xz  = Fz[:, iy_mid, :]

    ext = [z.min()*1e2, z.max()*1e2, x.min()*1e2, x.max()*1e2]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Gorkov potential
    ax = axes[0]
    vmax = np.percentile(np.abs(U_xz), 99)
    im   = ax.imshow(U_xz, extent=ext, origin="lower",
                     cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                     aspect="equal", interpolation="bilinear")
    plt.colorbar(im, ax=ax, label="U_rad [J normalised]")
    ax.set_xlabel("Z [cm]"); ax.set_ylabel("X [cm]")
    ax.set_title("Gorkov Potential U_rad  (XZ plane, y=0)")
    ax.scatter(trap_pos[2]*1e2, trap_pos[0]*1e2, c="cyan", s=120,
               zorder=5, label="Trap", marker="*")
    # sphere boundary
    th  = np.linspace(0, 2*np.pi, 200)
    ax.plot(np.cos(th)*cfg.R_SPHERE*1e2, np.sin(th)*cfg.R_SPHERE*1e2,
            "w--", lw=0.8, alpha=0.5)
    ax.legend(fontsize=8)

    # Force quiver (down-sampled for clarity)
    ax2   = axes[1]
    ds    = max(1, U_xz.shape[0] // 20)
    Xi    = x[::ds] * 1e2
    Zi    = z[::ds] * 1e2
    Fxs   = Fx_xz[::ds, ::ds]
    Fzs   = Fz_xz[::ds, ::ds]
    F_mag = np.sqrt(Fxs**2 + Fzs**2) + 1e-30
    Xm, Zm = np.meshgrid(Zi, Xi)
    ax2.quiver(Zm, Xm, Fzs/F_mag, Fxs/F_mag,
               F_mag, cmap="plasma", scale=30, width=0.003)
    ax2.set_xlabel("Z [cm]"); ax2.set_ylabel("X [cm]")
    ax2.set_title("Radiation Force Direction (XZ plane)")
    ax2.scatter(trap_pos[2]*1e2, trap_pos[0]*1e2, c="cyan", s=120, zorder=5, marker="*")
    ax2.plot(np.cos(th)*cfg.R_SPHERE*1e2, np.sin(th)*cfg.R_SPHERE*1e2,
             "k--", lw=0.8, alpha=0.5)
    ax2.set_aspect("equal")

    fig.suptitle("Gorkov Radiation-Force Potential (water droplets in air)", fontsize=12)
    plt.tight_layout()
    out = cfg.FIGURES_DIR / "gorkov_potential_xz.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out}")


def plot_trajectories(t_arr, pos_all, trap_pos):
    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection="3d")

    colors = plt.cm.tab10(np.linspace(0, 1, pos_all.shape[0]))
    for i, (traj, col) in enumerate(zip(pos_all, colors)):
        ax.plot(traj[:, 0]*1e2, traj[:, 1]*1e2, traj[:, 2]*1e2,
                color=col, lw=1.5, label=f"Droplet {i+1}")
        ax.scatter(*traj[0]*1e2, color=col, s=60, marker="o", zorder=5)  # start
        ax.scatter(*traj[-1]*1e2, color=col, s=60, marker="X", zorder=5)  # end

    ax.scatter(*trap_pos*1e2, c="red", s=200, marker="*", zorder=10, label="Trap")
    # Draw sphere wireframe
    u, v    = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    xs = cfg.R_SPHERE*1e2 * np.cos(u) * np.sin(v)
    ys = cfg.R_SPHERE*1e2 * np.sin(u) * np.sin(v)
    zs = cfg.R_SPHERE*1e2 * np.cos(v)
    ax.plot_wireframe(xs, ys, zs, color="grey", alpha=0.15, linewidth=0.5)

    ax.set_xlabel("X [cm]"); ax.set_ylabel("Y [cm]"); ax.set_zlabel("Z [cm]")
    ax.set_title("Droplet Trajectories Under Acoustic Radiation Force")
    ax.legend(fontsize=8, loc="upper left")

    out = cfg.FIGURES_DIR / "trajectories_3d.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out}")


def save_trajectory_animation(t_arr, pos_all, trap_pos, stride=5):
    """Save a lightweight animated GIF of droplet trajectories."""
    fig = plt.figure(figsize=(7, 7))
    ax  = fig.add_subplot(111, projection="3d")
    ax.set_xlim([-cfg.R_SPHERE*1e2, cfg.R_SPHERE*1e2])
    ax.set_ylim([-cfg.R_SPHERE*1e2, cfg.R_SPHERE*1e2])
    ax.set_zlim([-cfg.R_SPHERE*1e2, cfg.R_SPHERE*1e2])
    ax.set_xlabel("X [cm]"); ax.set_ylabel("Y [cm]"); ax.set_zlabel("Z [cm]")

    colors  = plt.cm.tab10(np.linspace(0, 1, pos_all.shape[0]))
    n_frames = len(t_arr) // stride
    scatter_handles = []
    for col in colors:
        sc, = ax.plot([], [], [], "o", color=col, markersize=6)
        scatter_handles.append(sc)
    ax.scatter(*trap_pos*1e2, c="red", s=200, marker="*", zorder=10)

    def update(frame):
        ti = frame * stride
        for i, sc in enumerate(scatter_handles):
            p = pos_all[i, ti] * 1e2
            sc.set_data([p[0]], [p[1]])
            sc.set_3d_properties([p[2]])
        ax.set_title(f"t = {t_arr[ti]:.3f} s")
        return scatter_handles

    ani = animation.FuncAnimation(fig, update, frames=n_frames,
                                   interval=50, blit=False)
    out = cfg.FIGURES_DIR / "trajectories_animation.gif"
    ani.save(str(out), writer="pillow", fps=20)
    plt.close(fig)
    print(f"  Saved -> {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Stage 3 — Gorkov Potential & Droplet Tracking")
    print("=" * 60)

    # Load pressure field from Stage 2
    pf_path = cfg.OUTPUT_DIR / "pressure_field.npz"
    if not pf_path.exists():
        print("  pressure_field.npz not found — running Stage 2 first …")
        from importlib import import_module
        stage2 = import_module("02_acoustic_field")
        stage2.main()

    pf = np.load(pf_path)
    x, y, z    = pf["grid_x"], pf["grid_y"], pf["grid_z"]
    p_complex  = pf["p_complex"]
    print(f"  Loaded pressure field  {p_complex.shape}")

    # ---- Gorkov potential ----
    U, Fx, Fy, Fz = compute_gorkov_potential(x, y, z, p_complex)

    ix_trap = np.argmin(np.abs(x - cfg.TRAP_POS[0]))
    iy_trap = np.argmin(np.abs(y - cfg.TRAP_POS[1]))
    iz_trap = np.argmin(np.abs(z - cfg.TRAP_POS[2]))
    U_at_trap = float(U[ix_trap, iy_trap, iz_trap])

    # Build a mask that restricts the search to the sphere interior
    # (r < 0.85 * R_SPHERE), excluding near-wall transducer near-field.
    Xg, Yg, Zg = np.meshgrid(x, y, z, indexing="ij")
    R_grid = np.sqrt(Xg**2 + Yg**2 + Zg**2)
    interior_mask = R_grid < 0.85 * cfg.R_SPHERE
    U_interior = np.where(interior_mask, U, np.inf)
    U_interior_min = float(U_interior.min())

    print(f"\n  U_rad at trap position    : {U_at_trap:.4e}")
    print(f"  U_rad min (sphere interior): {U_interior_min:.4e}")
    if U_at_trap <= U_interior_min * 1.10:
        print("  [OK] Gorkov potential minimised at trap interior — stable node trap.")
    else:
        min_idx  = np.unravel_index(U_interior.argmin(), U.shape)
        dist_to_min = np.linalg.norm(
            np.array(min_idx) - np.array([ix_trap, iy_trap, iz_trap])
        ) * (x[1]-x[0]) * 1e3
        print(f"  ⚠  Interior potential minimum is {dist_to_min:.1f} mm from trap centre.")
        print("      Consider more GS-PAT iterations (GS_ITERATIONS in config.py).")

    # ---- Build force interpolators ----
    print("\n  Building force interpolators …")
    interp_fx, interp_fy, interp_fz = build_force_interpolators(x, y, z, Fx, Fy, Fz)

    # ---- Droplet trajectory simulation ----
    print(f"\n  Simulating {cfg.N_DROPLETS} droplets for t_end={cfg.T_END} s …")
    print(f"  Droplet radius : {cfg.A_DROP*1e3:.1f} mm, mass : {cfg.M_DROP:.2e} kg")
    print(f"  Stokes drag    : {cfg.STOKES_COEFF:.2e} N·s/m")

    t_arr, pos_all, vel_all = simulate_droplets(
        interp_fx, interp_fy, interp_fz,
        n_droplets    = cfg.N_DROPLETS,
        t_end         = cfg.T_END,
        dt_out        = cfg.DT_OUTPUT,
        trap_pos      = cfg.TRAP_POS,
        entry_radius  = cfg.ENTRY_RADIUS,
        mass          = cfg.M_DROP,
        stokes        = cfg.STOKES_COEFF,
    )

    # ---- Detect impacts ----
    coal_dist = 2.0 * cfg.A_DROP   # two droplets touching
    impacts   = detect_impacts(t_arr, pos_all, vel_all, coal_dist)

    for ev in impacts:
        print(f"\n  COALESCENCE event: droplets {ev['i']+1} & {ev['j']+1}  "
              f"at t={ev['t']:.3f} s")
        print(f"    Impact velocity : {ev['v_impact']*1e3:.2f} mm/s")
        print(f"    Weber number    : {ev['Weber']:.4f}  "
              f"({'gentle coalescence [OK]' if ev['Weber'] < 5 else 'check We regime'})")

    # ---- Save ----
    save_results(x, y, z, U, Fx, Fy, Fz, t_arr, pos_all, vel_all, impacts)

    # ---- Plot ----
    plot_gorkov(x, y, z, U, Fx, Fz, cfg.TRAP_POS)
    plot_trajectories(t_arr, pos_all, cfg.TRAP_POS)
    try:
        save_trajectory_animation(t_arr, pos_all, cfg.TRAP_POS, stride=max(1, len(t_arr)//100))
    except Exception as e:
        print(f"  (Animation skipped: {e})")

    print("\nStage 3 complete.")
    return t_arr, pos_all, vel_all, impacts


if __name__ == "__main__":
    main()
