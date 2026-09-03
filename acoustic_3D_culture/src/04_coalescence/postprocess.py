"""
postprocess.py  —  prepare Stage 3 output for Basilisk and post-process Basilisk results
========================================================================================

Usage
-----
    # Before running Basilisk (export impact conditions):
    python postprocess.py --export

    # After Basilisk runs (plot results from log.csv):
    python postprocess.py --plot

"""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as cfg

HERE = pathlib.Path(__file__).parent


def export_impact_conditions():
    """
    Read impact_conditions.npz from Stage 3 and write a plain-text file
    that the Basilisk C code reads at startup.

    Writes:  output/impact_conditions.txt
    """
    ic_path = cfg.OUTPUT_DIR / "impact_conditions.npz"
    if not ic_path.exists():
        print("impact_conditions.npz not found.  Run Stage 3 first.")
        return

    ic = np.load(ic_path)
    v_impact = float(ic["v_impact"][0]) if "v_impact" in ic else 5e-4
    # Impact parameter b = lateral offset / droplet diameter
    # estimated from relative position of first impacting pair
    if "pos_i" in ic and "pos_j" in ic:
        dp = ic["pos_i"][0] - ic["pos_j"][0]
        dp_perp = np.sqrt(dp[1]**2 + dp[2]**2)   # lateral separation
        b = float(dp_perp / (2.0 * cfg.A_DROP))
    else:
        b = 0.0

    out = cfg.OUTPUT_DIR / "impact_conditions.txt"
    with open(out, "w") as f:
        f.write(f"{v_impact:.6e} {b:.6f}\n")

    print(f"Impact velocity : {v_impact*1e3:.4f} mm/s")
    print(f"Impact parameter: {b:.4f}")
    We = cfg.RHO_DROP * v_impact**2 * 2.0 * cfg.A_DROP / cfg.SIGMA
    print(f"Weber number    : {We:.4f}  ({'gentle coalescence ✓' if We < 5 else 'check We'})")
    print(f"Written → {out}")

    # Also copy to coalescence directory so Basilisk can read it relative to cwd
    import shutil
    dest = HERE / ".." / ".." / "output" / "impact_conditions.txt"
    shutil.copy(out, dest.resolve())
    print(f"Copied  → {dest.resolve()}")


def plot_basilisk_results():
    """
    Read log.csv produced by Basilisk and/or VTK snapshots and produce
    diagnostic plots.
    """
    coal_dir = HERE
    log_path = coal_dir / "log.csv"

    if not log_path.exists():
        # Try output directory
        log_path = cfg.OUTPUT_DIR / "coalescence" / "log.csv"

    if not log_path.exists():
        print(f"log.csv not found at {log_path}")
        print("Run Basilisk first (see README_wsl.md).")
        return

    data = np.loadtxt(log_path, delimiter=",", skiprows=1)
    t    = data[:, 0]
    vol  = data[:, 1]
    Ek   = data[:, 2]

    # Expected final volume = volume of two droplets merged
    V1 = (4/3) * np.pi * cfg.A_DROP**3
    V_expected = 2.0 * V1

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(t * 1e3, vol / V_expected, "b-", lw=1.5)
    axes[0].axhline(1.0, color="grey", linestyle="--", lw=0.8, label="Expected (merged)")
    axes[0].set_xlabel("Time [ms]")
    axes[0].set_ylabel("Droplet volume / V_merged")
    axes[0].set_title("Volume conservation (should ≈ 1.0 after coalescence)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t * 1e3, Ek, "r-", lw=1.5)
    axes[1].set_xlabel("Time [ms]")
    axes[1].set_ylabel("Kinetic energy [J]")
    axes[1].set_title("Kinetic energy (decay indicates damped oscillation)")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Basilisk Droplet Coalescence Diagnostics", fontsize=13)
    plt.tight_layout()

    cfg.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = cfg.FIGURES_DIR / "coalescence_diagnostics.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")

    print(f"\nFinal volume / Vmerged = {vol[-1]/V_expected:.4f}  (should ≈ 1.0)")
    print(f"Final kinetic energy   = {Ek[-1]:.3e} J  (should ≈ 0)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", action="store_true",
                        help="Export impact conditions for Basilisk")
    parser.add_argument("--plot",   action="store_true",
                        help="Plot Basilisk results from log.csv")
    args = parser.parse_args()

    if args.export:
        export_impact_conditions()
    elif args.plot:
        plot_basilisk_results()
    else:
        parser.print_help()
