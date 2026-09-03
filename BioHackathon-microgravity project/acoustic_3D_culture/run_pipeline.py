"""
run_pipeline.py — Master orchestration script
=============================================
Runs all simulation stages in sequence.  Each stage is independent and
writes its outputs to  output/  so any stage can be re-run in isolation.

Usage
-----
    python run_pipeline.py                     # run all stages 1–3 + 5
    python run_pipeline.py --stages 1 2        # run only stages 1 and 2
    python run_pipeline.py --backend kwave     # use k-Wave for Stage 2
    python run_pipeline.py --no-vis            # skip Stage 5

Stage 4 (Basilisk coalescence) is excluded from automatic execution because
it requires WSL 2 on Windows.  See src/04_coalescence/README_wsl.md.
"""

import sys
import io
import os
import argparse
import time
import pathlib
import traceback

# Ensure Unicode math symbols print correctly on Windows (cp1252 consoles).
# PYTHONUTF8=1 or -X utf8 is the preferred fix; this is a belt-and-suspenders
# fallback that only kicks in when the active encoding really is non-UTF-8 and
# a raw buffer is available (i.e. a real terminal, not an already-wrapped stream).
for _stream_name in ("stdout", "stderr"):
    _s = getattr(sys, _stream_name)
    if (getattr(_s, "encoding", "utf-8") or "utf-8").lower().replace("-", "") != "utf8":
        _buf = getattr(_s, "buffer", None)
        if _buf is not None:
            setattr(sys, _stream_name,
                    io.TextIOWrapper(_buf, encoding="utf-8",
                                     errors="replace", line_buffering=True))

# Always run from the project root (the folder containing this file),
# regardless of where PowerShell / cmd is currently sitting.
ROOT = pathlib.Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))


def run_stage(name: str, fn, *args, **kwargs):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    t0 = time.perf_counter()
    try:
        result = fn(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        print(f"\n  [OK]  {name} completed in {elapsed:.1f} s")
        return result
    except Exception:
        elapsed = time.perf_counter() - t0
        print(f"\n  [X]  {name} FAILED after {elapsed:.1f} s")
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Acoustic 3D GBM culture simulation pipeline"
    )
    parser.add_argument(
        "--stages", nargs="+", type=int, default=[1, 2, 3, 5],
        help="Stages to run (default: 1 2 3 5 — stage 4 requires WSL 2)"
    )
    parser.add_argument(
        "--backend", choices=["analytical", "kwave"], default="analytical",
        help="Stage 2 pressure-field backend (default: analytical)"
    )
    parser.add_argument(
        "--no-vis", action="store_true",
        help="Skip Stage 5 visualisation"
    )
    args = parser.parse_args()

    stages = set(args.stages)
    if args.no_vis:
        stages.discard(5)

    print("Acoustic 3D GBM Culture — Simulation Pipeline")
    print(f"Stages to run: {sorted(stages)}")
    print(f"Stage 2 backend: {args.backend}")

    total_start = time.perf_counter()

    if 1 in stages:
        from importlib import import_module
        s1 = import_module("01_phase_computation")
        run_stage("Stage 1 — GS-PAT Phase Computation", s1.main)

    if 2 in stages:
        from importlib import import_module
        s2 = import_module("02_acoustic_field")
        run_stage("Stage 2 — Acoustic Pressure Field",
                  s2.main, backend=args.backend)

    if 3 in stages:
        from importlib import import_module
        s3 = import_module("03_gorkov_tracking")
        run_stage("Stage 3 — Gorkov Potential & Droplet Tracking", s3.main)

    if 4 in stages:
        print("\n" + "="*60)
        print("  Stage 4 — Droplet Coalescence (Basilisk)")
        print("="*60)
        print("  Stage 4 runs in WSL 2.  Follow the instructions in:")
        print("  src/04_coalescence/README_wsl.md")
        print("\n  To export impact conditions for Basilisk, run:")
        print("    python src/04_coalescence/postprocess.py --export")

    if 5 in stages:
        from importlib import import_module
        s5 = import_module("05_visualize")
        run_stage("Stage 5 — Visualisation", s5.main)

    total = time.perf_counter() - total_start
    print(f"\n{'='*60}")
    print(f"  Pipeline complete in {total:.1f} s")
    print(f"  Outputs: {ROOT / 'output'}")
    print(f"  Figures: {ROOT / 'output' / 'figures'}")
    print("="*60)


if __name__ == "__main__":
    main()
