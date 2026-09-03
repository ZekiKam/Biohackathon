# Stage 4: Droplet Coalescence — Basilisk Setup (WSL 2)

Basilisk is a free, open-source PDE solver for two-phase flow with surface tension.
It runs on Linux (or WSL 2 on Windows) and is ideal for simulating droplet coalescence
because it uses the Volume-of-Fluid method with adaptive mesh refinement.

---

## 1. Install WSL 2 (Windows Subsystem for Linux)

Open PowerShell as Administrator and run:

```powershell
wsl --install -d Ubuntu-22.04
```

Restart your machine when prompted.

---

## 2. Install Basilisk inside WSL 2

Open an Ubuntu terminal and run these commands:

```bash
# Install build tools and dependencies
sudo apt update
sudo apt install -y build-essential gcc make darcs libgl1-mesa-dev ffmpeg

# Download Basilisk source
darcs clone http://basilisk.fr/basilisk

# Set up environment
echo 'export BASILISK=$HOME/basilisk' >> ~/.bashrc
echo 'export PATH=$PATH:$BASILISK' >> ~/.bashrc
source ~/.bashrc

# Build the qcc compiler
cd $BASILISK
make

# Test installation
qcc --version
```

If `darcs` is not available:
```bash
sudo apt install -y darcs
```

Alternatively, download the Basilisk tarball from http://basilisk.fr

---

## 3. Build and Run the Coalescence Simulation

From inside WSL 2, navigate to the coalescence directory:

```bash
# Navigate to the project (adjust path to match your Windows username)
cd /mnt/c/Users/YOUR_USERNAME/OneDrive\ -\ University\ of\ Edinburgh/\
files/Professional\ Development/acoustic_3D_culture/src/04_coalescence

# Build
make

# Export impact conditions from Stage 3 (run from project root)
cd ../..
python src/04_coalescence/postprocess.py --export

# Run the simulation
cd src/04_coalescence
./coalescence
```

The simulation creates:
- `snapshots/snapshot-NNNN.vtk` — 3D VTK snapshots (open in ParaView)
- `log.csv` — time, volume, and kinetic energy diagnostics

---

## 4. Check Results

```bash
# Quick plot of diagnostics (run from the project root on Windows)
python src/04_coalescence/postprocess.py --plot
```

Open the VTK snapshots in ParaView (Windows):
1. File → Open → navigate to `src/04_coalescence/snapshots/`
2. Select all `snapshot-*.vtk` files
3. Click Apply → Play animation

---

## 5. Expected Physical Results

For our system:
- Droplet radius: 1.0 mm
- Impact velocity: ~0.5 mm/s (from Stage 3)
- Weber number: We ≈ 0.007 → **gentle coalescence** (no shattering/bouncing)

After coalescence, the merged droplet oscillates at the Rayleigh frequency:
$$f_R = \frac{1}{2\pi}\sqrt{\frac{8\sigma}{\rho_p D_\text{merged}^3}}$$

For a 1.26 mm radius merged droplet (volume = 2 × original):
$f_R \approx 165\,\text{Hz}$

This oscillation damps over ~10 ms due to viscosity, leaving a stable spherical
droplet at the acoustic trap centre — now containing both GBM spheroids fused
into a single tumoroid.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `qcc: command not found` | Source `~/.bashrc` or add `$BASILISK` to PATH |
| `output_vtk` undefined | Use Basilisk version ≥ 2022; update with `darcs pull` |
| Build fails with GL errors | `sudo apt install -y libgl1-mesa-dev` |
| Slow simulation | Reduce `MAXLEVEL` from 8 to 7 in the C file |
| ParaView can't read VTK | In ParaView, select "Legacy VTK" reader |
