# setup.ps1 — Windows environment setup for the acoustic 3D culture pipeline
# Run this once from the project root in PowerShell:
#   Set-ExecutionPolicy -Scope Process Bypass
#   .\setup.ps1

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Acoustic 3D Culture — Environment Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# ---------------------------------------------------------------------------
# 1. Check Python version
# ---------------------------------------------------------------------------
try {
    $pyver = python --version 2>&1
    Write-Host "`n[1] Python found: $pyver" -ForegroundColor Green
}
catch {
    Write-Host "`n[1] Python not found. Install from https://www.python.org" -ForegroundColor Red
    exit 1
}

# ---------------------------------------------------------------------------
# 2. Create virtual environment
# ---------------------------------------------------------------------------
$venv = ".venv"
if (-not (Test-Path $venv)) {
    Write-Host "`n[2] Creating virtual environment '$venv' ..." -ForegroundColor Yellow
    python -m venv $venv
}
else {
    Write-Host "`n[2] Virtual environment '$venv' already exists." -ForegroundColor Green
}

# Activate
. "$venv\Scripts\Activate.ps1"
Write-Host "    Activated: $venv" -ForegroundColor Green

# ---------------------------------------------------------------------------
# 3. Upgrade pip and install requirements
# ---------------------------------------------------------------------------
Write-Host "`n[3] Installing Python dependencies ..." -ForegroundColor Yellow
python -m pip install --upgrade pip --quiet
python -m pip install -r requirements.txt

Write-Host "    Dependencies installed." -ForegroundColor Green

# ---------------------------------------------------------------------------
# 4. Create output directories
# ---------------------------------------------------------------------------
Write-Host "`n[4] Creating output directories ..." -ForegroundColor Yellow
$dirs = @(
    "output",
    "output\figures",
    "output\paraview",
    "output\coalescence"
)
foreach ($d in $dirs) {
    if (-not (Test-Path $d)) {
        New-Item -ItemType Directory -Path $d | Out-Null
        Write-Host "    Created: $d" -ForegroundColor Green
    }
    else {
        Write-Host "    Exists : $d" -ForegroundColor DarkGray
    }
}

# ---------------------------------------------------------------------------
# 5. Optional: download k-Wave binary
# ---------------------------------------------------------------------------
Write-Host "`n[5] Checking k-Wave binary ..." -ForegroundColor Yellow
$kwave_ok = python -c "
try:
    import kwave
    print('ok')
except Exception as e:
    print(str(e))
" 2>&1

if ($kwave_ok -like "*ok*") {
    Write-Host "    k-wave-python is available." -ForegroundColor Green
}
else {
    Write-Host "    k-wave-python may need additional setup:" -ForegroundColor Yellow
    Write-Host "    $kwave_ok" -ForegroundColor DarkGray
    Write-Host "    The analytical backend (default) works without k-Wave." -ForegroundColor Cyan
}

# ---------------------------------------------------------------------------
# 6. WSL 2 check (for Stage 4 Basilisk)
# ---------------------------------------------------------------------------
Write-Host "`n[6] Checking WSL 2 for Basilisk (Stage 4) ..." -ForegroundColor Yellow
$wsl = wsl --status 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "    WSL 2 is available." -ForegroundColor Green
    Write-Host "    See src/04_coalescence/README_wsl.md to install Basilisk." -ForegroundColor Cyan
}
else {
    Write-Host "    WSL 2 not detected. Install with:" -ForegroundColor Yellow
    Write-Host "      wsl --install -d Ubuntu-22.04" -ForegroundColor White
    Write-Host "    Stage 4 (coalescence) requires WSL 2 + Basilisk." -ForegroundColor Cyan
}

# ---------------------------------------------------------------------------
# 7. Done
# ---------------------------------------------------------------------------
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  Setup complete!  Run the pipeline with:" -ForegroundColor Cyan
Write-Host ""
Write-Host "    . .venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "    python run_pipeline.py" -ForegroundColor White
Write-Host ""
Write-Host "  Or run individual stages:" -ForegroundColor Cyan
Write-Host "    python src\01_phase_computation.py" -ForegroundColor White
Write-Host "    python src\02_acoustic_field.py" -ForegroundColor White
Write-Host "    python src\03_gorkov_tracking.py" -ForegroundColor White
Write-Host "    python src\05_visualize.py" -ForegroundColor White
Write-Host "========================================" -ForegroundColor Cyan
