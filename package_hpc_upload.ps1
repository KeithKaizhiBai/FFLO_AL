$ErrorActionPreference = "Stop"

$dateTag = Get-Date -Format "yyyyMMdd_HHmmss"
$bundleName = "hpc_upload_qdelta_discovery_512seed_256x50_$dateTag"
$archive = "$bundleName.tar.gz"
$staging = Join-Path (Get-Location) $bundleName

if (Test-Path -LiteralPath $staging) {
    Remove-Item -LiteralPath $staging -Recurse -Force
}

New-Item -ItemType Directory -Force -Path $staging | Out-Null

$items = @(
    "ml_phase",
    "scripts",
    "report",
    "docs",
    "AGENTS.md",
    "eta_phase_diagram_cuda.py",
    "tfflo_1d_cuda.py",
    "plot_eta_phase_diagram.py",
    "finite_T_phase_diagram.m",
    "MODEL_SPEC.md",
    "PROJECT_SUMMARY.md",
    "QDELTA_REFINEMENT_EXECUTION_PLAN.md",
    "QDELTA_TARGET_LOGIC_CODE_REWRITE_PLAN.md",
    "RUN_ORDER_GBU_HPC.md",
    "Active_Learning_Phase_Boundary_Refinement_Plan.md",
    "hpc_active_loop.sh",
    "run_discovery_512x50.sh",
    "run_discovery_512x50_background.sh",
    "hpc_run_readme.md"
)

foreach ($item in $items) {
    if (-not (Test-Path -LiteralPath $item)) {
        Write-Warning "Skipping missing item: $item"
        continue
    }
    $dest = Join-Path $staging $item
    $destParent = Split-Path -Parent $dest
    if ($destParent -and -not (Test-Path -LiteralPath $destParent)) {
        New-Item -ItemType Directory -Force -Path $destParent | Out-Null
    }
    Copy-Item -LiteralPath $item -Destination $dest -Recurse -Force
}

Get-ChildItem -Path $staging -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force
Get-ChildItem -Path $staging -Recurse -Directory -Filter ".pytest_cache" | Remove-Item -Recurse -Force
Get-ChildItem -Path $staging -Recurse -File -Include "*.pyc","*.pyo","slurm-*.out" | Remove-Item -Force

$generatedDirs = @(
    "ml_phase/active_runs",
    "ml_phase/datasets",
    "ml_phase/figures",
    "ml_phase/hpc_jobs",
    "ml_phase/models",
    "ml_phase/reports"
)

foreach ($dir in $generatedDirs) {
    $path = Join-Path $staging $dir
    if (Test-Path -LiteralPath $path) {
        Remove-Item -LiteralPath $path -Recurse -Force
    }
}

$reportDir = Join-Path $staging "report"
if (Test-Path -LiteralPath $reportDir) {
    Get-ChildItem -LiteralPath $reportDir -File |
        Where-Object { $_.Name -match '\.(aux|log|out|pdf|synctex\.gz)$' } |
        Remove-Item -Force
}

$reportFigures = Join-Path $staging "report/figures"
if (Test-Path -LiteralPath $reportFigures) {
    Get-ChildItem -LiteralPath $reportFigures -File |
        Where-Object { $_.Name -match '\.(png|pdf|jpg|jpeg)$' } |
        Remove-Item -Force
}

if (Test-Path -LiteralPath $archive) {
    Remove-Item -LiteralPath $archive -Force
}

tar -czf $archive $bundleName

Write-Host "Created archive: $archive"
Write-Host "Transfer this file to the cluster, then run:"
Write-Host "  tar -xzf $archive"
Write-Host "  cd $bundleName"
Write-Host "  export PROJECT_DIR=`$PWD"
Write-Host "  export PYTHON_BIN=/public_hw/home/sci_bfu/.conda/envs/my_env/bin/python"
Write-Host "  export EXCLUDE_NODES=gpuh01"
Write-Host "  bash run_discovery_512x50_background.sh"
Write-Host ""
Write-Host "Production defaults:"
Write-Host "  RUN_ID=active_boundary_discovery_512seed_256x50"
Write-Host "  RUN_MODE=discovery, CANDIDATE_DOMAIN_MODE=full"
Write-Host "  INITIAL_SEED_SIZE=512, BATCH_SIZE_MAX=256, N_ITERS=100, WORLD_SIZE=8"
Write-Host "  SELECTION_MODE=stochastic, ACTIVE_POOL_RULE=max_threshold, ACTIVE_POOL_REL_TO_P95=0.7"
Write-Host "  B_DELTA_GATE_MODE=normal_sc_competition, SAMPLING_POWER_SCHEDULE=piecewise"
Write-Host "  FINITE_T_BAND_WIDTH disabled"
