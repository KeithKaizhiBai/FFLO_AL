param(
    [string]$ProjectRoot = (Resolve-Path ".").Path
)

$ErrorActionPreference = "Stop"

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$historyRoot = Join-Path $ProjectRoot "project_history"
$manifestPath = Join-Path $historyRoot "root_reorganization_manifest_20260603.csv"
$readmePath = Join-Path $historyRoot "README.md"

$records = New-Object System.Collections.Generic.List[object]

function Ensure-Directory {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        New-Item -ItemType Directory -Force -Path $Path | Out-Null
    }
}

function Get-UniqueDestination {
    param(
        [string]$DestinationDirectory,
        [string]$Name
    )
    $candidate = Join-Path $DestinationDirectory $Name
    if (-not (Test-Path -LiteralPath $candidate)) {
        return $candidate
    }
    $stem = [System.IO.Path]::GetFileNameWithoutExtension($Name)
    $ext = [System.IO.Path]::GetExtension($Name)
    return (Join-Path $DestinationDirectory ("{0}_moved_{1}{2}" -f $stem, $timestamp, $ext))
}

function Move-RootItem {
    param(
        [string]$Name,
        [string]$DestinationRelative,
        [string]$Category,
        [string]$Note = ""
    )
    $source = Join-Path $ProjectRoot $Name
    $destDir = Join-Path $historyRoot $DestinationRelative
    Ensure-Directory $destDir

    if (-not (Test-Path -LiteralPath $source)) {
        $existing = Get-ChildItem -LiteralPath $historyRoot -Recurse -Force -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -eq $Name } |
            Select-Object -First 1
        if (-not $existing) {
            $privateRoot = Join-Path $ProjectRoot "private"
            if (Test-Path -LiteralPath $privateRoot) {
                $existing = Get-ChildItem -LiteralPath $privateRoot -Recurse -Force -ErrorAction SilentlyContinue |
                    Where-Object { $_.Name -eq $Name } |
                    Select-Object -First 1
            }
        }
        if ($existing) {
            $records.Add([pscustomobject]@{
                old_path = $Name
                new_path = (Resolve-Path -LiteralPath $existing.FullName).Path.Substring($ProjectRoot.Length + 1)
                category = $Category
                status = "already_moved"
                note = $Note
            })
            return
        }
        $records.Add([pscustomobject]@{
            old_path = $Name
            new_path = ""
            category = $Category
            status = "missing"
            note = $Note
        })
        return
    }

    $dest = Get-UniqueDestination -DestinationDirectory $destDir -Name (Split-Path -Leaf $source)
    try {
        Move-Item -LiteralPath $source -Destination $dest
        $records.Add([pscustomobject]@{
            old_path = $Name
            new_path = (Resolve-Path -LiteralPath $dest).Path.Substring($ProjectRoot.Length + 1)
            category = $Category
            status = "moved"
            note = $Note
        })
    } catch {
        $records.Add([pscustomobject]@{
            old_path = $Name
            new_path = ""
            category = $Category
            status = "failed"
            note = ($Note + " " + $_.Exception.Message).Trim()
        })
    }
}

function Move-RootGlob {
    param(
        [string]$Pattern,
        [string]$DestinationRelative,
        [string]$Category,
        [string]$Note = ""
    )
    $items = Get-ChildItem -LiteralPath $ProjectRoot -Force | Where-Object { $_.Name -like $Pattern }
    foreach ($item in $items) {
        Move-RootItem -Name $item.Name -DestinationRelative $DestinationRelative -Category $Category -Note $Note
    }
}

Ensure-Directory $historyRoot

# Plans, runbooks, and project notes that are not canonical specs.
$planFiles = @(
    "Active_Learning_Phase_Boundary_Refinement_Plan.md",
    "Full_reconstruct_plan.md",
    "ML_Guidance.md",
    "Phase_boundary_sharp_plan.md",
    "QDELTA_REFINEMENT_EXECUTION_PLAN.md",
    "QDELTA_TARGET_LOGIC_CODE_REWRITE_PLAN.md",
    "RUN_ORDER_GBU_HPC.md",
    "hpc_run_readme.md"
)
foreach ($file in $planFiles) {
    Move-RootItem -Name $file -DestinationRelative "plans_and_runbooks" -Category "plans_and_runbooks"
}

# Early exact data and legacy exploratory analysis artifacts.
Move-RootGlob -Pattern "eta_phase_diagram_nkt138_nja156*" -DestinationRelative "raw_exact_data" -Category "raw_exact_data"
Move-RootItem -Name "fflo_transition.ipynb" -DestinationRelative "legacy_analysis" -Category "legacy_analysis"
Move-RootItem -Name "finite_T_phase_diagram.m" -DestinationRelative "legacy_analysis" -Category "legacy_analysis"

# Historical HPC upload packages grouped by improvement stage.
Move-RootItem -Name "gbu_active_learning_hpc_upload_20260507_204427" -DestinationRelative "00_initial_hpc_uploads" -Category "hpc_initial_upload"

Move-RootGlob -Pattern "hpc_upload_qdelta_20260509*" -DestinationRelative "01_qdelta_warmup" -Category "hpc_qdelta_warmup"
Move-RootGlob -Pattern "hpc_upload_qdelta_20260510*" -DestinationRelative "01_qdelta_warmup" -Category "hpc_qdelta_warmup"
Move-RootGlob -Pattern "hpc_upload_qdelta_20260511*" -DestinationRelative "01_qdelta_warmup" -Category "hpc_qdelta_warmup"
Move-RootGlob -Pattern "hpc_upload_qdelta_20260512*" -DestinationRelative "01_qdelta_warmup" -Category "hpc_qdelta_warmup"
Move-RootGlob -Pattern "hpc_upload_qdelta_20260513*" -DestinationRelative "01_qdelta_warmup" -Category "hpc_qdelta_warmup"
Move-RootGlob -Pattern "hpc_upload_qdelta_warmup*" -DestinationRelative "01_qdelta_warmup" -Category "hpc_qdelta_warmup"

Move-RootGlob -Pattern "hpc_upload_qdelta_discovery_512seed_256x50_20260517*" -DestinationRelative "02_discovery_runs" -Category "hpc_discovery"
Move-RootGlob -Pattern "hpc_upload_qdelta_discovery_512seed_256x50_20260519*" -DestinationRelative "02_discovery_runs" -Category "hpc_discovery"
Move-RootGlob -Pattern "hpc_upload_qdelta_discovery_512seed_256x50_20260520*" -DestinationRelative "02_discovery_runs" -Category "hpc_discovery"

Move-RootGlob -Pattern "hpc_phase_qwindow_delta_refinement_20260525*" -DestinationRelative "03_phase_qwindow_delta_refinement" -Category "hpc_phase_qwindow_delta_refinement"

Move-RootGlob -Pattern "hpc_upload_robust_oracle_acq_compare_20260525*" -DestinationRelative "04_robust_oracle_acq_compare" -Category "hpc_robust_oracle_acq_compare"
Move-RootGlob -Pattern "hpc_upload_robust_oracle_label_closure_acq_compare_20260601*" -DestinationRelative "05_label_closure_ab" -Category "hpc_label_closure_ab"
Move-RootGlob -Pattern "hpc_upload_robust_incremental_qwindow_20260602_v3*" -DestinationRelative "06_incremental_qwindow" -Category "hpc_incremental_qwindow"
Move-RootItem -Name "hpc_packages" -DestinationRelative "hpc_packages" -Category "hpc_packages"

# Local active-learning smoke outputs.
$smokeDirs = @(
    "ML_Phase_acquisition_only_smoke",
    "ML_Phase_boundary_smoke",
    "ML_Phase_boundary_smoke2",
    "ML_Phase_discovery_smoke",
    "ML_Phase_local4090",
    "ML_Phase_stop_controller_smoke"
)
foreach ($dir in $smokeDirs) {
    Move-RootItem -Name $dir -DestinationRelative "smoke_runs" -Category "smoke_runs"
}

# Historical report folders. Keep root report/ and reports/ as active report roots.
$reportDirs = @(
    "report_active_learning_r0015_note",
    "report_numerical_reliability_audit_20260523",
    "report_phase_qwindow_delta_refinement_v1"
)
foreach ($dir in $reportDirs) {
    Move-RootItem -Name $dir -DestinationRelative "reports" -Category "historical_reports"
}

# External or private local-only files.
Move-RootGlob -Pattern "*.docx" -DestinationRelative "external_docs" -Category "external_docs"
Move-RootGlob -Pattern "*.pdf" -DestinationRelative "external_docs" -Category "external_docs" -Note "Root-level external PDFs only; report PDFs inside folders are unchanged."
Move-RootItem -Name "id_rsa" -DestinationRelative "..\private\keys" -Category "private_local" -Note "Private key moved out of root clutter; do not commit."

# Temporary files and local caches.
$tempItems = @(
    "tmp",
    "tmp_boundary_band_merge",
    ".history",
    ".matplotlib",
    ".pytest_cache",
    "__pycache__"
)
foreach ($item in $tempItems) {
    Move-RootItem -Name $item -DestinationRelative "local_temp_and_caches" -Category "local_temp_and_caches"
}

$records | Export-Csv -LiteralPath $manifestPath -NoTypeInformation -Encoding UTF8

@"
# Project History Archive

Created on 2026-06-03 to reduce first-level project clutter without changing
production source code or numerical behavior.

This folder groups historical generated artifacts by improvement stage:

- `00_initial_hpc_uploads/`
- `01_qdelta_warmup/`
- `02_discovery_runs/`
- `03_phase_qwindow_delta_refinement/`
- `04_robust_oracle_acq_compare/`
- `05_label_closure_ab/`
- `06_incremental_qwindow/`
- `hpc_packages/`
- `raw_exact_data/`
- `smoke_runs/`
- `reports/`
- `plans_and_runbooks/`
- `legacy_analysis/`
- `external_docs/`
- `local_temp_and_caches/`

See `root_reorganization_manifest_20260603.csv` for exact old and new paths.

Core source and canonical project files were intentionally left in place,
including `ml_phase/`, `scripts/`, `tests/`, `docs/`, `AGENTS.md`,
`MODEL_SPEC.md`, root solver files, and root launch scripts.
"@ | Set-Content -LiteralPath $readmePath -Encoding UTF8

Write-Host "Wrote manifest: $manifestPath"
Write-Host "Moved items:" ($records | Where-Object { $_.status -eq "moved" }).Count
