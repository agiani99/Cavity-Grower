param(
    [string]$PythonExe = ".\.venv\Scripts\python.exe",
    [string]$SpecPath = "pdb_du_preparer.spec",
    [switch]$SkipPyInstallerInstall
)

$ErrorActionPreference = "Stop"

function Resolve-ExistingPath {
    param([string]$Candidate)

    if ([string]::IsNullOrWhiteSpace($Candidate)) {
        return $null
    }
    if (Test-Path -LiteralPath $Candidate) {
        return (Resolve-Path -LiteralPath $Candidate).Path
    }
    return $null
}

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $projectRoot

try {
    $pythonPath = Resolve-ExistingPath $PythonExe
    if (-not $pythonPath) {
        throw "Python executable not found: $PythonExe"
    }

    $resolvedSpec = Resolve-ExistingPath (Join-Path $projectRoot $SpecPath)
    if (-not $resolvedSpec) {
        throw "Spec file not found: $SpecPath"
    }

    if (-not $SkipPyInstallerInstall) {
        & $pythonPath -m pip install pyinstaller
    }

    & $pythonPath -m PyInstaller --noconfirm --clean $resolvedSpec
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller build failed with exit code $LASTEXITCODE"
    }

    $appRoot = Join-Path $projectRoot "dist\pdb-du-preparer"
    if (-not (Test-Path -LiteralPath $appRoot)) {
        throw "Expected dist folder not found: $appRoot"
    }

    Write-Host "Packaged app ready:" (Join-Path $appRoot "pdb-du-preparer.exe")
}
finally {
    Pop-Location
}
