param(
    [string]$PythonExe = ".\.venv\Scripts\python.exe",
    [string]$DistName = "cavity-grower",
    [string]$VinaExe = "",
    [string]$AdfrSuiteDir = "",
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

function Resolve-VinaExe {
    param([string]$Preferred)

    $resolved = Resolve-ExistingPath $Preferred
    if ($resolved) {
        return $resolved
    }

    $command = Get-Command vina -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Source
    }

    $candidates = @(
        "C:\Program Files (x86)\Vina\vina.exe",
        "C:\Program Files\Vina\vina.exe"
    )
    foreach ($candidate in $candidates) {
        $resolved = Resolve-ExistingPath $candidate
        if ($resolved) {
            return $resolved
        }
    }
    return $null
}

function Resolve-AdfrSuiteDir {
    param([string]$Preferred)

    $resolved = Resolve-ExistingPath $Preferred
    if ($resolved) {
        return $resolved
    }

    $candidates = @(
        "C:\Program Files (x86)\ADFRsuite-1.1dev",
        "C:\Program Files\ADFRsuite-1.1dev"
    )
    foreach ($candidate in $candidates) {
        $resolved = Resolve-ExistingPath $candidate
        if ($resolved) {
            return $resolved
        }
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

    if (-not $SkipPyInstallerInstall) {
        & $pythonPath -m pip install pyinstaller
    }

    & $pythonPath -m PyInstaller --noconfirm --clean "cavity_grower.spec"
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller build failed with exit code $LASTEXITCODE"
    }

    $distRoot = Join-Path $projectRoot "dist"
    $appRoot = Join-Path $distRoot $DistName
    if (-not (Test-Path -LiteralPath $appRoot)) {
        throw "Expected dist folder not found: $appRoot"
    }

    $toolsRoot = Join-Path $appRoot "tools"
    New-Item -ItemType Directory -Force -Path $toolsRoot | Out-Null

    $resolvedVina = Resolve-VinaExe $VinaExe
    if ($resolvedVina) {
        $vinaTarget = Join-Path $toolsRoot "vina"
        New-Item -ItemType Directory -Force -Path $vinaTarget | Out-Null
        Copy-Item -LiteralPath (Split-Path -Parent $resolvedVina) -Destination $vinaTarget -Recurse -Force
    }
    else {
        Write-Warning "Vina executable not found. The packaged app will only work without --vina-enable unless you add vina.exe later."
    }

    $resolvedAdfr = Resolve-AdfrSuiteDir $AdfrSuiteDir
    if ($resolvedAdfr) {
        $adfrTarget = Join-Path $toolsRoot "adfr"
        if (Test-Path -LiteralPath $adfrTarget) {
            Remove-Item -LiteralPath $adfrTarget -Recurse -Force
        }
        Copy-Item -LiteralPath $resolvedAdfr -Destination $adfrTarget -Recurse -Force
    }
    else {
        Write-Warning "ADFR Suite not found. The packaged app can still use Meeko or an externally installed ADFR backend."
    }

    Write-Host "Packaged app ready:" (Join-Path $appRoot "cavity-grower.exe")
    Write-Host "Runtime tools folder:" $toolsRoot
}
finally {
    Pop-Location
}
