param(
    [string]$DistRoot = ".\dist",
    [string]$ReleaseDir = ".\release",
    [string]$PlatformTag = "windows-x64",
    [string]$Version = "",
    [string[]]$PackageNames = @("cavity-grower", "pdb-du-preparer")
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

function Get-ReleaseFileName {
    param(
        [string]$BaseName,
        [string]$Platform,
        [string]$VersionTag
    )

    if ([string]::IsNullOrWhiteSpace($VersionTag)) {
        return "$BaseName-$Platform.zip"
    }
    return "$BaseName-$VersionTag-$Platform.zip"
}

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $projectRoot

try {
    $resolvedDistRoot = Resolve-ExistingPath (Join-Path $projectRoot $DistRoot)
    if (-not $resolvedDistRoot) {
        throw "Dist folder not found: $DistRoot"
    }

    $releaseDirPath = Join-Path $projectRoot $ReleaseDir
    New-Item -ItemType Directory -Force -Path $releaseDirPath | Out-Null
    $resolvedReleaseDir = (Resolve-Path -LiteralPath $releaseDirPath).Path

    $packages = @(
        @{ Folder = "cavity-grower"; BaseName = "cavity-grower" },
        @{ Folder = "pdb-du-preparer"; BaseName = "pdb-du-preparer" }
    )

    foreach ($package in $packages) {
        if ($PackageNames -notcontains $package.Folder) {
            continue
        }

        $sourceDir = Join-Path $resolvedDistRoot $package.Folder
        if (-not (Test-Path -LiteralPath $sourceDir)) {
            throw "Required packaged app folder not found: $sourceDir"
        }

        $zipName = Get-ReleaseFileName -BaseName $package.BaseName -Platform $PlatformTag -VersionTag $Version
        $zipPath = Join-Path $resolvedReleaseDir $zipName
        if (Test-Path -LiteralPath $zipPath) {
            Remove-Item -LiteralPath $zipPath -Force
        }

        Compress-Archive -Path $sourceDir -DestinationPath $zipPath -CompressionLevel Optimal
        if (-not (Test-Path -LiteralPath $zipPath)) {
            throw "Archive creation did not produce the expected file: $zipPath"
        }
        Write-Host "Created release asset:" $zipPath
    }
}
finally {
    Pop-Location
}
