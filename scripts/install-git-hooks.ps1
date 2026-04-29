param(
    [string]$GitDir
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot

if (-not $GitDir) {
    if (Test-Path (Join-Path $repoRoot ".git-store\config")) {
        $GitDir = Join-Path $repoRoot ".git-store"
    } else {
        $GitDir = git -C $repoRoot rev-parse --git-dir
        if (-not [System.IO.Path]::IsPathRooted($GitDir)) {
            $GitDir = Join-Path $repoRoot $GitDir
        }
    }
}

$hookRoot = Join-Path $GitDir "hooks"

if (-not (Test-Path $hookRoot)) {
    New-Item -ItemType Directory -Path $hookRoot | Out-Null
}

$hook = @'
#!/usr/bin/env sh
python scripts/scan_sensitive.py
'@

Set-Content -Path (Join-Path $hookRoot "pre-commit") -Value $hook -NoNewline -Encoding ascii
Set-Content -Path (Join-Path $hookRoot "pre-push") -Value $hook -NoNewline -Encoding ascii

Write-Host "Installed sensitive-data hooks in .git-store/hooks"
