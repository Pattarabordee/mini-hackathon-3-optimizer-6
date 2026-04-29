param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$GitPushArgs
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$gitArgs = @()

if (Test-Path (Join-Path $repoRoot ".git-store\config")) {
    $gitArgs = @("--git-dir=.git-store", "--work-tree=.")
}

Push-Location $repoRoot
try {
    python scripts/scan_sensitive.py
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }

    $blockedNames = @(
        "employees.csv",
        "questions.csv",
        "train_labels.json",
        "FahMai Directory Q&A.pdf"
    )

    $reachable = git @gitArgs rev-list --objects HEAD
    $blocked = $reachable | Where-Object {
        $line = $_
        $blockedNames | Where-Object { $line.EndsWith(" $_") }
    }

    if ($blocked) {
        Write-Error "Push blocked: protected data is reachable from HEAD.`n$($blocked -join "`n")"
        exit 1
    }

    git @gitArgs push --no-verify @GitPushArgs
} finally {
    Pop-Location
}
