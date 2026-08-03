param(
    [string]$RunId = ""
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
if (-not $RunId) {
    $RunId = "alignment-lab-demo-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
}
$runDir = Join-Path $repoRoot "runs\$RunId"
$pilotPath = Join-Path $repoRoot "runs\$RunId-pilot.json"

Push-Location $repoRoot
try {
    Write-Host "[1/6] Verify the installed CLI"
    miniverl --version

    Write-Host "[2/6] Run the bounded pilot"
    miniverl pilot recipes/alignment_tool_policy_toy.yaml --out $pilotPath
    Get-Content -LiteralPath $pilotPath

    Write-Host "[3/6] Resolve the post-SFT stage graph without loading a model"
    miniverl align recipes/alignment_tool_policy_toy.yaml --dry-run

    Write-Host "[4/6] Run the short CPU alignment machinery demo"
    miniverl align recipes/alignment_tool_policy_toy.yaml --run-id $RunId

    Write-Host "[5/6] Inspect typed token provenance"
    miniverl inspect (Join-Path $runDir "trajectories.jsonl")

    Write-Host "[6/6] Show the privacy-safe Alignment Card and export-ready checkpoint"
    Get-Content -LiteralPath (Join-Path $runDir "alignment-card.md")
    Get-ChildItem -LiteralPath (Join-Path $runDir "checkpoints\final") |
        Select-Object Name, Length

    Write-Host "Reviewed result figures: docs\alignment-lab\"
    Write-Host "Demo artifacts: $runDir"
}
finally {
    Pop-Location
}
