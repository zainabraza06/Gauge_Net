param(
    [string]$DataDir = "",
    [string[]]$Datasets = @("FD002", "FD004"),
    [string[]]$Models = @("ridge", "rf", "xgboost"),
    [switch]$UseTemporalFeatures
)

if ($DataDir -eq "") {
    Write-Host "Please provide -DataDir" -ForegroundColor Red
    exit 1
}

$env:PYTHONPATH = "src"

$temporalFlag = ""
if ($UseTemporalFeatures) {
    $temporalFlag = "--use-temporal-features"
}

python -m ml_baseline.run --data-dir "$DataDir" --datasets $Datasets --models $Models --save-summary $temporalFlag
