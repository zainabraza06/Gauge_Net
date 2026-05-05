param(
    [string]$DataDir = "",
    [string[]]$Datasets = @("FD001", "FD002", "FD003", "FD004")
)

if ($DataDir -eq "") {
    Write-Host "Please provide -DataDir" -ForegroundColor Red
    exit 1
}

$env:PYTHONPATH = "src"
python -m dl_baseline.run --data-dir "$DataDir" --datasets $Datasets
