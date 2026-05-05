# DL_Baseline

Modular deep learning baseline for CMAPSS RUL prediction with NASA (PHM08) scoring.

## Quick start

1. Create and activate a Python environment.
2. Install dependencies:

```
pip install -r requirements.txt
```

3. Run experiments:

```
$env:PYTHONPATH = "src"
python -m dl_baseline.run --data-dir "<path-to-cmapss>" --datasets FD001 FD002 FD003 FD004
```

## Outputs

- Results: `DL_Baseline/results/`

## Notes

- Reproducibility is enforced through global seeding and deterministic Torch settings.
- Paths are local by default; pass `--data-dir` to point to the CMAPSS folder.
