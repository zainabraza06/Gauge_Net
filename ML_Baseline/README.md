# ML_Baseline

Modular baseline pipeline for CMAPSS RUL prediction with GroupKFold CV and PHM08 scoring.

## Quick start

1. Create and activate a Python environment.
2. Install dependencies:

```
pip install -r requirements.txt
```

3. Run experiments:

```
$env:PYTHONPATH = "src"
python -m ml_baseline.run --data-dir "<path-to-cmapss>" --datasets FD002 FD004
```

To enable the preset temporal feature engineer:

```
$env:PYTHONPATH = "src"
python -m ml_baseline.run --data-dir "<path-to-cmapss>" --datasets FD002 FD004 --use-temporal-features
```

## Outputs

- Models: `ML_Baseline/models/`
- Results: `ML_Baseline/results/`

## Notes

- Reproducibility is enforced through global seeding and fixed model seeds.
- Paths are local by default; pass `--data-dir` to point to the CMAPSS folder.
