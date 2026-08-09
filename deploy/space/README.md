---
title: GaugeNet RUL
emoji: ✈️
colorFrom: blue
colorTo: gray
sdk: gradio
app_file: space_app.py
pinned: false
license: mit
---

# GaugeNet — Remaining Useful Life

Upload a raw NASA C-MAPSS trajectory file and get one RUL estimate per engine.

A single 746K-parameter geometry-aware attention network, applied unchanged
across all four C-MAPSS subsets. This Space runs the reported configuration
for each subset: the same five seeds, the same ensembling technique, the same
MC-dropout averaging as the published tables.

**Input** — `test_FD00X.txt` from the C-MAPSS archive, or the same 26 columns
as CSV (`unit, cycle, op1-op3, s1-s21`), with or without a header row. Pick the
subset the file came from: the normalisation and the healthy geometric
reference are subset-specific and cannot be inferred from the upload.

**Note on speed** — the MC-dropout slider defaults to 25 passes here rather
than the 100 the reported numbers use. On a CPU Space, 100 passes costs about
a second per engine, which is several minutes for a full FD002 file. Raise it
to 100 to reproduce the published figures exactly.

Code, weights, ablation study and full results:
https://github.com/zainabraza06/TurboEngine_RLU_Prediction
