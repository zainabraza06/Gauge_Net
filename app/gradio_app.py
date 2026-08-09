"""
GaugeNet RUL demo -- upload a C-MAPSS trajectory file, get one RUL per engine.

    pip install gradio
    python -m app.gradio_app

Serves whichever subsets have a bundle in app/bundles/. A subset with
checkpoints but no bundle is not offered: the weights alone cannot put an
upload on the scale they were trained on (see app/bundle.py).

The demo reports the reported configuration for each subset -- the same
checkpoints, the same ensembling technique, the same MC-dropout averaging as
the tables in the README -- so a number here is the number the study would
produce for that engine.
"""

import os
import traceback

import gradio as gr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from model.config import MC_PASSES_TEST, RUL_CAP
from .predict import available_subsets, predict_csv, BUNDLE_DIR

BLUE, ORANGE, RED = '#4C72B0', '#DD8452', '#C44E52'

SUBSET_NOTE = {
    'FD001': "one operating condition, one fault mode — global z-score",
    'FD002': "six operating conditions, one fault mode — per-regime z-score",
    'FD003': "one operating condition, two fault modes — global z-score",
    'FD004': "six operating conditions, two fault modes — per-regime z-score, "
             "FD002→FD004 frozen-backbone transfer arm",
}


def plot_results(df, subset):
    """Predicted RUL per engine, sorted, with the cap marked.

    Sorted rather than in engine order: the useful reading is the shape of the
    fleet's health, and engine numbering carries no information.
    """
    d = df.sort_values('predicted_RUL').reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(d))
    ax.bar(x, d['predicted_RUL'], color=BLUE, width=0.8)
    ax.errorbar(x, d['predicted_RUL'], yerr=d['seed_spread_std'], fmt='none',
                ecolor=ORANGE, elinewidth=1, capsize=2, alpha=0.8,
                label='spread across seeds (1 s.d.)')
    ax.axhline(RUL_CAP, color=RED, ls='--', lw=1,
               label=f'RUL cap ({RUL_CAP:.0f} cycles)')
    ax.set_xticks(x)
    ax.set_xticklabels(d['engine'].astype(int), rotation=90, fontsize=7)
    ax.set_xlabel('engine (sorted by predicted RUL)')
    ax.set_ylabel('predicted RUL (cycles)')
    ax.set_title(f'{subset}: predicted remaining useful life, most urgent first')
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def run(file_obj, subset, mc_passes, progress=gr.Progress()):
    if file_obj is None:
        return None, None, "Upload a C-MAPSS trajectory file first."
    # gr.File hands back a path on current Gradio and a file-like object on
    # older ones; accept either rather than pinning a Gradio version.
    path = file_obj if isinstance(file_obj, str) else getattr(file_obj, 'name', file_obj)
    try:
        df, info = predict_csv(path, subset, mc_passes=int(mc_passes),
                               progress=lambda f, m: progress(f, desc=m))
    except Exception as exc:
        return None, None, f"**Failed.**\n\n```\n{exc}\n```\n"

    urgent = df.nsmallest(min(3, len(df)), 'predicted_RUL')
    lines = [
        f"**{info['n_engines']} engines** scored on **{info['subset']}** "
        f"({SUBSET_NOTE.get(subset, '')}).",
        "",
        f"- ensemble: `{info['technique']}` over {info['n_seeds']} seeds",
        f"- {info['mc_passes']} MC-dropout passes per seed, on `{info['device']}`",
        f"- window: last {info['window']} cycles per engine; "
        f"predictions capped in training at {info['rul_cap']:.0f}",
    ]
    if info['regimes']:
        lines.append(f"- {info['regimes']} operating regimes found in the upload")
    lines += [
        "",
        "**Most urgent:** " + ", ".join(
            f"engine {int(r.engine)} at {r.predicted_RUL:.0f} cycles"
            for r in urgent.itertuples()),
        "",
        f"Median predicted RUL {df['predicted_RUL'].median():.0f} cycles; "
        f"mean cross-seed spread {df['seed_spread_std'].mean():.1f} cycles "
        f"(a wide spread means the seeds disagree about that engine).",
    ]
    return df, plot_results(df, subset), "\n".join(lines)


def build_ui():
    subsets = available_subsets()
    intro = """
# GaugeNet — Remaining Useful Life

Upload a raw **C-MAPSS trajectory file** (`test_FD00X.txt`, or the same 26
columns as CSV: `unit, cycle, op1-op3, s1-s21`) and get one RUL estimate per
engine, from the engine's most recent 72 cycles.

Everything is causal — each prediction uses only that engine's own history —
and every normalisation and geometric reference was fitted on the training
split, so nothing in your upload influences its own scaling.
"""
    if not subsets:
        with gr.Blocks(title="GaugeNet RUL") as demo:
            gr.Markdown(intro)
            gr.Markdown(
                f"### No subsets are servable yet\n\n"
                f"No deployment bundles found in `{BUNDLE_DIR}`. A checkpoint alone "
                f"cannot serve predictions: the normalisation statistics, the healthy "
                f"geometric reference and the fleet median life were all fitted on the "
                f"training split and are not stored in the weights.\n\n"
                f"Build them where the C-MAPSS data lives:\n\n"
                f"```bash\npython -m preprocessing FD001\npython -m app.bundle FD001\n```")
        return demo

    with gr.Blocks(title="GaugeNet RUL") as demo:
        gr.Markdown(intro)
        with gr.Row():
            with gr.Column(scale=1):
                file_in = gr.File(label="C-MAPSS trajectory file",
                                  file_types=['.txt', '.csv', '.dat'])
                subset_in = gr.Dropdown(subsets, value=subsets[0], label="Subset",
                                        info="Must match the file — the normalisation "
                                             "and healthy reference are subset-specific")
                mc_in = gr.Slider(10, 100, value=MC_PASSES_TEST, step=5,
                                  label="MC-dropout passes",
                                  info="100 reproduces the reported numbers; lower is "
                                       "proportionally faster and slightly noisier")
                go = gr.Button("Predict", variant="primary")
                gr.Markdown(
                    "\n".join(f"- **{s}** — {SUBSET_NOTE.get(s, '')}" for s in subsets))
            with gr.Column(scale=2):
                summary = gr.Markdown()
                plot = gr.Plot(label="Predicted RUL per engine")
                table = gr.Dataframe(label="Per-engine results", wrap=True)

        go.click(run, [file_in, subset_in, mc_in], [table, plot, summary])
    return demo


if __name__ == '__main__':
    build_ui().launch()
