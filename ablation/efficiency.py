"""
What one prediction costs: parameters, MACs, latency, memory.

    python -m ablation.efficiency

Profiles the settled model and writes a summary table plus a bar figure to
<out-root>/figures/efficiency.png.

Measured ONCE per distinct model shape, not once per subset. Every subset runs
the same architecture on the same 18 features, so the parameter count and the
MAC count are identical by construction -- reporting them per subset invites
the reader to explain a difference that cannot exist. Latency is measured once
too, for the same reason: it depends on the shape and the machine, not on
which dataset's weights are loaded.

If a profile ever shows the same architecture with different MACs on different
subsets, that is a measurement artifact -- typically an exact count from a
profiler on one row and a hand estimate on another -- not a property of the
model. Mixing the two in one table makes the columns incomparable, so this
module measures every row the same way and says which way that was.
"""

import argparse
import json
import os
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

from model.config import DEVICE, OUT_ROOT, W, MC_PASSES_TEST

from .configs import config_by_name

SETTLED_CONFIG = 'sensor_branch_linear_far'
N_FEATURES = 18          # 14 sensors + 4 geometry, on every subset
BLUE, ORANGE, GREEN = '#4C72B0', '#DD8452', '#55A868'


def profile(n_features=N_FEATURES, config=SETTLED_CONFIG, n_repeats=200, warmup=20):
    from model.gaugenet import build_model
    model = build_model(n_features, 100.0, config_by_name(config))
    model.eval()
    x = torch.randn(1, W, n_features, device=DEVICE)
    lf = torch.zeros(1, device=DEVICE)

    try:
        from thop import profile as thop_profile
        macs, macs_source = float(thop_profile(model, inputs=(x, lf), verbose=False)[0]), 'thop'
    except ImportError:
        macs, macs_source = None, 'not measured (pip install thop)'

    with torch.no_grad():
        for _ in range(warmup):
            model(x, lf)
        if DEVICE == 'cuda':
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        for _ in range(n_repeats):
            model(x, lf)
        if DEVICE == 'cuda':
            torch.cuda.synchronize()
        per_forward_ms = (time.time() - t0) / n_repeats * 1000

    return dict(
        config=config, n_features=n_features, device=DEVICE,
        total_params=sum(p.numel() for p in model.parameters()),
        macs=macs, macs_source=macs_source,
        per_forward_ms=per_forward_ms,
        per_prediction_ms=per_forward_ms * MC_PASSES_TEST, mc_passes=MC_PASSES_TEST,
        peak_allocated_mb=(torch.cuda.max_memory_allocated() / 1024 ** 2
                           if DEVICE == 'cuda' else None))


def print_table(prof):
    print(f"\n{'=' * 78}\nEFFICIENCY -- {prof['config']} on {prof['device']}\n{'=' * 78}")
    print(f"  parameters            {prof['total_params']:>14,}")
    if prof['macs']:
        print(f"  MACs / forward pass   {prof['macs']:>14,.0f}  (measured by thop)")
    else:
        print(f"  MACs / forward pass   {prof['macs_source']:>14}")
    print(f"  latency / forward     {prof['per_forward_ms']:>14.2f} ms")
    print(f"  latency / prediction  {prof['per_prediction_ms']:>14.0f} ms  "
          f"({prof['mc_passes']} MC-dropout passes)")
    if prof['peak_allocated_mb'] is not None:
        print(f"  peak GPU memory       {prof['peak_allocated_mb']:>14.1f} MB")
    print(f"\n  One row, not one per subset: every subset runs this architecture on the")
    print(f"  same {prof['n_features']} features, so these numbers cannot differ between them.")
    print(f"  A prediction costs {prof['mc_passes']}x one forward pass -- the MC-dropout")
    print(f"  averaging, not the model, is what dominates deployment cost.")


def plot(prof, out_root):
    d = os.path.join(out_root, 'figures')
    os.makedirs(d, exist_ok=True)
    bars = [('parameters\n(millions)', prof['total_params'] / 1e6, BLUE),
            ('latency\n(ms / forward)', prof['per_forward_ms'], ORANGE),
            ('one prediction\n(ms, %d MC passes)' % prof['mc_passes'],
             prof['per_prediction_ms'], GREEN)]
    if prof['peak_allocated_mb'] is not None:
        bars.append(('peak memory\n(MB)', prof['peak_allocated_mb'], '#C44E52'))

    fig, axes = plt.subplots(1, len(bars), figsize=(3.1 * len(bars), 4))
    for ax, (label, value, colour) in zip(axes, bars):
        ax.bar([0], [value], color=colour, width=0.6)
        ax.text(0, value, f'{value:,.2f}' if value < 100 else f'{value:,.0f}',
                ha='center', va='bottom', fontsize=10)
        ax.set_xticks([])
        ax.set_title(label, fontsize=10)
        ax.set_ylim(0, value * 1.25)
    fig.suptitle(f"{prof['config']} on {prof['device']} -- identical on every subset")
    fig.tight_layout()
    path = os.path.join(d, 'efficiency.png')
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def per_subset_table(prof, picks):
    """The same profile shown per subset, labelled by where its model came from.

    Resource cost is the one thing that IS comparable across subsets -- unlike
    NASA, which is a sum over differently sized test sets. But the comparison
    has a boring answer here and the table says so: the transfer arm is the
    same architecture at the same width, so FD004 costs exactly what the
    directly trained subsets cost. Bars that differ would mean the profiles
    were measured differently, not that the models differ.
    """
    print(f"\n{'=' * 88}\nEFFICIENCY PER SUBSET -- best pick per subset\n{'=' * 88}")
    print(f"  {'subset':<8}{'source':<12}{'params':>12}{'MACs':>16}"
          f"{'ms/forward':>12}{'ms/prediction':>16}")
    macs = f"{prof['macs']:,.0f}" if prof['macs'] else 'not measured'
    for subset, pick in picks.items():
        print(f"  {subset:<8}{pick['source']:<12}{prof['total_params']:>12,}{macs:>16}"
              f"{prof['per_forward_ms']:>12.2f}{prof['per_prediction_ms']:>16.0f}")
    print(f"\n  Every row is the same measurement: the transfer arm is the same")
    print(f"  architecture at the same width, so its cost is identical by")
    print(f"  construction. Measured once and repeated rather than profiled four")
    print(f"  times, so no row can drift from another through measurement noise")
    print(f"  or through one row being an estimate and another exact.")


def plot_per_subset(prof, picks, out_root):
    d = os.path.join(out_root, 'figures')
    os.makedirs(d, exist_ok=True)
    subsets = list(picks)
    labels = [f"{s}\n({picks[s]['source']})" for s in subsets]
    colours = [ORANGE if picks[s]['source'] == 'transfer' else BLUE for s in subsets]
    panels = [('parameters (M)', prof['total_params'] / 1e6),
              ('ms / forward', prof['per_forward_ms']),
              (f"ms / prediction ({prof['mc_passes']} MC passes)", prof['per_prediction_ms'])]
    if prof['peak_allocated_mb'] is not None:
        panels.append(('peak memory (MB)', prof['peak_allocated_mb']))

    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4.2))
    for ax, (title, value) in zip(axes, panels):
        ax.bar(labels, [value] * len(subsets), color=colours)
        ax.set_title(title, fontsize=10)
        ax.tick_params(labelsize=8)
        ax.set_ylim(0, value * 1.3)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=BLUE, label='directly trained'),
                        Patch(color=ORANGE, label='transfer arm')],
               loc='upper center', ncol=2, fontsize=9, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle('Deployed cost per subset -- equal by construction, '
                 'the transfer arm shares the architecture', y=0.94, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    path = os.path.join(d, 'efficiency_per_subset.png')
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument('--config', default=SETTLED_CONFIG)
    ap.add_argument('--features', type=int, default=N_FEATURES)
    ap.add_argument('--out-root', default=OUT_ROOT)
    ap.add_argument('--per-subset', action='store_true',
                    help="also show the profile per subset, labelled by whether that "
                         "subset reports a directly trained or a transferred model")
    args = ap.parse_args()

    prof = profile(args.features, args.config)
    print_table(prof)
    print(f"\n  figure -> {plot(prof, args.out_root)}")
    if args.per_subset:
        from .figures import BEST_PICKS
        picks = {s: dict(source=('transfer' if p['source'] == 'transfer' else 'direct'))
                 for s, p in BEST_PICKS.items()}
        per_subset_table(prof, picks)
        print(f"\n  figure -> {plot_per_subset(prof, picks, args.out_root)}")
    out = os.path.join(args.out_root, 'figures', 'efficiency.json')
    with open(out, 'w') as f:
        json.dump(prof, f, indent=2)
    print(f"  profile -> {out}")


if __name__ == '__main__':
    main()
