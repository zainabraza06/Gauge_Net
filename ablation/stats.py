"""
Paired comparison between two rungs of the ladder.

Every rung is trained on the SAME seed list, and a seed fixes the validation
split, the end-of-life trim and the initialisation. So rung A's seed 53 and
rung B's seed 53 differ by the component under test and nothing else, which is
what makes a paired test the right one: it removes the seed-to-seed variance
that dominates an unpaired comparison at this sample size.

What is reported, and why more than one thing:

  paired t-test    the conventional first check. Assumes the differences are
                   normal, which five points cannot support.
  Wilcoxon         distribution-free, but nearly powerless here: with five
                   pairs the smallest attainable two-sided p is 0.0625, so it
                   CANNOT return p < 0.05 however consistent the direction.
                   A non-significant Wilcoxon at n=5 is not evidence of no
                   effect.
  Cohen's d        effect size on the paired differences, independent of n.
  bootstrap CI     a distribution-free 95% interval on the mean difference.

With five seeds all of these are low-powered. Read the effect size and whether
the interval excludes zero; do not read a p-value as a verdict.
"""

import numpy as np
from scipy import stats as scipy_stats

# Sign convention: diff = previous - current, so a POSITIVE mean_diff means
# the metric went down when the component was added. For NASA, RMSE, MAE and
# MAPE -- everything the delta table compares -- lower is better, so positive
# means the component helped.
DIFF_CONVENTION = 'previous - current (positive = metric fell = component helped)'


def paired_comparison(prev_vals, cur_vals, n_boot=10000, seed=0):
    """Compares two rungs' per-seed values of one metric, seed for seed.

    Inputs must be ordered by the SAME seed list; pairing two differently
    ordered lists silently compares unrelated runs.
    """
    a = np.asarray(prev_vals, dtype=np.float64)
    b = np.asarray(cur_vals, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"paired comparison needs equal-length inputs, got "
                         f"{a.shape} and {b.shape}")
    diffs = a - b
    n = len(diffs)

    if n < 2 or np.allclose(diffs, 0):
        # Nothing to test: one pair, or two identical runs.
        t_stat = p_t = w_stat = float('nan')
        p_w = 1.0 if n >= 2 else float('nan')
    else:
        t_stat, p_t = scipy_stats.ttest_rel(a, b)
        try:
            w_stat, p_w = scipy_stats.wilcoxon(a, b)
        except ValueError:
            w_stat, p_w = float('nan'), float('nan')

    sd = float(diffs.std(ddof=1)) if n > 1 else 0.0
    cohens_d = float(diffs.mean() / sd) if sd > 1e-12 else 0.0

    rng = np.random.default_rng(seed)
    boot = rng.choice(diffs, size=(n_boot, n), replace=True).mean(axis=1)
    ci_lo, ci_hi = (float(v) for v in np.percentile(boot, [2.5, 97.5]))

    return dict(n=n, mean_diff=float(diffs.mean()), shapiro_p=normality_p(a, b),
                t_stat=float(t_stat), p_ttest=float(p_t),
                wilcoxon_stat=float(w_stat), p_wilcoxon=float(p_w),
                cohens_d=cohens_d, bootstrap_ci95=(ci_lo, ci_hi),
                ci_excludes_zero=bool(ci_lo > 0 or ci_hi < 0),
                convention=DIFF_CONVENTION)


def normality_p(prev_vals, cur_vals):
    """Shapiro-Wilk on the paired differences.

    Diagnostic, not a gate: it says whether the t-test's normality assumption
    is defensible on these five points. At n=5 the test has almost no power
    to reject, so a large p is weak evidence FOR normality -- it mostly rules
    out the blatant cases.
    """
    diffs = np.asarray(prev_vals, dtype=np.float64) - np.asarray(cur_vals, dtype=np.float64)
    if len(diffs) < 3 or np.allclose(diffs, diffs[0]):
        return None
    try:
        return float(scipy_stats.shapiro(diffs)[1])
    except ValueError:
        return None


def friedman(per_group_values):
    """Omnibus test across three or more rungs measured on the same seeds.

    Answers "do these rungs differ at all", which the pairwise steps do not:
    walking a five-rung ladder means four comparisons, and testing each at
    0.05 makes a false positive somewhere likely. A non-significant omnibus
    is a reason to read the individual steps as suggestive rather than
    settled.

    Kendall's W rescales the statistic to [0, 1] -- how consistently the
    seeds rank the rungs the same way, which is what the ladder claims.
    """
    groups = list(per_group_values)
    if len(groups) < 3:
        return None
    arrays = [np.asarray(per_group_values[g], dtype=np.float64) for g in groups]
    n = len(arrays[0])
    if n < 3 or any(len(a) != n for a in arrays):
        return None
    try:
        chi2, p = scipy_stats.friedmanchisquare(*arrays)
    except ValueError:
        return None
    k = len(groups)
    return dict(groups=groups, n=n, k=k, chi2=float(chi2), p=float(p),
                kendalls_w=float(chi2) / (n * (k - 1)))


def effect_size_label(d):
    ad = abs(d)
    if ad < 0.2:
        return 'negligible'
    if ad < 0.5:
        return 'small'
    if ad < 0.8:
        return 'medium'
    return 'large'


def format_comparison(metric, cmp_, precision=2):
    """One line summarising a paired comparison, for the delta table."""
    lo, hi = cmp_['bootstrap_ci95']
    f = f'.{precision}f'
    shapiro = ('' if cmp_.get('shapiro_p') is None
               else f"   shapiro p={cmp_['shapiro_p']:.3f}")
    return (f"{metric.upper():<5} {cmp_['mean_diff']:+{f}}   "
            f"paired-t p={cmp_['p_ttest']:.4f}   wilcoxon p={cmp_['p_wilcoxon']:.4f}   "
            f"d={cmp_['cohens_d']:+.2f} ({effect_size_label(cmp_['cohens_d'])})   "
            f"boot 95% CI=({lo:+{f}}, {hi:+{f}})"
            f"{'  [excludes 0]' if cmp_['ci_excludes_zero'] else ''}{shapiro}")
