"""
Streamlit Community Cloud entry point.

At the repository root because Streamlit Cloud takes a path to one file and
runs it as a script; keeping it here means the deploy form takes
`streamlit_app.py` with no directory to configure.

Named streamlit_app.py rather than app.py for the same reason the Spaces entry
is space_app.py: the serving code is a PACKAGE called `app/`, and a module of
that name beside it makes `import app.predict` ambiguous.

The UI is a thin wrapper. Everything that decides a prediction -- the frozen
normalisation, the healthy geometric reference, the five seeds, the ensembling
technique -- lives in app/predict.py and is shared with the Gradio front end,
so the two cannot drift into serving different numbers.
"""

import io
import os

os.environ.setdefault('OMP_NUM_THREADS', '2')     # Community Cloud is ~1 CPU

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import streamlit as st

from app.predict import available_subsets, predict_csv

# The hosted default. 100 passes reproduces the reported numbers and costs
# about a second per engine on a shared CPU, which is minutes for a full FD002
# upload; 25 is the tradeoff the README recommends when budget is tight. The
# slider still reaches 100.
DEFAULT_MC = 25

SUBSET_BLURB = {
    'FD001': 'one operating condition, one fault mode',
    'FD002': 'six operating conditions, one fault mode',
    'FD003': 'one operating condition, two fault modes',
    'FD004': 'six operating conditions, two fault modes',
}

st.set_page_config(page_title='GaugeNet RUL', page_icon='✈️', layout='wide')


def plot_results(df):
    """Predicted RUL per engine, most urgent first, with the seed spread."""
    d = df.sort_values('predicted_RUL')
    fig, ax = plt.subplots(figsize=(10, max(3.0, 0.26 * len(d))))
    ax.barh([str(u) for u in d['engine']], d['predicted_RUL'],
            xerr=d['seed_spread_std'], color='#4C72B0',
            error_kw=dict(ecolor='#C44E52', lw=1, capsize=2))
    ax.set_xlabel('predicted RUL (cycles)')
    ax.set_ylabel('engine')
    ax.set_title('Predicted remaining useful life, most urgent first '
                 '(bars = cross-seed spread)')
    ax.margins(y=0.01)
    fig.tight_layout()
    return fig


st.title('GaugeNet — Remaining Useful Life')
st.markdown(
    'Upload a raw NASA C-MAPSS trajectory file and get one RUL estimate per '
    'engine. A single 746K-parameter geometry-aware attention network, applied '
    'unchanged across all four subsets. This app runs the **reported '
    'configuration** for each subset: the same five seeds, the same ensembling '
    'technique, the same MC-dropout averaging as the published tables.')

subsets = available_subsets()
if not subsets:
    st.error(
        'No prediction bundles found. The weights alone cannot serve: an '
        'upload has to be put on the scale the model was trained on, and the '
        'normalisation statistics, healthy geometric reference and fleet '
        'median life all come from the training split. Build them with '
        '`python -m app.bundle` and redeploy.')
    st.stop()

left, right = st.columns([1, 2])

with left:
    subset = st.selectbox(
        'Subset the file came from', subsets,
        format_func=lambda s: f'{s} — {SUBSET_BLURB.get(s, "")}')
    st.caption(
        'This cannot be inferred from the upload. The normalisation and the '
        'healthy reference are subset-specific, and choosing the wrong one '
        'silently produces wrong numbers rather than an error.')

    mc = st.slider('MC-dropout passes', 5, 100, DEFAULT_MC, step=5,
                   help='100 reproduces the reported numbers. Lower is '
                        'proportionally faster and slightly noisier.')
    if mc < 100:
        st.caption(f'{mc} passes — faster than the reported setting, and '
                   f'slightly noisier than it.')

    uploaded = st.file_uploader(
        'C-MAPSS trajectory file', type=['txt', 'csv', 'dat'],
        help='test_FD00X.txt from the C-MAPSS archive, or the same 26 columns '
             'as CSV: unit, cycle, op1-op3, s1-s21. Header optional.')

    go = st.button('Predict', type='primary', disabled=uploaded is None,
                   use_container_width=True)

with right:
    if not go:
        st.info(
            'Upload a file and press **Predict**.\n\n'
            'Expected columns: `unit, cycle, op1, op2, op3, s1 … s21` — '
            'whitespace or comma separated, header optional. Only the 14 '
            'sensors the study uses are read; the rest may be present and are '
            'ignored.\n\n'
            'Predictions are capped at 125 cycles, the cap the model was '
            'trained under, so 125 means "at least 125" rather than exactly '
            '125.')
    else:
        bar = st.progress(0.0, text='starting')
        try:
            df, info = predict_csv(
                io.BytesIO(uploaded.getvalue()), subset, mc_passes=int(mc),
                progress=lambda f, m: bar.progress(min(max(f, 0.0), 1.0), text=m))
        except Exception as exc:
            bar.empty()
            st.error(f'{type(exc).__name__}: {exc}')
            st.stop()
        bar.empty()

        urgent = df.nsmallest(3, 'predicted_RUL')
        st.success(
            f"**{info['n_engines']} engines** scored on **{info['subset']}** "
            f"({SUBSET_BLURB.get(info['subset'], '')}).")

        c1, c2, c3 = st.columns(3)
        c1.metric('Most urgent engine', f"{int(urgent.iloc[0]['engine'])}",
                  f"{urgent.iloc[0]['predicted_RUL']:.0f} cycles left",
                  delta_color='inverse')
        c2.metric('Median predicted RUL',
                  f"{df['predicted_RUL'].median():.0f} cycles")
        c3.metric('Mean cross-seed spread',
                  f"{df['seed_spread_std'].mean():.1f} cycles",
                  help='A wide spread means the seeds disagree about that '
                       'engine, which is worth more than the point estimate '
                       'alone.')

        st.pyplot(plot_results(df))

        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button('Download predictions (CSV)',
                           df.to_csv(index=False).encode(),
                           file_name=f"gaugenet_{info['subset']}_predictions.csv",
                           mime='text/csv')

        with st.expander('How this was computed'):
            st.markdown(
                f"- ensemble: `{info['technique']}` over {info['n_seeds']} seeds\n"
                f"- {info['mc_passes']} MC-dropout passes per seed, on "
                f"`{info['device']}`\n"
                f"- window: last {info['window']} cycles per engine\n"
                f"- predictions capped in training at {info['rul_cap']:.0f} cycles\n"
                f"- normalisation: {info['scheme']}"
                + (f", {info['regimes']} operating regimes found in the upload"
                   if info.get('regimes') else ''))

st.divider()
st.caption(
    'Code, weights, five-rung ablation and full results: '
    'https://github.com/zainabraza06/Gauge_Net')
