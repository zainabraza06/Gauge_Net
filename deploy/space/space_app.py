"""
Hugging Face Spaces entry point.

Named space_app.py rather than app.py deliberately: the serving code lives in
a PACKAGE called `app/`, and a module `app.py` beside a package `app/` in the
same directory makes `import app.predict` ambiguous. The Space's README points
`app_file` here instead.

The hosted demo lowers the MC-dropout default. 100 passes reproduces the
reported numbers and costs about a second per engine on a Space's CPU, which
is four minutes for a full FD002 upload; 25 passes is the tradeoff the README
already recommends when inference budget is tight. The slider still goes to
100 for anyone who wants the reported figure.
"""

import os

os.environ.setdefault('OMP_NUM_THREADS', '4')

from app import gradio_app

gradio_app.MC_PASSES_TEST = 25          # hosted default; the slider still reaches 100

demo = gradio_app.build_ui()

if __name__ == '__main__':
    # queue() serialises requests: five checkpoints held in memory at once is
    # fine, five concurrent uploads each loading their own set is not, on a
    # Space's 16 GB.
    demo.queue(max_size=8).launch(server_name='0.0.0.0', server_port=7860)
