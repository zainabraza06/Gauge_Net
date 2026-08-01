"""CLI: python -m preprocessing FD001 [--data-root ...] [--output-dir ...]"""

import argparse
import os

from .pipeline import build, VERSION, EMIT_PHASE


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('subset', choices=sorted(VERSION), help='C-MAPSS subset to preprocess')
    ap.add_argument('--data-root', default=os.environ.get('CMAPPS_DATA_ROOT',
                                                          '/content/drive/MyDrive/cmapps'),
                    help='directory holding train_<subset>.txt / test_<subset>.txt / RUL_<subset>.txt')
    ap.add_argument('--output-dir', default=None,
                    help='where to write output_<subset>_<version>.pkl '
                         '(default: <data-root>/pipeline_output)')
    phase = ap.add_mutually_exclusive_group()
    phase.add_argument('--emit-phase', dest='emit_phase', action='store_true', default=None,
                       help='emit onset/T_i/phase columns; CHANGES the trainer loss weighting '
                            f'relative to the historical default (per subset: {EMIT_PHASE})')
    phase.add_argument('--no-emit-phase', dest='emit_phase', action='store_false',
                       help='omit onset/T_i/phase columns')
    args = ap.parse_args()
    output_dir = args.output_dir or os.path.join(args.data_root, 'pipeline_output')
    build(args.subset, args.data_root, output_dir, emit_phase=args.emit_phase)


if __name__ == '__main__':
    main()
