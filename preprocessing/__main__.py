"""CLI: python -m preprocessing FD001 [--data-root ...] [--output-dir ...]"""

import argparse
import os

from .pipeline import build, VERSION


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('subset', choices=sorted(VERSION), help='C-MAPSS subset to preprocess')
    ap.add_argument('--data-root', default=os.environ.get('CMAPPS_DATA_ROOT',
                                                          '/content/drive/MyDrive/cmapps'),
                    help='directory holding train_<subset>.txt / test_<subset>.txt / RUL_<subset>.txt')
    ap.add_argument('--output-dir', default=None,
                    help='where to write output_<subset>_<version>.pkl '
                         '(default: <data-root>/pipeline_output)')
    args = ap.parse_args()
    output_dir = args.output_dir or os.path.join(args.data_root, 'pipeline_output')
    build(args.subset, args.data_root, output_dir)


if __name__ == '__main__':
    main()
