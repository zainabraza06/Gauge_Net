"""CLI: python -m preprocessing FD001 [--data-root ...] [--output-dir ...]"""

import argparse
import os
import sys

from model.config import DATA_ROOT

from .pipeline import build, VERSION


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('subset', choices=sorted(VERSION), help='C-MAPSS subset to preprocess')
    ap.add_argument('--data-root', default=DATA_ROOT,
                    help=f'directory holding train_<subset>.txt / test_<subset>.txt / '
                         f'RUL_<subset>.txt (default: {DATA_ROOT})')
    ap.add_argument('--output-dir', default=None,
                    help='where to write output_<subset>_<version>.pkl '
                         '(default: <data-root>/pipeline_output)')
    args = ap.parse_args()

    missing = [f for f in (f'train_{args.subset}.txt', f'test_{args.subset}.txt',
                           f'RUL_{args.subset}.txt')
               if not os.path.exists(os.path.join(args.data_root, f))]
    if missing:
        sys.exit(
            f"Missing {len(missing)} input file(s) in {args.data_root}:\n"
            + "".join(f"  {f}\n" for f in missing)
            + "\nDownload the C-MAPSS archive from the NASA Prognostics Data Repository\n"
              "  https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/\n"
              "and extract those files there, or pass --data-root pointing at them.")

    output_dir = args.output_dir or os.path.join(args.data_root, 'pipeline_output')
    build(args.subset, args.data_root, output_dir)


if __name__ == '__main__':
    main()
