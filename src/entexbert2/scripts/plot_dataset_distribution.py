#!/usr/bin/env python3

import argparse
import os

from entexbert2.visualizations import (
    plot_label_distribution,
    plot_sequence_length_distribution,
)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot label and sequence-length distributions for an entexBERT-2 dataset."
    )
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--bins", type=int, default=50)
    return parser.parse_args()

def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    plot_label_distribution(
        args.data_dir,
        output_path=os.path.join(args.output_dir, "label_distribution.png"),
        bins=args.bins,
        title="Target label distribution",
        by_split=True,
    )

    plot_sequence_length_distribution(
        args.data_dir,
        output_path=os.path.join(args.output_dir, "sequence_length_distribution.png"),
        bins=args.bins,
    )

    print(f"Saved plots to {args.output_dir}")


if __name__ == "__main__":
    main()