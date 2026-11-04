"""Plot rolling means for new SB3 Monitor logs (not the thesis PDF plots)."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('monitor_csv', type=Path)
    parser.add_argument('--window-short', type=int, default=10)
    parser.add_argument('--window-long', type=int, default=200)
    parser.add_argument('--output', type=Path, default=Path('outputs/monitor.png'))
    args = parser.parse_args()
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    df = pd.read_csv(args.monitor_csv, comment='#')
    if df.empty:
        parser.error('No completed episodes in monitor file')
    # SB3 Monitor uses r, l, t; the original ad hoc plot used different labels.
    cols = [c for c in ('r', 'l', 'soc_min', 'soc_mean', 'soc_dev', 'gate', 'success') if c in df]
    nrows = int(np.ceil(len(cols) / 2))
    fig, axes = plt.subplots(nrows, 2, figsize=(13, 3.5*nrows), squeeze=False)
    for ax, key in zip(axes.ravel(), cols):
        series = pd.to_numeric(df[key], errors='coerce')
        ax.plot(series, alpha=0.25, label='raw')
        ax.plot(series.rolling(args.window_short, min_periods=1).mean(), label=f'MA{args.window_short}')
        ax.plot(series.rolling(args.window_long, min_periods=1).mean(), label=f'MA{args.window_long}')
        ax.set(title=key, xlabel='Episode')
        ax.grid(alpha=0.3)
        ax.legend()
    for ax in axes.ravel()[len(cols):]:
        ax.set_visible(False)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170)
    print(f'New experiment chart saved to {args.output}')


if __name__ == '__main__':
    main()
