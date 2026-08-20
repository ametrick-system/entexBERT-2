#!/usr/bin/env python
"""
Plot the ISM position-control (jitter) experiment.

Takes one ISM .npz per jitter offset delta (from ism_saliency.py run on the rolled windows),
computes the by-position mean saliency curve for each, and overlays them. If the saliency peak
sits at center+delta for every delta, the model localizes the motif wherever the anchor is placed
-- i.e. the center-token mean-pool is NOT conflating position with saliency.

A companion panel plots measured peak position vs delta; slope ~1 (peak = center+delta) is the
result. Deviation from slope 1 (e.g. peaks stuck at center) would indicate a pooling artifact.

Inputs: --npz delta=path pairs, e.g.
    --npz "-60=ism_jit-60.npz" "0=ism_jit0.npz" "60=ism_jit60.npz"
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def smooth(y, k=11):
    if k <= 1:
        return y
    ker = np.ones(k) / k
    return np.convolve(np.pad(y, k // 2, mode="edge"), ker, "same")[k // 2:-(k // 2)] if k > 1 else y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_dir", required=True,
                    help="directory holding one ISM npz per delta, named <npz_prefix><delta>.npz")
    ap.add_argument("--npz_prefix", default="ism_jit",
                    help="filename prefix; file for delta d is <npz_prefix><d>.npz (e.g. ism_jit-30.npz)")
    ap.add_argument("--deltas", default="-60,-30,0,30,60",
                    help="comma list of jitter offsets (matches make_jitter_windows --deltas)")
    ap.add_argument("--smooth", type=int, default=11)
    ap.add_argument("--tf_label", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import os
    deltas = [int(x) for x in a.deltas.split(",")]
    items = []
    for dlt in deltas:
        path = os.path.join(a.npz_dir, f"{a.npz_prefix}{dlt}.npz")
        if not os.path.exists(path):
            print(f"[warn] missing {path} -- skipping delta {dlt}")
            continue
        items.append((dlt, path))
    if not items:
        raise SystemExit(f"no npz found in {a.npz_dir} for deltas {deltas}")
    items.sort(key=lambda t: t[0])

    curves = []
    L = None
    for dlt, path in items:
        d = np.load(path, allow_pickle=True)
        imp = d["importance"]  # (N, L)
        L = imp.shape[1]
        curves.append((dlt, imp.mean(axis=0)))

    center = L // 2
    x = np.arange(L) - center  # position relative to center

    # color by delta with a diverging map centered at 0 (semantic zero = no jitter)
    deltas = [d for d, _ in curves]
    vmax = max(abs(min(deltas)), abs(max(deltas))) or 1
    cmap = plt.cm.coolwarm

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 4.0),
                                  gridspec_kw={"width_ratios": [2.4, 1]})

    measured = []
    for dlt, cur in curves:
        col = cmap(0.5 + 0.5 * dlt / vmax)
        ys = smooth(cur, a.smooth)
        ax.plot(x, ys, color=col, lw=2.4, label=f"jitter {dlt:+d} bp")
        # expected peak at center+delta -> in centered coords, at x=delta
        ax.axvline(dlt, color=col, ls=":", lw=1.2, alpha=0.7)
        measured.append((dlt, x[cur.argmax()]))

    ax.set_xlabel("position relative to window center (bp)")
    ax.set_ylabel("ISM saliency (mean Δ score / substitution)")
    ax.axhline(0, color="0.7", lw=0.6)
    ttl = f"{a.tf_label} " if a.tf_label else ""
    ax.set_title(f"{ttl}ISM saliency peak travels with anchor position\n"
                 f"(dotted = expected peak at center+jitter)")
    ax.legend(frameon=False, fontsize=8)

    # peak-position vs delta: slope 1 is the result
    md = np.array(measured, float)
    ax2.plot(md[:, 0], md[:, 1], "o-", color="#333", lw=1.5, ms=6, label="measured peak")
    lim = [md[:, 0].min() - 8, md[:, 0].max() + 8]
    ax2.plot(lim, lim, ls="--", color="#b0304a", lw=1.5, label="peak = center+jitter (slope 1)")
    ax2.set_xlabel("jitter offset δ (bp)")
    ax2.set_ylabel("measured saliency-peak position (bp)")
    ax2.set_title("peak position vs jitter")
    ax2.legend(frameon=False, fontsize=8)
    ax2.set_aspect("equal", adjustable="box")

    fig.tight_layout()
    fig.savefig(a.out, dpi=150, bbox_inches="tight")
    # report the fit
    slope = np.polyfit(md[:, 0], md[:, 1], 1)[0]
    print(f"[jitter-plot] saved {a.out} | peak-vs-delta slope = {slope:.3f} (1.0 = perfect tracking)")
    for dlt, pk in measured:
        print(f"    delta={dlt:+4d} -> peak at {pk:+4d} (expected {dlt:+4d})")


if __name__ == "__main__":
    main()