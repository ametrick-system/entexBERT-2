#!/usr/bin/env python
"""attn_query_profiles.py -- overlay the mean attention profile from DIFFERENT query perspectives to
test whether the central peak is just the variant token's SELF-attention.

Dump the same windows with dump_attention_descriptors.py --query {variant,cls,meanq} into separate
npzs, then overlay their mean (over loci) attention profiles here, centered on the variant.

Reads:
  * variant-query flat away from center, SPIKES at 0  -> the peak is the variant token's self/local
    attention (diagonal). If CLS-query is FLAT at 0, that CONFIRMS 'it's just self-attention'.
  * CLS-query (or meanq) ALSO peaks at 0             -> the variant is a genuine attention HUB other
    tokens look at (a real signal, not just self).
Prints, per npz: peak offset (argmax - center) and a central/flank enrichment ratio
(mean within ±ctr_bp / mean at |pos-center|>flank_bp).

  python attn_query_profiles.py --npzs variant=attn_variant.npz cls=attn_cls.npz meanq=attn_meanq.npz \\
      --out attn_query_profiles_ctcf.png
"""
import argparse
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = ["#6a9fd8", "#d1495b", "#55a868", "#e8843c", "#7a5cad"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npzs", nargs="+", required=True, help="label=path.npz (uses 'embeddings' (N,L))")
    ap.add_argument("--normalize", default="max", choices=["max", "zscore", "none"],
                    help="per-profile normalization for SHAPE comparison (default max)")
    ap.add_argument("--ctr_bp", type=int, default=10); ap.add_argument("--flank_bp", type=int, default=40)
    ap.add_argument("--title", default=""); ap.add_argument("--out", required=True)
    a = ap.parse_args()

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for i, spec in enumerate(a.npzs):
        assert "=" in spec, f"expected label=path, got {spec}"
        lab, path = spec.split("=", 1)
        d = np.load(path, allow_pickle=True)
        prof = d["embeddings"].mean(0)                        # mean over loci -> (L,)
        L = len(prof); c = L // 2; x = np.arange(L) - c
        off = np.abs(x)
        ctr = prof[off <= a.ctr_bp].mean(); fl = prof[off > a.flank_bp].mean()
        ratio = ctr / fl if fl != 0 else np.inf
        print(f"[{lab:8}] peak offset={int(np.argmax(prof))-c:+d} bp | central/flank enrichment={ratio:.2f} "
              f"(ctr={ctr:.4g} flank={fl:.4g})")
        y = prof.copy()
        if a.normalize == "max":
            y = y / (np.abs(y).max() or 1.0)
        elif a.normalize == "zscore":
            y = (y - y.mean()) / (y.std() or 1.0)
        ax.plot(x, y, color=COLORS[i % len(COLORS)], lw=1.4, label=f"{lab} (enrich {ratio:.1f}x)")
    ax.axvline(0, color="0.8", lw=0.8, zorder=0)
    ax.set_xlabel("position - variant (bp)")
    ax.set_ylabel({"max": "attention / max", "zscore": "attention (z)", "none": "mean attention"}[a.normalize])
    ax.legend(frameon=False, fontsize=9)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    if a.title:
        ax.set_title(a.title, fontsize=11)
    fig.tight_layout(); fig.savefig(a.out, dpi=200, bbox_inches="tight")
    print(f"[fig] wrote {a.out}")


if __name__ == "__main__":
    main()
