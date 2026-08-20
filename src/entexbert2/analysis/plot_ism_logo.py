#!/usr/bin/env python
"""
Plot an averaged ISM contribution logo for the entexBERT-2 binding trunk and overlay the
JASPAR CTCF PWM (MA0139.1) aligned to the recovered motif.

Inputs:
  --ism      : .npz from ism_saliency.py  (onehot, contrib, importance, seqs)
  --pwm_json : jaspar2024_core_vert_pfms.json  (JASPAR PFMs keyed by matrix_id; entry['pfm'][base])
Because ISM windows are not motif-centered, we align by cross-correlating each window's
per-position importance against the PWM information profile, roll each to a common center,
then average the contribution logos. The top panel is the model's learned logo; the bottom
panel is the JASPAR reference logo at the same width.
"""
import argparse, json, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import logomaker

BASES = "ACGT"


def pfm_to_prob(pfm):
    """JASPAR PFM dict {A:[...],C:[...],G:[...],T:[...]} -> (W,4) probability matrix."""
    M = np.array([pfm[b] for b in BASES], dtype=np.float64).T  # (W,4)
    M = M + 1e-9
    return M / M.sum(axis=1, keepdims=True)


def prob_to_ic(P):
    """probabilities -> per-base information content (bits), for the reference logo + profile."""
    ic_pos = 2.0 + (P * np.log2(P)).sum(axis=1)          # (W,)
    return P * ic_pos[:, None]                            # (W,4)


def best_offset(imp, profile, W):
    """Slide the PWM info-profile across a window's importance; return offset of best correlation."""
    L = imp.shape[0]
    best, bo = -np.inf, 0
    prof = (profile - profile.mean()) / (profile.std() + 1e-9)
    for off in range(0, L - W + 1):
        seg = imp[off:off + W]
        seg = (seg - seg.mean()) / (seg.std() + 1e-9)
        c = float((seg * prof).sum())
        if c > best:
            best, bo = c, off
    return bo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ism", required=True)
    ap.add_argument("--pwm_json", required=True)
    ap.add_argument("--matrix_id", default="MA0139.2",
                    help="JASPAR matrix id present in --pwm_json (CTCF core = MA0139.2)")
    ap.add_argument("--flank", type=int, default=4, help="bp of flank to show around the motif")
    ap.add_argument("--tf_label", default="", help="TF name for the title (defaults to JASPAR entry name)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    d = np.load(a.ism, allow_pickle=True)
    onehot, contrib, imp = d["onehot"], d["contrib"], d["importance"]
    N, L, _ = onehot.shape

    pj = json.load(open(a.pwm_json))
    entry = pj[a.matrix_id]
    pfm = entry.get("pfm") or entry.get("counts")
    P = pfm_to_prob(pfm); W = P.shape[0]
    ref_ic = prob_to_ic(P)
    profile = ref_ic.sum(axis=1)                          # (W,) info profile to align against

    span = W + 2 * a.flank
    agg = np.zeros((span, 4), dtype=np.float64)
    used = 0
    for w in range(N):
        off = best_offset(imp[w], profile, W)
        s = off - a.flank; e = off + W + a.flank
        if s < 0 or e > L:
            continue
        agg += contrib[w, s:e, :]
        used += 1
    agg /= max(used, 1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(6, span * 0.35), 4.4),
                                   gridspec_kw={"height_ratios": [2, 1]})
    lm = pd.DataFrame(agg, columns=list(BASES))
    logomaker.Logo(lm, ax=ax1, color_scheme="classic")
    _tf = a.tf_label or entry.get("name", "")
    ax1.set_title(f"entexBERT-2 trunk — ISM contribution logo (avg of {used}/{N} {_tf} windows)",
                  fontsize=10)
    ax1.set_ylabel("ISM contribution")
    ax1.axvspan(a.flank - 0.5, a.flank + W - 0.5, color="0.9", zorder=-5)

    rl = pd.DataFrame(np.vstack([np.zeros((a.flank, 4)), ref_ic, np.zeros((a.flank, 4))]),
                      columns=list(BASES))
    logomaker.Logo(rl, ax=ax2, color_scheme="classic")
    ax2.set_title(f"JASPAR {a.matrix_id} ({entry.get('name','CTCF')}) — reference (bits)", fontsize=10)
    ax2.set_ylabel("bits"); ax2.set_xlabel("position (motif shaded)")
    fig.tight_layout()
    fig.savefig(a.out, dpi=150, bbox_inches="tight")
    print(f"[plot] saved {a.out} | {used}/{N} windows aligned | motif width {W} + {a.flank}bp flank")


if __name__ == "__main__":
    main()
