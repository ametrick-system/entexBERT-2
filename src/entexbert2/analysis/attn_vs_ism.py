#!/usr/bin/env python
"""attn_vs_ism.py -- does the model ATTEND to the positions ISM says it's causally SENSITIVE to?

Joins a center->all attention descriptor npz (dump_attention_descriptors.py: embeddings (N,L) [+ optional
attn_perhead (N,H,L)], chrom, anchor, labels) to an ISM saliency npz (ism_saliency.py: importance (N,L),
chr/chrom, anchor) on (chrom, anchor), both variant-centered at L//2, and measures per-position alignment
of attention with |ISM|.

The honest framing: attention != attribution. A WEAK alignment is the common/expected result (still a
finding); a STRONG one (attention concentrates on the ISM peak / motif) is the interesting mechanism.

Metrics printed:
  1. aggregate: mean attention profile vs mean |ISM| profile (do they peak at the same offsets?)
  2. per-locus Spearman(attention, |ISM|): median over loci  (the headline alignment number)
  3. attends-to-peak: per-locus z-scored attention AT the ISM-argmax position vs at flanks (paired) --
     "does the variant token attend to where mutation matters?"
  4. AS vs non-AS split of the per-locus alignment
  5. per-head (if attn_perhead present): per-head median alignment -> which heads are 'motif heads'

  python attn_vs_ism.py --attn attn_ctcf_ENC-002.npz --ism ism_trunk.npz --out attn_vs_ism_ctcf.png
"""
import argparse, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, wilcoxon


def _key(d, *names):
    for n in names:
        if n in d.files:
            return d[n]
    raise KeyError(f"none of {names} in {d.files}")


def zrow(X):
    m = X.mean(1, keepdims=True); s = X.std(1, keepdims=True); s[s == 0] = 1.0
    return (X - m) / s


def per_locus_corr(A, I, posmask=None):
    idx = np.flatnonzero(posmask) if posmask is not None else np.arange(A.shape[1])
    return np.array([spearmanr(A[i, idx], I[i, idx]).correlation for i in range(len(A))])


def attends_to_peak(A, I, center, flank_bp, posmask=None):
    """per-locus z-scored attention (z over KEPT positions) at the ISM-argmax vs mean at the distal
    flank. posmask restricts BOTH the peak search and the z-scoring to non-excluded positions, so with
    --exclude_center_bp this becomes 'does attention track the strongest FLANK saliency', a real test."""
    L = A.shape[1]
    idx = np.flatnonzero(posmask) if posmask is not None else np.arange(L)
    zA = zrow(A[:, idx])                                  # z over kept positions only
    Ik = I[:, idx]
    off_kept = np.abs(idx - center)
    fl = off_kept > flank_bp                              # distal-flank baseline within kept region
    if fl.sum() == 0:
        fl = np.ones(len(idx), dtype=bool)
    peak = zA[np.arange(len(A)), np.argmax(Ik, axis=1)]
    flank = zA[:, fl].mean(1)
    return peak, flank


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attn", required=True, help="attention descriptor npz (dump_attention_descriptors)")
    ap.add_argument("--ism", required=True, help="ISM saliency npz (ism_saliency: importance,chr,anchor)")
    ap.add_argument("--flank_bp", type=int, default=40, help="'flank' = |pos-center|>this for the peak test")
    ap.add_argument("--exclude_center_bp", type=int, default=0,
                    help="CONTROL: drop |pos-variant|<=this bp before correlating -> tests FLANK alignment "
                         "beyond the trivial shared central peak (both attn & |ISM| peak at the variant)")
    ap.add_argument("--detrend", action="store_true",
                    help="CONTROL: subtract the mean profile (over loci) from each locus's attn AND |ISM| "
                         "before correlating -> removes the shared central shape, correlates the residuals")
    ap.add_argument("--title", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    da = np.load(a.attn, allow_pickle=True); di = np.load(a.ism, allow_pickle=True)
    Aatt = _key(da, "embeddings"); ca = _key(da, "chrom", "chr").astype(str); aa = _key(da, "anchor").astype(np.int64)
    lab = _key(da, "labels").astype(int)
    Aism = np.abs(_key(di, "importance", "embeddings")); ci = _key(di, "chr", "chrom").astype(str); ai = _key(di, "anchor").astype(np.int64)
    ph = da["attn_perhead"] if "attn_perhead" in da.files else None

    # join on (chrom, anchor)
    ism_idx = {(c, int(p)): j for j, (c, p) in enumerate(zip(ci, ai))}
    rows_a, rows_i = [], []
    for j, (c, p) in enumerate(zip(ca, aa)):
        k = (c, int(p))
        if k in ism_idx:
            rows_a.append(j); rows_i.append(ism_idx[k])
    rows_a = np.array(rows_a); rows_i = np.array(rows_i)
    hit = len(rows_a) / max(len(ca), 1)
    print(f"[join] attn={len(ca)} ism={len(ci)} common={len(rows_a)} (hit-rate {hit:.2f})")
    assert len(rows_a) >= 20, "too few common loci to align (check both npz built on the same windows)"
    A = Aatt[rows_a]; I = Aism[rows_i]; y = lab[rows_a]
    L = A.shape[1]; c = L // 2

    # 1. aggregate profiles
    mA, mI = A.mean(0), I.mean(0)
    print(f"[aggregate] attention peak offset = {int(np.argmax(mA))-c:+d} bp | "
          f"|ISM| peak offset = {int(np.argmax(mI))-c:+d} bp")

    # ---- controls: exclude the shared central peak and/or detrend the shared mean profile ----
    posmask = np.abs(np.arange(L) - c) > a.exclude_center_bp        # True = keep (mA/mI above stay RAW)
    if a.detrend:
        A = A - A.mean(0, keepdims=True)
        I = I - I.mean(0, keepdims=True)
    ctl = ([f"exclude ±{a.exclude_center_bp}bp"] if a.exclude_center_bp > 0 else []) + (["detrend"] if a.detrend else [])
    print(f"[controls] {', '.join(ctl) if ctl else 'NONE (raw — shared central peak not removed)'}")

    # 2. per-locus alignment
    corr = per_locus_corr(A, I, posmask)
    print(f"[align] per-locus Spearman(attn,|ISM|): median={np.nanmedian(corr):+.3f} "
          f"mean={np.nanmean(corr):+.3f}  (frac>0: {np.mean(corr>0):.2f})")

    # 3. attends-to-peak
    peak, flank = attends_to_peak(A, I, c, a.flank_bp, posmask)
    dpk = peak - flank
    try:
        _, pw = wilcoxon(peak, flank)
    except ValueError:
        pw = np.nan
    print(f"[peak] z-attn at ISM-peak - flank: median={np.nanmedian(dpk):+.3f} "
          f"frac>0={np.mean(dpk>0):.2f}  Wilcoxon p={pw:.1e}")

    # 4. AS vs non-AS
    for lv, nm in [(1, "AS"), (0, "non-AS")]:
        m = y == lv
        if m.sum() > 5:
            print(f"[split] {nm} (n={int(m.sum())}): median corr={np.nanmedian(corr[m]):+.3f} "
                  f"peak-flank median={np.nanmedian(dpk[m]):+.3f}")

    # 5. per-head
    head_align = None
    if ph is not None:
        Aph = ph[rows_a]                                  # (n, H, L)
        if a.detrend:
            Aph = Aph - Aph.mean(0, keepdims=True)         # detrend each head over the joined loci
        H = Aph.shape[1]
        head_align = np.array([np.nanmedian(per_locus_corr(Aph[:, h, :], I, posmask)) for h in range(H)])
        order = np.argsort(-head_align)
        print(f"[per-head] {H} heads | median-alignment top3: "
              + ", ".join(f"h{int(o)}={head_align[o]:+.3f}" for o in order[:3])
              + f" | range [{head_align.min():+.3f}, {head_align.max():+.3f}]")

    # verdict
    med = np.nanmedian(corr)
    if med > 0.2 and np.nanmedian(dpk) > 0 and (pw < 1e-3):
        verdict = f"attention IS aligned with ISM (median corr {med:+.2f}, attends to the peak) -- mechanistic panel."
    elif np.nanmedian(dpk) > 0 and (pw < 1e-3):
        verdict = (f"weak global corr ({med:+.2f}) but attention DOES concentrate at the ISM peak "
                   f"(peak>flank, p={pw:.0e}) -- targeted alignment, worth showing.")
    else:
        verdict = (f"attention is NOT aligned with ISM (median corr {med:+.2f}, no peak enrichment) -- "
                   f"the expected 'attention != attribution' result; report as such.")
    print("[verdict]", verdict)

    # ---- figure ----
    npan = 4 if head_align is not None else 3
    fig, ax = plt.subplots(1, npan, figsize=(4.2 * npan, 3.6))
    x = np.arange(L) - c
    axT = ax[0]; axT.plot(x, mA, color="#6a9fd8", label="attention"); axT.set_ylabel("mean attention", color="#6a9fd8")
    axT.tick_params(axis="y", colors="#6a9fd8"); axT.axvline(0, color="0.8", lw=0.8)
    axS = axT.twinx(); axS.plot(x, mI, color="#d1495b", label="|ISM|"); axS.set_ylabel("mean |ISM|", color="#d1495b")
    axS.tick_params(axis="y", colors="#d1495b"); axT.set_xlabel("position - variant (bp)")
    if a.exclude_center_bp > 0:
        axT.axvspan(-a.exclude_center_bp, a.exclude_center_bp, color="0.85", alpha=0.5, zorder=0)  # excluded from corr
    axT.set_title("aggregate: attention vs |ISM| (raw profiles)", fontsize=10)

    ax[1].hist(corr[np.isfinite(corr)], bins=30, color="#7a7a7a")
    ax[1].axvline(np.nanmedian(corr), color="#c44e52", lw=1.5, label=f"median {np.nanmedian(corr):+.2f}")
    ax[1].axvline(0, color="0.6", lw=0.8, ls=":"); ax[1].set_xlabel("per-locus Spearman(attn,|ISM|)")
    ax[1].set_ylabel("loci"); ax[1].set_title("per-locus alignment", fontsize=10); ax[1].legend(frameon=False, fontsize=8)

    ax[2].boxplot([peak, flank], showfliers=False)
    ax[2].set_xticks([1, 2]); ax[2].set_xticklabels(["ISM-peak", "flank"])
    ax[2].set_ylabel("z-scored attention"); ax[2].set_title(f"attends-to-peak (p={pw:.0e})", fontsize=10)

    if head_align is not None:
        ax[3].bar(np.arange(len(head_align)), head_align[np.argsort(-head_align)], color="#55a868")
        ax[3].set_xlabel("head (sorted)"); ax[3].set_ylabel("median align")
        ax[3].set_title("per-head alignment", fontsize=10); ax[3].axhline(0, color="0.6", lw=0.8)

    for a_ in ax:
        for sp in ("top", "right"):
            a_.spines[sp].set_visible(False)
    suptitle = a.title + (f"   [controls: {', '.join(ctl)}]" if ctl else "")
    if suptitle.strip():
        fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94 if suptitle.strip() else 1.0))
    fig.savefig(a.out, dpi=200, bbox_inches="tight")
    print(f"[fig] wrote {a.out}")


if __name__ == "__main__":
    main()
