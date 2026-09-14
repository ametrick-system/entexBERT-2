#!/usr/bin/env python3
"""
fig_ism_motif_example.py -- ONE ISM window: per-base allelic saliency vs JASPAR motif hits,
plus a quantitative saliency<->motif OVERLAP metric. Standalone (reads only the npz + a JASPAR
PFM json; no model stack).

Top panel : gray = per-position motif-hit COUNT over the chosen motifs (left y);
            blue = ISM saliency track (right y). Dotted line = the variant at window center.
Lower panel: motif-hit heatmap -- one row per motif with a hit, bars at [start,end] colored by
            -log10 p (viridis). Capped to --max_rows most-significant motifs.

Which motifs to include is YOUR choice: --motifs is a comma-separated list of JASPAR TF-name
substrings (e.g. "CTCF,CTCFL") or matrix ids, or "ALL". The count track uses every selected
motif; the heatmap shows the top --max_rows by significance.

OVERLAP metric (printed for all three; the figure annotates Spearman rho + enrichment):
  spearman   : Spearman rho(per-position saliency, per-position motif count)
  enrichment : mean saliency at motif-covered positions / mean saliency elsewhere
  mass       : fraction of total |saliency| falling within motif-covered positions

Window selection: --select top_depth (highest total_reads), or index:<N>, or chr:<anchor>.

  python fig_ism_motif_example.py --npz ism_ref_AS.npz --jaspar jaspar2024_core_vert_pfms.json \
      --motifs CTCF --select top_depth --out fig_ism_motif_ctcf_topdepth.png
"""
import argparse, json
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.ndimage import gaussian_filter1d
from scipy.stats import spearmanr

B = {"A": 0, "C": 1, "G": 2, "T": 3}
plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200, "font.size": 8, "axes.titlesize": 8,
    "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.4,
})
C_SAL = "#2c6fbb"


def select_pwms(jaspar_json, motifs_arg, pseudo=0.1):
    """{matrix_id: (pfm(4,W), display_name)} for motifs whose name/id matches any --motifs token."""
    J = json.load(open(jaspar_json))
    toks = [t.strip().upper() for t in motifs_arg.split(",") if t.strip()]
    allm = (len(toks) == 1 and toks[0] == "ALL")
    out = {}
    for mid, v in J.items():
        name = v.get("name", mid)
        if allm or any(t in name.upper() or t == mid.upper() for t in toks):
            p = v["pfm"]; M = np.array([p["A"], p["C"], p["G"], p["T"]], dtype=np.float64) + pseudo
            out[mid] = (M / M.sum(0, keepdims=True), name)
    return out


def scan(seq, pwms, threshold):
    """fimo over one sequence -> list of (name, start, end, neglogp). neglogp from the p-value
    column if present, else the score (label adapts)."""
    from memelite import fimo
    X = np.zeros((1, 4, len(seq)), dtype=np.float32)
    for i, ch in enumerate(seq.upper()):
        if ch in B:
            X[0, B[ch], i] = 1.0
    flat = {mid: pwm for mid, (pwm, _) in pwms.items()}
    name_of = {mid: nm for mid, (_, nm) in pwms.items()}
    res = fimo(flat, X, threshold=threshold)
    hits, pcol = [], None
    for df in res:
        if len(df) == 0:
            continue
        if pcol is None:
            pcol = next((c for c in df.columns if "p-value" in c.lower() or c.lower() in ("pval", "pvalue")), None)
        for _, r in df.iterrows():
            mid = str(r["motif_name"])
            s, e = int(r["start"]), int(r["end"])
            if pcol is not None and np.isfinite(r[pcol]) and r[pcol] > 0:
                val = -np.log10(float(r[pcol]))
            else:
                val = float(r["score"])
            hits.append((name_of.get(mid, mid), s, e, val))
    return hits, ("−log10 p" if pcol is not None else "motif score")


def pick_window(z, select):
    n = z["importance"].shape[0]
    if select == "top_depth":
        tr = z["total_reads"].astype(float) if "total_reads" in z else np.zeros(n)
        return int(np.argmax(tr))
    if select.startswith("index:"):
        return int(select.split(":", 1)[1])
    if select.startswith("chr:"):
        an = int(select.split(":", 1)[1]); anc = z["anchor"].astype(np.int64)
        w = np.where(anc == an)[0]
        if len(w) == 0:
            raise SystemExit(f"no window with anchor {an}")
        return int(w[0])
    raise SystemExit(f"--select must be top_depth | index:N | chr:ANCHOR, got {select!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--jaspar", required=True)
    ap.add_argument("--motifs", default="ALL", help="comma-sep JASPAR TF-name substrings / matrix ids, or ALL")
    ap.add_argument("--select", default="top_depth", help="top_depth | index:N | chr:ANCHOR")
    ap.add_argument("--sal_key", default="importance")
    ap.add_argument("--fimo_threshold", type=float, default=1e-4)
    ap.add_argument("--max_rows", type=int, default=25, help="heatmap rows (top motifs by significance)")
    ap.add_argument("--smooth", type=float, default=2.0)
    ap.add_argument("--logo_key", default="contrib",
                    help="npz array for the ISM logo (contrib = onehot * mean-centered delta)")
    ap.add_argument("--zoom_bp", type=int, default=0,
                    help="crop ALL panels to +/- this many bp around the variant (0 = full window); "
                         "use a small value (e.g. 30) to make the ISM logo letters legible")
    ap.add_argument("--title", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    z = np.load(a.npz, allow_pickle=True)
    w = pick_window(z, a.select)
    seqs = z["seqs"]; seq = str(seqs[w])
    sal = z[a.sal_key][w].astype(np.float64)          # (L,)
    L = len(seq); center = L // 2; x = np.arange(L) - center
    # per-base ISM contribution for the logo panel (onehot * mean-centered delta); fall back if absent
    if a.logo_key in z:
        contrib = z[a.logo_key][w].astype(np.float64)             # (L, 4)
    elif "onehot" in z and "hyp_scores" in z:
        contrib = (z["onehot"][w] * z["hyp_scores"][w]).astype(np.float64)
    else:
        contrib = None
    anchor = int(z["anchor"][w]) if "anchor" in z else -1
    chrom = str(z["chr"][w]) if "chr" in z else "?"
    print(f"[window] idx={w} {chrom}:{anchor} L={L}")

    pwms = select_pwms(a.jaspar, a.motifs)
    if not pwms:
        raise SystemExit(f"--motifs {a.motifs!r} matched no JASPAR entries")
    hits, plabel = scan(seq, pwms, a.fimo_threshold)
    print(f"[scan] {len(pwms)} motifs selected -> {len(hits)} hits (thr p<{a.fimo_threshold})")

    # per-position motif COUNT track + motif-covered mask
    count = np.zeros(L); covered = np.zeros(L, dtype=bool)
    for _, s, e, _ in hits:
        e = min(e, L); count[s:e] += 1.0; covered[s:e] = True

    # ---- OVERLAP metrics: saliency track vs motif count ----------------------------------------
    absal = np.abs(sal)
    rho = spearmanr(sal, count).correlation if count.any() else np.nan
    enr = (absal[covered].mean() / absal[~covered].mean()
           if covered.any() and (~covered).any() and absal[~covered].mean() > 0 else np.nan)
    mass = absal[covered].sum() / absal.sum() if absal.sum() > 0 else np.nan
    print(f"[overlap] spearman(sal,count)={rho:.3f} | enrichment(in/out)={enr:.3f} | "
          f"saliency-mass-in-motifs={mass:.3f} | covered={covered.mean():.2f}")

    # ---- figure: track panel (dual y) + motif heatmap ------------------------------------------
    have_logo = contrib is not None
    nrow = 3 if have_logo else 2
    hr = [1.0, 1.6, 0.9] if have_logo else [1.0, 1.6]
    fig = plt.figure(figsize=(8.4, 5.6 + (1.4 if have_logo else 0)))
    gs = fig.add_gridspec(nrow, 2, width_ratios=[1.0, 0.03], height_ratios=hr,
                          hspace=0.16, wspace=0.02)
    axT = fig.add_subplot(gs[0, 0])
    axH = fig.add_subplot(gs[1, 0], sharex=axT)
    axL = fig.add_subplot(gs[2, 0], sharex=axT) if have_logo else None
    cax = fig.add_subplot(gs[1, 1])            # colorbar occupies the heatmap row only -> panels align
    # top: gray motif count (left), blue saliency (right)
    axT.fill_between(x, gaussian_filter1d(count, a.smooth) if a.smooth else count,
                     color="0.75", linewidth=0, zorder=1)
    axT.set_ylabel("motif count", color="0.4"); axT.tick_params(axis="y", colors="0.4")
    axT.margins(x=0.02)
    axS = axT.twinx(); axS.spines["top"].set_visible(False)
    axS.plot(x, gaussian_filter1d(sal, a.smooth) if a.smooth else sal, color=C_SAL, zorder=3)
    axS.set_ylabel("ISM saliency", color=C_SAL); axS.tick_params(axis="y", colors=C_SAL)
    for ax in (axT, axS):
        ax.axvline(0, color="0.4", ls=":", lw=0.8, zorder=2)
    ttl = (a.title or f"{chrom}:{anchor}").strip()
    axT.set_title(f"{ttl}  —  ρ(saliency,motif count)={rho:.2f}, in/out enrichment={enr:.2f}", loc="left")

    # bottom: motif-hit heatmap, top rows by significance
    by_motif = {}
    for name, s, e, val in hits:
        by_motif.setdefault(name, []).append((s, e, val))
    order = sorted(by_motif, key=lambda m: -max(v for _, _, v in by_motif[m]))[:a.max_rows]
    vals = [v for m in order for _, _, v in by_motif[m]]
    norm = Normalize(min(vals), max(vals)) if vals else Normalize(0, 1)
    cmap = plt.get_cmap("viridis")
    for row, name in enumerate(order):
        yv = len(order) - 1 - row
        for s, e, val in by_motif[name]:
            axH.broken_barh([(x[s], (min(e, L) - s))], (yv - 0.35, 0.7),
                            facecolors=cmap(norm(val)), edgecolor="none")
    axH.set_yticks(range(len(order)))
    axH.set_yticklabels(list(reversed(order)))
    axH.set_ylim(-0.6, len(order) - 0.4)
    axH.axvline(0, color="0.4", ls=":", lw=0.8)
    if not have_logo:
        axH.set_xlabel("position relative to variant (bp)")
    axH.margins(x=0.02)
    if len(by_motif) > len(order):
        axH.set_title(f"top {len(order)} of {len(by_motif)} motifs with a hit", loc="left")
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_label(f"motif hit {plabel}")

    # ---- bottom: ISM contribution logo for THIS window (letters = the motif the model reads) -----
    if have_logo:
        try:
            import logomaker
        except ImportError:
            raise SystemExit("logomaker is required for the ISM logo panel "
                             "(pip install logomaker; it is in the same env as plot_ism_logo)")
        lm = pd.DataFrame(contrib, index=x, columns=list("ACGT"))     # index=x -> glyphs align with tracks
        logomaker.Logo(lm, ax=axL, color_scheme="classic")
        axL.axvline(0, color="0.4", ls=":", lw=0.8)
        axL.set_ylabel("ISM contribution")
        axL.set_xlabel("position relative to variant (bp)")
        axL.spines["top"].set_visible(False); axL.spines["right"].set_visible(False)

    if a.zoom_bp > 0:
        axT.set_xlim(-a.zoom_bp, a.zoom_bp)              # sharex -> crops every panel; legible letters

    fig.savefig(a.out, bbox_inches="tight")   # GridSpec hspace/wspace set the spacing; logomaker axes break tight_layout
    print(f"[fig] wrote {a.out}")


if __name__ == "__main__":
    main()
