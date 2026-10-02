#!/usr/bin/env python3
"""
fig_ism_motif_example.py -- ONE ISM window: per-base allelic saliency vs JASPAR motif hits.
Standalone (reads only the npz + a JASPAR PFM json; no model stack).

Top panel : gray = per-position motif-hit COUNT over the chosen motifs (left y, "Motif Count");
            blue = ISM saliency track (right y). Dotted line = the variant at window center.
Lower panel: motif-hit lanes -- one row per motif with a hit (name at left), bars at [start,end]
            colored by -log10 p (viridis). Capped to --max_rows most-significant motifs.

Which motifs to include is YOUR choice: --motifs is a comma-separated list of JASPAR TF-name
substrings (e.g. "CTCF,CTCFL") or matrix ids, or "ALL".

Title: "{--title}\n{chr}: {start}-{end}" -- pass e.g. --title "Ind. 1, ATACseq"; the genomic
window span is appended automatically. Overlap metrics (spearman/enrichment/mass) are printed
to stdout (not drawn on the figure).

Window selection: --select top_depth (highest total_reads), or index:<N>, or chr:<anchor>.

  python fig_ism_motif_example.py --npz ism_ref_AS.npz --jaspar jaspar2024_core_vert_pfms.json \
      --motifs CTCF --select top_depth --title "Ind. 1, CTCF" --zoom_bp 75 --out fig_ism_motif_ctcf.png
"""
import argparse, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.ndimage import gaussian_filter1d
from scipy.stats import spearmanr

B = {"A": 0, "C": 1, "G": 2, "T": 3}
plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200, "font.size": 8, "axes.titlesize": 9,
    "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 6,
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
    return hits, ("\u2212log10 p" if pcol is not None else "motif score")


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
    ap.add_argument("--motifs", default="ALL", help="motifs shown as LANES: comma-sep JASPAR TF-name substrings / matrix ids, or ALL")
    ap.add_argument("--count_motifs", default=None,
                    help="motifs used for the gray COUNT track (known cofactors); comma-sep substrings/ids. "
                         "Default = same as --motifs. ALL is nearly flat (~3 hits/base genome-wide), so a "
                         "curated cofactor set makes the count track carry signal.")
    ap.add_argument("--select", default="top_depth", help="top_depth | index:N | chr:ANCHOR")
    ap.add_argument("--sal_key", default="importance")
    ap.add_argument("--fimo_threshold", type=float, default=1e-4)
    ap.add_argument("--max_rows", type=int, default=25, help="lane rows (top motifs by significance)")
    ap.add_argument("--smooth", type=float, default=2.0)
    ap.add_argument("--lane_lw", type=float, default=2.5, help="motif-hit line thickness in the lanes")
    ap.add_argument("--zoom_bp", type=int, default=0,
                    help="crop both panels to +/- this many bp around the variant (0 = full window; "
                         "use 75 to match the target layout)")
    ap.add_argument("--sal_label", default="ISM saliency", help="right-axis label for the blue track")
    ap.add_argument("--panel_label", default="", help="optional panel letter drawn top-left (e.g. B)")
    ap.add_argument("--title", default="",
                    help="top title line (e.g. 'Ind. 1, ATACseq'); the chr:start-end line is appended")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    z = np.load(a.npz, allow_pickle=True)
    w = pick_window(z, a.select)
    seq = str(z["seqs"][w])
    sal = z[a.sal_key][w].astype(np.float64)                 # (L,)
    L = len(seq); center = L // 2; x = np.arange(L) - center
    anchor = int(z["anchor"][w]) if "anchor" in z else -1
    chrom = str(z["chr"][w]) if "chr" in z else "?"
    gstart, gend = anchor - center, anchor - center + L      # genomic window span
    print(f"[window] idx={w} {chrom}:{anchor} L={L} span={gstart}-{gend}")

    lane_pwms = select_pwms(a.jaspar, a.motifs)
    if not lane_pwms:
        raise SystemExit(f"--motifs {a.motifs!r} matched no JASPAR entries")
    count_pwms = select_pwms(a.jaspar, a.count_motifs) if a.count_motifs else lane_pwms
    if a.count_motifs and not count_pwms:
        raise SystemExit(f"--count_motifs {a.count_motifs!r} matched no JASPAR entries")
    lane_names = {nm for _, nm in lane_pwms.values()}
    count_names = {nm for _, nm in count_pwms.values()}
    hits, plabel = scan(seq, {**count_pwms, **lane_pwms}, a.fimo_threshold)   # one fimo pass over the union
    print(f"[scan] lanes={len(lane_pwms)} count-motifs={len(count_pwms)} -> {len(hits)} hits "
          f"(thr p<{a.fimo_threshold})")

    # per-position motif COUNT track + covered mask -- built ONLY from the count-motif (cofactor) set
    count = np.zeros(L); covered = np.zeros(L, dtype=bool)
    for name, s, e, _ in hits:
        if name in count_names:
            e = min(e, L); count[s:e] += 1.0; covered[s:e] = True
    if not covered.any():
        print(f"[warn] the count-motif set matched 0 hits in this window -> gray count track is EMPTY. "
              f"Pick --count_motifs that occur here (this window's lanes: "
              f"{', '.join(sorted({nm for nm, _, _, _ in hits if nm in lane_names})[:12])})")

    # ---- OVERLAP metrics (printed only) --------------------------------------------------------
    absal = np.abs(sal)
    rho = spearmanr(sal, count).correlation if count.any() else np.nan
    enr = (absal[covered].mean() / absal[~covered].mean()
           if covered.any() and (~covered).any() and absal[~covered].mean() > 0 else np.nan)
    mass = absal[covered].sum() / absal.sum() if absal.sum() > 0 else np.nan
    print(f"[overlap] spearman(sal,count)={rho:.3f} | enrichment(in/out)={enr:.3f} | "
          f"saliency-mass-in-motifs={mass:.3f} | covered={covered.mean():.2f}")

    # ---- figure: top (count + saliency), bottom motif-hit lanes --------------------------------
    fig = plt.figure(figsize=(8.8, 5.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.02], height_ratios=[1.0, 1.7],
                          hspace=0.18, wspace=0.02)
    axT = fig.add_subplot(gs[0, 0])                          # col 0 -> axT and axH stay x-aligned
    axH = fig.add_subplot(gs[1, 0], sharex=axT)
    cax = fig.add_subplot(gs[1, 1])                          # colorbar beside the lanes only

    # top: gray motif count (left), blue saliency (right)
    axT.fill_between(x, gaussian_filter1d(count, a.smooth) if a.smooth else count,
                     color="0.75", linewidth=0, zorder=1)
    axT.set_ylabel("Motif Count", color="0.4"); axT.tick_params(axis="y", colors="0.4")
    axT.set_ylim(bottom=0)                                    # counts are non-negative
    axT.margins(x=0.02)
    axS = axT.twinx(); axS.spines["top"].set_visible(False)
    axS.plot(x, gaussian_filter1d(sal, a.smooth) if a.smooth else sal, color=C_SAL, zorder=3)
    axS.set_ylabel(a.sal_label, color=C_SAL); axS.tick_params(axis="y", colors=C_SAL)
    for ax in (axT, axS):
        ax.axvline(0, color="0.4", ls=":", lw=0.8, zorder=2)
    title = ((a.title.strip() + "\n") if a.title.strip() else "") + f"{chrom}: {gstart}-{gend}"
    axT.set_title(title)

    # bottom: motif-hit lanes -- one row per motif (name at left), bars colored by -log10 p (no colorbar)
    by_motif = {}
    for name, s, e, val in hits:
        if name in lane_names:
            by_motif.setdefault(name, []).append((s, e, val))
    if a.zoom_bp > 0:                                        # drop motifs whose hits all fall outside the crop
        by_motif = {m: hs for m, hs in by_motif.items()
                    if any(x[s] <= a.zoom_bp and x[min(e, L) - 1] >= -a.zoom_bp for s, e, _ in hs)}
    order = sorted(by_motif, key=lambda m: -max(v for _, _, v in by_motif[m]))[:a.max_rows]
    vals = [v for m in order for _, _, v in by_motif[m]]
    norm = Normalize(min(vals), max(vals)) if vals else Normalize(0, 1)
    cmap = plt.get_cmap("viridis")
    for row, name in enumerate(order):
        yv = len(order) - 1 - row
        for s, e, val in by_motif[name]:
            axH.hlines(yv, x[s], x[min(e, L) - 1], color=cmap(norm(val)), lw=a.lane_lw)
    axH.set_yticks(range(len(order)))
    axH.set_yticklabels(list(reversed(order)))
    axH.set_ylim(-0.6, len(order) - 0.4)
    axH.axvline(0, color="0.4", ls=":", lw=0.8)
    axH.set_xlabel("position relative to variant (bp)")
    axH.margins(x=0.02)
    axH.tick_params(axis="y", length=0)
    sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cb = fig.colorbar(sm, cax=cax); cb.set_label(f"motif hit {plabel}", fontsize=7)
    cb.ax.tick_params(labelsize=6)

    if a.zoom_bp > 0:
        axT.set_xlim(-a.zoom_bp, a.zoom_bp)                  # sharex -> crops both panels
    if a.panel_label:
        fig.text(0.005, 0.99, a.panel_label, fontsize=16, fontweight="bold", va="top", ha="left")

    fig.savefig(a.out, bbox_inches="tight")
    print(f"[fig] wrote {a.out}  (lanes colored by {plabel})")


if __name__ == "__main__":
    main()
