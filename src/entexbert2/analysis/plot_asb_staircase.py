#!/usr/bin/env python
"""
Fig A -- ASB staircase (per-TF): AUROC of the Stage-2 ASB head across the ablation arms,
read from the per-seed score files a replicate run writes. One figure per eval set
(--eval adastra | hetsnv), with BOTH regimes (leak-free and full) drawn as grouped bars
per rung -- the published DNN benchmarks were not necessarily leak-filtered, so the full
set is the fairer point of comparison for the ceiling line.

TERMINOLOGY: every arm performs Stage-2 supervised fine-tuning (SFT) of the ASB head, so
none is "no-SFT". What varies is (a) whether Stage-1 intermediate-task binding transfer
was applied to the backbone, and (b) whether the backbone is frozen or fine-tuned in
Stage-2. Rungs (left -> right = expected weakest -> strongest):
  no Stage-1 / frozen backbone     (arm 'nosft')     raw pretrained backbone, distance head only
  no Stage-1 / backbone fine-tuned (arm 'nosft_ft')  raw pretrained backbone, fully fine-tuned
  Stage-1 binding / frozen backbone(arm 'sft')       intermediate-task binding trunk, frozen
Optional leftmost ref_single (region-propensity) rung if its score files exist.

ADASTRA reads score_<arm>_seed<N>_adastra_metrics.json (results[].regime auroc).
EN-TEx  reads score_<arm>_seed<N>_hetsnv_summary.csv, matched-donor ALL-tissue row per regime.
Multi-seed arm -> bar = seed MEAN, error = seed sd (n-1). Single-seed arm -> error = the
bootstrap 95% CI half-width (auroc_hi - auroc) from the score file.

Self-contained: stdlib + numpy + matplotlib only. No entexbert2 import.
"""
import argparse, csv, glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _stat(aurocs, cis):
    """(mean, error, n): error = seed sd for >1 seed, else the single bootstrap CI half-width."""
    au = np.array(aurocs, float)
    if len(au) == 0:
        return None
    if len(au) > 1:
        return au.mean(), au.std(ddof=1), len(au)
    return au[0], (cis[0] if len(cis) else 0.0), 1


def load_adastra(exp_dir, arm, seeds, regime):
    aur, cis = [], []
    for s in seeds:
        hits = glob.glob(os.path.join(exp_dir, f"score_{arm}_seed{s}_adastra_metrics.json"))
        if not hits:
            continue
        d = json.load(open(hits[0]))
        rec = next((r for r in d["results"] if r["regime"] == regime), None) \
            or next((r for r in d["results"] if r["regime"] == "full"), None)
        if rec is None:
            continue
        aur.append(float(rec["auroc"]))
        cis.append(float(rec.get("auroc_hi", rec["auroc"])) - float(rec["auroc"]))
    return _stat(aur, cis)


def load_hetsnv(exp_dir, arm, seeds, regime, matched_donor, suffix="hetsnv",
                donor_kind="matched"):
    """Per-seed EN-TEx AUROC from score_<arm>_seed<N>_<suffix>_summary.csv, ALL-tissue rows.
    donor_kind='matched'    -> the matched_donor row (one value per seed).
    donor_kind='cross-donor'-> mean AUROC over the cross-donor donors present (donor != matched,
                               or donor_kind column == 'cross-donor') within each seed, then the
                               usual mean/sd across seeds. Needs the _hetsnv_xdonor files (all 4
                               donors); the matched-only _hetsnv files have no cross-donor rows."""
    aur, cis = [], []
    for s in seeds:
        hits = glob.glob(os.path.join(exp_dir, f"score_{arm}_seed{s}_{suffix}_summary.csv"))
        if not hits:
            continue
        rows = [r for r in csv.DictReader(open(hits[0])) if r.get("tissue") == "ALL"]

        def pick(reg):
            if donor_kind == "matched":
                return [r for r in rows if r.get("donor") == matched_donor and r.get("regime") == reg]
            # cross-donor: prefer the explicit donor_kind column, else "donor != matched"
            xd = [r for r in rows if r.get("regime") == reg
                  and (r.get("donor_kind") == "cross-donor"
                       or (r.get("donor") and r.get("donor") != matched_donor))]
            return xd

        recs = pick(regime) or pick("full")              # fall back to full if regime absent
        if not recs:
            continue
        a = np.array([float(r["auroc"]) for r in recs], float)
        # per-seed error: matched -> the single bootstrap half-width; cross-donor -> spread across donors
        if len(recs) == 1:
            c = float(recs[0].get("auroc_hi", recs[0]["auroc"])) - float(recs[0]["auroc"])
        else:
            c = float(a.std(ddof=1))
        aur.append(float(a.mean()))
        cis.append(c)
    return _stat(aur, cis)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_dir", required=True,
                    help="experiments/<tf> dir holding the score_* files")
    ap.add_argument("--tf", default="EP300", help="TF label for the title")
    ap.add_argument("--eval", default="adastra", choices=["adastra", "hetsnv"],
                    help="which eval set's score files to read")
    ap.add_argument("--matched_donor", default="ENC-001",
                    help="EN-TEx matched donor (EP300=ENC-001, CTCF=ENC-002); --eval hetsnv only")
    ap.add_argument("--hetsnv_suffix", default="hetsnv",
                    help="EN-TEx score-file suffix: 'hetsnv' (matched-only run) or "
                         "'hetsnv_xdonor' (cross-donor rescore). --eval hetsnv only")
    ap.add_argument("--donor_kind", default="matched", choices=["matched", "cross-donor"],
                    help="EN-TEx: score the matched training donor, or mean over the cross-donor "
                         "donors (needs the hetsnv_xdonor files). --eval hetsnv only")
    ap.add_argument("--seeds", default="20,21,22", help="comma list of frozen-head seeds")
    ap.add_argument("--ft_seed", default="20", help="single full-FT seed")
    ap.add_argument("--refsingle_arm", default="refsingle",
                    help="arm name for an optional ref_single rung; skipped if no score file")
    ap.add_argument("--reference_csv", default=None,
                    help="optional *_AUROC benchmark CSV -> ceiling line (ADASTRA benchmarks only)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    seeds = [s.strip() for s in a.seeds.split(",") if s.strip()]
    # (display label, arm key, seed list) -- ordered weakest -> strongest
    rung_defs = [
        ("ref_single\n(region propensity)",   a.refsingle_arm, seeds),
        ("no Stage-1\nfrozen backbone",        "nosft",         seeds),
        ("no Stage-1\nbackbone fine-tuned",    "nosft_ft",      [a.ft_seed]),
        ("Stage-1 binding\nfrozen backbone",   "sft",           seeds),
    ]

    def load(arm, sd, regime):
        if a.eval == "adastra":
            return load_adastra(a.exp_dir, arm, sd, regime)
        return load_hetsnv(a.exp_dir, arm, sd, regime, a.matched_donor,
                           suffix=a.hetsnv_suffix, donor_kind=a.donor_kind)

    labels, lf, full, is_ft = [], [], [], []
    for disp, arm, sd in rung_defs:
        s_lf = load(arm, sd, "leak_free")
        s_full = load(arm, sd, "full")
        if s_lf is None and s_full is None:
            print(f"[skip] {arm}: no {a.eval} score files -- rung omitted")
            continue
        labels.append(disp)
        lf.append(s_lf)          # (mean, err, n) or None
        full.append(s_full)
        is_ft.append("fine-tuned" in disp)
        print(f"[rung] {arm:10s} {a.eval}: "
              f"leak_free={None if s_lf is None else round(s_lf[0],4)} "
              f"full={None if s_full is None else round(s_full[0],4)}")

    if not labels:
        raise SystemExit(f"No rungs found under {a.exp_dir} (--eval {a.eval}) -- check paths/arm names.")

    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(1.9 * len(labels) + 1.8, 5.4))

    def draw(stats, offset, color, label, hatch_ft):
        xs, ys, es, ns, hatches = [], [], [], [], []
        for xi, (st, ft) in enumerate(zip(stats, is_ft)):
            if st is None:
                continue
            xs.append(xi + offset); ys.append(st[0]); es.append(st[1]); ns.append(st[2])
            hatches.append("///" if (hatch_ft and ft) else "")
        bars = ax.bar(xs, ys, width=w, yerr=es, capsize=4, color=color, edgecolor="black",
                      linewidth=0.8, label=label, error_kw=dict(elinewidth=1.1, ecolor="#333333"))
        for b, h in zip(bars, hatches):
            if h:
                b.set_hatch(h)
        for xi_, y_, e_, n_ in zip(xs, ys, es, ns):
            tag = f"{y_:.3f}" + ("" if n_ > 1 else "*")
            ax.text(xi_, y_ + e_ + 0.005, tag, ha="center", va="bottom", fontsize=8)
        # return the annotation tops (bar + error + label headroom) so ylim can fit them
        return [y_ + e_ + 0.012 for y_, e_ in zip(ys, es)]

    yfull = draw(full, -w / 2, "#9ecae1", "full set", hatch_ft=True)
    ylf = draw(lf, +w / 2, "#3182bd", "leak-free", hatch_ft=True)
    tops = [v for v in (yfull + ylf)]          # annotation tops (bar + error + headroom)
    bar_heights = [st[0] for st in (full + lf) if st is not None]

    ax.axhline(0.5, ls=":", c="grey", lw=1.0, zorder=0)
    ax.text(len(labels) - 0.5, 0.502, "chance", fontsize=8, color="grey", ha="right", va="bottom")

    # benchmark ceiling: only meaningful for ADASTRA (published DNN numbers are on that set)
    if a.eval == "adastra" and a.reference_csv and os.path.exists(a.reference_csv):
        rows = list(csv.DictReader(open(a.reference_csv)))
        acol = next((c for c in rows[0] if c.endswith("_AUROC")), None)
        if acol:
            top = max(rows, key=lambda r: float(r[acol]))
            yv = float(top[acol]); tops.append(yv + 0.012)
            ax.axhline(yv, ls="--", c="#e6550d", lw=1.3, zorder=1)
            ax.text(len(labels) - 0.5, yv + 0.004,
                    f"best benchmark: {top['model']} {yv:.3f}",
                    fontsize=8, color="#e6550d", ha="right", va="bottom")

    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    if a.eval == "adastra":
        evname = "ADASTRA"
    elif a.donor_kind == "matched":
        evname = f"EN-TEx hetSNV (matched donor {a.matched_donor})"
    else:
        evname = f"EN-TEx hetSNV (cross-donor mean, vs {a.matched_donor})"
    ax.set_ylabel(f"ASB AUROC  ({evname}, balanced)")
    ax.set_title(f"{a.tf}: ASB prediction across the two-stage ablation")
    ax.set_ylim(min(0.48, min(bar_heights) - 0.02), max(tops) + 0.015)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.annotate("* single seed (error = bootstrap 95% CI); others = seed sd, n=3",
                xy=(0, -0.16), xycoords="axes fraction", fontsize=7.5, color="#555555")
    fig.tight_layout()
    fig.savefig(a.out, dpi=200, bbox_inches="tight")
    print(f"[done] wrote {a.out}")


if __name__ == "__main__":
    main()
