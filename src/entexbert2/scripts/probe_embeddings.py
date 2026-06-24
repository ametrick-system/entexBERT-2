#!/usr/bin/env python3
"""
Embedding-probe analysis for entexBERT-2 (extended).

Attribution asks input->output sensitivity (flat on TP/TN). This asks what the pooled
EMBEDDING encodes. It takes analyze.py's PCA coordinates, attaches interpretable covariates
(sequence composition, repeat/complexity, genomic position/chromosome, and optional
mappability/repeat tracks), and answers:

  1. What does each PC encode?  Signed Spearman corr (continuous covariates) + chromosome
     eta^2 (categorical), and the DECISIVE multivariate CV R^2 of each PC from
     composition / repeat / position / ALL measured covariates.
  2. Is the label separable in embedding space, and is that separation any measured shortcut?
     CV AUROC of label from GC / composition / repeat / position / ALL-covariates / PCs.

If PC1 (the label axis) has near-zero R^2 from ALL measured covariates AND no covariate
family decodes the label, the embedding's discriminative axis is not a measured shortcut --
genuinely learned, non-trivial structure worth chasing. If some covariate explains it, you
found the shortcut.

This is a REPRESENTATION probe (what's encoded), not a causal claim about the decision
(that's ISM). State results as "the embedding encodes X", combined with the ISM result.

Runs on analyze.py outputs: pca.csv (+ predictions.csv). External tracks are optional hooks.
"""

import argparse
import bisect
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

try:
    import pyBigWig
    HAVE_PYBIGWIG = True
except Exception:
    HAVE_PYBIGWIG = False

BASES = ["A", "C", "G", "T"]

CHROM_COLS = ["chrom", "chromosome", "chr", "seqnames"]
START_COLS = ["window_start", "start", "chromStart", "win_start"]
END_COLS = ["window_end", "end", "chromEnd", "win_end"]
SNVPOS_COLS = ["snv_pos", "pos", "position", "snp_pos", "variant_pos"]
LABEL_RELATED_COLS = ["n_tissues", "num_tissues", "tissue_count", "n_as_tissues",
                      "total_reads", "ref_reads", "alt_reads", "imbalance_significance",
                      "min_total_reads", "n_significant", "as_tissue_count"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pca_csv", required=True)
    p.add_argument("--predictions_csv", required=True, help="Needs 'sequence' + any meta/coords.")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--label_col", default="target")
    p.add_argument("--n_pcs", type=int, default=5)
    p.add_argument("--correct_only", action="store_true",
                   help="Restrict to correct calls (TP+TN).")
    p.add_argument("--cv_folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    # window-coordinate resolution (for chrom eta^2 + external tracks)
    p.add_argument("--chrom_col", default=None)
    p.add_argument("--start_col", default=None)
    p.add_argument("--end_col", default=None)
    p.add_argument("--snv_pos_col", default=None,
                   help="If only a SNV coordinate is present, window = [pos-left_bp, pos+right_bp].")
    p.add_argument("--left_bp", type=int, default=256)
    p.add_argument("--right_bp", type=int, default=256)
    # optional external tracks
    p.add_argument("--mappability_bigwig", default=None, help="bigWig; mean per window (needs pyBigWig).")
    p.add_argument("--repeats_bed", default=None, help="BED; fraction of window overlapping repeats.")
    return p.parse_args()


# ----------------------- sequence covariates -----------------------
def seq_composition(seq):
    s = str(seq).upper()
    n = len(s)
    if n == 0:
        return {}
    counts = {b: s.count(b) for b in BASES}
    fracs = {f"{b.lower()}_frac": counts[b] / n for b in BASES}
    gc = (counts["G"] + counts["C"]) / n
    exp_cpg = (counts["C"] * counts["G"] / n) if n > 0 else 0.0
    cpg_oe = (s.count("CG") / exp_cpg) if exp_cpg > 0 else np.nan
    purine = (counts["A"] + counts["G"]) / n
    ps = np.array([counts[b] / n for b in BASES], dtype=float)
    ps = ps[ps > 0]
    entropy = float(-(ps * np.log2(ps)).sum())
    out = {"gc_frac": gc, "cpg_oe": cpg_oe, "shannon_entropy": entropy, "purine_frac": purine}
    out.update(fracs)
    return out


def seq_repeat_complexity(raw_seq):
    """Repeat/low-complexity proxies. lowercase_frac is a free RepeatMasker proxy IF the
    reference FASTA is soft-masked (lowercase = repeat); it's ~0 and uninformative otherwise."""
    s = str(raw_seq)
    n = len(s)
    if n == 0:
        return {}
    lower = sum(1 for c in s if c.islower()) / n
    u = s.upper()
    longest = 0
    in_run4 = 0
    i = 0
    while i < n:
        j = i
        while j < n and u[j] == u[i]:
            j += 1
        runlen = j - i
        longest = max(longest, runlen)
        if runlen >= 4:
            in_run4 += runlen
        i = j
    return {"lowercase_frac": lower,
            "longest_homopolymer_frac": longest / n,
            "low_complexity_frac": in_run4 / n}


# ----------------------- genomic-track covariates -----------------------
def resolve_windows(df, args):
    """Return (chroms, starts, ends) arrays or (None, None, None) if unresolvable."""
    def pick(cands, override):
        if override and override in df.columns:
            return override
        for c in cands:
            if c in df.columns:
                return c
        return None
    chrom_c = pick(CHROM_COLS, args.chrom_col)
    if chrom_c is None:
        return None, None, None
    chroms = df[chrom_c].astype(str).to_numpy()
    start_c, end_c = pick(START_COLS, args.start_col), pick(END_COLS, args.end_col)
    if start_c and end_c:
        return chroms, df[start_c].to_numpy(), df[end_c].to_numpy()
    snv_c = pick(SNVPOS_COLS, args.snv_pos_col)
    if snv_c:
        pos = df[snv_c].to_numpy()
        return chroms, pos - args.left_bp, pos + args.right_bp
    return chroms, None, None  # chrom present (for eta^2) but no window for tracks


def parse_bed(path):
    d = defaultdict(list)
    with open(path) as f:
        for line in f:
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            p = line.split()
            if len(p) < 3:
                continue
            try:
                d[p[0]].append((int(p[1]), int(p[2])))
            except ValueError:
                continue
    for c in d:
        d[c].sort()
    return d


def repeat_overlap_frac(intervals_by_chrom, chrom, wstart, wend, max_elt=50000):
    wstart, wend = int(wstart), int(wend)
    L = wend - wstart
    if L <= 0:
        return np.nan
    ivs = intervals_by_chrom.get(chrom)
    if not ivs:
        return 0.0
    starts = [s for s, _ in ivs]
    lo = bisect.bisect_left(starts, wstart - max_elt)
    hi = bisect.bisect_right(starts, wend)
    covered = 0
    for i in range(lo, hi):
        s, e = ivs[i]
        ov = min(e, wend) - max(s, wstart)
        if ov > 0:
            covered += ov
    return covered / L


def mappability_means(bw_path, chroms, starts, ends):
    bw = pyBigWig.open(bw_path)
    out = []
    for c, s, e in zip(chroms, starts, ends):
        try:
            v = bw.stats(str(c), int(s), int(e), type="mean")[0]
        except Exception:
            v = None
        out.append(np.nan if v is None else float(v))
    bw.close()
    return np.array(out, dtype=float)


# ----------------------- stats helpers -----------------------
def eta_squared(values, groups):
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups)
    ok = np.isfinite(values)
    values, groups = values[ok], groups[ok]
    if len(values) < 3:
        return np.nan
    grand = values.mean()
    ss_tot = ((values - grand) ** 2).sum()
    if ss_tot == 0:
        return np.nan
    ss_between = sum(len(values[groups == g]) * (values[groups == g].mean() - grand) ** 2
                     for g in np.unique(groups))
    return float(ss_between / ss_tot)


def cv_auroc(X, y, folds, seed):
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    ok = np.isfinite(X).all(axis=1)
    X, yy = X[ok], np.asarray(y)[ok]
    if X.shape[1] == 0 or len(np.unique(yy)) < 2 or len(yy) < folds * 2:
        return np.nan, np.nan
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    s = cross_val_score(pipe, X, yy, cv=skf, scoring="roc_auc")
    return float(s.mean()), float(s.std())


def cv_r2(X, y, folds, seed):
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    yy = np.asarray(y, dtype=float)
    ok = np.isfinite(X).all(axis=1) & np.isfinite(yy)
    X, yy = X[ok], yy[ok]
    if X.shape[1] == 0 or len(yy) < folds * 2:
        return np.nan
    pipe = make_pipeline(StandardScaler(), LinearRegression())
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    return float(cross_val_score(pipe, X, yy, cv=kf, scoring="r2").mean())


def onehot(values):
    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    return enc.fit_transform(np.asarray(values).reshape(-1, 1))


# ----------------------- plots -----------------------
def plot_corr_heatmap(corr, out_path, dpi):
    fig, ax = plt.subplots(figsize=(1.0 * corr.shape[1] + 2, 0.5 * corr.shape[0] + 2))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    ax.set_xticks(range(corr.shape[1])); ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(corr.shape[0])); ax.set_yticklabels(corr.index, fontsize=8)
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            v = corr.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="black" if abs(v) < 0.6 else "white")
    fig.colorbar(im, ax=ax, label="Spearman r")
    ax.set_title("PC vs covariate correlation")
    fig.tight_layout(); fig.savefig(out_path, dpi=dpi); plt.close(fig); print(f"Saved {out_path}")


def plot_pc_vs_gc(df, pc_col, label, out_path, dpi):
    fig, ax = plt.subplots(figsize=(7, 5))
    for lab, color in [(0, "#1f77b4"), (1, "#ff7f0e")]:
        m = label == lab
        ax.scatter(df.loc[m, "gc_frac"], df.loc[m, pc_col], s=12, alpha=0.6, color=color, label=f"label={lab}")
    ax.set_xlabel("GC fraction of window"); ax.set_ylabel(pc_col)
    ax.set_title(f"{pc_col} vs GC content (is the separation just composition?)")
    ax.legend(); fig.tight_layout(); fig.savefig(out_path, dpi=dpi); plt.close(fig); print(f"Saved {out_path}")


def plot_auroc_bar(rows, out_path, dpi):
    names = [r["features"] for r in rows]
    means = [r["cv_auroc"] for r in rows]; stds = [r["cv_auroc_std"] for r in rows]
    fig, ax = plt.subplots(figsize=(max(7, 1.1 * len(names)), 5))
    x = np.arange(len(names))
    ax.bar(x, means, yerr=stds, capsize=4, color="#4c72b0")
    ax.axhline(0.5, linestyle="--", linewidth=1, color="grey")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylabel("CV AUROC (label)"); ax.set_ylim(0.4, 1.0)
    ax.set_title("How well does each feature set decode the label?")
    fig.tight_layout(); fig.savefig(out_path, dpi=dpi); plt.close(fig); print(f"Saved {out_path}")


def plot_pc_explained(pc_expl_df, out_path, dpi):
    pcs = pc_expl_df["pc"].tolist()
    fig, ax = plt.subplots(figsize=(max(7, 1.2 * len(pcs)), 5))
    x = np.arange(len(pcs)); w = 0.2
    for off, col, lbl in [(-1.5*w, "r2_from_composition", "composition"),
                          (-0.5*w, "r2_from_repeat", "repeat/complexity"),
                          (0.5*w, "chrom_eta2", "chromosome (eta^2)"),
                          (1.5*w, "r2_from_all", "ALL measured")]:
        if col in pc_expl_df:
            ax.bar(x + off, pc_expl_df[col].fillna(0).to_numpy(), width=w, label=lbl)
    ax.set_xticks(x); ax.set_xticklabels(pcs)
    ax.set_ylabel("variance of PC explained (CV R^2 / eta^2)")
    ax.set_ylim(0, 1); ax.set_title("How much of each PC is explained by measured covariates?")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(out_path, dpi=dpi); plt.close(fig); print(f"Saved {out_path}")


# ----------------------- main -----------------------
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    pca = pd.read_csv(args.pca_csv)
    pred = pd.read_csv(args.predictions_csv)
    if "example_id" not in pca.columns or "example_id" not in pred.columns:
        raise ValueError("Both pca.csv and predictions.csv need 'example_id'.")
    if pred["example_id"].duplicated().any():
        raise ValueError("example_id not unique in predictions.csv.")

    pc_cols = [c for c in pca.columns if c.startswith("PC")]
    pc_cols = pc_cols[: args.n_pcs] if args.n_pcs > 0 else pc_cols
    coord_cols = [c for c in pred.columns if c in CHROM_COLS + START_COLS + END_COLS + SNVPOS_COLS]
    keep = ["example_id", "sequence"] + [c for c in pred.columns
                                         if c in LABEL_RELATED_COLS + coord_cols]
    keep = list(dict.fromkeys(keep))
    df = pca.merge(pred[keep].drop_duplicates("example_id"), on="example_id", how="left")

    if args.correct_only and "confusion_category" in df.columns:
        df = df[df["confusion_category"].isin(["TP", "TN"])].copy()
        print(f"Restricted to correct calls (TP+TN): {len(df)} rows")
    if args.label_col not in df.columns:
        raise ValueError(f"label column '{args.label_col}' not in pca.csv.")
    label = df[args.label_col].astype(int).to_numpy()

    # --- build covariates ---
    comp = pd.DataFrame([seq_composition(s) for s in df["sequence"]]).reset_index(drop=True)
    rep = pd.DataFrame([seq_repeat_complexity(s) for s in df["sequence"]]).reset_index(drop=True)
    families = {c: "composition" for c in comp.columns}
    families.update({c: "repeat" for c in rep.columns})
    cov = pd.concat([comp, rep], axis=1)

    chroms, wstart, wend = resolve_windows(df, args)
    chrom_vals = chroms if chroms is not None else None

    if wstart is not None and wend is not None:
        cov["window_mid"] = (np.asarray(wstart, float) + np.asarray(wend, float)) / 2.0
        families["window_mid"] = "position"
        if args.repeats_bed:
            try:
                iv = parse_bed(args.repeats_bed)
                cov["repeat_overlap_frac"] = [repeat_overlap_frac(iv, c, s, e)
                                              for c, s, e in zip(chroms, wstart, wend)]
                families["repeat_overlap_frac"] = "repeat"
                print(f"Loaded repeats from {args.repeats_bed}")
            except Exception as ex:
                print(f"WARNING: repeats_bed failed ({ex}); skipping.")
        if args.mappability_bigwig:
            if not HAVE_PYBIGWIG:
                print("WARNING: pyBigWig not available; skipping mappability.")
            else:
                try:
                    cov["mappability"] = mappability_means(args.mappability_bigwig, chroms, wstart, wend)
                    families["mappability"] = "position"
                    print(f"Loaded mappability from {args.mappability_bigwig}")
                except Exception as ex:
                    print(f"WARNING: mappability failed ({ex}); skipping.")
    elif args.repeats_bed or args.mappability_bigwig:
        print("WARNING: could not resolve window coordinates (need chrom + start/end or a SNV pos "
              "column); external tracks skipped. Pass --chrom_col/--start_col/--end_col or --snv_pos_col.")

    for c in LABEL_RELATED_COLS:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            cov[c] = df[c].to_numpy(); families[c] = "label_related"

    cov = cov.reset_index(drop=True)
    work = pd.concat([df[pc_cols].reset_index(drop=True), cov], axis=1)
    work[args.label_col] = label
    cov_cols = list(cov.columns)
    comp_cols = [c for c in cov_cols if families[c] == "composition"]
    rep_cols = [c for c in cov_cols if families[c] == "repeat"]
    pos_cols = [c for c in cov_cols if families[c] == "position"]
    lab_cols = [c for c in cov_cols if families[c] == "label_related"]
    numeric_cov = comp_cols + rep_cols + pos_cols  # excludes label-related (tautological)

    # --- 1. correlation heatmap (continuous covariates) ---
    corr = pd.DataFrame(index=pc_cols, columns=numeric_cov, dtype=float)
    for pc in pc_cols:
        for cvn in numeric_cov:
            corr.loc[pc, cvn] = work[[pc, cvn]].corr(method="spearman").iloc[0, 1]
    corr.to_csv(os.path.join(args.output_dir, "pc_covariate_correlation.csv"))
    if numeric_cov:
        plot_corr_heatmap(corr, os.path.join(args.output_dir, "pc_covariate_correlation.png"), 150)

    # --- 2. how much of each PC is explained (the decisive multivariate R^2) ---
    chrom_oh = onehot(chrom_vals) if chrom_vals is not None else None
    all_num_X = work[numeric_cov].to_numpy() if numeric_cov else np.empty((len(work), 0))
    all_X = np.concatenate([all_num_X, chrom_oh], axis=1) if chrom_oh is not None else all_num_X
    rows = []
    for pc in pc_cols:
        y = work[pc].to_numpy()
        top = corr.loc[pc].abs().idxmax() if numeric_cov else None
        rows.append({
            "pc": pc,
            "r2_from_composition": cv_r2(work[comp_cols].to_numpy(), y, args.cv_folds, args.seed) if comp_cols else np.nan,
            "r2_from_repeat": cv_r2(work[rep_cols].to_numpy(), y, args.cv_folds, args.seed) if rep_cols else np.nan,
            "r2_from_position": cv_r2(work[pos_cols].to_numpy(), y, args.cv_folds, args.seed) if pos_cols else np.nan,
            "chrom_eta2": eta_squared(y, chrom_vals) if chrom_vals is not None else np.nan,
            "r2_from_all": cv_r2(all_X, y, args.cv_folds, args.seed) if all_X.shape[1] else np.nan,
            "top_covariate": top,
            "top_abs_spearman": float(abs(corr.loc[pc, top])) if top is not None else np.nan,
        })
    pc_expl_df = pd.DataFrame(rows)
    pc_expl_df.to_csv(os.path.join(args.output_dir, "pc_explained_by_covariates.csv"), index=False)
    plot_pc_explained(pc_expl_df, os.path.join(args.output_dir, "pc_explained_by_covariates.png"), 150)

    # --- 3. label decodability per feature set ---
    feature_sets = [("GC only", work[["gc_frac"]].to_numpy()),
                    ("composition", work[comp_cols].to_numpy() if comp_cols else np.empty((len(work), 0)))]
    if rep_cols:
        feature_sets.append(("repeat/complexity", work[rep_cols].to_numpy()))
    if pos_cols or chrom_oh is not None:
        pos_X = work[pos_cols].to_numpy() if pos_cols else np.empty((len(work), 0))
        pos_X = np.concatenate([pos_X, chrom_oh], axis=1) if chrom_oh is not None else pos_X
        feature_sets.append(("position+chrom", pos_X))
    if all_X.shape[1]:
        feature_sets.append(("ALL covariates (no embedding)", all_X))
    feature_sets.append(("PCs", work[pc_cols].to_numpy()))
    if lab_cols:
        feature_sets.append(("label-related meta", work[lab_cols].to_numpy()))
    auroc_rows = []
    for name, X in feature_sets:
        m, sd = cv_auroc(X, label, args.cv_folds, args.seed)
        auroc_rows.append({"features": name, "cv_auroc": m, "cv_auroc_std": sd})
    auroc_df = pd.DataFrame(auroc_rows)
    auroc_df.to_csv(os.path.join(args.output_dir, "label_decode_auroc.csv"), index=False)
    plot_auroc_bar(auroc_rows, os.path.join(args.output_dir, "label_decode_auroc.png"), 150)

    if pc_cols:
        plot_pc_vs_gc(work, pc_cols[0], label, os.path.join(args.output_dir, f"{pc_cols[0]}_vs_gc.png"), 150)

    print("\n=== How much of each PC is explained by measured covariates? ===")
    print(pc_expl_df.to_string(index=False))
    print("\n=== Label decodability (CV AUROC) ===")
    print(auroc_df.to_string(index=False))
    if lab_cols:
        print(f"\nNOTE: 'label-related meta' ({', '.join(lab_cols)}) is near-tautological with the label.")
    print("\nDECISIVE READ: if PC1's r2_from_all and chrom_eta2 are ~0, AND 'ALL covariates "
          "(no embedding)' decodes the label near 0.5 while 'PCs' decodes high, then the embedding's "
          "label axis is NOT any measured shortcut -- genuinely learned, non-trivial structure. "
          "Representation probe, not causation.")
    print(f"\nOutput: {args.output_dir}")


if __name__ == "__main__":
    main()
