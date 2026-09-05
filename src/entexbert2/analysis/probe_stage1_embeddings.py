#!/usr/bin/env python
"""
Probe Stage-1 binding-trunk REFERENCE-window embeddings for AS vs non-AS separation, and — the
point of the exercise — decide whether any separation is REAL or a coverage/composition shortcut.

Input: the npz from dump_stage1_ref_embeddings.py (embeddings, labels, total_reads, mu, sequence,
chrom, anchor). Produces:
  * a PCA scatter (z-scored dims) colored by AS/non-AS, by log10 coverage, and by binding score mu
    -- if the AS/non-AS split visually tracks the coverage coloring, the axis IS coverage.
  * the DECISIVE check: 5-fold CV AUROC of the AS label decoded from each of
      coverage (total_reads) | GC | binding score mu | top-k PCs | PCs RESIDUALIZED on coverage+mu.
    If PCs beat coverage AND the residualized PCs still separate -> genuine learned regional signal.
    If residualized PCs collapse to ~0.5 -> the embedding's separation was the depth shortcut.

CPU-only; runs anywhere.
"""
import argparse, numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler

AS_C, NON_C = "#7b2d8e", "#2fb5d6"


def gc_frac(seqs):
    out = np.zeros(len(seqs))
    for i, s in enumerate(seqs):
        s = str(s).upper(); n = len(s) or 1
        out[i] = (s.count("G") + s.count("C")) / n
    return out


def cv_auroc(X, y, seed=0):
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    X = StandardScaler().fit_transform(X)
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    clf = LogisticRegression(max_iter=2000, C=1.0)
    return float(np.mean(cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")))


def residualize(X, C):
    """Remove the linear component predictable from confounds C from each column of X."""
    C = np.asarray(C, dtype=float)
    if C.ndim == 1:
        C = C[:, None]
    C = StandardScaler().fit_transform(C)
    return X - LinearRegression().fit(C, X).predict(C)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", default="stage1_ref_probe.png")
    ap.add_argument("--n_pcs", type=int, default=10, help="PCs for the CV decode (scatter uses PC1/PC2)")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    d = np.load(a.npz, allow_pickle=True)
    emb = d["embeddings"].astype(float); y = d["labels"].astype(int)
    cov = d["total_reads"].astype(float); mu = d["mu"].astype(float)
    seqs = d["sequence"]; gc = gc_frac(seqs)
    logcov = np.log10(np.clip(cov, 1, None))

    Z = StandardScaler().fit_transform(emb)          # z-score dims (fix the scale axis)
    k = min(a.n_pcs, Z.shape[1], Z.shape[0] - 1)
    pcs = PCA(n_components=k, random_state=0).fit(Z)
    P = pcs.transform(Z); evr = pcs.explained_variance_ratio_

    # ---- the decisive shortcut check ----
    # mu may be a constant placeholder (attention descriptors have no binding head) -> use it as a
    # confound only when it actually varies; otherwise residualize on coverage alone.
    mu_used = float(np.ptp(mu)) > 1e-9
    conf = np.c_[logcov, mu] if mu_used else logcov[:, None]
    resid_label = "PCs | resid(cov,mu)" if mu_used else "PCs | resid(cov)"
    res = {
        "coverage(total_reads)": cv_auroc(logcov, y, a.seed),
        "GC content":            cv_auroc(gc, y, a.seed),
        f"top-{k} PCs":          cv_auroc(P, y, a.seed),
        resid_label:             cv_auroc(residualize(P, conf), y, a.seed),
    }
    if mu_used:
        res["binding score mu"] = cv_auroc(mu, y, a.seed)
    print("=== AS-vs-nonAS decodability (5-fold CV AUROC) ===")
    for name, v in res.items():
        print(f"  {name:22s}: {v:.3f}")
    # The test is whether the EMBEDDING separates the classes BEYOND coverage/mu -- i.e. whether the
    # residualized-PC AUROC clears chance by a margin. Coverage's own AUROC is NOT the bar (the label
    # correlating with depth is expected; the question is what the embedding adds on top).
    resid_auroc = res[resid_label]
    verdict = ("REAL regional signal (embedding separates beyond coverage+mu; resid AUROC "
               f"{resid_auroc:.3f} > 0.55)" if resid_auroc > 0.55 else
               "SHORTCUT: separation not retained beyond coverage/binding-strength (resid AUROC "
               f"{resid_auroc:.3f} <= 0.55)")
    print(f"[verdict] {verdict}")

    # ---- figure ----
    fig, ax = plt.subplots(1, 3, figsize=(16, 5.2))
    x1, x2 = P[:, 0], P[:, 1]
    ax[0].scatter(x1[y == 1], x2[y == 1], s=9, alpha=.35, c=AS_C, label=f"AS (n={int(y.sum())})")
    ax[0].scatter(x1[y == 0], x2[y == 0], s=9, alpha=.35, c=NON_C, label=f"non-AS (n={int((y==0).sum())})")
    ax[0].legend(loc="upper right", fontsize=9)
    ax[0].set_title(f"AS vs non-AS\nPC-decode AUROC={res[f'top-{k} PCs']:.3f} | resid={resid_auroc:.3f}")
    for axi, val, lab in [(ax[1], logcov, "log10 coverage"), (ax[2], mu, "binding score mu")]:
        sc = axi.scatter(x1, x2, s=9, alpha=.45, c=val, cmap="viridis")
        fig.colorbar(sc, ax=axi, fraction=.046, pad=.04, label=lab)
        axi.set_title(f"colored by {lab}\n(if this matches the AS/non-AS split, the axis IS {lab})")
    for a_ in ax:
        a_.set_xlabel(f"PC1 ({evr[0]*100:.1f}%)"); a_.set_ylabel(f"PC2 ({evr[1]*100:.1f}%)")
    fig.suptitle(f"Stage-1 trunk reference-window embeddings — {verdict}", fontweight="bold")
    fig.tight_layout()
    fig.savefig(a.out, dpi=a.dpi, bbox_inches="tight")
    print(f"[wrote] {a.out}")


if __name__ == "__main__":
    main()
