#!/usr/bin/env python
"""
Compare entexBERT-2 de novo motifs (MoDISco CWMs) to JASPAR reference logos.

For each discovered pattern (from modisco_results.h5) whose top TOMTOM hit falls in a
requested TF family, render two logos side by side:
    LEFT  : discovered CWM (contribution-weighted matrix) -- what the MODEL actually used.
            Letters scaled by per-base contribution; can go negative (one-sided head-ISM).
    RIGHT : the matched JASPAR motif as an information-content (bits) sequence logo.

Families (name-substring match against the pattern's top TOMTOM hit):
    AP-1  : FOS, FOSL, FOSB, JUN, JUNB, JUND, JDP2, BATF
    CEBP  : CEBP, DDIT3
    NR    : RXR, PPAR, NR3C, NR4A, AR, PGR, ESR, RAR, THR, VDR, Ar, Pgr, Nr

Inputs: --h5 modisco_results.h5   --json discovered_motifs.json   --jaspar <jaspar json>
Runs LOCALLY (logomaker). Set NUMBA_CACHE_DIR is NOT needed (no memelite import here).
"""
import argparse, json, os
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logomaker
import pandas as pd

BASES = list("ACGT")

FAMILIES = {
    "AP-1": ["FOS", "FOSL", "FOSB", "JUN", "JUNB", "JUND", "JDP2", "BATF"],
    "CEBP": ["CEBP", "DDIT3"],
    "NR":   ["RXR", "PPAR", "NR3C", "NR4A", "ESR", "RAR", "THR",
             "VDR", "AR", "PGR", "Ar", "Pgr", "Nr", "Esr"],
}


def which_family(tf_name):
    """Return family key if the JASPAR hit name matches a family, else None."""
    if not tf_name:
        return None
    up = tf_name.upper()
    for fam, keys in FAMILIES.items():
        for k in keys:
            ku = k.upper()
            # word-ish match: substring is fine for these (FOSL1 contains FOS etc.)
            if ku in up:
                return fam
    return None


def load_h5_patterns(h5_path):
    """{name: {ppm(W,4), cwm(W,4), n_seqlets}} for every pattern in the h5."""
    out = {}
    with h5py.File(h5_path, "r") as f:
        for grp_name in ("pos_patterns", "neg_patterns"):
            if grp_name not in f:
                continue
            grp = f[grp_name]
            for pat in grp:
                p = grp[pat]
                ppm = np.array(p["sequence"][:]) if "sequence" in p else None
                cwm = np.array(p["contrib_scores"][:]) if "contrib_scores" in p else None
                n = None
                if "seqlets" in p and "n_seqlets" in p["seqlets"]:
                    n = int(np.asarray(p["seqlets"]["n_seqlets"]).reshape(-1)[0])
                out[f"{grp_name}/{pat}"] = dict(ppm=ppm, cwm=cwm, n_seqlets=n)
    return out


def load_jaspar(path):
    """{NAME_upper: (W,4) prob matrix} -- first matrix_id per name."""
    db = json.load(open(path))
    byname = {}
    for k, v in db.items():
        name = v.get("name", k)
        M = np.array([v["pfm"][b] for b in BASES], dtype=np.float64).T  # (W,4) counts
        M = M + 1e-6
        M = M / M.sum(axis=1, keepdims=True)
        byname.setdefault(name.upper(), M)
    return byname


def trim_cwm(cwm, frac=0.2):
    """Trim flanks to the informative core: keep positions with |contrib| >= frac*max."""
    if cwm is None or cwm.shape[0] == 0:
        return cwm, 0, 0
    per_pos = np.abs(cwm).sum(axis=1)
    if per_pos.max() <= 0:
        return cwm, 0, cwm.shape[0]
    keep = np.where(per_pos >= frac * per_pos.max())[0]
    if len(keep) == 0:
        return cwm, 0, cwm.shape[0]
    lo, hi = keep.min(), keep.max() + 1
    return cwm[lo:hi], lo, hi


def ppm_to_bits(ppm):
    """Info-content (bits) matrix for a sequence logo."""
    p = np.clip(ppm, 1e-9, 1.0)
    ic = 2.0 + (p * np.log2(p)).sum(axis=1)  # per-position information (bits)
    return ppm * ic[:, None]


def draw_cwm(ax, cwm, title):
    df = pd.DataFrame(cwm, columns=BASES)
    logomaker.Logo(df, ax=ax, color_scheme="classic")
    ax.set_title(title, fontsize=9)
    ax.set_ylabel("contribution", fontsize=8)
    ax.axhline(0, color="0.6", lw=0.6)
    ax.set_xticks([])


def draw_ref(ax, pwm, title):
    bits = ppm_to_bits(pwm)
    df = pd.DataFrame(bits, columns=BASES)
    logomaker.Logo(df, ax=ax, color_scheme="classic")
    ax.set_title(title, fontsize=9)
    ax.set_ylabel("bits", fontsize=8)
    ax.set_ylim(0, 2)
    ax.set_xticks([])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True, help="modisco_results.h5")
    ap.add_argument("--json", required=True, help="discovered_motifs.json (for top TOMTOM hit per pattern)")
    ap.add_argument("--jaspar", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tf_label", default="EP300")
    ap.add_argument("--only_families", default="AP-1,CEBP,NR",
                    help="comma list of family keys to include (default all cofactor families)")
    ap.add_argument("--max_patterns", type=int, default=8)
    ap.add_argument("--cwm_flip", action="store_true",
                    help="flip CWM sign so a one-sided negative motif reads upright")
    a = ap.parse_args()

    want = set(x.strip() for x in a.only_families.split(",") if x.strip())
    pats_h5 = load_h5_patterns(a.h5)
    jaspar = load_jaspar(a.jaspar)
    disc = json.load(open(a.json))  # list of {name, n_seqlets, top_hits:[{name,q_value,...}], ...}

    # pair each json pattern with its h5 CWM + family of its top hit
    rows = []
    for d in disc:
        name = d.get("name")
        hits = d.get("top_hits") or []
        top = hits[0] if hits else {}
        # modisco_discover.py writes each hit as {matrix_id, tf, p_value, q_value, strand};
        # accept "name" too for hand-built/older jsons.
        tf = top.get("tf") or top.get("name")
        q = top.get("q_value")
        fam = which_family(tf)
        if fam is None or fam not in want:
            continue
        h = pats_h5.get(name, {})
        cwm = h.get("cwm")
        if cwm is None:
            continue
        rows.append(dict(name=name, tf=tf, q=q, fam=fam,
                         n_seqlets=h.get("n_seqlets"), cwm=cwm))
    # order by family then by significance (small q first)
    fam_order = {f: i for i, f in enumerate(["AP-1", "CEBP", "NR"])}
    rows.sort(key=lambda r: (fam_order.get(r["fam"], 9), r["q"] if r["q"] is not None else 1))
    rows = rows[:a.max_patterns]

    if not rows:
        print("[warn] no discovered patterns matched the requested families:", want)
        # still emit an empty figure so the driver doesn't error
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, f"No {a.tf_label} patterns matched {sorted(want)}",
                ha="center", va="center"); ax.axis("off")
        fig.savefig(a.out, dpi=150, bbox_inches="tight"); return

    n = len(rows)
    fig, axes = plt.subplots(n, 2, figsize=(9, 1.9 * n), squeeze=False)
    for i, r in enumerate(rows):
        cwm = -r["cwm"] if a.cwm_flip else r["cwm"]
        cwm_t, _, _ = trim_cwm(cwm)
        qtxt = f"q={r['q']:.1e}" if r["q"] is not None else "q=NA"
        draw_cwm(axes[i][0], cwm_t,
                 f"{a.tf_label} de novo {r['name'].split('/')[-1]}  (n={r['n_seqlets']})")
        ref = jaspar.get((r["tf"] or "").upper())
        if ref is not None:
            draw_ref(axes[i][1], ref, f"JASPAR {r['tf']}  [{r['fam']}, {qtxt}]")
        else:
            axes[i][1].text(0.5, 0.5, f"{r['tf']} not in JASPAR json",
                            ha="center", va="center"); axes[i][1].axis("off")
    fig.suptitle(f"{a.tf_label} head-ISM de novo motifs vs cofactor references", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(a.out, dpi=150, bbox_inches="tight")
    print(f"[done] {n} patterns -> {a.out}")
    for r in rows:
        print(f"  {r['name']}  {r['fam']}:{r['tf']}  q={r['q']}  n={r['n_seqlets']}")


if __name__ == "__main__":
    main()