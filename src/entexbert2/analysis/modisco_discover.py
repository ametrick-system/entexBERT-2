#!/usr/bin/env python
"""
De novo motif discovery from entexBERT-2 trunk ISM scores.

Stage 1  TF-MoDISco : cluster ISM seqlets -> de novo motifs (CWMs/PPMs), no prior.
Stage 2  TOMTOM     : match each discovered motif to JASPAR CORE vertebrates (memelite, no
                      MEME binary; reverse-complement aware) -> named TF hits + q-values.

Runs LOCALLY (CPU) on the ISM .npz (onehot + hyp_scores). Set NUMBA_CACHE_DIR before import.

Outputs:
  <out>/modisco_results.h5        (raw MoDISco output)
  <out>/discovered_motifs.json    (per-pattern PPM, IC, n_seqlets, top TOMTOM hits)
"""
import os
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.makedirs(os.environ["NUMBA_CACHE_DIR"], exist_ok=True)
import argparse, json, h5py, numpy as np

BASES = "ACGT"


def load_jaspar(path):
    """{id: {name, pfm{A,C,G,T}}} -> (ids, names, list of (W,4) prob matrices)."""
    db = json.load(open(path))
    ids, names, mats = [], [], []
    for k, v in db.items():
        M = np.array([v["pfm"][b] for b in BASES], dtype=np.float64).T  # (W,4) counts
        M = M + 1e-6
        mats.append(M / M.sum(axis=1, keepdims=True))
        ids.append(v.get("id", k)); names.append(v.get("name", k))
    return ids, names, mats


def cwm_to_ppm(cwm):
    """A trimmed contribution-weight matrix -> a probability matrix for TOMTOM (softmax over |contrib|)."""
    a = np.abs(cwm)
    s = a.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return a / s


def extract_patterns(h5_path):
    """Pull per-pattern PPM (from the seqlet one-hot), CWM, and n_seqlets from a modisco h5."""
    out = []
    with h5py.File(h5_path, "r") as f:
        for grp_name in ("pos_patterns", "neg_patterns"):
            if grp_name not in f:
                continue
            grp = f[grp_name]
            # sort pattern_0, pattern_1, ... numerically (MoDISco orders by seqlet support)
            pats_sorted = sorted(grp, key=lambda s: int(s.split("_")[-1]) if s.split("_")[-1].isdigit() else 1e9)
            for pat in pats_sorted:
                p = grp[pat]
                ppm = np.array(p["sequence"][:]) if "sequence" in p else None      # PPM (W,4)
                cwm = np.array(p["contrib_scores"][:]) if "contrib_scores" in p else None
                n = None
                if "seqlets" in p and "n_seqlets" in p["seqlets"]:
                    n = int(np.asarray(p["seqlets"]["n_seqlets"]).reshape(-1)[0])
                out.append(dict(group=grp_name, name=f"{grp_name}/{pat}",
                                ppm=ppm, cwm=cwm, n_seqlets=n))
    return out


def info_content(ppm):
    p = np.clip(ppm, 1e-9, 1)
    return float((2.0 + (p * np.log2(p)).sum(axis=1)).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ism", required=True, help=".npz with onehot + hyp_scores (N,L,4)")
    ap.add_argument("--jaspar", required=True, help="jaspar PFMs json")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--sliding_window_size", type=int, default=21)
    ap.add_argument("--min_metacluster_size", type=int, default=100)
    ap.add_argument("--max_seqlets", type=int, default=20000)
    ap.add_argument("--top_hits", type=int, default=5)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    import modiscolite as ml
    from memelite import tomtom

    # --- guard: one-sided attribution (e.g. head-ISM AS loci, where disrupting a driver base
    # moves the ASB logit in a single direction) leaves ONE polarity with zero sliding-window
    # sums. MoDISco's _isotonic_thresholds is called unconditionally for both polarities and
    # divides by len(values) -> ZeroDivisionError / "sample weights all zero" when a side is
    # empty. Wrap it so an empty side simply rejects everything (threshold=+/-inf) instead of
    # crashing; the non-empty side clusters normally.
    from modiscolite import extract_seqlets as _es
    _orig_isotonic = _es._isotonic_thresholds
    def _safe_isotonic(values, null_values, increasing, target_fdr, min_frac_neg=0.95):
        if len(values) == 0:
            return np.inf if increasing else -np.inf
        return _orig_isotonic(values, null_values, increasing, target_fdr, min_frac_neg)
    _es._isotonic_thresholds = _safe_isotonic

    d = np.load(a.ism, allow_pickle=True)
    onehot = d["onehot"].astype(np.float32)
    hyp = d["hyp_scores"].astype(np.float32)
    N, L, _ = onehot.shape
    print(f"[modisco] {N} windows x L={L}; running TF-MoDISco "
          f"(window={a.sliding_window_size}, min_metacluster={a.min_metacluster_size})")

    pos, neg = ml.tfmodisco.TFMoDISco(
        one_hot=onehot, hypothetical_contribs=hyp,
        sliding_window_size=a.sliding_window_size, flank_size=10,
        min_metacluster_size=a.min_metacluster_size,
        max_seqlets_per_metacluster=a.max_seqlets, verbose=True)
    h5 = os.path.join(a.out, "modisco_results.h5")
    ml.io.save_hdf5(h5, pos, neg, window_size=a.sliding_window_size)
    print(f"[modisco] saved {h5}")

    pats = extract_patterns(h5)
    print(f"[modisco] {len(pats)} patterns discovered")
    if not pats:
        json.dump([], open(os.path.join(a.out, "discovered_motifs.json"), "w"))
        print("[modisco] no patterns -- likely too few seqlets; increase N or lower min_metacluster_size")
        return

    jid, jname, jmats = load_jaspar(a.jaspar)
    # memelite.tomtom expects PWMs as (4, W) -- bases on axis 0. Our matrices are (W, 4).
    Qs = [(cwm_to_ppm(p["cwm"]) if p["cwm"] is not None else p["ppm"]).T for p in pats]
    Ts = [m.T for m in jmats]
    print(f"[tomtom] matching {len(Qs)} motifs vs {len(Ts)} JASPAR vertebrate PWMs")
    p_vals, scores, offsets, overlaps, strands = tomtom(Qs, Ts)

    def bh_qvals(pv):
        """Benjamini-Hochberg q-values across the target set for one query."""
        pv = np.asarray(pv, dtype=np.float64); m = len(pv)
        order = np.argsort(pv); ranked = pv[order]
        q = ranked * m / (np.arange(m) + 1)
        q = np.minimum.accumulate(q[::-1])[::-1]          # enforce monotonicity
        out = np.empty(m); out[order] = np.clip(q, 0, 1)
        return out

    results = []
    for i, p in enumerate(pats):
        qv = bh_qvals(p_vals[i])
        order = np.argsort(p_vals[i])[:a.top_hits]
        hits = [dict(matrix_id=jid[j], tf=jname[j], p_value=float(p_vals[i][j]),
                     q_value=float(qv[j]), strand=("-" if strands[i][j] else "+")) for j in order]
        results.append(dict(
            name=p["name"], group=p["group"], n_seqlets=p["n_seqlets"],
            info_content=round(info_content(p["ppm"]), 2) if p["ppm"] is not None else None,
            ppm=(p["ppm"].tolist() if p["ppm"] is not None else None),
            top_hits=hits))
        top = hits[0] if hits else {}
        print(f"  {p['name']:16s} n_seqlets={p['n_seqlets']}  "
              f"top: {top.get('tf','?')} (p={top.get('p_value',float('nan')):.2e}, "
              f"q={top.get('q_value',float('nan')):.2e})")
    json.dump(results, open(os.path.join(a.out, "discovered_motifs.json"), "w"))
    print(f"[done] -> {a.out}/discovered_motifs.json")


if __name__ == "__main__":
    main()
