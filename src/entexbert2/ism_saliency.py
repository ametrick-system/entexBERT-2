#!/usr/bin/env python
"""
In-silico mutagenesis (ISM) saliency for the entexBERT-2 binding trunk (Stage-1, regression).

For each input window of length L we score the reference sequence, then every single-base
substitution (3 alts x L positions), re-TOKENIZING each mutant (BPE is re-run per mutant, so
the attribution is tokenization-agnostic -- the whole reason ISM is preferred here over
gradient/attention). Output is a per-window (L, 4) effect matrix and MoDISco-ready arrays.

Runs on the cluster (GPU + checkpoint). Plot/overlay happens locally from the saved .npz.

Conventions:
  delta[w,i,b] = score(mutate pos i -> base b) - score(ref)    (ref-base entry = 0)
  importance[w,i] = -mean_{b != ref} delta[w,i,b]              (>0 where mutating AWAY hurts
                     binding, i.e. motif positions; the trunk score is predicted fold-change)
  hyp_scores[w,i,b] = delta[w,i,b] - mean_b delta[w,i,b]       (MoDISco hypothetical-contribution
                     convention: mean-centered per position, so the REF base carries -mean(delta)
                     = +importance at conserved sites. NOTE: -delta is WRONG for MoDISco -- delta
                     at the ref base is 0, so onehot*(-delta) projects to all zeros and seqlet
                     extraction finds nothing.)
  contrib[w,i,:]    = onehot[w,i,:] * hyp_scores[w,i,:]        (actual per-base contribution =
                     the ref base's mean-centered hypothetical score; MoDISco input)
"""
import argparse, numpy as np, pandas as pd

BASES = "ACGT"
B2I = {b: i for i, b in enumerate(BASES)}


def build_mutant_list(seqs):
    """Flatten [ref, all single mutants] with an index of (win, pos, base_idx); pos=-1 = baseline."""
    inputs, index = [], []
    for w, s in enumerate(seqs):
        inputs.append(s); index.append((w, -1, -1))
        for i, ch in enumerate(s):
            for b in BASES:
                if b == ch:
                    continue
                inputs.append(s[:i] + b + s[i + 1:])
                index.append((w, i, B2I[b]))
    return inputs, index


def assemble(seqs, index, scores):
    """Turn flat scores back into per-window onehot / delta / importance / hyp / contrib arrays."""
    N = len(seqs); L = len(seqs[0])
    base = np.full(N, np.nan, dtype=np.float64)
    delta = np.zeros((N, L, 4), dtype=np.float32)
    onehot = np.zeros((N, L, 4), dtype=np.float32)
    for w, s in enumerate(seqs):
        for i, ch in enumerate(s):
            if ch in B2I:
                onehot[w, i, B2I[ch]] = 1.0
    for (w, i, b), sc in zip(index, scores):
        if i == -1:
            base[w] = sc
    for (w, i, b), sc in zip(index, scores):
        if i == -1:
            continue
        delta[w, i, b] = sc - base[w]
    # importance = -mean over the 3 non-ref bases (ref entry is 0, so sum/3)
    importance = -(delta.sum(axis=2) / 3.0).astype(np.float32)
    # MoDISco hypothetical contributions: mean-center delta per position so the ref base carries
    # -mean(delta) (positive at conserved sites). -delta alone is WRONG: it's 0 at the ref base,
    # so onehot*(-delta) is all zeros and seqlet extraction fails.
    hyp = (delta - delta.mean(axis=2, keepdims=True)).astype(np.float32)
    contrib = (onehot * hyp).astype(np.float32)
    return dict(onehot=onehot, delta=delta, importance=importance,
                hyp_scores=hyp, contrib=contrib, base_score=base.astype(np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_dir", required=True, help="Stage-1 binding trunk (regression)")
    ap.add_argument("--windows_csv", required=True, help="CSV with a sequence column")
    ap.add_argument("--seq_col", default="sequence")
    ap.add_argument("--rank_col", default="binding_label_raw",
                    help="rank windows desc by this col and take top n_windows (skip if absent)")
    ap.add_argument("--feature_col", default="feature_type",
                    help="keep only rows whose value contains 'peak' (skip if absent)")
    ap.add_argument("--n_windows", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from entexbert2.model_io import run_inference   # cluster-only import

    df = pd.read_csv(a.windows_csv)
    if a.feature_col in df.columns:
        df = df[df[a.feature_col].astype(str).str.contains("peak", case=False, na=False)]
    if a.rank_col in df.columns:
        df = df.sort_values(a.rank_col, ascending=False)
    df = df.head(a.n_windows).reset_index(drop=True)
    seqs = df[a.seq_col].astype(str).str.upper().str.strip().tolist()
    if not seqs:
        raise SystemExit("no windows selected -- check --seq_col/--feature_col/--rank_col")
    L = len(seqs[0])
    bad = [i for i, s in enumerate(seqs) if len(s) != L]
    if bad:
        raise SystemExit(f"windows must be equal length for a stacked ISM array; {len(bad)} differ (L0={L})")

    inputs, index = build_mutant_list(seqs)
    print(f"[ISM] {len(seqs)} windows x L={L}  -> {len(inputs)} forward inputs "
          f"({len(inputs)-len(seqs)} mutants + {len(seqs)} baselines)")
    logits, _emb, rc = run_inference(a.checkpoint_dir, inputs, batch_size=a.batch_size, device=a.device)
    scores = np.asarray(logits, dtype=np.float64).reshape(-1)
    if scores.shape[0] != len(inputs):
        raise SystemExit(f"scored {scores.shape[0]} != {len(inputs)} inputs (multi-track trunk? need num_labels=1)")

    arr = assemble(seqs, index, scores)
    np.savez_compressed(a.out, seqs=np.array(seqs), **arr)
    imp = arr["importance"]
    print(f"[ISM] saved -> {a.out} | importance range [{imp.min():.3f},{imp.max():.3f}] "
          f"mean per-window peak {imp.max(axis=1).mean():.3f}")


if __name__ == "__main__":
    main()
