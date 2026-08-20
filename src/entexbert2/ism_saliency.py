#!/usr/bin/env python
"""
In-silico mutagenesis (ISM) saliency for entexBERT-2. Works on ANY checkpoint -- the task is
read from run_config.json and the scoring target is chosen accordingly:

  regression  (Stage-1 binding trunk):  single-seq forward, target = binding score mu.
      delta[w,i,b] = mu(mutate pos i -> b) - mu(ref)          (ref entry = 0)
      importance[w,i] = -mean_{b!=ref} delta[w,i,b]           (>0 where mutating AWAY hurts
                        binding -> motif positions; mu is predicted fold-change)

  classification (Stage-2 ASB head, SFT-head OR no-SFT-head): TWIN forward. We mutate window1
      and HOLD window2 = the unmutated reference window, then score the ASB logit
      ell = a*||z1 - z2|| + b. Baseline ell(ref,ref) is the minimum contrast (a*0 + b). A base
      whose mutation the head reads as allele-specific pushes z1 away from z_ref and RAISES ell.
      delta[w,i,b] = ell(mut,ref) - ell(ref,ref)              (ref entry = 0, delta >= 0-ish)
      importance[w,i] = +mean_{b!=ref} delta[w,i,b]           (>0 where a variant would create
                        the largest allelic-binding contrast -> the head's motif-sensitivity map)
      NOTE: the head is symmetric (||z1-z2||), so ell-ISM is SIGN-BLIND -- it flags positions
      that CHANGE binding, not the direction. That is exactly right for motif LOCALIZATION
      (the aggregate-density and logo overlays care about *where*), which is what these figures
      replicate. A signed head-ISM would need a fixed projection axis; not done here.

Either way the output .npz carries the SAME arrays (onehot/delta/importance/hyp_scores/contrib/
base_score) so every downstream plotter (aggregate density, logo, MoDISco) is unchanged.

Both re-TOKENIZE each mutant (BPE re-run per mutant -> tokenization-agnostic attribution, the
whole reason ISM beats gradient/attention here). Runs on the cluster (GPU + checkpoint);
plot/overlay happens locally from the saved .npz.

MoDISco convention:
  hyp_scores[w,i,b] = delta[w,i,b] - mean_b delta[w,i,b]  (mean-centered per position; the REF
                      base carries -mean(delta) = +importance at conserved sites. -delta alone is
                      WRONG: it's 0 at the ref base, so onehot*(-delta) is all zeros.)
  contrib[w,i,:]    = onehot[w,i,:] * hyp_scores[w,i,:]
"""
import argparse, numpy as np, pandas as pd

BASES = "ACGT"
B2I = {b: i for i, b in enumerate(BASES)}


def build_mutant_list(seqs):
    """Flatten [ref, all single mutants] with an index of (win, pos, base_idx); pos=-1 = baseline.
    Returns single-sequence inputs (regression)."""
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


def build_mutant_pairs(seqs, partners=None):
    """Twin-ISM inputs: mutate window1, HOLD window2 = a fixed partner. Each element is a
    [window1, window2] pair; baseline (pos=-1) is [seq, partner]. Same (win,pos,base) index.

    partners=None  -> window2 = a copy of the reference window (baseline ell(ref,ref) = minimum
                      contrast; symmetric motif-sensitivity map, cleaner to aggregate).
    partners=list  -> window2 = the actual paired alt window for that locus (baseline is the real
                      observed allelic contrast, locus-dependent; saliency around the true variant)."""
    inputs, index = [], []
    for w, s in enumerate(seqs):
        p = s if partners is None else partners[w]
        inputs.append([s, p]); index.append((w, -1, -1))       # baseline: ell(seq, partner)
        for i, ch in enumerate(s):
            for b in BASES:
                if b == ch:
                    continue
                inputs.append([s[:i] + b + s[i + 1:], p])      # mutate w1, hold w2 = partner
                index.append((w, i, B2I[b]))
    return inputs, index


def assemble(seqs, index, scores, task):
    """Flat scores -> per-window onehot / delta / importance / hyp / contrib arrays.
    task='regression': importance = -mean delta (mutating away hurts binding).
    task='classification': importance = +mean delta (mutating creates allelic contrast)."""
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
    # importance = mean over the 3 non-ref bases (ref entry is 0, so sum/3), sign by task:
    #   regression     -> -mean (mutating AWAY from a motif base lowers binding => delta<0 => +imp)
    #   classification -> +mean (a variant at a read-out base RAISES the allelic contrast ell)
    sign = -1.0 if task == "regression" else +1.0
    importance = (sign * delta.sum(axis=2) / 3.0).astype(np.float32)
    # MoDISco hypothetical contributions: mean-center delta per position (ref base carries -mean).
    hyp = (delta - delta.mean(axis=2, keepdims=True)).astype(np.float32)
    contrib = (onehot * hyp).astype(np.float32)
    return dict(onehot=onehot, delta=delta, importance=importance,
                hyp_scores=hyp, contrib=contrib, base_score=base.astype(np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_dir", required=True,
                    help="any checkpoint; task auto-detected from run_config.json "
                         "(regression -> binding-trunk ISM, classification -> twin/head ISM)")
    ap.add_argument("--windows_csv", required=True, help="CSV with a sequence column")
    ap.add_argument("--seq_col", default="sequence")
    ap.add_argument("--rank_col", default="binding_label_raw",
                    help="rank windows desc by this col and take top n_windows (skip if absent)")
    ap.add_argument("--feature_col", default="feature_type",
                    help="keep only rows whose value contains this substring (skip if absent)")
    ap.add_argument("--feature_keep", default="peak",
                    help="value the feature_col must match (substring by default; e.g. 'peak')")
    ap.add_argument("--feature_exact", action="store_true",
                    help="require feature_col == feature_keep EXACTLY (use for as_label so "
                         "'AS' does not also match 'nonAS')")
    ap.add_argument("--n_windows", type=int, default=30)
    ap.add_argument("--twin_baseline", default="ref", choices=["ref", "alt"],
                    help="classification only. 'ref': hold window2 = copy of the reference window "
                         "(min-contrast baseline, symmetric motif map). 'alt': hold window2 = the "
                         "actual paired alt window from --partner_col (anchored at the real "
                         "observed allelic contrast).")
    ap.add_argument("--partner_col", default="sequence2",
                    help="column holding the alt/hap2 window; used only when --twin_baseline alt")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from entexbert2.model_io import run_inference, load_run_config   # cluster-only import

    task = load_run_config(a.checkpoint_dir).get("task", "regression")
    print(f"[ISM] checkpoint task = {task}  ->  "
          f"{'binding-trunk (single-seq mu)' if task == 'regression' else 'twin/head (ASB logit ell)'} ISM")

    df = pd.read_csv(a.windows_csv)
    if a.feature_col in df.columns:
        if a.feature_exact:
            df = df[df[a.feature_col].astype(str).str.strip() == a.feature_keep]
        else:
            df = df[df[a.feature_col].astype(str).str.contains(a.feature_keep, case=False, na=False)]
    if a.rank_col in df.columns:
        df = df.sort_values(a.rank_col, ascending=False)
    # n_windows <= 0 -> keep ALL selected rows (used for the AS/non-AS split figures)
    df = (df if a.n_windows <= 0 else df.head(a.n_windows)).reset_index(drop=True)
    seqs = df[a.seq_col].astype(str).str.upper().str.strip().tolist()
    if not seqs:
        raise SystemExit("no windows selected -- check --seq_col/--feature_col/--rank_col")
    L = len(seqs[0])
    bad = [i for i, s in enumerate(seqs) if len(s) != L]
    if bad:
        raise SystemExit(f"windows must be equal length for a stacked ISM array; {len(bad)} differ (L0={L})")

    if task == "classification":
        partners = None
        if a.twin_baseline == "alt":
            if a.partner_col not in df.columns:
                raise SystemExit(f"--twin_baseline alt needs '{a.partner_col}' in the CSV; "
                                 f"columns are {list(df.columns)}")
            partners = df[a.partner_col].astype(str).str.upper().str.strip().tolist()
            if any(len(p) != L for p in partners):
                raise SystemExit("partner windows must match window1 length L for stacked ISM")
        inputs, index = build_mutant_pairs(seqs, partners)
        kind = f"twin pairs [mut_w1, {a.twin_baseline}_w2]"
    else:
        inputs, index = build_mutant_list(seqs)
        kind = "single-seq"
    print(f"[ISM] {len(seqs)} windows x L={L}  -> {len(inputs)} forward inputs ({kind}; "
          f"{len(inputs)-len(seqs)} mutants + {len(seqs)} baselines)")

    logits, _emb, rc = run_inference(a.checkpoint_dir, inputs, batch_size=a.batch_size, device=a.device)
    scores = np.asarray(logits, dtype=np.float64).reshape(-1)
    if scores.shape[0] != len(inputs):
        raise SystemExit(f"scored {scores.shape[0]} != {len(inputs)} inputs "
                         f"(multi-track trunk? classification needs num_labels=1 contrast logit)")

    arr = assemble(seqs, index, scores, task)
    tb = a.twin_baseline if task == "classification" else "n/a"
    np.savez_compressed(a.out, seqs=np.array(seqs), task=task, twin_baseline=tb, **arr)
    imp = arr["importance"]
    print(f"[ISM] saved -> {a.out} | task={task} | importance range [{imp.min():.3f},{imp.max():.3f}] "
          f"mean per-window peak {imp.max(axis=1).mean():.3f}")


if __name__ == "__main__":
    main()
