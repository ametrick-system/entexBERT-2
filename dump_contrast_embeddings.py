#!/usr/bin/env python
"""
Dump per-locus contrast representations for a trained CLASSIFICATION contrast head, at TWO
stages, so we can ask whether the head separates AS from non-AS loci more than the raw trunk:

  * TRUNK contrast : h1 - h2      (pool_ref - pool_alt, hidden dim, e.g. 768)  -- what the frozen
                     trunk already produces, BEFORE the learned projection.
  * HEAD  contrast : P(h1) - P(h2) (proj_dim, e.g. 128)                        -- the space the
                     distance s = ||P(h1) - P(h2)|| is actually taken in.

Also saved: the model logit ell (= a*s + b = logit P(ASB)) and the binary AS label.

Input is a Stage-2 build CSV (columns: sequence1, sequence2, label[, depth]) -- the SAME hap_pair
windows the head was trained/evaluated on, so no FASTA or window-rebuilding is needed. Use the
held-out test split for an honest picture.

Run ON THE CLUSTER (needs transformers + the trained checkpoint + a GPU is nice but CPU works).

  python dump_contrast_embeddings.py \
    --checkpoint_dir $WORK/runs/clf_2a_s20 \
    --data_csv       $WORK/inputs_clf/test.csv \
    --out            $WORK/pca_contrast_test.npz \
    --device cuda --batch_size 64 --max_rows 0
"""
import argparse, csv, sys
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_dir", required=True, help="trained classification Stage-2 run dir")
    ap.add_argument("--data_csv", required=True, help="Stage-2 build CSV (sequence1,sequence2,label)")
    ap.add_argument("--out", required=True, help="output .npz")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_rows", type=int, default=0,
                    help="0 = all rows; else cap (keeps ALL positives + a random negative subset).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label_col", default="label",
                    help="label column; numeric (>0.5=pos) OR categorical AS/nonAS (as_label)")
    args = ap.parse_args()

    # entexbert2 must be importable (repo on PYTHONPATH)
    from entexbert2.model_io import run_inference

    with open(args.data_csv) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit(f"empty CSV: {args.data_csv}")
    cols = rows[0].keys()
    for need in ("sequence1", "sequence2", args.label_col):
        if need not in cols:
            sys.exit(f"{args.data_csv} missing column '{need}'; has {list(cols)}")

    def _lab(v):
        s = str(v).strip()
        if s.lower() in ("as", "1", "true"):  return 1
        if s.lower() in ("nonas", "non-as", "0", "false"):  return 0
        return int(float(s) > 0.5)   # numeric fallback
    labels = np.array([_lab(r[args.label_col]) for r in rows], dtype=int)
    idx = np.arange(len(rows))

    # optional subsample: keep every positive, randomly downsample negatives, to bound compute.
    if args.max_rows and args.max_rows < len(rows):
        rng = np.random.default_rng(args.seed)
        pos = idx[labels == 1]
        neg = idx[labels == 0]
        n_neg_keep = max(0, args.max_rows - len(pos))
        neg_keep = rng.choice(neg, size=min(n_neg_keep, len(neg)), replace=False)
        idx = np.sort(np.concatenate([pos, neg_keep]))
        rows = [rows[i] for i in idx]
        labels = labels[idx]
        print(f"[subsample] kept {len(rows)} rows ({int((labels==1).sum())} pos, "
              f"{int((labels==0).sum())} neg) of original")

    texts = [[r["sequence1"], r["sequence2"]] for r in rows]
    print(f"[in] {len(texts)} loci | pos={int((labels==1).sum())} neg={int((labels==0).sum())}")

    # dump_pools=True -> (logits, emb, pool_ref, pool_alt, run_config)
    #   emb       = P(h1) - P(h2)   (head contrast, proj_dim)
    #   pool_ref  = h1, pool_alt = h2   (trunk pools, hidden dim)
    logits, emb, pool_ref, pool_alt, rc = run_inference(
        args.checkpoint_dir, texts, batch_size=args.batch_size,
        device=args.device, dump_pools=True)

    logits = np.asarray(logits, dtype=np.float32).reshape(-1)     # ell = logit P(ASB)
    head_contrast = np.asarray(emb, dtype=np.float32)             # (N, proj_dim)
    pool_ref = np.asarray(pool_ref, dtype=np.float32)             # (N, H)
    pool_alt = np.asarray(pool_alt, dtype=np.float32)             # (N, H)
    trunk_contrast = pool_ref - pool_alt                          # (N, H)

    assert head_contrast.shape[0] == trunk_contrast.shape[0] == len(labels), "row mismatch"
    print(f"[dump] trunk_contrast {trunk_contrast.shape} | head_contrast {head_contrast.shape} "
          f"| task={rc.get('task')} proj_dim={rc.get('proj_dim')}")

    # Stash the trained logistic-link scalars a>0, b so the plot can draw the decision radius
    # delta* = -b/a (p=0.5 boundary) without re-loading the checkpoint. Read the raw parameters
    # dist_a, dist_b straight from the state dict (tiny tensors) and apply a = softplus(a_raw).
    extra = {}
    try:
        import torch, os, glob
        sd_path = os.path.join(args.checkpoint_dir, "pytorch_model.bin")
        if not os.path.exists(sd_path):
            cands = glob.glob(os.path.join(args.checkpoint_dir, "*.bin"))
            sd_path = cands[0] if cands else None
        if sd_path:
            sd = torch.load(sd_path, map_location="cpu")
            if "dist_a" in sd and "dist_b" in sd:
                a_raw = float(sd["dist_a"].reshape(-1)[0])
                b = float(sd["dist_b"].reshape(-1)[0])
                a = float(torch.nn.functional.softplus(torch.tensor(a_raw)))
                extra = {"a_raw": np.float32(a_raw), "a": np.float32(a), "b": np.float32(b)}
                print(f"[dump] head scalars: a_raw={a_raw:.4g} -> a={a:.4g} | b={b:.4g} "
                      f"| decision radius -b/a={-b/a:.4g}")
    except Exception as e:
        print(f"[dump] could not read a/b from checkpoint ({e}); plot will need --a_raw/--b")

    np.savez_compressed(
        args.out,
        labels=labels.astype(np.int8),
        ell=logits,
        trunk_contrast=trunk_contrast,
        head_contrast=head_contrast,
        trunk_norm=np.linalg.norm(trunk_contrast, axis=1).astype(np.float32),
        head_norm=np.linalg.norm(head_contrast, axis=1).astype(np.float32),
        checkpoint_dir=np.array(args.checkpoint_dir),
        data_csv=np.array(args.data_csv),
        **extra,
    )
    print(f"[write] {args.out}")


if __name__ == "__main__":
    main()
