#!/usr/bin/env python
"""
enformer-pytorch predictor -- torch port of hdm2020/benchmark's enformer.predict.py.
Loads the SAME published Enformer weights (EleutherAI/enformer-official-rough) and computes the
IDENTICAL variant score: per human track, delta = mean_bins(P(alt)) - mean_bins(P(ref)).

Output matches the benchmark's TSV exactly (header=False): col0 = snp (chr_pos_ref_alt),
cols 1..5313 = per-track delta for human track index 0..5312 -> score_enformer_asb.py reads it as-is.

Genome-agnostic: extracts windows from whatever fasta you pass (use hg38.fa with hg38 SNP coords;
no liftover). SNP file: one `chr_pos_ref_alt` per line (1-based pos, + strand ref), from
build_enformer_snpfile.py.
"""
import argparse, sys, numpy as np, torch, pyfaidx
from enformer_pytorch import Enformer, str_to_one_hot, SEQUENCE_LENGTH  # SEQUENCE_LENGTH = 196608

def read_snps(path):
    out = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        chrom, pos, ref, alt = line.split("_")
        out.append((line, chrom, int(pos), ref.upper(), alt.upper()))
    return out

def window(fa, chrom, pos1, L):
    """L-bp window centered so the 1-based variant `pos1` sits at index L//2 (0-based). Pads with N."""
    c = L // 2
    start0 = (pos1 - 1) - c            # 0-based inclusive
    end0 = start0 + L                  # exclusive
    clen = len(fa[chrom])
    lo, hi = max(start0, 0), min(end0, clen)
    seq = str(fa[chrom][lo:hi]).upper()
    seq = ("N" * max(-start0, 0)) + seq + ("N" * max(end0 - clen, 0))
    return seq  # len L; variant at index c

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--snp", required=True)
    ap.add_argument("-g", "--genome", required=True, help="fasta path (hg38.fa)")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--model", default="EleutherAI/enformer-official-rough")
    ap.add_argument("--batch_size", type=int, default=2, help="variants per forward (x2 seqs: ref+alt)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: only first N variants (0 = all)")
    args = ap.parse_args()

    fa = pyfaidx.Fasta(args.genome)
    snps = read_snps(args.snp)
    if args.limit:
        snps = snps[:args.limit]
    print(f"[enformer] {len(snps)} variants; window L={SEQUENCE_LENGTH}", flush=True)

    model = Enformer.from_pretrained(args.model, use_tf_gamma=True).to(args.device).eval()
    c = SEQUENCE_LENGTH // 2
    n_mismatch = 0
    rows_snp, rows_delta = [], []

    def flush_batch(batch):
        nonlocal n_mismatch
        seqs, names = [], []
        for (name, chrom, pos1, ref, alt) in batch:
            ref_seq = window(fa, chrom, pos1, SEQUENCE_LENGTH)
            if ref_seq[c] != ref:
                n_mismatch += 1
            alt_seq = ref_seq[:c] + alt + ref_seq[c + 1:]
            seqs.append(ref_seq); seqs.append(alt_seq); names.append(name)
        x = str_to_one_hot(seqs).to(args.device)                # (2B, L, 4)
        with torch.no_grad():
            out = model(x, head="human")
            if isinstance(out, dict):
                out = out["human"]
            mean_bins = out.mean(dim=1).float().cpu().numpy()   # (2B, 5313)
        ref_pred = mean_bins[0::2]; alt_pred = mean_bins[1::2]   # de-interleave
        delta = alt_pred - ref_pred                             # (B, 5313)
        for j, name in enumerate(names):
            rows_snp.append(name); rows_delta.append(delta[j])

    batch = []
    for i, s in enumerate(snps):
        batch.append(s)
        if len(batch) == args.batch_size:
            flush_batch(batch); batch = []
            if (i + 1) % 200 < args.batch_size:
                print(f"[enformer] {i+1}/{len(snps)}", flush=True)
    if batch:
        flush_batch(batch)

    D = np.vstack(rows_delta)
    if n_mismatch:
        frac = n_mismatch / len(snps)
        print(f"[enformer] WARNING: ref-allele mismatch at center for {n_mismatch}/{len(snps)} "
              f"({100*frac:.1f}%) -- check coord convention if high (>5%).", flush=True)
        if frac > 0.05:
            sys.exit(f"[enformer] FATAL: {100*frac:.1f}% ref mismatches -> coordinate/strand convention is wrong.")
    with open(args.out, "w") as f:
        for name, d in zip(rows_snp, D):
            f.write(name + "\t" + "\t".join(f"{v:.6g}" for v in d) + "\n")
    print(f"[enformer] wrote {args.out}  ({D.shape[0]} variants x {D.shape[1]} tracks)", flush=True)

if __name__ == "__main__":
    main()
