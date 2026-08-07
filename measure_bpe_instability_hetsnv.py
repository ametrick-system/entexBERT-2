#!/usr/bin/env python3
"""
measure_bpe_instability_hetsnv.py — BPE tokenization instability on the FULL EN-TEx hetSNV set.

For each UNIQUE hetSNV locus (chr, ref_start, ref_allele, hap1_allele, hap2_allele), build the
hap1 and hap2 257bp windows exactly as score_hetsnv.py does (center base <- hap allele) and
tokenize both. The twin Delta = head(hap2)-head(hap1), so instability is measured hap1-vs-hap2:
  1. tok_differs      : do the token-ID sequences differ at all?
  2. dlen             : len(tokens_hap2) - len(tokens_hap1)   (nonzero => misaligned twin)
  3. disruption_span  : bp from the SNV to the FARTHEST changed token boundary
  4. center_tok_len   : length (bp) of the hap1 token containing the SNV

No subsampling. Deduplicated to unique loci (tokenization is a pure function of the sequence);
per-locus label = 1 if the site is imbalance_significant in ANY (donor,tissue) measurement.

Run in eb2 from repo root:
  python measure_bpe_instability_hetsnv.py \
      --hetsnv_tsv /home/asm242/entex_data/hetSNVs.tsv \
      --ref_fasta  /home/asm242/reference_genome/hg38.fa \
      --model_dir  /home/asm242/entexBERT-2/DNABERT-2-117M-attention \
      --assay ALL --min_total_reads 0 --left_bp 128 --right_bp 128 --out bpe_hetsnv
"""
import argparse
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

_BASECOL = {"A": "cA", "C": "cC", "G": "cG", "T": "cT"}

def changed_span_bp(off_ref, off_alt, center):
    b_ref = set()
    for s, e in off_ref: b_ref.add(s); b_ref.add(e)
    b_alt = set()
    for s, e in off_alt: b_alt.add(s); b_alt.add(e)
    diff = b_ref.symmetric_difference(b_alt)
    return 0.0 if not diff else float(max(abs(p - center) for p in diff))

def center_token_len(off_ref, center):
    for s, e in off_ref:
        if s <= center < e:
            return e - s
    return 0

def load_unique_loci(path, assay, min_total_reads):
    """Load hetSNV TSV (score_hetsnv.py schema), derive counts+label, dedup to unique loci."""
    usecols = ["chr", "ref_start", "ref_allele", "hap1_allele", "hap2_allele",
               "assay", "cA", "cC", "cG", "cT", "imbalance_significance"]
    df = pd.read_csv(path, sep="\t", usecols=lambda c: c in usecols)
    n_rows = len(df)
    if assay and assay.upper() != "ALL":
        df = df[df["assay"].astype(str).str.contains(assay, case=False, na=False)]
    df = df.reset_index(drop=True)
    if min_total_reads:
        def bc(row, col):
            c = _BASECOL.get(str(row[col]).upper())
            return float(row[c]) if c in row and pd.notna(row[c]) else 0.0
        tot = df.apply(lambda r: bc(r, "hap1_allele") + bc(r, "hap2_allele"), axis=1)
        df = df[tot >= min_total_reads].reset_index(drop=True)
    df["label"] = df["imbalance_significance"].astype(int)
    # dedup to unique locus; label = any-significant across measurements
    key = ["chr", "ref_start", "ref_allele", "hap1_allele", "hap2_allele"]
    uni = (df.groupby(key, observed=True)["label"].max().reset_index())
    print(f"[load] {n_rows} rows (assay={assay}, min_total_reads={min_total_reads}) "
          f"-> {len(uni)} unique loci; pos(any-sig)={int(uni.label.sum())}")
    return uni

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hetsnv_tsv", required=True)
    ap.add_argument("--ref_fasta", required=True)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--assay", default="ALL")
    ap.add_argument("--min_total_reads", type=int, default=0)
    ap.add_argument("--left_bp", type=int, default=128)
    ap.add_argument("--right_bp", type=int, default=128)
    ap.add_argument("--out", default="bpe_hetsnv")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    from pyfaidx import Fasta
    tok = AutoTokenizer.from_pretrained(a.model_dir, trust_remote_code=True)
    fa = Fasta(a.ref_fasta, sequence_always_upper=True)

    def _supports_offsets():
        try:
            t = tok("ACGTACGT", return_offsets_mapping=True, add_special_tokens=False)
            return "offset_mapping" in t and len(t["offset_mapping"]) > 0
        except Exception:
            return False
    USE_OFFSETS = _supports_offsets()
    def tokenize_batch(seqs):
        """Batch-tokenize a list of sequences -> list of (input_ids, [(start,end),...])."""
        if USE_OFFSETS:
            e = tok(list(seqs), return_offsets_mapping=True, add_special_tokens=False)
            return [(ids, [(a_, b_) for (a_, b_) in offs if b_ > a_])
                    for ids, offs in zip(e["input_ids"], e["offset_mapping"])]
        enc = tok(list(seqs), add_special_tokens=False)["input_ids"]
        out = []
        for ids in enc:
            offs, cur = [], 0
            for p in tok.convert_ids_to_tokens(ids):
                p = p.replace("##", ""); offs.append((cur, cur + len(p))); cur += len(p)
            out.append((ids, offs))
        return out
    print(f"[tokenizer] offset_mapping supported: {USE_OFFSETS}"
          + ("" if USE_OFFSETS else "  (string-reconstruction fallback)"))

    uni = load_unique_loci(a.hetsnv_tsv, a.assay, a.min_total_reads)
    W = a.left_bp + 1 + a.right_bp
    c = a.left_bp
    BATCH = 4096

    # Pass 1: build all hap1/hap2 windows (cheap FASTA slicing), keep aligned metadata.
    meta, h1_seqs, h2_seqs = [], [], []
    n_oob = n_badchrom = 0
    for chrom, p0, ref_a, h1, h2, label in zip(
        uni["chr"], uni["ref_start"], uni["ref_allele"],
        uni["hap1_allele"], uni["hap2_allele"], uni["label"]
    ):
        chrom = str(chrom)
        if chrom not in fa: n_badchrom += 1; continue
        p0 = int(p0); start = p0 - a.left_bp; end = p0 + a.right_bp + 1
        if start < 0 or end > len(fa[chrom]): n_oob += 1; continue
        seq = str(fa[chrom][start:end])
        if len(seq) != W: n_oob += 1; continue
        h1_seqs.append(seq[:c] + str(h1).upper() + seq[c+1:])
        h2_seqs.append(seq[:c] + str(h2).upper() + seq[c+1:])
        meta.append((chrom, p0, int(label)))
    print(f"[windows] built {len(meta)} loci (dropped {n_oob} oob, {n_badchrom} bad-chrom)")

    # Pass 2: batch-tokenize hap1 and hap2 windows, compute the four metrics.
    rows = []
    n = len(meta)
    for i in range(0, n, BATCH):
        j = min(i + BATCH, n)
        t1 = tokenize_batch(h1_seqs[i:j])
        t2 = tokenize_batch(h2_seqs[i:j])
        for k in range(j - i):
            ids1, off1 = t1[k]; ids2, off2 = t2[k]
            chrom, p0, label = meta[i + k]
            rows.append(dict(
                chr=chrom, ref_start=p0, label=label,
                tok_differs=int(ids1 != ids2),
                dlen=len(ids2) - len(ids1),
                disruption_span=changed_span_bp(off1, off2, c),
                center_tok_len=center_token_len(off1, c),
            ))
        if (i // BATCH) % 20 == 0:
            print(f"[tokenize] {j}/{n} loci", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(f"{a.out}_perVariant.csv.gz", index=False, compression="gzip")

    def summarize(sub, name):
        return (f"{name:9s} n={len(sub):7d}  tok_differs={sub.tok_differs.mean():.3f}  "
                f"dlen!=0={(sub.dlen!=0).mean():.3f}  "
                f"span med={sub.disruption_span.median():.1f} p90={sub.disruption_span.quantile(.9):.1f} "
                f"max={sub.disruption_span.max():.1f}  ctr_tok med={sub.center_tok_len.median():.1f}")
    print(summarize(d, "ALL"))
    print(summarize(d[d.label==1], "positive"))
    print(summarize(d[d.label==0], "negative"))

    POS, NEG = "#c0392b", "#8fb0d0"
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(8.6, 3.8), dpi=300)
    dpos, dneg = d[d.label==1], d[d.label==0]
    hi = float(d.disruption_span.quantile(0.99)) or 1.0
    bins = np.linspace(0, max(hi, 1.0), 40)
    axA.hist(dneg.disruption_span, bins=bins, density=True, color=NEG, alpha=0.7,
             label=f"ASB neg (n={len(dneg)})")
    axA.hist(dpos.disruption_span, bins=bins, density=True, histtype="step", lw=1.6,
             color=POS, label=f"ASB pos (n={len(dpos)})")
    axA.axvline(a.left_bp, color="#666", lw=0.8, ls=":")
    axA.set_xlabel("disruption span: bp from SNV to farthest changed token boundary", fontsize=7.5)
    axA.set_ylabel("density", fontsize=8)
    axA.set_title("A  BPE disruption span (hap1 vs hap2, EN-TEx hetSNVs)", fontsize=8, loc="left")
    axA.legend(fontsize=6.5, frameon=False); axA.tick_params(labelsize=6)
    for s in ("top","right"): axA.spines[s].set_visible(False)

    m = int(min(6, d.dlen.abs().max()))
    xs = np.arange(0, m+1)
    wpos = [ (dpos.dlen.abs()==k).mean() for k in xs ]
    wneg = [ (dneg.dlen.abs()==k).mean() for k in xs ]
    w = 0.38
    axB.bar(xs-w/2, wneg, w, color=NEG, label="ASB neg")
    axB.bar(xs+w/2, wpos, w, color=POS, label="ASB pos")
    axB.set_xlabel("|token-count change|  (|len(hap2)-len(hap1)|)", fontsize=7.5)
    axB.set_ylabel("fraction of loci", fontsize=8)
    axB.set_title(f"B  token-count change  (dlen!=0: {(d.dlen!=0).mean():.0%} of loci)",
                  fontsize=8, loc="left")
    axB.set_xticks(xs); axB.legend(fontsize=6.5, frameon=False); axB.tick_params(labelsize=6)
    for s in ("top","right"): axB.spines[s].set_visible(False)

    fig.tight_layout()
    fig.savefig(f"{a.out}.png", bbox_inches="tight")
    print(f"[wrote] {a.out}.png and {a.out}_perVariant.csv.gz")

if __name__ == "__main__":
    main()
