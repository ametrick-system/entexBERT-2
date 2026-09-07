#!/usr/bin/env python
"""
personal_seq.py -- lift hg38 reference loci into an individual's phased haplotype genomes and
extract PERSONAL windows (which carry all in-cis variants) for the entexBERT-2 personal arm.

Two vcf2diploid liftOver chains per individual map the REFERENCE to each haplotype:
    ENC00X_maternal.chain   REF -> maternal    (hap1)
    ENC00X_paternal.chain   REF -> paternal    (hap2)

Personal-genome README (MJ 12/4/2024): hap1 = maternal, hap2 = paternal for all chromosomes,
with per-sample sex-chromosome exceptions (e.g. chrX in a male may come from paternal). We do NOT
special-case those exceptions: the hap1<->hap2 assignment is derived PER LOCUS by allele-match
(the two haplotypes of a heterozygous SNV carry the two different alleles), which auto-corrects any
swap and simultaneously validates the coordinate math.

For a het SNV at REF (chrom, pos0) [0-based] with alleles a1 (hap1_allele) / a2 (hap2_allele):
  1. lift pos0 through each chain -> maternal coord / paternal coord   (None if deleted there)
  2. extract a FIXED-length window (+/- flank bp) from each haplotype FASTA, re-centered on the
     lifted variant -> both windows are (left+right+1) bp with the variant at offset `left`.
     Indels within the window shift how much REF span it covers but NOT its length, so the
     downstream equal-length / shared-anchor-offset invariant in add_sequence_inputs is preserved.
  3. read the focal base in each personal window; ASSIGN maternal/paternal -> hap1/hap2 by matching
     the focal bases to (a1, a2); flag mismatches (the 'personal allele mismatch' guard, mirroring
     add_sequence_inputs' reference-allele check).

UCSC chain format (0-based, half-open; target = REF, query = haplotype):
    chain score tName tSize tStrand tStart tEnd qName qSize qStrand qStart qEnd id
    size dt dq        # aligned block `size`, then target-gap dt and query-gap dq before next block
    ...
    size              # final block (no trailing gaps)
    <blank line>
vcf2diploid emits +/+ strands; we assert that rather than silently mis-lift a '-' query.

NOTE on reference identity: the chain target sizes here are NOT stock hg38 (e.g. chr1 target =
248,387,328 vs hg38 248,956,422). The lift is only valid if the hetSNV calls and vcf2diploid used
the SAME reference. `PersonalGenome.stats` reports the allele-match rate so a real run empirically
confirms this -- a low rate means the references differ and must be reconciled first.
"""
from collections import defaultdict

# ---------------------------------------------------------------------------------------------
# Chain parsing + point lift
# ---------------------------------------------------------------------------------------------
class _Chain:
    """One `chain` record: aligned blocks mapping target [tStart,tEnd) -> query [qStart,qEnd)."""
    __slots__ = ("t_name", "t_size", "t_start", "t_end", "q_name", "q_size",
                 "q_strand", "q_start", "q_end", "blocks")

    def __init__(self, header_fields):
        # header_fields = tokens AFTER the literal 'chain' keyword
        (_score, self.t_name, self.t_size, t_strand, self.t_start, self.t_end,
         self.q_name, self.q_size, self.q_strand, self.q_start, self.q_end, *_id) = header_fields
        self.t_size = int(self.t_size); self.t_start = int(self.t_start); self.t_end = int(self.t_end)
        self.q_size = int(self.q_size); self.q_start = int(self.q_start); self.q_end = int(self.q_end)
        if t_strand != "+":
            raise ValueError(f"chain {self.t_name}->{self.q_name}: target strand {t_strand!r} != '+'")
        self.blocks = []   # list of (size, dt, dq)

    def lift(self, tpos):
        """Lift a 0-based TARGET position to a 0-based QUERY position, or None if it falls in a
        target-only gap (i.e. the base is deleted in this haplotype)."""
        if tpos < self.t_start or tpos >= self.t_end:
            return None
        t = self.t_start
        q = self.q_start
        for (size, dt, dq) in self.blocks:
            if tpos < t + size:                      # inside an aligned block -> 1:1
                qpos = q + (tpos - t)
                if self.q_strand == "-":             # not expected here, but handle correctly
                    return self.q_size - qpos - 1
                return qpos
            t += size + dt                           # skip the aligned block + the target gap
            q += size + dq
            if tpos < t and tpos >= t - dt:          # tpos landed in the target-gap dt -> deleted in query
                return None
        return None


def parse_chain_file(path):
    """Parse a (possibly multi-chromosome) UCSC chain file -> {target_chrom: [_Chain, ...]}."""
    index = defaultdict(list)
    cur = None
    with open(path) as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                cur = None
                continue
            if line.startswith("chain"):
                cur = _Chain(line.split()[1:])
                index[cur.t_name].append(cur)
            elif cur is not None:
                parts = line.split()
                if len(parts) == 1:
                    cur.blocks.append((int(parts[0]), 0, 0))       # final block
                else:
                    cur.blocks.append((int(parts[0]), int(parts[1]), int(parts[2])))
    return dict(index)


def lift_point(chain_index, chrom, pos0):
    """Lift 0-based (chrom, pos0) through the first covering chain. -> (q_name, q_pos0) or None."""
    for ch in chain_index.get(chrom, ()):
        q = ch.lift(pos0)
        if q is not None:
            return (ch.q_name, q)
    return None


# ---------------------------------------------------------------------------------------------
# Haplotype window extraction (FASTA-agnostic: anything supporting fa[chrom][a:b] -> str)
# ---------------------------------------------------------------------------------------------
def _resolve_chrom(fasta, q_name):
    """The chain query is e.g. 'chr1_maternal'; the haplotype FASTA may use that or a renamed
    'chr1'. Return the key present in the FASTA, or None."""
    keys = set(getattr(fasta, "keys", lambda: [])())
    if q_name in keys:
        return q_name
    for suf in ("_maternal", "_paternal", "_hap1", "_hap2"):
        if q_name.endswith(suf) and q_name[: -len(suf)] in keys:
            return q_name[: -len(suf)]
    return None


def extract_window(fasta, q_name, q_pos0, left, right):
    """Fixed-length window [q_pos0-left, q_pos0+right] (length left+right+1), variant at offset
    `left`. Returns (seq_upper, offset) or None if it resolves no chrom / runs off the end."""
    chrom = _resolve_chrom(fasta, q_name)
    if chrom is None:
        return None
    start = q_pos0 - left
    end = q_pos0 + right + 1
    if start < 0:
        return None
    seq = str(fasta[chrom][start:end]).upper()
    if len(seq) != left + right + 1:                 # ran off the chromosome end
        return None
    return seq, left


# ---------------------------------------------------------------------------------------------
# PersonalGenome: bind (hap1 chain+FASTA, hap2 chain+FASTA) and build validated allele-matched pairs
# ---------------------------------------------------------------------------------------------
class PersonalGenome:
    """
    hap1 = (maternal chain, hap1 FASTA), hap2 = (paternal chain, hap2 FASTA) per the README default.
    pair_windows() lifts a REF het SNV into both haplotypes, extracts variant-centered windows, and
    assigns them to hap1/hap2 by ALLELE-MATCH (so a sex-chrom mat/pat swap is auto-corrected).
    """
    def __init__(self, hap1_chain_index, hap1_fasta, hap2_chain_index, hap2_fasta, donor=None):
        self.h1_chain, self.h1_fa = hap1_chain_index, hap1_fasta
        self.h2_chain, self.h2_fa = hap2_chain_index, hap2_fasta
        self.donor = donor
        self.stats = defaultdict(int)   # ok / gap / offchrom / mismatch / swap / total

    def pair_windows(self, chrom, pos0, a1, a2, left=128, right=128):
        """
        chrom, pos0 (0-based) : REF locus.  a1 = hap1_allele, a2 = hap2_allele (single bases).
        Returns dict(seq1, seq2, offset, swapped) on success, or (None, reason) on failure.
        seq1 is the window whose focal base == a1 (hap1), seq2 == a2 (hap2).
        """
        self.stats["total"] += 1
        a1, a2 = str(a1).upper(), str(a2).upper()

        # lift through BOTH chains (maternal / paternal), extract a variant-centered window from each
        def side(chain_index, fasta):
            lifted = lift_point(chain_index, chrom, pos0)
            if lifted is None:
                return None
            q_name, q_pos0 = lifted
            return extract_window(fasta, q_name, q_pos0, left, right)

        m = side(self.h1_chain, self.h1_fa)    # maternal side (nominal hap1)
        p = side(self.h2_chain, self.h2_fa)    # paternal side (nominal hap2)
        if m is None or p is None:
            self.stats["gap_or_offchrom"] += 1
            return None, "variant deleted in a haplotype or window off chromosome end"

        (m_seq, off), (p_seq, _) = m, p
        m_base, p_base = m_seq[off], p_seq[off]

        # assign by allele-match; auto-correct a maternal/paternal <-> hap1/hap2 swap (sex chroms)
        if m_base == a1 and p_base == a2:
            self.stats["ok"] += 1
            return {"seq1": m_seq, "seq2": p_seq, "offset": off, "swapped": False}, None
        if m_base == a2 and p_base == a1:
            self.stats["ok"] += 1; self.stats["swap"] += 1
            return {"seq1": p_seq, "seq2": m_seq, "offset": off, "swapped": True}, None

        self.stats["mismatch"] += 1
        return None, (f"personal allele mismatch at {chrom}:{pos0} "
                      f"(maternal base {m_base!r}, paternal base {p_base!r}; expected {{{a1},{a2}}})")

    def match_rate(self):
        t = self.stats["total"]
        return (self.stats["ok"] / t) if t else float("nan")
