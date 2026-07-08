#!/usr/bin/env python3
"""
make_task_manifest.py — turn a generate_all_inputs batch manifest into a SLURM task manifest
(one line per cell x LUPI-arm) that run_ref_single.sbatch reads.

Each output line is TAB-separated:  <run_tag>\t<data_dir>\t<aux_arm>
  run_tag  = <cell_id>__<arm>          (unique per run)
  data_dir = that cell's fold output dir (from the batch manifest)
  aux_arm  = baseline | lupi

Usage:
  python make_task_manifest.py inputs/<run>/generate_all_inputs_manifest.json \
      --arms baseline lupi > tasks_ref_single.tsv
  # then set --array=0-(N-1) in run_ref_single.sbatch, N = number of lines printed to stderr.
"""
import argparse, json, sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest", help="generate_all_inputs_manifest.json")
    ap.add_argument("--arms", nargs="+", default=["baseline", "lupi"],
                    choices=["baseline", "lupi"], help="which LUPI arms to emit per cell")
    ap.add_argument("--only_ok", action="store_true",
                    help="only cells whose generate status was 'ok' or 'skipped'")
    args = ap.parse_args()

    m = json.load(open(args.manifest))
    n = 0
    for cell in m["cells"]:
        if args.only_ok and cell["status"] not in ("ok", "skipped"):
            continue
        for arm in args.arms:
            print(f"{cell['cell_id']}__{arm}\t{cell['output_dir']}\t{arm}")
            n += 1
    print(f"{n} task line(s) written; set --array=0-{n-1}", file=sys.stderr)

if __name__ == "__main__":
    main()
