#!/usr/bin/env python3
"""
make_task_manifest.py — turn a generate_all_inputs batch manifest into a ref_single task list
(one entry per dataset x LUPI-arm). Two output formats:

  --format tsv (default): TAB-separated lines run_ref_single.sbatch reads, to STDOUT (or --out):
      <run_tag>\t<data_dir>\t<aux_arm>
  --format dsq: a dead-Simple-Queue job file, ONE self-contained command per line (needs --out):
      bash run_ref_single_job.sh <run_tag> <data_dir> <aux_arm>
      Submit with:  module load dSQ && dsq --job-file <out> --partition gpu --account <acct> \
                        --gpus 1 --cpus-per-task 4 --mem 48G --time 12:00:00 --output logs/%A_%a.out
      Re-run only failures later:  dsqa -j <arrayjobid> > rerun_ref_single.txt   (dSQAutopsy)

  run_tag  = <dataset_id>__<arm>          (unique per run)
  data_dir = that dataset's fold output dir (from the batch manifest)
  aux_arm  = baseline | lupi

Usage:
  # classic array path (unchanged):
  python make_task_manifest.py inputs/<run>/generate_all_inputs_manifest.json \
      --arms baseline lupi > tasks_ref_single.tsv
  # dSQ path:
  python make_task_manifest.py inputs/<run>/generate_all_inputs_manifest.json \
      --arms baseline lupi --format dsq --out jobs_ref_single.txt
"""
import argparse, json, sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest", help="generate_all_inputs_manifest.json")
    ap.add_argument("--arms", nargs="+", default=["baseline", "lupi"],
                    choices=["baseline", "lupi"], help="which LUPI arms to emit per dataset")
    ap.add_argument("--only_ok", action="store_true",
                    help="only datasets whose generate status was 'ok' or 'skipped'")
    ap.add_argument("--format", choices=["tsv", "dsq"], default="tsv",
                    help="tsv = TAB lines for run_ref_single.sbatch (stdout unless --out); "
                         "dsq = job file, one command per line (requires --out)")
    ap.add_argument("--runner", default="run_ref_single_job.sh",
                    help="dsq only: per-job runner script referenced in each command line")
    ap.add_argument("--out", default=None, help="write here instead of stdout (required for --format dsq)")
    args = ap.parse_args()

    if args.format == "dsq" and not args.out:
        ap.error("--format dsq requires --out (a job file path)")

    m = json.load(open(args.manifest))
    lines = []
    for dataset in m["datasets"]:
        if args.only_ok and dataset["status"] not in ("ok", "skipped"):
            continue
        for arm in args.arms:
            run_tag = f"{dataset['dataset_id']}__{arm}"
            if args.format == "tsv":
                lines.append(f"{run_tag}\t{dataset['output_dir']}\t{arm}")
            else:
                lines.append(f"bash {args.runner} {run_tag} {dataset['output_dir']} {arm}")

    out = open(args.out, "w") if args.out else sys.stdout
    for ln in lines:
        out.write(ln + "\n")
    if args.out:
        out.close()

    kind = "task line(s)" if args.format == "tsv" else "dsq job(s)"
    print(f"{len(lines)} {kind} written; "
          + (f"set --array=0-{len(lines)-1}" if args.format == "tsv"
             else f"submit with dsq --job-file {args.out}"), file=sys.stderr)

if __name__ == "__main__":
    main()
