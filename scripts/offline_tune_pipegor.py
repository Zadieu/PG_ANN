#!/usr/bin/env python3
import argparse
import csv
import itertools
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_GORGEOUS_EXE = "/home/dell/projects/Gorgeous-baseline/build/tests/search_disk_index"
DEFAULT_PIPEGOR_EXE = "/home/dell/projects/PipeGor_ANN/build/tests/search_disk_index"
DEFAULT_RESULT_ROOT = "/home/dell/projects/pipegor_offline_tuning"
DEFAULT_GRAPH_REP_COMPAT = "/home/dell/data/gorgeous/sift1M/M4_R64_L128/GRAPH_CACHE_INDEX/"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Offline profile PipeGorANN pipeline knobs against a pipeline-off baseline. "
            "The default baseline is PipeGorANN itself with pipeline disabled."
        )
    )
    parser.add_argument("--baseline-kind", choices=["pipegor", "gorgeous"], default="pipegor")
    parser.add_argument("--baseline-exe", default=None,
                        help="Override baseline binary. Defaults depend on --baseline-kind.")
    parser.add_argument("--pipegor-exe", default=DEFAULT_PIPEGOR_EXE)
    parser.add_argument("--graph-rep-index-prefix", default=DEFAULT_GRAPH_REP_COMPAT)
    parser.add_argument("--result-root", default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--threads", nargs="+", type=int, default=[8, 16, 32, 64])
    parser.add_argument("--Ls", nargs="+", type=int, default=[15, 20, 25, 30, 35, 40])
    parser.add_argument("--limit", type=int, default=12,
                        help="Max pipeline knob configs per T/L. Use 0 for full grid.")
    parser.add_argument("--config-file", default=None,
                        help="JSON/CSV recommendation file. Entries may include T/L.")
    parser.add_argument("--honor-recommended-mode", action="store_true",
                        help="If config entries include recommended_mode=baseline, skip pipeline candidates for that case.")
    parser.add_argument("--repeats", type=int, default=1,
                        help="Repeat every candidate and average metrics.")
    parser.add_argument("--max-recall-drop", type=float, default=1.0)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--pipe-max-values", nargs="+", default=["auto", "4", "6", "8"])
    parser.add_argument("--qd-budget-values", nargs="+", default=["128", "256"])
    parser.add_argument("--pipe-start-values", nargs="+", default=["auto", "4"])
    parser.add_argument("--ramp-step-values", nargs="+", default=["auto", "1", "2"])
    parser.add_argument("--feedback-window-values", nargs="+", default=["4"])
    parser.add_argument("--min-marker-values", nargs="+", default=["2"])
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolved_baseline_exe(args):
    if args.baseline_exe:
        return args.baseline_exe
    if args.baseline_kind == "gorgeous":
        return DEFAULT_GORGEOUS_EXE
    return args.pipegor_exe


def common_args(thread_count, l_value):
    return [
        "--data_type", "float",
        "--dist_fn", "l2",
        "--index_path_prefix", "/home/dell/data/gorgeous/sift1M/M4_R64_L128/",
        "--pq_path_prefix", "/home/dell/data/gorgeous/sift1M/PQ/C4/",
        "--query_file", "/home/dell/data/sift/sift_query.fbin",
        "--gt_file", "/home/dell/data/sift/computed_gt_1000_sift1m.bin",
        "-K", "10",
        "-L", str(l_value),
        "-T", str(thread_count),
        "-W", "8",
        "--mem_L", "10",
        "--sector_len", "4096",
        "--mem_index_path", "/home/dell/data/gorgeous/sift1M/M4_R64_L128/_mem.index",
        "--mem_sample_path", "/home/dell/data/gorgeous/sift1M/M4_R64_L128/_sample_data.bin",
        "--use_page_search", "1",
        "--use_ratio", "0.3",
        "--pq_ratio", "1",
        "--disk_file_path", "/home/dell/data/gorgeous/sift1M/M4_R64_L128/_disk.index",
        "--disk_graph_prefix", "/home/dell/data/gorgeous/sift1M/M4_R64_L128/GRAPH/",
        "--deco_impl", "1",
        "--use_graph_rep_index", "0",
        "--mem_graph_use_ratio", "0.1",
        "--mem_emb_use_ratio", "0.0",
        "--emb_search_ratio", "0.4",
    ]


def parse_metrics(log_path):
    metrics = None
    for line in Path(log_path).read_text(errors="ignore").splitlines():
        parts = line.split()
        if len(parts) >= 18 and parts[0].isdigit():
            metrics = {
                "L": int(parts[0]),
                "BW": int(parts[1]),
                "qps": float(parts[2]),
                "mean_latency": float(parts[3]),
                "p999_latency": float(parts[4]),
                "graph_io": float(parts[5]),
                "emb_io": float(parts[6]),
                "recall": float(parts[17]),
                "pipe_sub": float(parts[18]) if len(parts) > 18 else 0.0,
                "pipe_use_pct": float(parts[19]) if len(parts) > 19 else 0.0,
                "pipe_wst": float(parts[20]) if len(parts) > 20 else 0.0,
                "pipe_stl": float(parts[21]) if len(parts) > 21 else 0.0,
                "pipe_w": float(parts[22]) if len(parts) > 22 else 0.0,
                "pipe_max": float(parts[23]) if len(parts) > 23 else 0.0,
            }
    if metrics is None:
        raise RuntimeError(f"No result row found in {log_path}")
    return metrics


def run_and_parse(cmd, env, log_path, timeout, dry_run=False):
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        Path(log_path).write_text("DRY RUN\n" + " ".join(cmd) + "\n")
        return None
    with open(log_path, "w") as log:
        completed = subprocess.run(
            cmd,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with code {completed.returncode}; see {log_path}")
    return parse_metrics(log_path)


def run_repeated(cmd, env, case_dir, log_stem, repeats, timeout, dry_run=False):
    metrics_list = []
    log_paths = []
    for repeat_id in range(repeats):
        suffix = f"_r{repeat_id}" if repeats > 1 else ""
        log_path = case_dir / f"{log_stem}{suffix}.log"
        metrics = run_and_parse(cmd, env, log_path, timeout, dry_run)
        log_paths.append(str(log_path))
        if metrics is not None:
            metrics_list.append(metrics)
    if dry_run:
        return None, log_paths
    return aggregate_metrics(metrics_list), log_paths


def aggregate_metrics(metrics_list):
    if not metrics_list:
        raise RuntimeError("No metrics to aggregate")
    out = {}
    numeric_keys = metrics_list[0].keys()
    for key in numeric_keys:
        values = [m[key] for m in metrics_list]
        if key in {"L", "BW"}:
            out[key] = values[0]
        else:
            out[key] = statistics.mean(values)
    return out


def baseline_env():
    env = os.environ.copy()
    env.update({
        "GORGEOUS_DYNAMIC_GRAPH_CACHE_RATIO": "0",
        "GORGEOUS_PIPEANN_STATE_MACHINE": "0",
        "GORGEOUS_PIPEANN_STATE_SCHEDULER": "0",
        "GORGEOUS_PIPEANN_DYNAMIC_PIPE_WIDTH": "0",
        "GORGEOUS_PIPEANN_RESOURCE_ADAPTIVE": "0",
        "GORGEOUS_PIPELINED_GRAPH_IO": "0",
        "GORGEOUS_PIPELINED_GRAPH_DYNAMIC_WIDTH": "0",
        "GORGEOUS_PIPELINED_REFINE_IO": "0",
        "GORGEOUS_EARLY_REFINE_PREFETCH": "0",
    })
    return env


def pipegor_env(knob):
    env = baseline_env()
    env.update({
        "GORGEOUS_PIPELINED_GRAPH_IO": "1",
        "GORGEOUS_PIPELINED_GRAPH_DYNAMIC_WIDTH": "1",
        "GORGEOUS_PIPELINED_GRAPH_PIPE_MAX": knob["pipe_max"],
        "GORGEOUS_PIPELINED_GRAPH_QD_BUDGET": knob["qd_budget"],
        "GORGEOUS_PIPELINED_GRAPH_RAMP_STEP": knob["ramp_step"],
        "GORGEOUS_PIPEANN_PIPE_START": knob["pipe_start"],
        "GORGEOUS_PIPEANN_PIPE_MIN": "1",
        "GORGEOUS_PIPEANN_L_AWARE_PIPE_START": "1",
        "GORGEOUS_PIPEANN_L_AWARE_LOW_L": "12",
        "GORGEOUS_PIPEANN_L_AWARE_HIGH_L": "18",
        "GORGEOUS_PIPEANN_PIPE_FEEDBACK": "1",
        "GORGEOUS_PIPEANN_PIPE_FEEDBACK_WINDOW": knob["feedback_window"],
        "GORGEOUS_PIPEANN_PIPE_MIN_MARKER": knob["min_marker"],
        "GORGEOUS_PIPEANN_PIPE_WASTE_THRESHOLD": "0.10",
        "GORGEOUS_PIPEANN_PIPE_DOWN_WASTE_THRESHOLD": "0.35",
    })
    return env


def knob_grid(args):
    configs = []
    # Keep high-impact knobs innermost so limited searches cover widths and QD budgets first.
    product = itertools.product(
        args.pipe_start_values,
        args.ramp_step_values,
        args.feedback_window_values,
        args.min_marker_values,
        args.pipe_max_values,
        args.qd_budget_values,
    )
    for pipe_start, ramp_step, feedback_window, min_marker, pipe_max, qd_budget in product:
        configs.append({
            "pipe_max": pipe_max,
            "qd_budget": qd_budget,
            "pipe_start": pipe_start,
            "ramp_step": ramp_step,
            "feedback_window": feedback_window,
            "min_marker": min_marker,
        })
    if args.limit > 0:
        return configs[:args.limit]
    return configs

def load_config_file(path):
    config_path = Path(path)
    if config_path.suffix.lower() == ".json":
        payload = json.loads(config_path.read_text())
        entries = payload.get("recommendations", payload) if isinstance(payload, dict) else payload
    else:
        with config_path.open(newline="") as f:
            entries = list(csv.DictReader(f))

    configs = []
    for entry in entries:
        config = {
            "pipe_max": str(entry.get("pipe_max", entry.get("pipe_max_knob", "auto"))),
            "qd_budget": str(entry.get("qd_budget", "256")),
            "pipe_start": str(entry.get("pipe_start", "auto")),
            "ramp_step": str(entry.get("ramp_step", "auto")),
            "feedback_window": str(entry.get("feedback_window", "4")),
            "min_marker": str(entry.get("min_marker", "2")),
        }
        if entry.get("T") not in (None, ""):
            config["T"] = int(entry["T"])
        if entry.get("L") not in (None, ""):
            config["L"] = int(entry["L"])
        if entry.get("recommended_mode") not in (None, ""):
            config["recommended_mode"] = str(entry["recommended_mode"])
        configs.append(config)
    return configs


def configs_for_case(configs, thread_count, l_value, honor_recommended_mode=False):
    selected = []
    seen = set()
    for config in configs:
        if "T" in config and config["T"] != thread_count:
            continue
        if "L" in config and config["L"] != l_value:
            continue
        if honor_recommended_mode and config.get("recommended_mode") == "baseline":
            continue
        knob = {
            "pipe_max": config["pipe_max"],
            "qd_budget": config["qd_budget"],
            "pipe_start": config["pipe_start"],
            "ramp_step": config["ramp_step"],
            "feedback_window": config["feedback_window"],
            "min_marker": config["min_marker"],
        }
        key = (
            knob["pipe_max"], knob["qd_budget"], knob["pipe_start"],
            knob["ramp_step"], knob["feedback_window"], knob["min_marker"],
        )
        if key in seen:
            continue
        seen.add(key)
        selected.append(knob)
    return selected


def choose_best(rows, max_recall_drop):
    valid = [r for r in rows if r["recall_drop"] <= max_recall_drop]
    if valid:
        return max(valid, key=lambda r: (r["qps"], -r["mean_latency"]))
    return max(rows, key=lambda r: (r["recall"], r["qps"]))


def make_baseline_row(thread_count, l_value, baseline, baseline_log_paths):
    return {
        "T": thread_count,
        "L": l_value,
        "config_id": "baseline",
        "mode": "baseline",
        "valid": True,
        "qps": baseline["qps"],
        "mean_latency": baseline["mean_latency"],
        "p999_latency": baseline["p999_latency"],
        "recall": baseline["recall"],
        "baseline_qps": baseline["qps"],
        "baseline_recall": baseline["recall"],
        "recall_drop": 0.0,
        "qps_gain_pct": 0.0,
        "graph_io": baseline["graph_io"],
        "pipe_sub": 0.0,
        "pipe_w": 0.0,
        "pipe_max": 0.0,
        "pipe_max_knob": "baseline",
        "qd_budget": "",
        "pipe_start": "",
        "ramp_step": "",
        "feedback_window": "",
        "min_marker": "",
        "log_path": ";".join(baseline_log_paths),
    }


def make_pipeline_row(thread_count, l_value, config_id, knob, baseline, metrics, log_paths, max_recall_drop):
    recall_drop = baseline["recall"] - metrics["recall"]
    qps_gain_pct = 0.0
    if baseline["qps"] != 0:
        qps_gain_pct = 100.0 * (metrics["qps"] - baseline["qps"]) / baseline["qps"]
    return {
        "T": thread_count,
        "L": l_value,
        "config_id": config_id,
        "mode": "pipegor_pipeline",
        "valid": recall_drop <= max_recall_drop,
        "qps": metrics["qps"],
        "mean_latency": metrics["mean_latency"],
        "p999_latency": metrics["p999_latency"],
        "recall": metrics["recall"],
        "baseline_qps": baseline["qps"],
        "baseline_recall": baseline["recall"],
        "recall_drop": recall_drop,
        "qps_gain_pct": qps_gain_pct,
        "graph_io": metrics["graph_io"],
        "pipe_sub": metrics["pipe_sub"],
        "pipe_w": metrics["pipe_w"],
        "pipe_max": metrics["pipe_max"],
        "pipe_max_knob": knob["pipe_max"],
        "qd_budget": knob["qd_budget"],
        "pipe_start": knob["pipe_start"],
        "ramp_step": knob["ramp_step"],
        "feedback_window": knob["feedback_window"],
        "min_marker": knob["min_marker"],
        "log_path": ";".join(log_paths),
    }


def profile_entry(best):
    return {
        "config_id": best["config_id"],
        "mode": best["mode"],
        "valid": best["valid"],
        "knobs": {
            "pipe_max": best["pipe_max_knob"],
            "qd_budget": best["qd_budget"],
            "pipe_start": best["pipe_start"],
            "ramp_step": best["ramp_step"],
            "feedback_window": best["feedback_window"],
            "min_marker": best["min_marker"],
        },
        "metrics": {
            "qps": best["qps"],
            "mean_latency": best["mean_latency"],
            "p999_latency": best["p999_latency"],
            "recall": best["recall"],
            "recall_drop": best["recall_drop"],
            "baseline_qps": best["baseline_qps"],
            "baseline_recall": best["baseline_recall"],
            "qps_gain_pct": best["qps_gain_pct"],
        },
    }


def write_profile_outputs(profile, profile_path, summary_path):
    profile_path.write_text(json.dumps(profile, indent=2))
    fields = [
        "T", "L", "mode", "config_id", "qps", "baseline_qps", "qps_gain_pct",
        "recall", "baseline_recall", "recall_drop",
        "pipe_max", "qd_budget", "pipe_start", "ramp_step",
    ]
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for key, value in profile.items():
            t_value, l_value = key[1:].split("_L")
            metrics = value["metrics"]
            knobs = value["knobs"]
            writer.writerow({
                "T": t_value,
                "L": l_value,
                "mode": value["mode"],
                "config_id": value["config_id"],
                "qps": f"{metrics['qps']:.2f}",
                "baseline_qps": f"{metrics['baseline_qps']:.2f}",
                "qps_gain_pct": f"{metrics['qps_gain_pct']:.2f}",
                "recall": f"{metrics['recall']:.2f}",
                "baseline_recall": f"{metrics['baseline_recall']:.2f}",
                "recall_drop": f"{metrics['recall_drop']:.2f}",
                "pipe_max": knobs["pipe_max"],
                "qd_budget": knobs["qd_budget"],
                "pipe_start": knobs["pipe_start"],
                "ramp_step": knobs["ramp_step"],
            })


def write_run_config(args, run_dir, configs, baseline_exe):
    payload = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "baseline_kind": args.baseline_kind,
        "baseline_exe": baseline_exe,
        "pipegor_exe": args.pipegor_exe,
        "result_root": args.result_root,
        "run_dir": str(run_dir),
        "threads": args.threads,
        "Ls": args.Ls,
        "limit": args.limit,
        "repeats": args.repeats,
        "config_file": args.config_file,
        "honor_recommended_mode": args.honor_recommended_mode,
        "max_recall_drop": args.max_recall_drop,
        "timeout": args.timeout,
        "config_count": len(configs),
        "knob_grid": configs,
    }
    (run_dir / "run_config.json").write_text(json.dumps(payload, indent=2))


def main():
    args = parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.result_root) / f"run_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    baseline_exe = resolved_baseline_exe(args)
    configs = load_config_file(args.config_file) if args.config_file else knob_grid(args)
    record_path = run_dir / "record.csv"
    profile_path = run_dir / "best_profile.json"
    summary_path = run_dir / "best_summary.csv"
    profile = {}

    write_run_config(args, run_dir, configs, baseline_exe)

    fields = [
        "T", "L", "config_id", "mode", "valid", "qps", "mean_latency", "p999_latency",
        "recall", "baseline_qps", "baseline_recall", "recall_drop", "qps_gain_pct",
        "graph_io", "pipe_sub", "pipe_w", "pipe_max",
        "pipe_max_knob", "qd_budget", "pipe_start", "ramp_step",
        "feedback_window", "min_marker", "log_path",
    ]

    with record_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for thread_count in args.threads:
            for l_value in args.Ls:
                case_dir = run_dir / f"T{thread_count}_L{l_value}"
                base_cmd = [
                    baseline_exe,
                    *common_args(thread_count, l_value),
                    "--graph_rep_index_prefix",
                    args.graph_rep_index_prefix,
                    "--result_path",
                    str(case_dir / "baseline_result"),
                ]
                print(f"[BASE] T={thread_count} L={l_value} kind={args.baseline_kind}", flush=True)
                baseline, baseline_logs = run_repeated(
                    base_cmd, baseline_env(), case_dir, "baseline",
                    args.repeats, args.timeout, args.dry_run
                )
                if args.dry_run:
                    continue

                rows = [make_baseline_row(thread_count, l_value, baseline, baseline_logs)]
                writer.writerow(rows[0])
                f.flush()

                case_configs = configs_for_case(
                    configs, thread_count, l_value, args.honor_recommended_mode
                )
                for config_id, knob in enumerate(case_configs):
                    log_stem = f"pipegor_config_{config_id}"
                    cmd = [
                        args.pipegor_exe,
                        *common_args(thread_count, l_value),
                        "--result_path",
                        str(case_dir / f"pipegor_config_{config_id}_result"),
                    ]
                    print(f"[RUN] T={thread_count} L={l_value} cfg={config_id} {knob}", flush=True)
                    metrics, log_paths = run_repeated(
                        cmd, pipegor_env(knob), case_dir, log_stem,
                        args.repeats, args.timeout, args.dry_run
                    )
                    row = make_pipeline_row(
                        thread_count, l_value, config_id, knob, baseline,
                        metrics, log_paths, args.max_recall_drop
                    )
                    rows.append(row)
                    writer.writerow(row)
                    f.flush()

                best = choose_best(rows, args.max_recall_drop)
                profile[f"T{thread_count}_L{l_value}"] = profile_entry(best)
                write_profile_outputs(profile, profile_path, summary_path)
                print(
                    f"[BEST] T={thread_count} L={l_value}: cfg={best['config_id']} "
                    f"qps={best['qps']:.2f} recall={best['recall']:.2f} "
                    f"drop={best['recall_drop']:.2f} gain={best['qps_gain_pct']:.2f}% "
                    f"valid={best['valid']} mode={best['mode']}",
                    flush=True,
                )

    print(f"RUN_DIR={run_dir}")
    print(f"RECORD={record_path}")
    print(f"PROFILE={profile_path}")
    print(f"SUMMARY={summary_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise
