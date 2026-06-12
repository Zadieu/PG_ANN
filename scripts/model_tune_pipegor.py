#!/usr/bin/env python3
import argparse
import csv
import itertools
import json
import math
import statistics
import time
from pathlib import Path


DEFAULT_RESULT_ROOT = "/home/dell/projects/pipegor_model_tuning"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "VDTuner-style surrogate recommender for PipeGorANN. "
            "It reads historical record.csv files and recommends unseen pipeline knobs."
        )
    )
    parser.add_argument("--records", nargs="+", required=True,
                        help="One or more offline_tune_pipegor.py record.csv files.")
    parser.add_argument("--result-root", default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--threads", nargs="+", type=int, default=[8, 16, 32, 64])
    parser.add_argument("--Ls", nargs="+", type=int, default=[15, 20, 25, 30, 35, 40])
    parser.add_argument("--top-k", type=int, default=4,
                        help="Recommendations per T/L.")
    parser.add_argument("--neighbors", type=int, default=8)
    parser.add_argument("--max-recall-drop", type=float, default=1.0)
    parser.add_argument("--exploration-weight", type=float, default=0.25,
                        help="Score bonus from neighbor disagreement.")
    parser.add_argument("--pipe-max-values", nargs="+", default=["auto", "4", "6", "8"])
    parser.add_argument("--qd-budget-values", nargs="+", default=["128", "256"])
    parser.add_argument("--pipe-start-values", nargs="+", default=["auto", "4"])
    parser.add_argument("--ramp-step-values", nargs="+", default=["auto", "1", "2"])
    parser.add_argument("--feedback-window-values", nargs="+", default=["4"])
    parser.add_argument("--min-marker-values", nargs="+", default=["2"])
    parser.add_argument("--dataset-name", default="sift1m",
                        help="Metadata label for future multi-dataset records.")
    return parser.parse_args()


def knob_grid(args):
    configs = []
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
    return configs


def parse_float(value, default=0.0):
    if value is None or value == "":
        return default
    return float(value)


def load_records(paths, dataset_name):
    rows = []
    for path in paths:
        record_path = Path(path)
        with record_path.open(newline="") as f:
            for row in csv.DictReader(f):
                if row.get("mode") != "pipegor_pipeline":
                    continue
                sample = {
                    "dataset": dataset_name,
                    "record_path": str(record_path),
                    "T": int(row["T"]),
                    "L": int(row["L"]),
                    "pipe_max": row["pipe_max_knob"],
                    "qd_budget": row["qd_budget"],
                    "pipe_start": row["pipe_start"],
                    "ramp_step": row["ramp_step"],
                    "feedback_window": row["feedback_window"],
                    "min_marker": row["min_marker"],
                    "qps": parse_float(row["qps"]),
                    "baseline_qps": parse_float(row["baseline_qps"]),
                    "qps_gain_pct": parse_float(row["qps_gain_pct"]),
                    "recall": parse_float(row["recall"]),
                    "baseline_recall": parse_float(row["baseline_recall"]),
                    "recall_drop": parse_float(row["recall_drop"]),
                    "mean_latency": parse_float(row["mean_latency"]),
                    "p999_latency": parse_float(row["p999_latency"]),
                    "graph_io": parse_float(row["graph_io"]),
                    "pipe_w": parse_float(row["pipe_w"]),
                    "pipe_max_observed": parse_float(row["pipe_max"]),
                }
                rows.append(sample)
    return rows


def knob_key(t_value, l_value, knob):
    return (
        int(t_value),
        int(l_value),
        str(knob["pipe_max"]),
        str(knob["qd_budget"]),
        str(knob["pipe_start"]),
        str(knob["ramp_step"]),
        str(knob["feedback_window"]),
        str(knob["min_marker"]),
    )


def sample_key(sample):
    return knob_key(sample["T"], sample["L"], sample)


def encode_numeric(value, mapping):
    if value in mapping:
        return mapping[value]
    try:
        return float(value)
    except ValueError:
        return 0.0


def feature_vector(t_value, l_value, knob):
    pipe_max = encode_numeric(str(knob["pipe_max"]), {"auto": 0.0, "baseline": 0.0})
    qd_budget = encode_numeric(str(knob["qd_budget"]), {"": 0.0})
    pipe_start = encode_numeric(str(knob["pipe_start"]), {"auto": 0.0, "": 0.0})
    ramp_step = encode_numeric(str(knob["ramp_step"]), {"auto": 0.0, "": 0.0})
    feedback_window = encode_numeric(str(knob["feedback_window"]), {"": 0.0})
    min_marker = encode_numeric(str(knob["min_marker"]), {"": 0.0})
    return [
        math.log2(float(t_value)),
        float(l_value) / 10.0,
        pipe_max / 8.0,
        qd_budget / 256.0,
        pipe_start / 8.0,
        ramp_step / 4.0,
        feedback_window / 8.0,
        min_marker / 8.0,
    ]


def distance(left, right):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def weighted_prediction(samples, t_value, l_value, knob, neighbors):
    target = feature_vector(t_value, l_value, knob)
    ranked = []
    for sample in samples:
        vec = feature_vector(sample["T"], sample["L"], sample)
        dist = distance(target, vec)
        ranked.append((dist, sample))
    ranked.sort(key=lambda item: item[0])
    selected = ranked[:max(1, min(neighbors, len(ranked)))]

    weights = [1.0 / (dist + 1e-6) for dist, _ in selected]
    total_weight = sum(weights)

    def pred(field):
        return sum(w * sample[field] for w, (_, sample) in zip(weights, selected)) / total_weight

    gain_values = [sample["qps_gain_pct"] for _, sample in selected]
    recall_values = [sample["recall_drop"] for _, sample in selected]
    return {
        "pred_gain_pct": pred("qps_gain_pct"),
        "pred_recall_drop": pred("recall_drop"),
        "pred_mean_latency": pred("mean_latency"),
        "pred_p999_latency": pred("p999_latency"),
        "gain_std": statistics.pstdev(gain_values) if len(gain_values) > 1 else 0.0,
        "recall_drop_std": statistics.pstdev(recall_values) if len(recall_values) > 1 else 0.0,
        "neighbor_count": len(selected),
        "nearest_distance": selected[0][0],
    }


def score_prediction(pred, max_recall_drop, exploration_weight):
    if pred["pred_recall_drop"] > max_recall_drop:
        violation = pred["pred_recall_drop"] - max_recall_drop
        return pred["pred_gain_pct"] - 100.0 * violation
    return pred["pred_gain_pct"] + exploration_weight * pred["gain_std"]


def recommend(samples, args):
    configs = knob_grid(args)
    observed = {sample_key(sample) for sample in samples}
    recommendations = []

    for thread_count in args.threads:
        for l_value in args.Ls:
            case_candidates = []
            for knob in configs:
                key = knob_key(thread_count, l_value, knob)
                if key in observed:
                    continue
                pred = weighted_prediction(samples, thread_count, l_value, knob, args.neighbors)
                row = {
                    "T": thread_count,
                    "L": l_value,
                    **knob,
                    **pred,
                }
                row["score"] = score_prediction(
                    pred, args.max_recall_drop, args.exploration_weight
                )
                row["pred_valid"] = pred["pred_recall_drop"] <= args.max_recall_drop
                case_candidates.append(row)

            case_candidates.sort(key=lambda row: row["score"], reverse=True)
            recommendations.extend(case_candidates[:args.top_k])
    return recommendations


def write_outputs(args, samples, recommendations):
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.result_root) / f"run_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    rec_csv = run_dir / "recommendations.csv"
    fields = [
        "T", "L", "pipe_max", "qd_budget", "pipe_start", "ramp_step",
        "feedback_window", "min_marker", "score", "pred_valid",
        "pred_gain_pct", "pred_recall_drop", "gain_std", "recall_drop_std",
        "pred_mean_latency", "pred_p999_latency", "nearest_distance",
        "neighbor_count",
    ]
    with rec_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in recommendations:
            writer.writerow({
                field: f"{row[field]:.6f}" if isinstance(row.get(field), float) else row.get(field)
                for field in fields
            })

    rec_json = run_dir / "recommendations.json"
    rec_json.write_text(json.dumps(recommendations, indent=2))

    metadata = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_name": args.dataset_name,
        "records": args.records,
        "sample_count": len(samples),
        "recommendation_count": len(recommendations),
        "threads": args.threads,
        "Ls": args.Ls,
        "top_k": args.top_k,
        "neighbors": args.neighbors,
        "max_recall_drop": args.max_recall_drop,
        "exploration_weight": args.exploration_weight,
    }
    meta_json = run_dir / "model_tune_config.json"
    meta_json.write_text(json.dumps(metadata, indent=2))

    return run_dir, rec_csv, rec_json, meta_json


def main():
    args = parse_args()
    samples = load_records(args.records, args.dataset_name)
    if not samples:
        raise RuntimeError("No pipeline samples found in input records")
    recommendations = recommend(samples, args)
    run_dir, rec_csv, rec_json, meta_json = write_outputs(args, samples, recommendations)

    print(f"RUN_DIR={run_dir}")
    print(f"RECOMMENDATIONS_CSV={rec_csv}")
    print(f"RECOMMENDATIONS_JSON={rec_json}")
    print(f"CONFIG={meta_json}")
    print(f"SAMPLES={len(samples)} RECOMMENDATIONS={len(recommendations)}")
    with rec_csv.open() as f:
        for idx, line in enumerate(f):
            print(line.rstrip())
            if idx >= min(10, len(recommendations)):
                break


if __name__ == "__main__":
    main()
