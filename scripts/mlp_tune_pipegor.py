#!/usr/bin/env python3
import argparse
import csv
import itertools
import json
import math
import random
import time
from pathlib import Path


DEFAULT_RESULT_ROOT = "/home/dell/projects/pipegor_mlp_tuning"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Small pure-Python MLP recommender for PipeGorANN tuning."
    )
    parser.add_argument("--records", nargs="+", required=True)
    parser.add_argument("--result-root", default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--threads", nargs="+", type=int, default=[8, 16, 32, 64])
    parser.add_argument("--Ls", nargs="+", type=int, default=[15, 20, 25, 30, 35, 40])
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--max-recall-drop", type=float, default=1.0)
    parser.add_argument("--prefer-observed-best", dest="prefer_observed_best", action="store_true", default=True,
                        help="Prefer the best measured historical config for target T/L when available.")
    parser.add_argument("--no-prefer-observed-best", dest="prefer_observed_best", action="store_false",
                        help="Always use the MLP surrogate to rank candidates, even for observed T/L.")
    parser.add_argument("--min-observed-gain", type=float, default=0.0)
    parser.add_argument("--min-pred-gain", type=float, default=5.0)
    parser.add_argument("--recall-drop-margin", type=float, default=0.05)
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=2500)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--pipe-max-values", nargs="+", default=["auto", "4", "6", "8"])
    parser.add_argument("--qd-budget-values", nargs="+", default=["128", "256"])
    parser.add_argument("--pipe-start-values", nargs="+", default=["auto", "4"])
    parser.add_argument("--ramp-step-values", nargs="+", default=["auto", "1", "2"])
    parser.add_argument("--feedback-window-values", nargs="+", default=["4"])
    parser.add_argument("--min-marker-values", nargs="+", default=["2"])
    parser.add_argument("--dataset-name", default="sift1m")
    return parser.parse_args()


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
                rows.append({
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
                    "qps_gain_pct": parse_float(row["qps_gain_pct"]),
                    "recall_drop": parse_float(row["recall_drop"]),
                })
    return rows


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
        math.log2(float(t_value)) / 6.0,
        float(l_value) / 40.0,
        pipe_max / 8.0,
        qd_budget / 256.0,
        pipe_start / 8.0,
        ramp_step / 4.0,
        feedback_window / 8.0,
        min_marker / 8.0,
    ]


def mean(values):
    return sum(values) / len(values)


def std(values):
    if len(values) <= 1:
        return 1.0
    mu = mean(values)
    return math.sqrt(sum((v - mu) ** 2 for v in values) / len(values)) or 1.0


def prepare_dataset(samples):
    x_rows = [feature_vector(s["T"], s["L"], s) for s in samples]
    gain_values = [s["qps_gain_pct"] for s in samples]
    recall_values = [s["recall_drop"] for s in samples]
    target_mean = [mean(gain_values), mean(recall_values)]
    target_std = [std(gain_values), std(recall_values)]
    y_rows = [
        [
            (s["qps_gain_pct"] - target_mean[0]) / target_std[0],
            (s["recall_drop"] - target_mean[1]) / target_std[1],
        ]
        for s in samples
    ]
    return x_rows, y_rows, target_mean, target_std


def tanh(x):
    if x > 20:
        return 1.0
    if x < -20:
        return -1.0
    return math.tanh(x)


class SmallMLP:
    def __init__(self, input_size, hidden_size, output_size, rng):
        scale1 = math.sqrt(2.0 / (input_size + hidden_size))
        scale2 = math.sqrt(2.0 / (hidden_size + output_size))
        self.w1 = [[rng.uniform(-scale1, scale1) for _ in range(input_size)] for _ in range(hidden_size)]
        self.b1 = [0.0 for _ in range(hidden_size)]
        self.w2 = [[rng.uniform(-scale2, scale2) for _ in range(hidden_size)] for _ in range(output_size)]
        self.b2 = [0.0 for _ in range(output_size)]

    def forward(self, x):
        hidden_pre = [
            self.b1[j] + sum(self.w1[j][i] * x[i] for i in range(len(x)))
            for j in range(len(self.b1))
        ]
        hidden = [tanh(v) for v in hidden_pre]
        out = [
            self.b2[k] + sum(self.w2[k][j] * hidden[j] for j in range(len(hidden)))
            for k in range(len(self.b2))
        ]
        return hidden, out

    def predict_scaled(self, x):
        return self.forward(x)[1]

    def train_epoch(self, x_rows, y_rows, learning_rate, weight_decay, order):
        total_loss = 0.0
        for idx in order:
            x = x_rows[idx]
            y = y_rows[idx]
            hidden, out = self.forward(x)
            d_out = [out[k] - y[k] for k in range(len(out))]
            total_loss += sum(v * v for v in d_out) / len(d_out)

            old_w2 = [row[:] for row in self.w2]

            for k in range(len(self.w2)):
                for j in range(len(self.w2[k])):
                    grad = d_out[k] * hidden[j] + weight_decay * self.w2[k][j]
                    self.w2[k][j] -= learning_rate * grad
                self.b2[k] -= learning_rate * d_out[k]

            d_hidden = []
            for j in range(len(hidden)):
                grad = sum(old_w2[k][j] * d_out[k] for k in range(len(d_out)))
                grad *= 1.0 - hidden[j] * hidden[j]
                d_hidden.append(grad)

            for j in range(len(self.w1)):
                for i in range(len(self.w1[j])):
                    grad = d_hidden[j] * x[i] + weight_decay * self.w1[j][i]
                    self.w1[j][i] -= learning_rate * grad
                self.b1[j] -= learning_rate * d_hidden[j]

        return total_loss / len(order)


def train_mlp(x_rows, y_rows, args):
    rng = random.Random(args.seed)
    model = SmallMLP(len(x_rows[0]), args.hidden_size, 2, rng)
    order = list(range(len(x_rows)))
    losses = []
    for epoch in range(args.epochs):
        rng.shuffle(order)
        loss = model.train_epoch(x_rows, y_rows, args.learning_rate, args.weight_decay, order)
        if epoch == 0 or (epoch + 1) % max(1, args.epochs // 10) == 0:
            losses.append({"epoch": epoch + 1, "loss": loss})
    return model, losses


def predict(model, target_mean, target_std, t_value, l_value, knob):
    x = feature_vector(t_value, l_value, knob)
    scaled = model.predict_scaled(x)
    gain = scaled[0] * target_std[0] + target_mean[0]
    recall_drop = scaled[1] * target_std[1] + target_mean[1]
    return {
        "pred_gain_pct": gain,
        "pred_recall_drop": recall_drop,
    }


def score_prediction(pred, max_recall_drop):
    if pred["pred_recall_drop"] > max_recall_drop:
        return pred["pred_gain_pct"] - 100.0 * (pred["pred_recall_drop"] - max_recall_drop)
    return pred["pred_gain_pct"]


def observed_best_by_case(samples, max_recall_drop):
    best = {}
    observed_cases = set()
    for sample in samples:
        case_key = (sample["T"], sample["L"])
        observed_cases.add(case_key)
        if sample["recall_drop"] > max_recall_drop:
            continue
        current = best.get(case_key)
        if current is None or sample["qps_gain_pct"] > current["qps_gain_pct"]:
            best[case_key] = sample
    return best, observed_cases


def observed_row(sample):
    return {
        "T": sample["T"],
        "L": sample["L"],
        "pipe_max": sample["pipe_max"],
        "qd_budget": sample["qd_budget"],
        "pipe_start": sample["pipe_start"],
        "ramp_step": sample["ramp_step"],
        "feedback_window": sample["feedback_window"],
        "min_marker": sample["min_marker"],
        "pred_gain_pct": sample["qps_gain_pct"],
        "pred_recall_drop": sample["recall_drop"],
        "score": sample["qps_gain_pct"],
        "pred_valid": True,
        "source": "observed_best",
    }


def recommend(samples, model, target_mean, target_std, args):
    observed = {sample_key(sample) for sample in samples}
    best_observed, observed_cases = observed_best_by_case(samples, args.max_recall_drop)
    configs = knob_grid(args)
    recommendations = []
    for thread_count in args.threads:
        for l_value in args.Ls:
            case_candidates = []
            for knob in configs:
                if knob_key(thread_count, l_value, knob) in observed:
                    continue
                pred = predict(model, target_mean, target_std, thread_count, l_value, knob)
                row = {
                    "T": thread_count,
                    "L": l_value,
                    **knob,
                    **pred,
                }
                row["score"] = score_prediction(pred, args.max_recall_drop)
                row["pred_valid"] = pred["pred_recall_drop"] <= args.max_recall_drop
                row["source"] = "mlp_predicted"
                case_candidates.append(row)
            case_candidates.sort(key=lambda row: row["score"], reverse=True)
            case_key = (thread_count, l_value)
            if args.prefer_observed_best and case_key in best_observed:
                case_candidates.insert(0, observed_row(best_observed[case_key]))
            if not case_candidates:
                continue
            top = case_candidates[0]
            if args.prefer_observed_best and case_key in observed_cases and case_key not in best_observed:
                recommended_mode = "baseline"
                decision_reason = "no_valid_observed_pipeline"
            elif top.get("source") == "observed_best":
                if top["pred_gain_pct"] <= args.min_observed_gain:
                    recommended_mode = "baseline"
                    decision_reason = "low_observed_gain"
                else:
                    recommended_mode = "pipegor_pipeline"
                    decision_reason = "observed_best"
            elif not top["pred_valid"]:
                recommended_mode = "baseline"
                decision_reason = "pred_recall_drop_over_limit"
            elif top["pred_recall_drop"] > args.max_recall_drop - args.recall_drop_margin:
                recommended_mode = "baseline"
                decision_reason = "recall_drop_margin"
            elif top["pred_gain_pct"] < args.min_pred_gain:
                recommended_mode = "baseline"
                decision_reason = "low_pred_gain"
            else:
                recommended_mode = "pipegor_pipeline"
                decision_reason = "pipeline_candidate"

            selected = case_candidates[:args.top_k]
            for rank, row in enumerate(selected, 1):
                row["case_rank"] = rank
                row["recommended_mode"] = recommended_mode
                row["decision_reason"] = decision_reason
            recommendations.extend(selected)
    return recommendations


def write_outputs(args, samples, recommendations, losses, target_mean, target_std):
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.result_root) / f"run_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    rec_csv = run_dir / "recommendations.csv"
    fields = [
        "T", "L", "case_rank", "source", "recommended_mode", "decision_reason",
        "pipe_max", "qd_budget", "pipe_start", "ramp_step",
        "feedback_window", "min_marker", "score", "pred_valid",
        "pred_gain_pct", "pred_recall_drop",
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

    profile_rows = [row for row in recommendations if row.get("case_rank") == 1]
    profile_csv = run_dir / "predicted_profile.csv"
    profile_fields = [
        "T", "L", "source", "recommended_mode", "decision_reason",
        "pipe_max", "qd_budget", "pipe_start", "ramp_step",
        "feedback_window", "min_marker", "pred_gain_pct", "pred_recall_drop",
    ]
    with profile_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=profile_fields)
        writer.writeheader()
        for row in profile_rows:
            writer.writerow({
                field: f"{row[field]:.6f}" if isinstance(row.get(field), float) else row.get(field)
                for field in profile_fields
            })
    profile_json = run_dir / "predicted_profile.json"
    profile_json.write_text(json.dumps(profile_rows, indent=2))

    config = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": "pure_python_mlp",
        "dataset_name": args.dataset_name,
        "records": args.records,
        "sample_count": len(samples),
        "recommendation_count": len(recommendations),
        "predicted_profile_csv": str(profile_csv),
        "predicted_profile_json": str(profile_json),
        "threads": args.threads,
        "Ls": args.Ls,
        "top_k": args.top_k,
        "max_recall_drop": args.max_recall_drop,
        "prefer_observed_best": args.prefer_observed_best,
        "min_observed_gain": args.min_observed_gain,
        "min_pred_gain": args.min_pred_gain,
        "recall_drop_margin": args.recall_drop_margin,
        "hidden_size": args.hidden_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "target_mean": target_mean,
        "target_std": target_std,
        "losses": losses,
    }
    config_json = run_dir / "mlp_tune_config.json"
    config_json.write_text(json.dumps(config, indent=2))
    return run_dir, rec_csv, rec_json, profile_csv, profile_json, config_json


def main():
    args = parse_args()
    samples = load_records(args.records, args.dataset_name)
    if len(samples) < 8:
        raise RuntimeError("Need at least 8 pipeline samples to train the MLP")

    x_rows, y_rows, target_mean, target_std = prepare_dataset(samples)
    model, losses = train_mlp(x_rows, y_rows, args)
    recommendations = recommend(samples, model, target_mean, target_std, args)
    run_dir, rec_csv, rec_json, profile_csv, profile_json, config_json = write_outputs(
        args, samples, recommendations, losses, target_mean, target_std
    )

    print(f"RUN_DIR={run_dir}")
    print(f"RECOMMENDATIONS_CSV={rec_csv}")
    print(f"RECOMMENDATIONS_JSON={rec_json}")
    print(f"PREDICTED_PROFILE_CSV={profile_csv}")
    print(f"PREDICTED_PROFILE_JSON={profile_json}")
    print(f"CONFIG={config_json}")
    print(f"SAMPLES={len(samples)} RECOMMENDATIONS={len(recommendations)}")
    print("LOSSES=" + json.dumps(losses[-5:]))
    with rec_csv.open() as f:
        for idx, line in enumerate(f):
            print(line.rstrip())
            if idx >= min(10, len(recommendations)):
                break


if __name__ == "__main__":
    main()
