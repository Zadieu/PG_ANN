#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


DEFAULT_RESULT_ROOT = "/home/dell/projects/pipegor_torch_mlp_tuning"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sklearn + PyTorch MLP recommender for PipeGorANN tuning."
    )
    parser.add_argument("--records", nargs="+", required=True)
    parser.add_argument("--result-root", default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--threads", nargs="+", type=int, default=[8, 16, 32, 64])
    parser.add_argument("--Ls", nargs="+", type=int, default=[15, 20, 25, 30, 35, 40])
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-recall-drop", type=float, default=1.0)
    parser.add_argument("--recall-penalty", type=float, default=50.0,
                        help="Penalty multiplier for recall drop over --max-recall-drop.")
    parser.add_argument(
        "--prefer-observed-best",
        dest="prefer_observed_best",
        action="store_true",
        default=True,
        help="Use measured best config for target T/L when available.",
    )
    parser.add_argument(
        "--no-prefer-observed-best",
        dest="prefer_observed_best",
        action="store_false",
        help="Always rank by the learned model, even for observed T/L.",
    )
    parser.add_argument("--min-observed-gain", type=float, default=0.0)
    parser.add_argument("--min-pred-gain", type=float, default=5.0)
    parser.add_argument("--recall-drop-margin", type=float, default=0.05)
    parser.add_argument("--hidden-sizes", nargs="+", type=int, default=[64, 32])
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=120)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
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


def non_empty(row, key, default):
    value = row.get(key, default)
    if value is None or value == "":
        return default
    return value


def load_records(paths, dataset_name):
    rows = []
    for path in paths:
        record_path = Path(path)
        with record_path.open(newline="") as f:
            for row in csv.DictReader(f):
                if row.get("mode") != "pipegor_pipeline":
                    continue
                rows.append(
                    {
                        "dataset": dataset_name,
                        "record_path": str(record_path),
                        "T": int(row["T"]),
                        "L": int(row["L"]),
                        "pipe_max": non_empty(row, "pipe_max_knob", "auto"),
                        "qd_budget": non_empty(row, "qd_budget", "256"),
                        "pipe_start": non_empty(row, "pipe_start", "auto"),
                        "ramp_step": non_empty(row, "ramp_step", "auto"),
                        "feedback_window": non_empty(row, "feedback_window", "4"),
                        "min_marker": non_empty(row, "min_marker", "2"),
                        "qps_gain_pct": parse_float(row.get("qps_gain_pct")),
                        "recall_drop": parse_float(row.get("recall_drop")),
                    }
                )
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
        configs.append(
            {
                "pipe_max": pipe_max,
                "qd_budget": qd_budget,
                "pipe_start": pipe_start,
                "ramp_step": ramp_step,
                "feedback_window": feedback_window,
                "min_marker": min_marker,
            }
        )
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


def feature_dict(t_value, l_value, knob, dataset_name):
    t_float = float(t_value)
    l_float = float(l_value)
    out = {
        "dataset=" + str(dataset_name): 1.0,
        "log2_T": math.log2(max(t_float, 1.0)),
        "L": l_float,
        "L_over_T": l_float / max(t_float, 1.0),
        "T_over_L": t_float / max(l_float, 1.0),
    }
    for key in [
        "pipe_max",
        "qd_budget",
        "pipe_start",
        "ramp_step",
        "feedback_window",
        "min_marker",
    ]:
        out[f"{key}={knob[key]}"] = 1.0
    return out


def constrained_score(gain_pct, recall_drop, max_recall_drop, recall_penalty):
    return gain_pct - recall_penalty * max(0.0, recall_drop - max_recall_drop)


def attach_scores(samples, args):
    for sample in samples:
        sample["constrained_score"] = constrained_score(
            sample["qps_gain_pct"],
            sample["recall_drop"],
            args.max_recall_drop,
            args.recall_penalty,
        )


def observed_best_by_case(samples, max_recall_drop):
    best = {}
    observed_cases = set()
    for sample in samples:
        case_key = (sample["T"], sample["L"])
        observed_cases.add(case_key)
        if sample["recall_drop"] > max_recall_drop:
            continue
        current = best.get(case_key)
        if current is None or sample["constrained_score"] > current["constrained_score"]:
            best[case_key] = sample
    return best, observed_cases


def observed_row(sample):
    return {
        "T": sample["T"],
        "L": sample["L"],
        "source": "observed_best",
        "pipe_max": sample["pipe_max"],
        "qd_budget": sample["qd_budget"],
        "pipe_start": sample["pipe_start"],
        "ramp_step": sample["ramp_step"],
        "feedback_window": sample["feedback_window"],
        "min_marker": sample["min_marker"],
        "pred_gain_pct": sample["qps_gain_pct"],
        "pred_recall_drop": sample["recall_drop"],
        "pred_score": sample["constrained_score"],
        "score": sample["constrained_score"],
        "pred_valid": True,
    }


class TorchMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout):
        super().__init__()
        layers = []
        prev = input_size
        for hidden in hidden_sizes:
            layers.append(nn.Linear(prev, hidden))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = hidden
        layers.append(nn.Linear(prev, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def choose_device(args):
    if args.device == "cpu":
        return torch.device("cpu")
    if args.device == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_arrays(samples, args):
    vectorizer = DictVectorizer(sparse=False)
    feature_rows = [
        feature_dict(s["T"], s["L"], s, s.get("dataset", args.dataset_name))
        for s in samples
    ]
    x_raw = vectorizer.fit_transform(feature_rows).astype(np.float32)
    x_scaler = StandardScaler()
    x_scaled = x_scaler.fit_transform(x_raw).astype(np.float32)

    y_raw = np.array(
        [[s["qps_gain_pct"], s["recall_drop"], s["constrained_score"]] for s in samples],
        dtype=np.float32,
    )
    y_scaler = StandardScaler()
    y_scaled = y_scaler.fit_transform(y_raw).astype(np.float32)
    return x_scaled, y_scaled, vectorizer, x_scaler, y_scaler


def split_train_val(x, y, args):
    if len(x) < 16 or args.validation_fraction <= 0:
        return x, y, x, y
    return train_test_split(
        x,
        y,
        test_size=args.validation_fraction,
        random_state=args.seed,
        shuffle=True,
    )


def train_model(x, y, args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args)

    x_train, x_val, y_train, y_val = split_train_val(x, y, args)
    train_ds = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=min(args.batch_size, len(train_ds)),
        shuffle=True,
    )
    x_val_t = torch.tensor(x_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device)

    model = TorchMLP(x.shape[1], args.hidden_sizes, y.shape[1], args.dropout).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    loss_fn = nn.MSELoss()

    best_state = None
    best_val = float("inf")
    best_epoch = 0
    stale = 0
    losses = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        seen = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * len(xb)
            seen += len(xb)
        train_loss /= max(seen, 1)

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(x_val_t), y_val_t).item())

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            stale = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            stale += 1

        if epoch == 1 or epoch % max(1, args.epochs // 10) == 0:
            losses.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                }
            )

        if args.patience > 0 and stale >= args.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, losses, best_epoch, best_val, str(device)


def transform_features(feature_rows, vectorizer, x_scaler):
    x_raw = vectorizer.transform(feature_rows).astype(np.float32)
    return x_scaler.transform(x_raw).astype(np.float32)


def predict(model, vectorizer, x_scaler, y_scaler, t_value, l_value, knob, dataset_name, device):
    feature = feature_dict(t_value, l_value, knob, dataset_name)
    x = transform_features([feature], vectorizer, x_scaler)
    x_t = torch.tensor(x, dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        pred_scaled = model(x_t).cpu().numpy()
    pred = y_scaler.inverse_transform(pred_scaled)[0]
    return {
        "pred_gain_pct": float(pred[0]),
        "pred_recall_drop": float(pred[1]),
        "pred_score": float(pred[2]),
    }


def score_prediction(pred, max_recall_drop, recall_penalty):
    if "pred_score" in pred:
        return pred["pred_score"]
    return constrained_score(
        pred["pred_gain_pct"],
        pred["pred_recall_drop"],
        max_recall_drop,
        recall_penalty,
    )


def recommend(samples, model, vectorizer, x_scaler, y_scaler, args, device):
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
                pred = predict(
                    model,
                    vectorizer,
                    x_scaler,
                    y_scaler,
                    thread_count,
                    l_value,
                    knob,
                    args.dataset_name,
                    device,
                )
                row = {
                    "T": thread_count,
                    "L": l_value,
                    "source": "torch_mlp_predicted",
                    **knob,
                    **pred,
                }
                row["score"] = score_prediction(pred, args.max_recall_drop, args.recall_penalty)
                row["pred_valid"] = pred["pred_recall_drop"] <= args.max_recall_drop
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

            selected = case_candidates[: args.top_k]
            for rank, row in enumerate(selected, 1):
                row["case_rank"] = rank
                row["recommended_mode"] = recommended_mode
                row["decision_reason"] = decision_reason
            recommendations.extend(selected)
    return recommendations


def float_or_value(value):
    if isinstance(value, float):
        return f"{value:.6f}"
    return value


def write_csv(path, rows, fields):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: float_or_value(row.get(field, "")) for field in fields})


def write_outputs(args, samples, recommendations, model, vectorizer, x_scaler, y_scaler, losses, best_epoch, best_val, device):
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.result_root) / f"run_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    rec_fields = [
        "T",
        "L",
        "case_rank",
        "source",
        "recommended_mode",
        "decision_reason",
        "pipe_max",
        "qd_budget",
        "pipe_start",
        "ramp_step",
        "feedback_window",
        "min_marker",
        "score",
        "pred_valid",
        "pred_gain_pct",
        "pred_recall_drop",
        "pred_score",
    ]
    rec_csv = run_dir / "recommendations.csv"
    write_csv(rec_csv, recommendations, rec_fields)

    rec_json = run_dir / "recommendations.json"
    rec_json.write_text(json.dumps(recommendations, indent=2))

    profile_rows = [row for row in recommendations if row.get("case_rank") == 1]
    profile_fields = [
        "T",
        "L",
        "source",
        "recommended_mode",
        "decision_reason",
        "pipe_max",
        "qd_budget",
        "pipe_start",
        "ramp_step",
        "feedback_window",
        "min_marker",
        "pred_gain_pct",
        "pred_recall_drop",
        "pred_score",
    ]
    profile_csv = run_dir / "predicted_profile.csv"
    write_csv(profile_csv, profile_rows, profile_fields)
    profile_json = run_dir / "predicted_profile.json"
    profile_json.write_text(json.dumps(profile_rows, indent=2))

    model_path = run_dir / "torch_model.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "input_size": next(model.parameters()).shape[1],
            "hidden_sizes": args.hidden_sizes,
            "output_size": 3,
            "dropout": args.dropout,
        },
        model_path,
    )
    preprocess_path = run_dir / "preprocess.joblib"
    joblib.dump(
        {
            "vectorizer": vectorizer,
            "x_scaler": x_scaler,
            "y_scaler": y_scaler,
        },
        preprocess_path,
    )

    config = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": "sklearn_torch_mlp",
        "dataset_name": args.dataset_name,
        "records": args.records,
        "sample_count": len(samples),
        "recommendation_count": len(recommendations),
        "predicted_profile_csv": str(profile_csv),
        "predicted_profile_json": str(profile_json),
        "recommendations_csv": str(rec_csv),
        "recommendations_json": str(rec_json),
        "model_path": str(model_path),
        "preprocess_path": str(preprocess_path),
        "threads": args.threads,
        "Ls": args.Ls,
        "top_k": args.top_k,
        "max_recall_drop": args.max_recall_drop,
        "recall_penalty": args.recall_penalty,
        "prefer_observed_best": args.prefer_observed_best,
        "min_observed_gain": args.min_observed_gain,
        "min_pred_gain": args.min_pred_gain,
        "recall_drop_margin": args.recall_drop_margin,
        "hidden_sizes": args.hidden_sizes,
        "dropout": args.dropout,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "validation_fraction": args.validation_fraction,
        "patience": args.patience,
        "seed": args.seed,
        "device": device,
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "losses": losses,
    }
    config_json = run_dir / "torch_mlp_config.json"
    config_json.write_text(json.dumps(config, indent=2))
    return run_dir, rec_csv, rec_json, profile_csv, profile_json, model_path, preprocess_path, config_json


def main():
    args = parse_args()
    samples = load_records(args.records, args.dataset_name)
    attach_scores(samples, args)
    if len(samples) < 8:
        raise RuntimeError("Need at least 8 pipeline samples to train the MLP")

    x, y, vectorizer, x_scaler, y_scaler = prepare_arrays(samples, args)
    model, losses, best_epoch, best_val, device_name = train_model(x, y, args)
    device = torch.device(device_name)
    model.to(device)
    recommendations = recommend(samples, model, vectorizer, x_scaler, y_scaler, args, device)
    (
        run_dir,
        rec_csv,
        rec_json,
        profile_csv,
        profile_json,
        model_path,
        preprocess_path,
        config_json,
    ) = write_outputs(
        args,
        samples,
        recommendations,
        model.cpu(),
        vectorizer,
        x_scaler,
        y_scaler,
        losses,
        best_epoch,
        best_val,
        device_name,
    )

    print(f"RUN_DIR={run_dir}")
    print(f"RECOMMENDATIONS_CSV={rec_csv}")
    print(f"RECOMMENDATIONS_JSON={rec_json}")
    print(f"PREDICTED_PROFILE_CSV={profile_csv}")
    print(f"PREDICTED_PROFILE_JSON={profile_json}")
    print(f"MODEL={model_path}")
    print(f"PREPROCESS={preprocess_path}")
    print(f"CONFIG={config_json}")
    print(f"SAMPLES={len(samples)} RECOMMENDATIONS={len(recommendations)}")
    print(f"BEST_EPOCH={best_epoch} BEST_VAL_LOSS={best_val:.6f} DEVICE={device_name}")
    print("LOSSES=" + json.dumps(losses[-5:]))
    with rec_csv.open() as f:
        for idx, line in enumerate(f):
            print(line.rstrip())
            if idx >= min(10, len(recommendations)):
                break


if __name__ == "__main__":
    main()
