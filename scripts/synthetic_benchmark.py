#!/usr/bin/env python
from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import sys
import tempfile
import warnings
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path
from typing import Literal

# Allow running via `python scripts/...` without needing to set PYTHONPATH.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import mlflow
import numpy as np
import pandas as pd

from ml_monitoring_service.anomaly_detector import AnomalyDetector
from ml_monitoring_service.constants import MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI
from ml_monitoring_service.data_handling import (
    convert_to_model_input,
    get_microservice_data_from_file,
    get_ordered_timepoints,
)
from ml_monitoring_service.evaluation.synthetic import (
    InjectionSpec,
    inject_anomalies,
    set_seeds,
)

# Keep benchmark output compact.
warnings.filterwarnings("ignore", category=FutureWarning, module=r"mlflow\..*")


_LOG = logging.getLogger(__name__)


def _git_dir() -> Path | None:
    """Return the git directory path if this looks like a git worktree."""
    dot_git = _REPO_ROOT / ".git"
    if not dot_git.exists():
        return None
    if dot_git.is_dir():
        return dot_git
    if dot_git.is_file():
        # Worktree/submodule case: .git is a file containing `gitdir: <path>`.
        try:
            txt = dot_git.read_text(encoding="utf-8").strip()
        except Exception as exc:
            _LOG.debug("Unable to read .git file: %s", exc)
            return None
        prefix = "gitdir:"
        if not txt.lower().startswith(prefix):
            return None
        gitdir = txt[len(prefix) :].strip()
        gd = Path(gitdir)
        if not gd.is_absolute():
            gd = (_REPO_ROOT / gd).resolve()
        return gd
    return None


def _git_packed_ref(git_dir: Path, ref: str) -> str | None:
    packed = git_dir / "packed-refs"
    if not packed.exists():
        return None
    try:
        for line in packed.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("^"):
                continue
            sha, name = line.split(" ", 1)
            if name.strip() == ref:
                return sha.strip()
    except Exception as exc:
        _LOG.debug("Unable to parse packed-refs: %s", exc)
    return None


def _git_metadata() -> tuple[str | None, str | None]:
    """Return (commit_short, branch) best-effort, without calling git."""
    gd = _git_dir()
    if gd is None:
        return None, None

    head = gd / "HEAD"
    try:
        head_txt = head.read_text(encoding="utf-8").strip()
    except Exception as exc:
        _LOG.debug("Unable to read git HEAD: %s", exc)
        return None, None

    if head_txt.startswith("ref:"):
        ref = head_txt.split(":", 1)[1].strip()
        branch = ref.split("/")[-1] if ref else None
        ref_path = gd / ref
        sha = None
        if ref_path.exists():
            try:
                sha = ref_path.read_text(encoding="utf-8").strip()
            except Exception as exc:
                _LOG.debug("Unable to read ref %s: %s", ref, exc)
        if not sha:
            sha = _git_packed_ref(gd, ref)
        commit_short = sha[:7] if sha else None
        return commit_short, branch

    # Detached HEAD: HEAD contains the commit hash.
    commit_short = head_txt[:7] if head_txt else None
    return commit_short, None


@contextlib.contextmanager
def _suppress_project_warnings() -> Iterator[None]:
    """Temporarily suppress warning-level logs from the project during benchmarks."""
    target_names = [
        "ml_monitoring_service",
        "ml_monitoring_service.data_handling",
        "mlflow",
        "mlflow.system_metrics",
        "mlflow.system_metrics.system_metrics_monitor",
    ]
    targets = [logging.getLogger(name) for name in target_names]
    prev_levels = [t.level for t in targets]
    try:
        for t in targets:
            t.setLevel(logging.ERROR)
        yield
    finally:
        for t, lvl in zip(targets, prev_levels, strict=False):
            t.setLevel(lvl)


def _parse_variant(variant: str) -> tuple[Literal["grid", "event"], str | None]:
    """Parse a preprocessing variant string.

    Supported formats:
      - "event" -> ("event", None)
      - "grid" -> ("grid", None)
      - "grid:1s" -> ("grid", "1s")
      - "grid:1min" -> ("grid", "1min")
    """
    raw = variant.strip()
    if not raw:
        raise ValueError("Empty variant")

    if ":" in raw:
        approach, freq = raw.split(":", 1)
        approach = approach.strip().lower()
        freq = freq.strip()
    else:
        approach, freq = raw.lower(), None

    if approach not in ("grid", "event"):
        raise ValueError(
            f"Invalid variant '{variant}'. Use 'event' or 'grid[:<freq>]'."
        )
    if approach == "event":
        freq = None
    return approach, freq


def _auc_roc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute ROC AUC with a simple trapezoidal rule (no sklearn dependency)."""
    y_true = y_true.astype(bool)
    order = np.argsort(-y_score)
    y_true = y_true[order]
    y_score = y_score[order]

    pos = y_true.sum()
    neg = (~y_true).sum()
    if pos == 0 or neg == 0:
        return float("nan")

    tps = np.cumsum(y_true)
    fps = np.cumsum(~y_true)

    tpr = tps / pos
    fpr = fps / neg

    # Add (0,0) start
    tpr = np.concatenate([[0.0], tpr])
    fpr = np.concatenate([[0.0], fpr])

    return float(np.trapz(tpr, fpr))


def _best_f1_threshold(
    y_true: np.ndarray, y_score: np.ndarray
) -> tuple[float, float, float, float]:
    """Return (best_f1, threshold, precision, recall)."""
    y_true = y_true.astype(bool)
    # candidate thresholds: unique scores
    thresholds = np.unique(y_score)
    best = (-1.0, 0.0, 0.0, 0.0)
    for thr in thresholds:
        y_pred = y_score >= thr
        tp = np.sum(y_pred & y_true)
        fp = np.sum(y_pred & ~y_true)
        fn = np.sum(~y_pred & y_true)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        if f1 > best[0]:
            best = (float(f1), float(thr), float(precision), float(recall))
    return best


def _window_labels(
    point_labels: np.ndarray, window_size: int, stride: int
) -> np.ndarray:
    labels = []
    for i in range(0, len(point_labels) - window_size + 1, stride):
        labels.append(bool(point_labels[i : i + window_size].any()))
    return np.asarray(labels, dtype=bool)


def run_variant(
    *,
    dataset_path: Path,
    active_set: str,
    variant: str,
    seed: int,
    max_epochs: int,
    window_size: int,
    batch_size: int,
    stride: int,
) -> dict:
    set_seeds(seed)

    approach, freq = _parse_variant(variant)

    # NOTE: convert_to_model_input mutates the df in-place (to keep timepoints consistent).
    # Load a fresh df per variant to allow parallel comparisons.
    df = get_microservice_data_from_file(str(dataset_path))

    data, services, features = convert_to_model_input(
        active_set,
        df,
        approach=approach,
        time_grid_freq=freq,
    )
    timepoints = get_ordered_timepoints(df)

    # Split: train/val/infer
    # AnomalyDetector.train() requires:
    #   - len(train_data) > max(window_size + 1, batch_size)
    #   - len(val_data) > window_size
    # We also want inference long enough for at least one scoring window.
    n = len(data)
    min_train = max(window_size + 2, batch_size + 1)
    min_val = window_size + 1
    min_infer = window_size

    if n < (min_train + min_val + min_infer):
        raise ValueError(
            f"Variant '{variant}' yields too few timesteps for this benchmark: n={n}, "
            f"need >= {min_train + min_val + min_infer} (train>={min_train}, val>={min_val}, infer>={min_infer}). "
            "Use a smaller --window-size, smaller --batch-size, a finer grid (e.g. 'grid:1s'), or a larger dataset."
        )

    # Start from desired fractions, then clamp to satisfy minimum segment sizes.
    train_end = max(min_train, int(0.6 * n))
    val_end = max(train_end + min_val, int(0.8 * n))
    # Ensure enough room for inference.
    if n - val_end < min_infer:
        val_end = n - min_infer
    # Ensure val large enough.
    if val_end - train_end < min_val:
        train_end = val_end - min_val
    # Ensure train large enough.
    if train_end < min_train:
        train_end = min_train
        val_end = train_end + min_val
        if n - val_end < min_infer:
            raise ValueError(
                f"Variant '{variant}' cannot satisfy split constraints after adjustment: n={n}."
            )

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    infer_data = data[val_end:]

    infer_timepoints = timepoints[val_end:]

    detector = AnomalyDetector(
        num_services=len(services),
        num_features=len(features),
        window_size=window_size,
        config=None,
    )

    detector.train(
        train_data,
        val_data,
        df=df,
        active_set=active_set,
        max_epochs=max_epochs,
        timepoints=timepoints[:val_end],
        batch_size=batch_size,
        save_checkpoint=False,
    )
    detector.set_threshold(
        val_data, timepoints=timepoints[train_end:val_end], percentile=95
    )

    # Synthetic anomaly injection on inference segment
    # Keep it deterministic and simple: one spike + one dropout.
    if len(infer_data) < window_size:
        raise ValueError(
            f"Inference segment too short for scoring windows: len(infer)={len(infer_data)} < window_size={window_size}."
        )

    spike_start = min(max(0, len(infer_data) // 3), max(0, len(infer_data) - 2))
    dropout_start = min(max(0, len(infer_data) // 2), max(0, len(infer_data) - 2))
    specs = [
        InjectionSpec(
            kind="spike",
            service_idx=0,
            feature_idx=0,
            start_t=spike_start,
            duration=min(5, max(1, len(infer_data) - spike_start)),
            magnitude=2.5,
        ),
        InjectionSpec(
            kind="dropout",
            service_idx=min(1, len(services) - 1),
            feature_idx=min(2, len(features) - 1),
            start_t=dropout_start,
            duration=min(7, max(1, len(infer_data) - dropout_start)),
            magnitude=0.0,
        ),
    ]

    infer_injected, point_labels = inject_anomalies(infer_data, specs)

    window_scores = []
    window_time = []
    window_size = detector.window_size

    for i in range(0, len(infer_injected) - window_size + 1, stride):
        window = infer_injected[i : i + window_size]
        window_tp = infer_timepoints[i : i + window_size]
        res = detector.detect(window, window_tp)
        window_scores.append(float(res["error_score"]))
        window_time.append(str(pd.to_datetime(window_tp[0])))

    window_scores = np.asarray(window_scores, dtype=float)
    window_labels = _window_labels(point_labels, window_size, stride)

    auc = _auc_roc(window_labels, window_scores)
    best_f1, best_thr, best_p, best_r = _best_f1_threshold(window_labels, window_scores)

    persisted_thr = (
        float(detector.threshold) if detector.threshold is not None else float("nan")
    )
    persisted_pred = window_scores > persisted_thr
    tp = int(np.sum(persisted_pred & window_labels))
    fp = int(np.sum(persisted_pred & ~window_labels))
    fn = int(np.sum(~persisted_pred & window_labels))

    summary = {
        "dataset": str(dataset_path),
        "active_set": active_set,
        "variant": variant,
        "seed": seed,
        "stride": stride,
        "window_size": window_size,
        "batch_size": batch_size,
        "approach": approach,
        "time_grid_freq": freq,
        "num_services": len(services),
        "num_features": len(features),
        "features": features,
        "services": services,
        "num_windows": int(len(window_scores)),
        "roc_auc": float(auc),
        "best_f1": best_f1,
        "best_f1_threshold": best_thr,
        "best_f1_precision": best_p,
        "best_f1_recall": best_r,
        "persisted_threshold": persisted_thr,
        "persisted_tp": tp,
        "persisted_fp": fp,
        "persisted_fn": fn,
        "injections": [asdict(s) for s in specs],
    }

    return summary


def _print_comparison(results: list[dict]) -> None:
    print(
        "\n=== Synthetic benchmark (variant | windows | AUC | bestF1 | P | R | error) ==="
    )
    for r in results:
        v = str(r.get("variant"))
        err = r.get("error")
        if err:
            print(
                f"{v:10s} | {'-':>7s} | {'-':>6s} | {'-':>6s} | {'-':>6s} | {'-':>6s} | {err}"
            )
            continue
        w = int(r.get("num_windows", 0))
        auc = float(r.get("roc_auc", float("nan")))
        f1 = float(r.get("best_f1", float("nan")))
        p = float(r.get("best_f1_precision", float("nan")))
        rec = float(r.get("best_f1_recall", float("nan")))
        print(f"{v:10s} | {w:7d} | {auc:0.4f} | {f1:0.4f} | {p:0.4f} | {rec:0.4f} |")


def _log_results_to_mlflow(
    *, results: list[dict], args: argparse.Namespace, out: Path | None
) -> str:
    """Log a parent run plus one nested run per variant."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    parent_name = f"Synthetic benchmark: {args.active_set}"
    # MLflow 3.x prints UI links to stdout by default; suppress those here.
    with contextlib.redirect_stdout(io.StringIO()):
        with mlflow.start_run(run_name=parent_name) as parent:
            # Repro tags (best-effort; ignore if git info isn't available).
            try:
                git_commit, git_branch = _git_metadata()
                if git_commit:
                    mlflow.set_tag("git.commit", git_commit)
                if git_branch:
                    mlflow.set_tag("git.branch", git_branch)
            except Exception as exc:
                _LOG.debug("Skipping git metadata tags: %s", exc)

            mlflow.log_param("benchmark", "synthetic")
            mlflow.log_param("dataset", str(args.dataset))
            mlflow.log_param("active_set", str(args.active_set))
            mlflow.log_param("variants", str(args.variants))
            mlflow.log_param("seed", int(args.seed))
            mlflow.log_param("max_epochs", int(args.max_epochs))
            mlflow.log_param("window_size", int(args.window_size))
            mlflow.log_param("batch_size", int(args.batch_size))
            mlflow.log_param("stride", int(args.stride))

            # Log the combined results JSON once.
            payload = {"results": results}
            if out is not None:
                # If the user asked for an output file, log exactly that.
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(payload, indent=2))
                mlflow.log_artifact(str(out))
            else:
                # Otherwise, write a temporary artifact.
                with tempfile.TemporaryDirectory() as tmpdir:
                    p = Path(tmpdir) / "synthetic_benchmark_results.json"
                    p.write_text(json.dumps(payload, indent=2))
                    mlflow.log_artifact(str(p))

            # Nested run per variant with metrics.
            for r in results:
                variant = str(r.get("variant"))
                with mlflow.start_run(run_name=f"variant: {variant}", nested=True):
                    if "error" in r:
                        mlflow.set_tag("status", "error")
                        mlflow.set_tag("error", str(r.get("error")))
                        continue

                    mlflow.set_tag("status", "ok")
                    mlflow.log_param("variant", variant)
                    mlflow.log_param("approach", r.get("approach"))
                    if r.get("time_grid_freq") is not None:
                        mlflow.log_param("time_grid_freq", r.get("time_grid_freq"))

                    for key in (
                        "num_windows",
                        "roc_auc",
                        "best_f1",
                        "best_f1_precision",
                        "best_f1_recall",
                        "persisted_threshold",
                        "persisted_tp",
                        "persisted_fp",
                        "persisted_fn",
                    ):
                        if key in r and r[key] is not None:
                            val = r[key]
                            try:
                                mlflow.log_metric(key, float(val))
                            except Exception as exc:
                                # Keep logging robust; skip non-numeric values.
                                _LOG.debug(
                                    "Skipping metric %s=%r for variant %s: %s",
                                    key,
                                    val,
                                    variant,
                                    exc,
                                )

                    # Variant JSON as an artifact (handy for debugging).
                    with tempfile.TemporaryDirectory() as tmpdir:
                        vp = Path(tmpdir) / f"result_{variant.replace(':', '_')}.json"
                        vp.write_text(json.dumps(r, indent=2))
                        mlflow.log_artifact(str(vp))

            return parent.info.run_id


def main() -> None:
    p = argparse.ArgumentParser(description="Synthetic anomaly benchmark")
    p.add_argument(
        "--dataset",
        default="tests/resources/combined_dataset_test.json",
        help="Path to combined dataset JSON.",
    )
    p.add_argument("--active-set", default="transfer")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--max-epochs", type=int, default=10)
    p.add_argument("--window-size", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument(
        "--variants",
        default="event,grid:1s",
        help="Comma-separated list of variants: 'event' or 'grid[:<freq>]'.",
    )
    p.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Window stride during scoring (1 = max overlap).",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Optional path to write JSON results (single file containing all variants).",
    )

    args = p.parse_args()
    out = Path(args.out) if args.out else None

    variants = [v.strip() for v in str(args.variants).split(",") if v.strip()]
    results: list[dict] = []
    for variant in variants:
        try:
            # Keep terminal output compact by default; suppress internal prints.
            with (
                _suppress_project_warnings(),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                results.append(
                    run_variant(
                        dataset_path=Path(args.dataset),
                        active_set=args.active_set,
                        variant=variant,
                        seed=args.seed,
                        max_epochs=args.max_epochs,
                        window_size=args.window_size,
                        batch_size=args.batch_size,
                        stride=args.stride,
                    )
                )
        except Exception as e:
            results.append({"variant": variant, "error": str(e)})

    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"results": results}, indent=2))

    _print_comparison(results)

    run_id = _log_results_to_mlflow(results=results, args=args, out=out)
    print(
        f"Logged to MLflow experiment '{MLFLOW_EXPERIMENT_NAME}' (parent run_id={run_id})."
    )


if __name__ == "__main__":
    main()
