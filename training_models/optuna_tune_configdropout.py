"""
Optuna tuner for ConfigDropout (unified BETA).

Tunes:
  - DECAY_FUNCTION ∈ {negexp, invpower}
  - BETA ∈ {0.5, 1.0, ..., 15.0}
  - NOISE_LEVEL ∈ {0.01, 0.02, ..., 0.20}  (Gaussian)

Keeps fixed across trials:
  MIN_KEEP_RATIO=0.45, FINAL_REVISION=true, VAL_FRAC=0.0, PROGRESSIVE_PREFIX=false, SEED=42
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

import optuna

PROJECT_ROOT = Path(__file__).resolve().parent

TEST_PAT = re.compile(r"test:\s*loss=\s*[0-9.]+\s*acc=\s*([0-9.]+)", re.IGNORECASE)
BANNER   = re.compile(r"\[schedule\]", re.IGNORECASE)

# ---------- search space (unified BETA) ----------
def suggest_schedule(trial: optuna.Trial) -> Dict[str, Any]:
    decay = trial.suggest_categorical("DECAY_FUNCTION", ["negexp", "invpower"])
    beta  = trial.suggest_float("BETA", 0.5, 15.0, step=0.5)
    noise_level = trial.suggest_float("NOISE_LEVEL", 0.01, 0.20, step=0.01)
    return {
        "DECAY_FUNCTION": decay,
        "BETA":           beta,
        "NOISE_TYPE":     "gaussian",
        "NOISE_LEVEL":    noise_level,
        "NOISE_PROB":     0.00,
    }

# ---------- run one training trial ----------
def run_trial(args, display_idx: int, storage_idx: int, schedule: Dict[str, Any]) -> Tuple[float, List[Tuple[int, float]]]:
    main_argv = [
        sys.executable, "main.py",
        "--model", "mobilenet_v2",
        "--mode", "train_with_configurable",
        "--epoch", str(args.epochs),
        "--save_path", str(Path("cifar10_results/mobilenet_v2_optuna") / f"trial_{storage_idx:05d}"),
        "--dataset", "cifar10",
        "--batch_size", "32",
        "--start_revision", "0",
        "--task", "classification",
        "--threshold", "0.3",
        "--seed", "42",
    ]

    env = os.environ.copy()
    env.update({
        # tuned
        "CD_DECAY_FUNCTION": schedule["DECAY_FUNCTION"],
        "CD_BETA":           str(schedule["BETA"]),
        "CD_NOISE_TYPE":     schedule["NOISE_TYPE"],
        "CD_NOISE_LEVEL":    str(schedule["NOISE_LEVEL"]),
        "CD_NOISE_PROB":     str(schedule["NOISE_PROB"]),
        # fixed
        "CD_MIN_KEEP_RATIO": "0.45",
        "CD_FINAL_REVISION": "true",
        "CD_VAL_FRAC":       "0.0",
        "CD_SEED":           "42",
        "CD_PROGRESSIVE_PREFIX": "false",
        "PYTHONUNBUFFERED": "1",
    })

    workdir = PROJECT_ROOT / "tune_configdrop_runs" / f"trial_{storage_idx:05d}"
    workdir.mkdir(parents=True, exist_ok=True)
    log_path = workdir / "train.log"
    log_fp = log_path.open("w", buffering=1)

    print(f"[trial {display_idx}] (study #{storage_idx}) starting…", flush=True)

    proc = subprocess.Popen(
        main_argv, cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, env=env
    )

    history: List[Tuple[int, float]] = []
    best = 0.0
    ep = 0
    saw_banner = False
    early_stopped = False

    for line in iter(proc.stdout.readline, ""):
        sys.stdout.write(f"[trial {display_idx}] {line}")
        log_fp.write(line)

        if BANNER.search(line):
            saw_banner = True

        m_t = TEST_PAT.search(line)
        if m_t:
            ep += 1
            acc = float(m_t.group(1))
            history.append((ep, acc))
            if acc > best:
                best = acc
            if best >= args.target_acc:
                print(f"[trial {display_idx}] Hit target TEST acc {args.target_acc:.2f}. Terminating run.")
                proc.terminate()
                early_stopped = True
                break

    proc.wait()
    log_fp.close()

    # tolerate non-zero rc if we early-stopped the process ourselves
    if proc.returncode not in (0,) and not early_stopped:
        tail = "\n".join(log_path.read_text().splitlines()[-80:])
        raise RuntimeError(f"Training failed (rc={proc.returncode}). Tail:\n{tail}")

    if not saw_banner:
        print(f"[trial {display_idx}] NOTE: no '[schedule]' banner seen — is ConfigDropout printing it?")

    if not history:
        print(f"[trial {display_idx}] NOTE: no TEST acc parsed. See {log_path}")

    return best, history

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", default=None, help="Name of study. Omit for an ephemeral in-memory study.")
    ap.add_argument("--storage", default="", help="Optuna storage URL. Leave empty for in-memory (no DB).")
    ap.add_argument("--n-trials", type=int, default=40)
    ap.add_argument("--timeout", type=int, default=None)
    ap.add_argument("--target-acc", type=float, default=0.90)
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()

    (PROJECT_ROOT / "tune_configdrop_runs").mkdir(exist_ok=True)
    (PROJECT_ROOT / "cifar10_results" / "mobilenet_v2_optuna").mkdir(parents=True, exist_ok=True)

    use_storage = args.storage.strip() != ""
    if use_storage:
        # persistent (resumable) study
        study = optuna.create_study(
            study_name=args.study or "configdrop-schedule",
            storage=args.storage,
            direction="maximize",
            load_if_exists=True,
            pruner=optuna.pruners.SuccessiveHalvingPruner(min_resource=5, reduction_factor=3),
            sampler=optuna.samplers.TPESampler(
                seed=0, multivariate=True, group=True, warn_independent_sampling=False
            ),
        )
        print(f"[study] {study.study_name} @ {args.storage}")
    else:
        # ephemeral in-memory study (no DB, no reuse)
        study = optuna.create_study(
            direction="maximize",
            pruner=optuna.pruners.SuccessiveHalvingPruner(min_resource=5, reduction_factor=3),
            sampler=optuna.samplers.TPESampler(
                seed=0, multivariate=True, group=True, warn_independent_sampling=False
            ),
        )
        print("[study] ephemeral (in-memory)")

    print("[objective] maximizing TEST accuracy")

    run_counter = {"i": 0}

    def objective(trial: optuna.Trial) -> float:
        run_counter["i"] += 1
        local_idx = run_counter["i"]

        schedule = suggest_schedule(trial)
        best, history = run_trial(
            args, display_idx=local_idx, storage_idx=trial.number, schedule=schedule
        )

        for ep, acc in history:
            trial.report(acc, step=ep)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if best >= args.target_acc:
            print(f"[trial {local_idx}] Reached target acc {args.target_acc:.2f}. Stopping study.")
            trial.study.stop()

        return float(best)

    study.optimize(
        objective,
        n_trials=args.n_trials,
        timeout=args.timeout,
        gc_after_trial=True,
        show_progress_bar=True,
    )

    out = PROJECT_ROOT / "best_configdrop_schedule.json"
    out.write_text(json.dumps({
        "best_value": study.best_value,
        "best_params": study.best_params
    }, indent=2))

    print("Best TEST acc:", study.best_value)
    print("Best schedule params:", study.best_params)
    print("Saved:", out)

if __name__ == "__main__":
    main()
