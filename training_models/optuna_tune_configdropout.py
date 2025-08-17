"""
Optuna tuner for ConfigDropout SCHEDULE ONLY.

Per trial, this runs your real training:
  python main.py --model mobilenet_v2 --mode train_with_configurable \
    --epoch 30 --dataset cifar10 --batch_size 32 --threshold 0.3 --seed 123

It sets CD_* env vars (read by ConfigDropout.py via the tiny optional override block),
parses "test: ... acc=..." (default), and maximizes accuracy.

Session-local trial numbering is printed as:
  [trial 1], [trial 2], ...
Artifacts (save_path / logs) still use the study's global trial number for uniqueness.
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
VAL_PAT  = re.compile(r"val:\s*loss=\s*[0-9.]+\s*acc=\s*([0-9.]+)",  re.IGNORECASE)
TEST_PAT = re.compile(r"test:\s*loss=\s*[0-9.]+\s*acc=\s*([0-9.]+)", re.IGNORECASE)
BANNER   = re.compile(r"\[schedule\]", re.IGNORECASE)

# ---------- search space (schedule only) ----------
def suggest_schedule(trial: optuna.Trial) -> Dict[str, Any]:
    decay = trial.suggest_categorical("DECAY_FUNCTION", ["negexp", "invpower"])
    if decay == "negexp":
        beta = trial.suggest_float("BETA", 0.5, 15.0, log=True); p_beta = 0.0
    else:
        beta = 0.0; p_beta = trial.suggest_float("POWER_BETA", 0.0, 4.0)

    min_keep  = trial.suggest_float("MIN_KEEP_RATIO", 0.10, 0.90)
    final_rev = trial.suggest_categorical("FINAL_REVISION", [True, False])

    noise_type = trial.suggest_categorical("NOISE_TYPE", ["none", "gaussian", "uniform", "saltpepper"])
    if noise_type == "none":
        noise_level, noise_prob = 0.0, 0.0
    else:
        noise_level = trial.suggest_float("NOISE_LEVEL", 0.0, 0.15)
        noise_prob  = trial.suggest_float("NOISE_PROB", 0.0, 0.30) if noise_type == "saltpepper" else 0.0

    val_frac = trial.suggest_float("VAL_FRAC", 0.05, 0.20)

    return {
        "DECAY_FUNCTION": decay,
        "BETA": beta,
        "POWER_BETA": p_beta,
        "MIN_KEEP_RATIO": min_keep,
        "FINAL_REVISION": final_rev,
        "NOISE_TYPE": noise_type,
        "NOISE_LEVEL": noise_level,
        "NOISE_PROB": noise_prob,
        "VAL_FRAC": val_frac,
    }

# ---------- one trial (display_idx = local/session, storage_idx = global/study) ----------
def run_trial(args, display_idx: int, storage_idx: int, schedule: Dict[str, Any]) -> Tuple[float, List[Tuple[int, float]]]:
    # real training invocation
    main_argv = [
        sys.executable, "main.py",
        "--model", "mobilenet_v2",
        "--mode", "train_with_configurable",
        "--epoch", "30",
        "--save_path", str(Path("cifar10_results/mobilenet_v2_optuna") / f"trial_{storage_idx:05d}"),
        "--dataset", "cifar10",
        "--batch_size", "32",
        "--start_revision", "0",
        "--task", "classification",
        "--threshold", "0.3",
        "--seed", "123",
    ]

    # env for schedule overrides (only during tuning)
    env = os.environ.copy()
    env.update({
        "CD_DECAY_FUNCTION": schedule["DECAY_FUNCTION"],
        "CD_BETA":          str(schedule["BETA"]),
        "CD_POWER_BETA":    str(schedule["POWER_BETA"]),
        "CD_MIN_KEEP_RATIO":str(schedule["MIN_KEEP_RATIO"]),
        "CD_FINAL_REVISION":str(schedule["FINAL_REVISION"]),
        "CD_NOISE_TYPE":    schedule["NOISE_TYPE"],
        "CD_NOISE_LEVEL":   str(schedule["NOISE_LEVEL"]),
        "CD_NOISE_PROB":    str(schedule["NOISE_PROB"]),
        "CD_VAL_FRAC":      str(schedule["VAL_FRAC"]),
        "CD_SEED":          "123",
        "PYTHONUNBUFFERED": "1",
    })

    # logs/artifacts keyed by global trial number
    workdir = PROJECT_ROOT / "tune_configdrop_runs" / f"trial_{storage_idx:05d}"
    workdir.mkdir(parents=True, exist_ok=True)
    log_path = workdir / "train.log"
    log_fp = log_path.open("w", buffering=1)

    # map local↔global once at start
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
    pat = TEST_PAT if args.metric == "test" else VAL_PAT

    for line in iter(proc.stdout.readline, ""):
        # print with session-local numbering
        sys.stdout.write(f"[trial {display_idx}] {line}")
        log_fp.write(line)

        if BANNER.search(line):
            saw_banner = True

        m = pat.search(line)
        if m:
            ep += 1
            acc = float(m.group(1))
            history.append((ep, acc))
            best = max(best, acc)

    proc.wait()
    log_fp.close()

    if proc.returncode != 0:
        tail = "\n".join(log_path.read_text().splitlines()[-60:])
        raise RuntimeError(f"Training failed (rc={proc.returncode}). Tail:\n{tail}")

    if not saw_banner:
        print(f"[trial {display_idx}] NOTE: no '[schedule]' banner seen — verify your print line.")

    if not history:
        print(f"[trial {display_idx}] NOTE: no {args.metric} acc parsed. See {log_path}")

    return best, history

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", default="configdrop-schedule")
    ap.add_argument("--storage", default="sqlite:///optuna_studies/configdrop.db")
    ap.add_argument("--n-trials", type=int, default=40)
    ap.add_argument("--timeout", type=int, default=None)
    ap.add_argument("--target-acc", type=float, default=0.90)
    ap.add_argument("--metric", choices=["val","test"], default="test")  # default = TEST accuracy
    args = ap.parse_args()

    (PROJECT_ROOT / "optuna_studies").mkdir(exist_ok=True)
    (PROJECT_ROOT / "tune_configdrop_runs").mkdir(exist_ok=True)
    (PROJECT_ROOT / "cifar10_results" / "mobilenet_v2_optuna").mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        study_name=args.study,
        storage=args.storage,
        direction="maximize",
        load_if_exists=True,
        pruner=optuna.pruners.SuccessiveHalvingPruner(min_resource=5, reduction_factor=3),
        sampler=optuna.samplers.TPESampler(seed=0, multivariate=True, group=True),
    )

    print(f"[study] {study.study_name} @ {args.storage}")
    print("[objective] maximizing", args.metric.upper(), "accuracy")

    # session-local counter
    run_counter = {"i": 0}

    def objective(trial: optuna.Trial) -> float:
        run_counter["i"] += 1
        local_idx = run_counter["i"]          # 1,2,3,... within THIS run

        schedule = suggest_schedule(trial)

        # display_idx = session-local; storage_idx = global trial.number
        best, history = run_trial(
            args,
            display_idx=local_idx,
            storage_idx=trial.number,
            schedule=schedule
        )

        for ep, acc in history:
            trial.report(acc, step=ep)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if best >= args.target_acc:
            trial.study.stop()

        return float(best)

    study.optimize(
        objective,
        n_trials=args.n_trials,
        timeout=args.timeout,
        gc_after_trial=True,
        show_progress_bar=True
    )

    out = PROJECT_ROOT / "best_configdrop_schedule.json"
    out.write_text(json.dumps({
        "best_value": study.best_value,
        "best_params": study.best_params
    }, indent=2))

    print("Best acc:", study.best_value)
    print("Best schedule params:", study.best_params)
    print("Saved:", out)

if __name__ == "__main__":
    main()
