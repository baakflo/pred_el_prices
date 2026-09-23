"""Optuna search for the quantile network (`qnn-tune`).

Each trial backtests one configuration on the validation span (default 2025,
refit every 4 weeks, 2 seeds, rolling window and fuel scaling as in the level
fix) and scores the mean pinball loss over the 99 percentiles in EUR/MWh. 2026
stays untouched for the holdout comparison. Trials run one after another by
default, each spreading its refits x seeds over `n_jobs` worker processes
(`n_parallel` > 1 runs trials in threads that share joblib's process pool).

Artifacts: trials.csv, best_params.json, study.db (resumable).
"""

import json
from pathlib import Path

import optuna

from pred_el_prices.experiments.qnn_de import backtest, pinball_by_quantile
from pred_el_prices.models.qnn import QNNConfig


def run(
    out_dir: Path,
    n_trials: int = 60,
    n_parallel: int = 1,
    n_jobs: int = -1,
    val_start: str = "2025-01-01",
    val_end: str = "2025-12-31",
    n_seeds: int = 2,
    dataset_path: str = "data/dataset/hourly.parquet",
    train_start: str = "2018-12-01",
) -> dict:
    def objective(trial: optuna.Trial) -> float:
        n_layers = trial.suggest_int("n_layers", 1, 3)
        width = trial.suggest_categorical("width", [64, 128, 256, 512])
        config = QNNConfig(
            hidden=[width] * n_layers,
            dropout=trial.suggest_float("dropout", 0.0, 0.4),
            lr=trial.suggest_float("lr", 1e-4, 3e-3, log=True),
            weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            batch_size=trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
        )
        window_days = trial.suggest_categorical("window_days", [365, 548, 730, 1095, 1460])
        qdf = backtest(
            dataset_path,
            train_start,
            [],
            config,
            val_start,
            val_end,
            "4weeks",
            window_days,
            True,
            n_seeds,
            n_jobs,
            verbose=0,
        )
        score = float(pinball_by_quantile(qdf).mean())
        print(f"trial {trial.number}: pinball {score:.4f} {trial.params}", flush=True)
        return score

    study = optuna.create_study(
        study_name="qnn",
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=0),
        storage=f"sqlite:///{out_dir / 'study.db'}",
        load_if_exists=True,
    )
    # the level-fix configuration as the first trial, so the search starts from it
    study.enqueue_trial(
        {
            "n_layers": 2,
            "width": 256,
            "dropout": 0.1,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "batch_size": 32,
            "window_days": 730,
        }
    )
    study.optimize(objective, n_trials=n_trials, n_jobs=n_parallel)

    study.trials_dataframe().to_csv(out_dir / "trials.csv", index=False)
    best = {"pinball": study.best_value, **study.best_params}
    (out_dir / "best_params.json").write_text(json.dumps(best, indent=2))
    baseline = study.trials[0].value
    return {
        "n_trials": len(study.trials),
        "validation": [val_start, val_end],
        "baseline_pinball": baseline,
        "best": best,
    }
