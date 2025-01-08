"""
We train the classifier on the agreement data that we collected in pilot 1 and 2. In total, we classified 72 requirements and the agreement statistics are stored in data/all_agree_statistics.csv.

We create a classifier for each dimension. We use nested cross-validation in order to evaluate the performance of the classifier.

Inspiration:
https://machinelearningmastery.com/nested-cross-validation-for-machine-learning-with-python/
"""

import optuna
from typing import Dict, Union, Any
from setfit import TrainingArguments, SetFitModel, Trainer
from datasets import load_dataset, DatasetDict
from sklearn.model_selection import StratifiedKFold
from pathlib import Path
from datetime import datetime, timedelta
import torch
import numpy as np
import gc
import os
import logging
import time

logger = logging.getLogger()
logger.setLevel(logging.INFO)
slogger = logging.getLogger("setfit")
ologger = logging.getLogger("optuna")

file_handler = logging.FileHandler(f"../data/agreement_ncv-{datetime.now().strftime("%Y%m%d-%H%M%S")}.log")
formatter = logging.Formatter("%(asctime)s - %(name)s - %(funcName)s - %(message)s")
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)
slogger.addHandler(file_handler)
ologger.addHandler(file_handler)

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# -------------------------------------
# Configuration
# -------------------------------------
dataset = load_dataset("csv", data_files="../data/all_agree_statistics.csv", split="train")
k_outer = 10
k_inner = 5
n_trials = 100

dimensions = ["target", "nature", "interpretability", "reference"]

# -------------------------------------
# Time keeping and progress estimation
# -------------------------------------
total_fits = k_outer * k_inner * n_trials * len(dimensions)
done_fits = 0
fit_durations = []

def avg_fit_durations():
    return np.mean(fit_durations) if len(fit_durations) > 0 else 100

def estimated_time_left():
    return avg_fit_durations() * (total_fits - done_fits)

def estimated_time_finished():
    return datetime.now() + timedelta(seconds=estimated_time_left())

def human_readable_time(seconds):
    units = [
        ("days", 86400),  # 86400 seconds in a day
        ("hours", 3600),  # 3600 seconds in an hour
        ("minutes", 60),  # 60 seconds in a minute
        ("seconds", 1),
    ]
    parts = []
    for name, unit in units:
        value, seconds = divmod(seconds, unit)
        if value > 0:
            parts.append(f"{value:.0f} {name}")
    return ", ".join(parts)


# -------------------------------------
# Function to initialize a model
# -------------------------------------
def model_init(params: Dict[str, Any]) -> SetFitModel:
    params = params or {}
    max_iter = params.get("max_iter", 100)
    solver = params.get("solver", "liblinear")
    params = {
        "head_params": {
            "max_iter": max_iter,
            "solver": solver,
        }
    }

    logger.info(f"Model parameters: {params}")

    return SetFitModel.from_pretrained("KBLab/sentence-bert-swedish-cased", **params)

# -------------------------------------
# Function for inner hyperparameter tuning
# -------------------------------------
def inner_objective(trial, outer_train_dataset, k_inner):
    # Define search space for hyperparameters
    training_args = TrainingArguments(
        body_learning_rate = trial.suggest_float("body_learning_rate", 1e-6, 1e-3, log=True),
        num_epochs = trial.suggest_int("num_epochs", 1, 5),
        batch_size = trial.suggest_categorical("batch_size", [8, 16]),
        seed = trial.suggest_int("seed", 1, 40)
    )

    model_params = {
        "max_iter": trial.suggest_int("max_iter", 50, 300),
        "solver": trial.suggest_categorical("solver", ["newton-cg", "lbfgs", "liblinear"])
    }

    # Use K-fold CV on the inner training data
    inner_skf = StratifiedKFold(n_splits=k_inner, shuffle=True, random_state=1)
    inner_scores = []

    features = np.array(outer_train_dataset["text"])
    labels = np.array(outer_train_dataset["label"])

    for fold, (inner_train_idx, inner_val_idx) in enumerate(inner_skf.split(features, labels)):
        global done_fits
        start_time = time.perf_counter()
        logger.info(f"Fit {done_fits + 1}/{total_fits}")

        inner_train_dataset = outer_train_dataset.select(inner_train_idx)
        inner_val_dataset = outer_train_dataset.select(inner_val_idx)

        logger.info(f"Inner fold: {fold + 1}")
        logger.info(f"Train size: {len(inner_train_dataset)}")
        logger.info(f"Test size: {len(inner_val_dataset)}")
        logger.info(f"Train class distribution: {np.bincount(inner_train_dataset['label'])}")
        logger.info(f"Validation class distribution: {np.bincount(inner_val_dataset['label'])}")

        trainer = Trainer(
            model_init=lambda: model_init(model_params),
            args=training_args,
            train_dataset=inner_train_dataset,
            eval_dataset=inner_val_dataset,
            metric="accuracy"
        )

        # Train with the candidate hyperparameters()
        trainer.train()

        metrics = trainer.evaluate()
        logger.info(f"Inner evaluation (fold {fold + 1}): {metrics}")
        inner_scores.append(metrics["accuracy"])

        del trainer
        torch.cuda.empty_cache()
        gc.collect()
        end_time = time.perf_counter()
        fit_durations.append(end_time - start_time)
        done_fits = done_fits + 1
        logger.info(f"Estimated time left: {human_readable_time(estimated_time_left())}.")
        logger.info(f"Estimated finish time: {estimated_time_finished().strftime("%Y-%m-%d %H:%M:%S")}.")


    avg = np.mean(inner_scores)
    logger.info(f"Across inner fold average accuracy: {avg}")

    # Return the average accuracy across inner folds
    return avg

def tune_hyperparameters(outer_train_dataset, n_trials, k_inner):
    # Create an Optuna study for inner optimization
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: inner_objective(trial, outer_train_dataset, k_inner), n_trials=n_trials, gc_after_trial=True)
    return study.best_params

def log_result(dim, result):
    logger.info(f"Best runs found for {dim}:")
    for fold, run in enumerate(result["best_runs"]):
        logger.info(f"Best hyperparameters found in fold {fold + 1}: {run}")
    logger.info(f"Performance summary for {dim}:")
    logger.info(f"Nested CV average accuracy: {result['avg_accuracy']}")
    logger.info(f"Nested CV std deviation: {result['std_dev']}")

#----------------------------------------------------------
# Loop over dimensions, on each we perform cross-validation
#----------------------------------------------------------

results = {}

for dim in dimensions:
    logger.info(f"Cross-validation for dimension: {dim}")

    dataset = dataset.class_encode_column(dim)
    dim_dataset = dataset.select_columns(["text", dim])
    dim_dataset = dim_dataset.rename_column(dim, "label")

    features = np.array(dim_dataset["text"])
    labels = np.array(dim_dataset["label"])

    # -------------------------------------
    # Outer loop for nested cross-validation
    # -------------------------------------
    outer_skf = StratifiedKFold(n_splits=k_outer, shuffle=True, random_state=1)

    outer_scores = []
    best_runs = []

    for fold, (train_idx, test_idx) in enumerate(outer_skf.split(features, labels)):
        outer_train_dataset = dim_dataset.select(train_idx)
        outer_test_dataset = dim_dataset.select(test_idx)

        logger.info(f"Outer fold {fold + 1}")
        logger.info(f"Train size: {len(outer_train_dataset)}")
        logger.info(f"Test size: {len(outer_test_dataset)}")
        logger.info(f"Train class distribution: {np.bincount(outer_train_dataset['label'])}")
        logger.info(f"Test class distribution: {np.bincount(outer_test_dataset['label'])}")

        # -------------------------------------
        # Hyperparameter tuning (inner loop)
        # -------------------------------------
        best_run = tune_hyperparameters(outer_train_dataset, n_trials, k_inner)
        logger.info(f"Best hyperparameters found for {dim}: {best_run}")
        best_runs.append(best_run)

        start_time = time.perf_counter()
        logger.info(f"Fit {done_fits + 1}/{total_fits}")

        best_model_in_fold = model_init(
            {
                key: best_run[key] for key in ["max_iter", "solver"] if key in best_run
            }
        )

        best_training_args_in_fold = TrainingArguments(
            body_learning_rate = best_run["body_learning_rate"],
            num_epochs = best_run["num_epochs"],
            batch_size = best_run["batch_size"],
            seed = best_run["seed"]
        )

        trainer = Trainer(
            model=best_model_in_fold,
            args=best_training_args_in_fold,
            train_dataset=outer_train_dataset,
            eval_dataset=outer_test_dataset,
            metric="accuracy"
        )

        trainer.train()
        metrics = trainer.evaluate()
        logger.info(f"Outer evaluation (fold {fold + 1}): {metrics}")
        outer_scores.append(metrics["accuracy"])

        del trainer
        torch.cuda.empty_cache()
        gc.collect()

        end_time = time.perf_counter()
        fit_durations.append(end_time - start_time)
        done_fits = done_fits + 1
        logger.info(f"Estimated time left: {human_readable_time(estimated_time_left())}.")
        logger.info(f"Estimated finish time: {estimated_time_finished().strftime("%Y-%m-%d %H:%M:%S")}.")

    results[dim] = {
        "best_runs": best_runs,
        "avg_accuracy": np.mean(outer_scores),
        "std_dev": np.std(outer_scores)
    }

    log_result(dim, results[dim])

logger.info("------ Final results ------")
for dim in results:
    log_result(dim, results[dim])

