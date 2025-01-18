"""
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
from scipy import stats
import gc
import os
import logging
import time
import argparse

logger = logging.getLogger()
logger.setLevel(logging.INFO)
slogger = logging.getLogger("setfit")
ologger = logging.getLogger("optuna")

parser = argparse.ArgumentParser()
parser.add_argument("name", type=str, help="Name of the classification. Used as a prefix for the log file")
parser.add_argument("logpath", type=str, help="Path to store the log file")
parser.add_argument("datafile", type=str, help="The csv file with the training data")
parser.add_argument("dimensions", nargs="*", help="The dimensions in the data file to train on")
parser.add_argument("-o", "--outer", type=int, default=10, help="The number of outer folds")
parser.add_argument("-i", "--inner", type=int, default=5, help="The number of inner folds")
parser.add_argument("-t", "--trials", type=int, default=100, help="The number of trials")

args = parser.parse_args()

logFile = Path(args.logpath) / f'{args.name}_ncv-{datetime.now().strftime("%Y%m%d-%H%M%S")}.log'

file_handler = logging.FileHandler(logFile)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(funcName)s - %(message)s")
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)
slogger.addHandler(file_handler)
ologger.addHandler(file_handler)

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

args_str = f"Arguments: {vars(args)}"
print(args_str)
logger.info(args_str)

# -------------------------------------
# Instructions
#
# 1. Identify with nvidia-smi which GPUs are available
# 2. Limit, if needed, to run only on a specific GPU, e.g. with index 0,
#    by invoking CUDA_VISIBLE_DEVICES=0 python nested_crossvalidation.py ARGS
# -------------------------------------


# -------------------------------------
# Configuration
# -------------------------------------
dataset = load_dataset("csv", data_files=args.datafile, split="train")
k_outer = args.outer
k_inner = args.inner
n_trials = args.trials

dimensions = args.dimensions

# -------------------------------------
# Time keeping and progress estimation
# -------------------------------------
num_dimensions = len(dimensions)
inner_fits = k_outer * k_inner * n_trials * num_dimensions
outer_fits = k_outer * num_dimensions
total_fits = inner_fits + outer_fits
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

    for index, (inner_train_idx, inner_val_idx) in enumerate(inner_skf.split(features, labels), start=1):
        global done_fits
        start_time = time.perf_counter()
        logger.info(f"Fit {done_fits + 1}/{total_fits}")

        inner_train_dataset = outer_train_dataset.select(inner_train_idx)
        inner_val_dataset = outer_train_dataset.select(inner_val_idx)

        logger.info(f"Inner fold: {index}")
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
        logger.info(f"Inner evaluation (fold {index}): {metrics}")
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
    for index, (run, score) in enumerate(zip(result["best_runs"], result["scores"]), start=1):
        logger.info(f"Best hyperparameters found in fold {index} with accuracy {score}: {run}")
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

    for index, (train_idx, test_idx) in enumerate(outer_skf.split(features, labels), start=1):
        outer_train_dataset = dim_dataset.select(train_idx)
        outer_test_dataset = dim_dataset.select(test_idx)

        logger.info(f"Outer fold {index}")
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
        logger.info(f"Outer evaluation (fold {index}): {metrics}")
        outer_scores.append(metrics["accuracy"])

        del trainer
        torch.cuda.empty_cache()
        gc.collect()

        end_time = time.perf_counter()
        fit_durations.append(end_time - start_time)
        done_fits = done_fits + 1
        logger.info(f"Estimated time left: {human_readable_time(estimated_time_left())}.")
        logger.info(f"Estimated finish time: {estimated_time_finished().strftime("%Y-%m-%d %H:%M:%S")}.")

    outer_scores = np.round(outer_scores, decimals=3)
    results[dim] = {
        "best_runs": best_runs,
        "scores": outer_scores,
        "avg_accuracy": np.round(np.mean(outer_scores), decimals=3),
        "std_dev": np.round(np.std(outer_scores), decimals=3)
    }

    log_result(dim, results[dim])

logger.info("------ Parameters for final model training ------")
for dim in results:
    result = results[dim]
    best_runs = result["best_runs"]

    learning_rate = np.mean([r["body_learning_rate"] for r in best_runs])
    num_epochs = np.round(np.mean([r["num_epochs"] for r in best_runs]))
    batch_size = stats.mode([r["batch_size"] for r in best_runs])
    seed = np.round(np.mean([r["seed"] for r in best_runs]))
    max_iter = np.round(np.mean([r["max_iter"] for r in best_runs]))
    solvers = [r["solver"] for r in best_runs]
    unique_solvers, counts = np.unique(solvers, return_counts=True)
    solver = unique_solvers[np.argmax(counts)]

    logger.info(f"Dimension: {dim}")
    logger.info(f'Body learning rate: {learning_rate}')
    logger.info(f'Epochs: {num_epochs}')
    logger.info(f'Batch size: {batch_size.mode}')
    logger.info(f'Seed: {seed}')
    logger.info(f'Max iter: {max_iter}')
    logger.info(f'Solver: {solver}')
    logger.info("------")
