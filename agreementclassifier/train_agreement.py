"""
NOTE: this implementated is outdated and replaced by nested cross-validation.

We train the classifier on the agreement data that we collected in pilot 1 and 2. In total, we classified 72 requirements and the agreement statistics are stored in data/all_agree_statistics.csv.

We create a classifier for each dimension. The dataset is split into training (80%), validation (10%) and test data (10%). The test data will not be used for training nor parameter optimization.
"""


from optuna import Trial
from typing import Dict, Union, Any
from setfit import TrainingArguments, SetFitModel, Trainer
from datasets import load_dataset, DatasetDict
from pathlib import Path
from datetime import datetime
import torch

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
    return SetFitModel.from_pretrained("KBLab/sentence-bert-swedish-cased", **params)

def hp_space(trial: Trial) -> Dict[str, Union[float, int, str]]:
    return {
        "body_learning_rate": trial.suggest_float("body_learning_rate", 1e-6, 1e-3, log=True),
        "num_epochs": trial.suggest_int("num_epochs", 1, 5),
        "batch_size": trial.suggest_categorical("batch_size", [8, 16]),
        "seed": trial.suggest_int("seed", 1, 40),
        "max_iter": trial.suggest_int("max_iter", 50, 300),
        "solver": trial.suggest_categorical("solver", ["newton-cg", "lbfgs", "liblinear"]),
    }

current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
run_dir = "../models/agreement" / Path(current_time)

dataset = load_dataset("csv", data_files="../data/all_agree_statistics.csv")


dimensions = ["target", "nature", "interpretability", "reference"]

for dim in dimensions:
    print(f'Training for dimension: {dim}')

    dataset = dataset.class_encode_column(dim)

    train_testvalid = dataset['train'].train_test_split(test_size=0.2, stratify_by_column=dim)
    test_valid = train_testvalid['test'].train_test_split(test_size=0.5, stratify_by_column=dim)
    ttv_dataset = DatasetDict({
        'train': train_testvalid['train'],
        'test': test_valid['test'],
        'valid': test_valid['train']})

    trainer = Trainer(
        model_init=model_init,
        train_dataset=ttv_dataset["train"],
        eval_dataset=ttv_dataset["valid"],
        column_mapping={
            "text": "text",
            dim: "label"
        }
    )


    best_run = trainer.hyperparameter_search(direction="maximize", hp_space=hp_space, n_trials=200)
    print(best_run)

    trainer.apply_hyperparameters(best_run.hyperparameters, final_model=True)
    trainer.train()
    trainer.evaluate()
    trainer.evaluate(ttv_dataset['test'])

    model_path = run_dir / dim
    trainer.model.save_pretrained(model_path)

    del trainer
    torch.cuda.empty_cache()

