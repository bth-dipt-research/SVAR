#from typing import Dict, Union, Any
from setfit import TrainingArguments, SetFitModel, Trainer
from datasets import load_dataset, DatasetDict
from pathlib import Path
from datetime import datetime
import torch

# -------------------------------------
# Instructions
#
# 1. Identify with nvidia-smi which GPUs are available
# 2. Limit, if needed, to run only on a specific GPU, e.g. with index 0,
#    by invoking CUDA_VISIBLE_DEVICES=0 python train_agreement.py
# -------------------------------------

current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
run_dir = "../models/agreement" / Path(current_time)

dataset = load_dataset("csv", data_files="../data/all_agree_statistics.csv")

best_parameters = {}

best_parameters["target"] = {
    "body_learning_rate": 4.35E-05,
    "num_epochs": 3,
    "batch_size": 16,
    "seed": 24,
    "max_iter": 162,
    "solver": "newton-cg"
}

best_parameters["nature"] = {
    "body_learning_rate": 2.18E-04,
    "num_epochs": 3,
    "batch_size": 16,
    "seed": 21,
    "max_iter": 172,
    "solver": "liblinear"
}

best_parameters["interpretability"] = {
    "body_learning_rate": 0.000223843975299691,
    "num_epochs": 3,
    "batch_size": 8,
    "seed": 21,
    "max_iter": 206,
    "solver": "liblinear"
}

best_parameters["reference"] = {
    "body_learning_rate": 6.56E-05,
    "num_epochs": 4,
    "batch_size": 8,
    "seed": 22,
    "max_iter": 176,
    "solver": "newton-cg"
}


for key, value in best_parameters.items():
    dim = key
    params = value

    print(f'Training for dimension: {dim}')

    dataset = dataset.class_encode_column(dim)

    max_iter = params.get("max_iter", 100)
    solver = params.get("solver", "liblinear")
    model_params = {
        "head_params": {
            "max_iter": max_iter,
            "solver": solver,
        }
    }

    print(f'Model parameters: {model_params}')

    model = SetFitModel.from_pretrained("KBLab/sentence-bert-swedish-cased",
                                        **model_params)

    trainer = Trainer(
        model=model,
        train_dataset=dataset["train"],
        column_mapping={
            "text": "text",
            dim: "label"
        }
    )

    arguments = TrainingArguments.from_dict(params, ignore_extra=True)

    print(f'Training arguments: {arguments}')

    trainer.train(arguments)

    model_path = run_dir / dim
    trainer.model.save_pretrained(model_path)

    del trainer
    torch.cuda.empty_cache()
