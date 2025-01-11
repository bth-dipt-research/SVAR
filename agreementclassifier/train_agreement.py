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

# Best fold: 7
# Accuracy: 0.8571428571428571
# Nested CV average accuracy: 0.6321428571428571
# Nested CV std deviation: 0.14965947742683544
best_parameters["target"] = {
    "body_learning_rate": 1.3229142620339052e-06,
    "num_epochs": 4,
    "batch_size": 8,
    "seed": 31,
    "max_iter": 57,
    "solver": "liblinear"
}

# Best fold: 4
# Accuracy: 0.8571428571428571
# Nested CV average accuracy: 0.7357142857142857
# Nested CV std deviation: 0.0769309258162072
best_parameters["nature"] = {
    "body_learning_rate": 0.000955332145145954,
    "num_epochs": 2,
    "batch_size": 16,
    "seed": 34,
    "max_iter": 155,
    "solver": "liblinear"
}

# Best fold: 7
# Accuracy: 0.8571428571428571
# Nested CV average accuracy: 0.6678571428571428
# Nested CV std deviation: 0.08718968296952645
best_parameters["interpretability"] = {
    "body_learning_rate": 0.0006596212557964204,
    "num_epochs": 2,
    "batch_size": 16,
    "seed": 2,
    "max_iter": 215,
    "solver": "liblinear"
}

# Best fold: 3
# Accuracy: 1.0
# Nested CV average accuracy: 0.8053571428571429
# Nested CV std deviation: 0.11306547227340522
best_parameters["reference"] = {
    "body_learning_rate": 6.158460454793998e-06,
    "num_epochs": 5,
    "batch_size": 8,
    "seed": 10,
    "max_iter": 298,
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
