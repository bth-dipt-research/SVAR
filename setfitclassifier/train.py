from setfit import TrainingArguments
from setfit import SetFitModel
from setfit import SetFitModelCardData
from setfit import Trainer
from datasets import load_dataset
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
run_dir = "../models/verifiability" / Path(current_time)
run_dir.mkdir(parents=True, exist_ok=True)

dataset = load_dataset("csv", data_files="../data/cv_250_samples/Consolidated and validated results_20250409.csv")

best_parameters = {}

best_parameters["target"] = {
    "body_learning_rate": 1.4287462741998545e-05,
    "num_epochs": 2,
    "batch_size": 32,
    "seed": 23,
    "max_iter": 160,
    "solver": "lbfgs"
}

best_parameters["nature"] = {
    "body_learning_rate": 1.9511727001353293e-05,
    "num_epochs": 2,
    "batch_size": 16,
    "seed": 22,
    "max_iter": 194,
    "solver": "lbfgs"
}

best_parameters["interpretability"] = {
    "body_learning_rate": 0.00017650586856170257,
    "num_epochs": 2,
    "batch_size": 16,
    "seed": 22,
    "max_iter": 166,
    "solver": "liblinear"
}

best_parameters["reference"] = {
    "body_learning_rate": 6.27673051627565e-05,
    "num_epochs": 1,
    "batch_size": 16,
    "seed": 30,
    "max_iter": 121,
    "solver": "liblinear"
}


for key, value in best_parameters.items():
    dim = key
    params = value

    print(f'Training for dimension: {dim}')
    labels = set(dataset['train'][dim])

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
                                        model_card_data=SetFitModelCardData(
                                            language="sv",
                                            license="apache-2.0",
                                            dataset_name="TRVInfra 250 labels",
                                        ),
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
