from setfit import TrainingArguments
from setfit import SetFitModel
from setfit import SetFitModelCardData
from setfit import Trainer
from datasets import load_dataset
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
import statistics as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import MultipleLocator
from pathlib import Path
import torch
import logging
import argparse
import json
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def evaluate(y_pred, y_true):
    return {
        "f1_micro": f"{f1_score(y_true, y_pred, average='micro'):.3f}",
        "f1_macro": f"{f1_score(y_true, y_pred, average='macro'):.3f}",
        "f1_weighted": f"{f1_score(y_true, y_pred, average='weighted'):.3f}",
        "precision_micro": f"{precision_score(y_true, y_pred, average='micro', zero_division=0.0):.3f}",
        "precision_macro": f"{precision_score(y_true, y_pred, average='macro', zero_division=0.0):.3f}",
        "precision_weighted": f"{precision_score(y_true, y_pred, average='weighted', zero_division=0.0):.3f}",
        "recall_micro": f"{recall_score(y_true, y_pred, average='micro', zero_division=0.0):.3f}",
        "recall_macro": f"{recall_score(y_true, y_pred, average='macro', zero_division=0.0):.3f}",
        "recall_weighted": f"{recall_score(y_true, y_pred, average='weighted', zero_division=0.0):.3f}",
        "accuracy": f"{accuracy_score(y_true, y_pred):.3f}"
    }


def result_stats(scores):
    f1_micro = [float(d["f1_micro"]) for d in scores]
    f1_macro = [float(d["f1_macro"]) for d in scores]
    f1_weighted = [float(d["f1_weighted"]) for d in scores]
    precision_micro = [float(d["precision_micro"]) for d in scores]
    precision_macro = [float(d["precision_macro"]) for d in scores]
    precision_weighted = [float(d["precision_weighted"]) for d in scores]
    recall_micro = [float(d["recall_micro"]) for d in scores]
    recall_macro = [float(d["recall_macro"]) for d in scores]
    recall_weighted = [float(d["recall_weighted"]) for d in scores]
    accuracy = [float(d["accuracy"]) for d in scores]

    return {
        "f1_micro_avg": st.mean(f1_micro),
        "f1_micro_stdev": st.stdev(f1_micro),
        "f1_macro_avg": st.mean(f1_macro),
        "f1_macro_stdev": st.stdev(f1_macro),
        "f1_weighted_avg": st.mean(f1_weighted),
        "f1_weighted_stdev": st.stdev(f1_weighted),
        "precision_micro_avg": st.mean(precision_micro),
        "precision_micro_stdev": st.stdev(precision_micro),
        "precision_macro_avg": st.mean(precision_macro),
        "precision_macro_stdev": st.stdev(precision_macro),
        "precision_weighted_avg": st.mean(precision_weighted),
        "precision_weighted_stdev": st.stdev(precision_weighted),
        "recall_micro_avg": st.mean(recall_micro),
        "recall_micro_stdev": st.stdev(recall_micro),
        "recall_macro_avg": st.mean(recall_macro),
        "recall_macro_stdev": st.stdev(recall_macro),
        "recall_weighted_avg": st.mean(recall_weighted),
        "recall_weighted_stdev": st.stdev(recall_weighted),
        "accuracy_avg": st.mean(accuracy),
        "accuracy_stdev": st.stdev(accuracy)
    }

# The best parameters stem from the cross-validation study.

best_parameters = {}

best_parameters["target"] = {
    "body_learning_rate": 1.46948781073726e-05,
    "num_epochs": 2,
    "batch_size": 32,
    "seed": 29,
    "max_iter": 154,
    "solver": "lbfgs"
}

best_parameters["nature"] = {
    "body_learning_rate": 2.9110595020703845e-05,
    "num_epochs": 2,
    "batch_size": 16,
    "seed": 18,
    "max_iter": 100,
    "solver": "lbfgs"
}

best_parameters["interpretability"] = {
    "body_learning_rate": 4.653173233065894e-05,
    "num_epochs": 2,
    "batch_size": 16,
    "seed": 28,
    "max_iter": 225,
    "solver": "liblinear"
}

best_parameters["reference"] = {
    "body_learning_rate": 7.75502030284365e-05,
    "num_epochs": 2,
    "batch_size": 16,
    "seed": 14,
    "max_iter": 184,
    "solver": "lbfgs"
}

# We train the classifier, using the previously learned best parameters,
# with different amounts of data and evaluate the performance with cross-validation.
# Training increments are 20 samples and we start with atraining size of 20.
# The training is repeated 5 times with randomly sampled data to reduce sampling
# bias.

def training_size_study(path):
    logPath = path / 'training_set_size_study'
    logPath.mkdir(parents=True, exist_ok=True)

    logPrefix = f'{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    logFile = logPath / f'{logPrefix}.log'

    file_handler = logging.FileHandler(logFile)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(funcName)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    dataset = load_dataset("csv", data_files="../data/cv_250_samples/Consolidated and validated results_20250409.csv", split='train')

    training_increment = 20
    train_sizes = range(20, len(dataset), training_increment)
    n_repeats = 5
    k = 5

    logger.info(f'Train sizes: {list(train_sizes)}. Repeats: {n_repeats}. Folds: {k}')

    train_counter = 1
    models_to_train = len(best_parameters) * len(list(train_sizes)) * n_repeats * k
    results = []

    for key, value in best_parameters.items():
        dim = key
        params = value

        max_iter = params.get("max_iter", 100)
        solver = params.get("solver", "liblinear")
        model_params = {
            "head_params": {
                "max_iter": max_iter,
                "solver": solver,
            }
        }

        print(f'Training and evaluating dimension: {dim}')

        dataset = dataset.class_encode_column(dim)
        ds = dataset.select_columns(['text', dim])
        ds = ds.rename_column(dim, 'label')

        for size in train_sizes:
            for seed in range(n_repeats):
                ds_subset = ds.train_test_split(train_size=size,
                                                stratify_by_column='label',
                                                shuffle=True,
                                                seed=seed)['train']

                cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
                cv_result = []
                for index, (train_idx, test_idx) in enumerate(cv.split(ds_subset['text'], ds_subset['label']), start=1):
                    ds_train = ds_subset.select(train_idx)
                    ds_test = ds_subset.select(test_idx)

                    model = SetFitModel.from_pretrained("KBLab/sentence-bert-swedish-cased",
                                                        model_card_data=SetFitModelCardData(
                                                            language="sv",
                                                            license="apache-2.0",
                                                            dataset_name=f"TRVInfra {size} labels",
                                                        ),
                                                        **model_params)

                    trainer = Trainer(
                        model=model,
                        args=TrainingArguments.from_dict(params, ignore_extra=True),
                        train_dataset=ds_train,
                        eval_dataset=ds_test,
                        metric=evaluate
                    )

                    logger.info(f'*****{train_counter}/{models_to_train}*****')
                    logger.info(f'Dimension: {dim}')
                    logger.info(f'Data size: {size} (train: {len(ds_train)}, test: {len(ds_test)})')
                    logger.info(f'Seed: {seed}')
                    logger.info(f'CV iteration: {index}/{k}')
                    trainer.train()
                    train_counter += 1
                    cv_result.append(trainer.evaluate())

                    del trainer
                    torch.cuda.empty_cache()

                results.append({
                    'dimension': dim,
                    'train_size': size,
                    'seed': seed,
                    **result_stats(cv_result)
                })


    df = pd.DataFrame(results)
    mean_df = df.groupby(['dimension', 'train_size']).agg({
        'f1_macro_avg': 'mean', 'f1_macro_stdev': 'mean',
        'f1_micro_avg': 'mean', 'f1_micro_stdev': 'mean',
        'f1_weighted_avg': 'mean', 'f1_weighted_stdev': 'mean',
        'recall_macro_avg': 'mean', 'recall_macro_stdev': 'mean',
        'recall_micro_avg': 'mean', 'recall_micro_stdev': 'mean',
        'recall_weighted_avg': 'mean', 'recall_weighted_stdev': 'mean',
        'precision_macro_avg': 'mean', 'precision_macro_stdev': 'mean',
        'precision_micro_avg': 'mean', 'precision_micro_stdev': 'mean',
        'precision_weighted_avg': 'mean', 'precision_weighted_stdev': 'mean',
        'accuracy_avg': 'mean', 'accuracy_stdev': 'mean'
    }).reset_index()

    logging.info(mean_df.to_json(orient='records'))

    plt.figure(figsize=(10, 6))
    for name, group in mean_df.groupby('dimension'):
        plt.errorbar(group['train_size'],
                     group['f1_macro_avg'],
                     yerr=group['f1_macro_stdev'],
                     label=f'{name}',
                     fmt='-o', capsize=5)
    plt.title('Learning Curve')
    plt.xlabel('Training Set Size (n)')
    plt.ylabel('Macro F1 Score')
    plt.legend()
    plt.grid(True)
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_major_locator(MultipleLocator(10))
    plt.savefig(logPath / f'{logPrefix}.png')
    plt.savefig(logPath / f'{logPrefix}.pdf')


def create_learning_curves(logfile):
    with open(logfile, 'r') as f:
        lines = f.readlines()

    json_lines = [line.split(' - ')[-1].strip() for line in lines if line.split(' - ')[-1].strip().startswith('[{')]
    for i, json_str in enumerate(json_lines):
        try:
            data = json.loads(json_str)
            df = pd.DataFrame(data)
            df.to_csv(logfile.with_suffix('.csv'), index=False)
            # TODO: Create additional learning curve diagrams with the data, based on what we need in report/paper.
        except json.JSONDecodeError:
            print(f'Skipping invalid JSON at line {i}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("logpath", type=str, help="Path to the log file. If it is a file, it is parsed to generate learning curve diagrams.")
    args = parser.parse_args()
    path = Path(args.logpath)

    if path.is_file():
        create_learning_curves(Path(path))
    else:
        training_size_study(path)
