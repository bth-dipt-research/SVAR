from pathlib import Path
import json
import os
import re
import argparse
import pandas as pd
from pydantic import ValidationError
from sklearn.metrics import precision_score, recall_score, f1_score

from utils import Prediction

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('inputfile', type=str, help='Test set file')
    parser.add_argument('predictionsdir', type=str, help='Directory containing the predictions')

    args = parser.parse_args()
    inputfile = args.inputfile
    predictionsdir = Path(args.predictionsdir)

    ground_truth = pd.read_csv(inputfile, encoding='latin-1')
    dimensions = ['target', 'nature', 'interpretability', 'reference']
    evaluation = pd.DataFrame(columns=['model', 'dimension', 'average type', 'precision', 'recall', 'f1'])
    average_types = ['micro', 'macro', 'weighted']

    for file in predictionsdir.glob('predictions*.json'):
        match = re.search(r'predictions_(.+?)\.json', file.name)
        if match:
            model_name = match.group(1)
        else:
            model_name = 'unknown'

        with open(file, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        try:
            classifications = [Prediction(**item) for item in raw_data]
        except ValidationError:
            print(f'{file.name} contains invalid data')
            raise

        flat_data = [
            {
                'id': c.id,
                'requirement': c.requirement,
                'target': c.classification.target.value,
                'nature': c.classification.nature.value,
                'interpretability': c.classification.interpretability.value,
                'reference': c.classification.reference.value
            }
            for c in classifications
        ]

        predictions = pd.DataFrame(flat_data)

        for dimension in dimensions:
            y_true = ground_truth[dimension]
            y_pred = predictions[dimension]

            for avg in average_types:
                precision = precision_score(y_true, y_pred, average=avg)
                recall = recall_score(y_true, y_pred, average=avg)
                f1 = f1_score(y_true, y_pred, average=avg)

                evaluation.loc[len(evaluation)] = [model_name, dimension, avg, f'{precision:.3f}', f'{recall:.3f}', f'{f1:.3f}']


    evaluation.to_csv(predictionsdir / 'gptevaluation.csv', index=False)



