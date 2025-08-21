from setfit import SetFitModel
from sklearn.metrics import precision_score, recall_score, f1_score
import argparse
from pathlib import Path
import pandas as pd

def predict():
    for item in modelsdir.iterdir():
        if item.is_dir():
            model = SetFitModel.from_pretrained(item)
            dimension = item.name
            print(f'Predicting: {dimension}')
            predictions = model.predict(df['requirement'])
            df[f'{dimension} - predicted'] = predictions.tolist()

            del model

def evaluate():
    average_types = ['micro', 'macro', 'weighted']
    dimensions = ['target', 'nature', 'interpretability', 'reference']
    result = pd.DataFrame(columns=['dimension', 'average type', 'precision', 'recall', 'f1'])

    for dimension in dimensions:
        y_true = df[dimension]
        y_pred = df[f'{dimension} - predicted']
        for avg in average_types:
            precision = precision_score(y_true, y_pred, average=avg)
            recall = recall_score(y_true, y_pred, average=avg)
            f1 = f1_score(y_true, y_pred, average=avg)
            result.loc[len(result)] = [dimension, avg, f'{precision:.3f}', f'{recall:.3f}', f'{f1:.3f}']

    print(result)
    result.to_csv(evaluation, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('inputfile', type=str, help='Test set file')
    parser.add_argument('modelsdir', type=str, help='Directory containing the prediction models')
    parser.add_argument('predictions', type=str, help='File to write the predictions')
    parser.add_argument('evaluation', type=str, help='File to write the evaluation')

    args = parser.parse_args()
    inputfile = Path(args.inputfile)
    modelsdir = Path(args.modelsdir)
    predictions = Path(args.predictions)
    evaluation = Path(args.evaluation)

    df = pd.read_csv(inputfile, encoding='latin-1')
    predict()
    df.to_csv(predictions, index=False)
    evaluate()
