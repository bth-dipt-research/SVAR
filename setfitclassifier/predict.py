"""
The predictor

Parameter 1: a path to the models for different classification dimensions.
Parameter 2: a file containing a list of requirements IDs for wich we do
             not want to have predictions, e.g. because they have been
             used for training.
Parameter 3: The file containing the requirements for which we want to have
             predictions.
Parameter 4: The file where to write the result.
"""

from setfit import SetFitModel
import sys
import csv
import re
from pathlib import Path
import pandas as pd


def read_csv(filename, has_header=True):
    with open(filename, mode='r') as file:
        reader = csv.reader(file)
        if has_header:
            next(reader)
        return [row for row in reader]


def read_data(input_file, filter_file):
    input_file_content = read_csv(input_file)
    print(f'{len(input_file_content)} data points read.')

    filter_file_content = read_csv(filter_file)
    filter_ids = [row[0] for row in filter_file_content]
    print(f'{len(filter_ids)} data points filtered.')

    data = [row for row in input_file_content if row[0] not in filter_ids]
    print(f'{len(data)} data points remaining.')
    return data


if len(sys.argv) != 5:
    print("Usage: python predict.py <models path> <filter.csv> <input.csv> <output.csv>")
    sys.exit(1)

models_path = Path(sys.argv[1])
filter_file = sys.argv[2]
input_file = sys.argv[3]
output_file = sys.argv[4]


data = read_data(input_file, filter_file)

# Some requirements contain still the id, hence removing
for d in data:
    d[1] = re.sub(r'^K\d+ ', '', d[1])

df = pd.DataFrame(data, columns=['ID', 'requirement'])

for item in models_path.iterdir():
    if item.is_dir():
        model = SetFitModel.from_pretrained(item)
        dimension = item.name
        print(f'Predicting: {dimension}')
        predictions = model.predict(df['requirement'])
        df[dimension] = predictions.tolist()
        #agreements = sum(predictions_list)
        #total = len(predictions_list)
        #print(f'Classified {total} requirements with {agreements} '
        #      f'agreements and {total - agreements} disagreements '
        #      f'in dimension {dimension}.')

        #predictions_list.insert(0, dimension)
        #results = [old_tuple + (prediction,) for old_tuple, prediction
        #           in zip(results, predictions_list)]

        del model

df.to_csv(output_file, index=False)
