"""
The predictor

Parameter 1: a path to the models for different clasification dimensions.
Parameter 2: a file containing a list of requirements IDs for wich we do
             not want to have predictions, g.g. because they have been
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


def clean(requirements):
    return [re.sub(r'^K\d+ ', '', s) for s in requirements]


if len(sys.argv) != 5:
    print("Usage: python predict_agreement.py <models path> <filter.csv> <input.csv> <output.csv>")
    sys.exit(1)

models_path = Path(sys.argv[1])
filter_file = sys.argv[2]
input_file = sys.argv[3]
output_file = sys.argv[4]

with open(filter_file, mode='r') as file:
    csv_reader = csv.reader(file)
    filter_ids = [row[0] for row in csv_reader]

print(f'Filtering the following {len(filter_ids)} requirements: {filter_ids}')

with open(input_file, mode='r') as file:
    csv_reader = csv.reader(file)

    rows = [row for row in csv_reader if row[0] not in filter_ids]
    ids = [sublist[0] for sublist in rows]
    requirements = clean([sublist[1] for sublist in rows])

results = zip(ids, requirements)

for item in models_path.iterdir():
    if item.is_dir():
        model = SetFitModel.from_pretrained(item)

        predictions = model.predict(requirements[1:])
        predictions_list = predictions.tolist()

        dimension = item.name
        agreements = sum(predictions_list)
        total = len(predictions_list)
        print(f'Classified {total} requirements with {agreements} '
              f'agreements and {total - agreements} disagreements '
              f'in dimension {dimension}.')

        predictions_list.insert(0, dimension)
        results = [old_tuple + (prediction,) for old_tuple, prediction
                   in zip(results, predictions_list)]

        del model

with open(output_file, mode='w') as file:
    writer = csv.writer(file)
    writer.writerows(results)
