# Overview

This repository contains code developed for the [SVAR project](https://www.bth.se/eng/research/research-areas/software-engineering/requirements-engineering/systematic-verification-and-acceptance-of-requirements-svar/).

# Supervised and unsupervised classifier implementations

## SetFit classifier (supervised)
See the [README](setfitclassifier/README.md) in setfitclassifier for more information.

## GPT classifier (unsupervised)
See the [README](gptclassifier/README.md) in gptclassifier for more information.

## Classifier demonstrator
See the [README](classifierdemo/README.md) in classifierdemo for more information.

# Additional code
We have developed some auxiliary code to facilitate the main objectives.

## Requirements complete or not
The DOORS NG [export](data/TRVInfra_all_only_requirements_202312.xlsx) resulted in requirements that were truncated, making them incomplete and not useable for further processing. Our approach to filter out incomplete requirements was to train a classifier on 3000 manually labeled requirements. The labels are stored in the same file, different sheet.

On the test set we achieve an accuracy of 98%, which is sufficient to use the classifier as a predictor for the remaining 18300 requirements. The resulting [dataset](data/trvinfra_requirements_all_complete.csv) consists of 17622 complete requirements.

The code for this classifier can be found [here](rcon/).

## Agreement prediction
After two rounds of labeling, we have collected data on 72 requirements where the labelers agree or disagree. We use this information to train a classifier to predict agreement. We do this so that a single labeler can proceed and sample from the requirements that are predicted to have agreed labels.

We use again [nested cross-fold validation](setfitclassifier/nested_crossvalidation.py) to identify the hyper-parameters for the classifier and evaluate its performance. The [code to train](setfitclassifier/train_agreement.py) the model uses the learned hyper-parameters. The resulting model is then used to [predict](setfitclassifier/predict_agreement.py) the agreement.

The input and results of this analysis can be found [here](data/agreement_classification).

