# Overview

This repository contains code developed for the [SVAR project](https://www.bth.se/eng/research/research-areas/software-engineering/requirements-engineering/systematic-verification-and-acceptance-of-requirements-svar/).

# Supervised and unsupervised classifier implementations

## SETFIT classifier (supervised)


## GPT classifier (unsupervised)

## Classifier demonstrator

# Additional code
We have developed some auxiliary code to facilitate the main objectives.

## Requirements complete or not
The DOORS NG export resulted in requirements that were truncated, making them incomplete and not useable for further processing. Our approach to filter out incomplete requirements was to train a classifier on 3000 manually labeled requirements.

On the test set we achieve an accuracy of 98%, which is sufficient to use the classifier as a predictor for the remaining 18300 requirements. The resulting dataset consists of 17622 complete requirements.

The code for this classifier can be found [here](rcon/).



