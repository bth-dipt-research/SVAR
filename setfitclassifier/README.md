# About

The classifier is implemented with [SetFit](https://huggingface.co/docs/setfit/index) which uses contrastive learning for fine-tuning a few-shot classifier.

The trainer (both for cross-validation and final model training) expects a CSV file of the following format:

| text | dimension 1 | dimensions 2 | ... | dimension n |
|------|-------------|--------------|-----|-------------|
| abc  | value       | value        |     | value       |

A `text` column with the data to classify and `n` dimensions on which the data is classified.

Currently, we provide the following:

 - Nested cross-validation for hyper-parameter tuning and an unbiased estimation of the classifiers performance
 - Training of the final model
 - Prediction on unseen data

## Agreement classifier

A classifier, trained on the agreement from the requirements verifiability pilots, that predicts agreements/disagreements in classification. The purpose of this is to split the dataset into:

 1. requirements where the likelihood of classification agreement is high. For these, only one judge makes the classification.
 2. requirements where the likelihood of classification agreement is low. For these, all judges discuss the classification.

We create a classifier for each dimensions: target, nature, interpretability and reference.

We train the classifier on the agreement data that we collected in pilot 1 and 2. In total, we classified 72 requirements and the agreement statistics are stored in [data/all_agree_statistics.csv](../data/all_agree_statistics.csv). This file encodes, for each dimension, if all the judges agreed on the classification (`TRUE`) or not (`FALSE`).

We create a classifier for each dimension. We use nested cross-validation in order to evaluate the performance of the classifier.

# Nested cross-validation

[Inspiration](https://machinelearningmastery.com/nested-cross-validation-for-machine-learning-with-python/)

Nested cross-validation combines two important aspects:
 1. It allows hyper-parameter tuning. This is important for optimizing the performance of SetFit, given the small amount of training data.
 2. Cross-validation allows to better estimate the performance of the classifier on unseen data. With a small amount of labeled data, we do not want to let chance influence the performance on the train-test split. We want to use all data to understand the performance variance of the classifier.

Nested cross-validation has a cost: we need to train and evaluate many more models. Assume we run *n* trials for hyper-parameter tuning with *i* inner folds and *o* outer folds. We need to train and evaluate `n * i * o` models. Our implementation estimates the completion time for the nested cross-validation and stores that information, together with the results, in a log file.

The models created in nested cross-validation are thrown away. We do not need them anymore, once we have used them to estimate the performance of the classifier.

## IMPORTANT
SetFit has a bug where GPU memory is not released after model training, see this [issue report](https://github.com/huggingface/setfit/issues/567). Currently (January 2025), this issue is not resolved but a [work-around](https://github.com/huggingface/setfit/issues/567#issuecomment-2557352330) exists. Run this [test](./setfit_memory.py) to see if the memory leak is fixed or if you need to apply the work-around.

# Training

[Inspiration](https://machinelearningmastery.com/train-final-machine-learning-model/)

Once we have done hyper-parameter tuning and have estimated the performance of the classifier, we can decide if the performance is good enough for our task. If yes, we can use the identified optimal hyper-parameter that are recorded in the log file resulted from the nested cross-validation.

If we create *o* outer folds, we have *o* optimal hyper-parameters. There is no guarantee that these are all the same. In fact, it is likely that they are different and one has the following options to choose the final hyper-parameters for training:
 1. Choose a set of hyper-parameter that worked well in the majority of the folds.
 2. Choose the hyper-parameters that led to the best performance overall.

**Important:** For [training](./train_agreement.py) the final model with SetFit, we use ***all*** available labeled data.

# Prediction



# Environment setup

These are all steps needed to run the python notebook with the [initial experimental code](target_agreement_sv.ipynb) that only analysed the "target" dimension. Skip the Jupyter-specific steps if you want to run only the [cross-validation](./nested_crossvalidation.py), [training](./train_agreement.py) or [prediction](./predict_agreement.py) code.

1. Install miniconda
2. Switch to the base environment

   `conda activate base`
3. Install Python 3.10

   `conda install python=3.10`
4. Install Jupyter Notebook (see [guideline](https://towardsdatascience.com/how-to-set-up-anaconda-and-jupyter-notebook-the-right-way-de3b7623ea4a))

   `conda install -c conda-forge notebook`

   `conda install -c conda-forge nb_conda_kernels`
5. Create a new conda environment and switch to it

   `conda create --name setfitclassifier pip ipykernel`

    `conda activate setfitclassifier`
6. Install required packages

   `pip install -r requirements.txt`
7. Go back to the base env and start the notebook

   `conda deactivate`

    `jupyter-notebook --no-browser`
8. Create an ssh tunnel from the local machine to the server

   `ssh -L 8888:localhost:8888 username@machine`
9. Open the notebook on the local machine at

   `localhost:8888`
