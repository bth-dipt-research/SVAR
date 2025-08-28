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

# Nested cross-validation

[Inspiration](https://machinelearningmastery.com/nested-cross-validation-for-machine-learning-with-python/)

Nested cross-validation combines two important aspects:
 1. It allows hyper-parameter tuning. This is important for optimizing the performance of SetFit, given the small amount of training data.
 2. Cross-validation allows to better estimate the performance of the classifier on unseen data. With a small amount of labeled data, we do not want to let chance influence the performance on the train-test split. We want to use all data to understand the performance variance of the classifier.

Nested cross-validation has a cost: we need to train and evaluate many more models. Assume we run *n* trials for hyper-parameter tuning with *i* inner folds and *o* outer folds. We need to train and evaluate `n * i * o` models. Our implementation estimates the completion time for the nested cross-validation and stores that information, together with the results, in a log file.

The models created in nested cross-validation are thrown away. We do not need them anymore, once we have used them to estimate the performance of the classifier.

The code for nested cross-validation can be found [here](https://machinelearningmastery.com/nested-cross-validation-for-machine-learning-with-python/). We ran the nested cross-validation with [75](../data/cv_75_samples) and [250](../data/cv_250_samples) samples.

## IMPORTANT
SetFit has a bug where GPU memory is not released after model training, see this [issue report](https://github.com/huggingface/setfit/issues/567). Currently (January 2025), this issue is not resolved but a [work-around](https://github.com/huggingface/setfit/issues/567#issuecomment-2557352330) exists. Run this [test](./setfit_memory.py) to see if the memory leak is fixed or if you need to apply the work-around.

# Training set size study
We wanted to study how training set size affects performance. The code [here](training_set_size_study.py) uses the identified best hyper-parameters from the nested cross-validation with 250 samples to train models with training set sizes ranging from 20 to 240. The results of this investigation can be found [here](../data/training_set_size_study).

# Test set evaluation
In addition to nested cross-validation, we [evaluate](test_set_evaluation.py) the classifier on unseen data, an additional 250 requirements, that was not used for training at all. The results can be found [here](../data/test_set_evaluation/setfitclassifier). 

# Training

[Inspiration](https://machinelearningmastery.com/train-final-machine-learning-model/)

Once we have done hyper-parameter tuning and have estimated the performance of the classifier, we can decide if the performance is good enough for our task. If yes, we can use the identified optimal hyper-parameter that are recorded in the [log files](../data/cv_250_samples/200_50/) resulted from the nested cross-validation.

**Important:** For [training](./train.py) the final model with SetFit, we use ***all*** available labeled data.

# Prediction
For the [prediction](predict.py), we use the trained models on the [unseen data](../data/cv_250_samples/trvinfra_requirements_all_complete_250samples_predictions.xlsx).



