# About

This is a unsupervised classifier implemented with Generative Pre-trained Transformer (GPT) models. We use OpenAI's models but any other LLM provider or local models could be used instead.

# Prediction
For the prompt, we use a condensed version of the classification instructions that were used by the human labelers when creating the ground truth. The [prediction code](predict.py) creates structured output in the form of [json files](../data/test_set_evaluation/gptclassifier/).

# Evaluation
We used the same [250 requirements](../data/test_set_evaluation/SOLO\ Iteration\ 2\ data\ -\ Test\ set.csv) that were used to evaluate the supervised SetFit classifier on unseen data. The [evaluation](evaluate.py) calculates precision, recall, and f1-score (micro, macro, weighted) and stores the results [here](../data/test_set_evaluation/gptclassifier/gptevaluation.csv).
