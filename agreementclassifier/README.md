# About
A classifier, trained on the agreement from the requirements verifiability pilots, that predicts agreements/disagreements in classification. The purpose of this is to split the dataset into:
a) requirements were the likelihood of classification agreement is high -> only one judge makes the classification
b) requirements were the likelihood of classification agreement is low -> all judges discuss the classification

We create a classifier for each dimensions: target, nature, interpretability, reference, logic rule

# Environment setup
1. Install miniconda
2. Switch to the base environment

   `conda activate base`
3. Install Python 3.10

   `conda install python=3.10`
4. Install Jupyter Notebook (see [guideline](https://towardsdatascience.com/how-to-set-up-anaconda-and-jupyter-notebook-the-right-way-de3b7623ea4a))

   `conda install -c conda-forge notebook`

   `conda install -c conda-forge nb_conda_kernels`
5. Create a new conda environment and switch to it

   `conda create --name agreementclassifier pip ipykernel`
        
    `conda activate agreementclassifier`
6. Install required packages

   `pip install -r requirements.txt`
7. Go back to the base env and start the notebook

   `conda deactivate`

    `jupyter-notebook --no-browser`
8. Create an ssh tunnel from the local machine to the server

   `ssh -L 8888:localhost:8888 username@machine`
9. Open the notebook on the local machine at

   `localhost:8888`
