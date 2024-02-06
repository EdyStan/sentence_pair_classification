# Sentence Pair Classification

## Short description

The main task involves the classification of instances that have two seemingly unrelated sentences as features. The data set can be found at the following address: https://www.kaggle.com/competitions/sentence-pair-classification-pml-2023/data.

The code was written in Python and a great number of features preprocessing techniques were applied:

- Vectorization using Count Vectorizer.
- Oversampling using SMOTE.
- Counts of punctuation signs.
- Length of each sentence.
- A word relevance score calculated with the help of a TfidfVectorizer.
- Jaccard and Cosine similarity between the sentences.
- Length similarity (division of sentences lenghts).
- Topic distributions using Latent Dirichlet Allocation (LDA). 

The model I used to predict the labels is Logistic Regression.

A visual study of the hyperparameter tuning and the conclusions are presented in `Documentation.pdf`.

## About the files

`method1.py` and `method2.py` are two independent files which represent the final models I chose to use for this project. Explanatory comments were added. 
I recommend checking `quality_code.py` after the first two, since there are several other failed attempts to reach a high F1 score. Another important aspect of this file is that it contains studies meant to find the best hyperparameters for certain models and vectorizers used. The outputs are shown in comments after the code that generated them. There will be references for the most relevant studies in `method1.py` and `method2.py`. 
Lastly, check `heatmap.csv` to see the correlations between all the custom features created in `quality_code.py`. None of those features were actually used in the final two codes.