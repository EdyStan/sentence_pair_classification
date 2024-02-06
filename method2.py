# import the necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from imblearn.over_sampling import SMOTE

# Extract data from .json files and create a pd.DataFrame for each dataset. 
train_df = pd.read_json('/home/edstan/Desktop/master_AI/pml/final_projects/Stan_Eduard-George_407/data/train.json')
test_df = pd.read_json('/home/edstan/Desktop/master_AI/pml/final_projects/Stan_Eduard-George_407/data/test.json')

# define and call get_combine function that generates a new feature that combines the two sentences
def get_combined(dataframe):
    dataframe['combined'] = dataframe['sentence1'] + ' ' + dataframe['sentence2']

get_combined(test_df)
get_combined(train_df)

# assign the new feature to variable 'X' and the labels to 'y'
X = train_df['combined']
y = train_df['label']

# initialize CountVectorizer with the following parameters:
# - token_pattern="\w+", changed from the default r"(?u)\b\w\w+\b"
#   (I have changed this parameter because the default one includes 416 other symbols, single letters, and digits).
#   Here are some of them: 'ë', 'ń', 'ø', 'θ', 'В', 'ν', 'ㅣ', 'ρ', 'ѣ', 'Ă', ',', '橋', 'ष', 'ñ', 'è', 'ㆍ', 'ƴ', 'Д', '妃'.# - min_df=2, the vectorizer takes into account only the instances (one word or a group of words) that appear more than two times in the whole list of sentences.
#   (this action truncates the BoW by around 77%, from 1480284 words to 347826).
# - max_df=0.35, the vectorizer removes the instances that appear in more than 35% of the sentences.
#   (it only reduces 8 instances in total, but it's always good to remove these misleading occurrences. This parameter also gives the best final results).
# - ngram_range=(1,2), the vectorizer counts single words and groups of two words.
# a detailed study on how to choose the optimal ngram_range was realized:
# (file: quality_code.py, method: line 165, output: line 229)
# then, fit the vectorizer with the sentences in 'X', and assign the transformed X to 'X_vectorized'.
vectorizer = CountVectorizer(token_pattern="\w+", min_df=2, max_df=0.35, ngram_range=(1,2))
X_vectorized = vectorizer.fit_transform(X)

# for memory reduction, I chose to redistribute the memory for the data stored in X_vectorizer, and y. 
# the reason I didn't use 'int8' is because the maximum number of occurrences of a word in a sentence is ... 502.
X_vectorized = X_vectorized.astype('int16')
y = y.astype('int8')

# the data is unbalanced. Here is the number of sentences associated with their label 0:2666, 1:1372, 2:26857, 3:30278.
# this is why I decided to oversample the data with SMOTE. The first two lines just define the number of samples that will be added to labels 0 and 1.
len_majority = len(y[y == 2])
len_minority = int(0.5 * len_majority)
X_resampled, y_resampled = SMOTE(random_state=42, sampling_strategy={0: len_minority, 1: len_minority}).fit_resample(X_vectorized, y)

# I have chosen LinearSVC because SVC had an insane runtime of a couple of hours. I have realized a detailed study on the optimal hyperparameters.
# (file: quality_code.py, method: line 412, output: line 516).
# the study revealed that using SMOTE made a significant difference for LinearSVC model. Therefore, for other models it didn't make any improvement.
classifier = LinearSVC(max_iter=300, C=0.1, dual='auto', random_state=42)
classifier.fit(X_resampled, y_resampled)

# I vectorized the test set on the vectorizer fitted with the train+val set, I changed its type, and predicted the labels.
X_test = test_df['combined']
X_test_vectorized = vectorizer.transform(X_test)
X_test_vectorized = X_test_vectorized.astype('int16')
y_pred = classifier.predict(X_test_vectorized)

# export predicted data + guid to a csv file.
test_df['label'] = y_pred
test_df[['guid', 'label']].to_csv('/home/edstan/Desktop/master_AI/pml/final_projects/Stan_Eduard-George_407/data/sample_submission_C01.csv', sep=',', encoding='utf-8', index=False)