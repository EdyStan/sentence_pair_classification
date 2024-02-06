import pandas as pd
import numpy as np
import re
import seaborn as sn
from scipy.sparse import csr_array
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim import corpora, models
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, classification_report, precision_score, recall_score
from imblearn.over_sampling import SMOTE
from sklearn.svm import LinearSVC


train_df = pd.read_json('/home/edstan/Desktop/master_AI/pml/final_projects/Stan_Eduard-George_407/data/train.json')
val_df = pd.read_json('/home/edstan/Desktop/master_AI/pml/final_projects/Stan_Eduard-George_407/data/validation.json')
test_df = pd.read_json('/home/edstan/Desktop/master_AI/pml/final_projects/Stan_Eduard-George_407/data/test.json')

def get_combined(dataframe):
    dataframe['combined'] = dataframe['sentence1'] + ' ' + dataframe['sentence2']

    
get_combined(train_df)
get_combined(val_df)
get_combined(test_df)



def vectorize(vectorizer, train_df, test_df):
    X_train = vectorizer.fit_transform(train_df['combined'])
    X_test = vectorizer.transform(test_df['combined'])

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, train_df['label'])
    y_pred = clf.predict(X_test)
    return y_pred

def get_classification_report(y_val, y_pred):
    print(classification_report(y_val, y_pred))

def write_csv(y_pred):
    test_df['label'] = y_pred
    test_df[['guid', 'label']].to_csv('/home/edstan/Desktop/master_AI/pml/final_projects/Stan_Eduard-George_407/data/sample_submissionn.csv', sep=',', encoding='utf-8', index=False)


def generate_features(dataframe):

    get_punctuation(dataframe, 'sentence1')
    get_punctuation(dataframe, 'sentence2')
    get_punctuation(dataframe, 'combined')

    get_length_score(dataframe, 'sentence1')
    get_length_score(dataframe, 'sentence2')
    get_length_score(dataframe, 'combined')

    get_word_relevance_scores(dataframe, 'sentence1')
    get_word_relevance_scores(dataframe, 'sentence2')
    get_word_relevance_scores(dataframe, 'combined')

    get_jaccard_similarity(dataframe)
    length_similarity(dataframe)

    get_topics(dataframe)
    get_topic_similarity(dataframe)
    print_heatmap()


def get_punctuation(dataframe, sentence):
    punctuation = '[!\"#$%&\'()*+,-„”./:;<=>?@[\\]^_`{|}~]'
    punctuation_count = []
    sentence_list = dataframe[sentence].to_list()
    for character in punctuation:
        local_count = []
        for phrase in sentence_list:
            local_count.append(phrase.count(character))
        punctuation_count.append(local_count)
    
    punctuation_count = np.array(punctuation_count)
    punctuation_count = np.sum(punctuation_count, axis=0)
    # dataframe['punctuation_' + sentence] = MinMaxScaler().fit_transform(punctuation_count.reshape(-1, 1))
    dataframe['punctuation_' + sentence] = punctuation_count

def get_length_score(dataframe, sentence):
    # lengths = np.array(dataframe[sentence].str.len()).reshape(-1, 1)
    # dataframe['length_' + sentence] = MinMaxScaler().fit_transform(lengths)
    dataframe['length_' + sentence] = dataframe[sentence].str.len()


def get_word_relevance_scores(dataframe, sentence):
    vectorized_data = TfidfVectorizer().fit_transform(dataframe[sentence])
    words_count = []

    for i in range(len(dataframe[sentence])):
        words_count.append(len(re.findall(r"([\w]*\w)", dataframe[sentence][i])))
    words_count = np.array([words_count]).reshape(-1, 1)
    scores = np.array(vectorized_data.sum(axis=1)).reshape(-1, 1)
    words_count[words_count == 0] = 1e7
    scores_median = scores / words_count
    

    scores_scaled = MinMaxScaler().fit_transform(scores_median.reshape(-1, 1))
    dataframe['scores_' + sentence] = scores_scaled

def get_jaccard_similarity(dataframe):
    jaccard_sim = []
    for s1, s2 in zip(dataframe['sentence1'], dataframe['sentence2']):
        s1 = set(re.findall("\w+", s1.lower()))
        s2 = set(re.findall("\w+", s2.lower()))
        inter = len(s1.intersection(s2))
        union = len(s1.union(s2))
        jaccard_sim.append(inter/union)
    dataframe['jaccard_similarity'] = jaccard_sim

def get_cosine_similarity(dataframe):
    cosine_similarities = []
    for i, j in zip(dataframe['sentence1'], dataframe['sentence2']):
        sentences_transformed = CountVectorizer().fit_transform([i, j])
        cosine_similarities.append(cosine_similarity(sentences_transformed)[0,1])
    dataframe['cosine_similarity'] = cosine_similarities

def length_similarity(dataframe):
    dataframe['length_similarity_1'] = [len(s2)/len(s1) for s1, s2 in zip(dataframe['sentence1'], dataframe['sentence2'])]
    dataframe['length_similarity_2'] = [len(s1)/len(s2) for s1, s2 in zip(dataframe['sentence1'], dataframe['sentence2'])]

def get_topics(dataframe):
    tokenized_sentences = [re.findall("\w+", sentence) for sentence in dataframe['combined']]
    dictionary = corpora.Dictionary(tokenized_sentences)
    corpus = [dictionary.doc2bow(sentence) for sentence in tokenized_sentences]
    lda_model = models.LdaModel(corpus, num_topics=3, id2word=dictionary, random_state=42)
    topic_distributions = np.array([max(lda_model[doc], key=lambda x: x[1])[0] for doc in corpus])
    dataframe['topic_distributions'] = topic_distributions

def get_topic_similarity(dataframe):
    sentences_concat = np.concatenate([dataframe['sentence1'], dataframe['sentence2']])
    tokenized_sentences = [re.findall("\w+", sentence) for sentence in sentences_concat]
    dictionary = corpora.Dictionary(tokenized_sentences)
    corpus = [dictionary.doc2bow(sentence) for sentence in tokenized_sentences]

    num_topics = 4
    lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, random_state=42)
    topic_ratio_matrix = np.zeros((len(corpus), num_topics))
    for i, doc in enumerate(corpus):
        for topic, score in lda_model[doc]:
            topic_ratio_matrix[i, topic] = score
    trm_len = len(topic_ratio_matrix)
    topics_s1 = topic_ratio_matrix[(trm_len // 2):]
    topics_s2 = topic_ratio_matrix[:(trm_len // 2)]
    topic_difference = topics_s2 - topics_s1
    dataframe['topic_similarity'] = [np.linalg.norm(row) for row in topic_difference]

def print_heatmap():
    dataframe_for_corr = train_df.copy()
    dataframe_for_corr.drop(['sentence1', 'sentence2', 'combined', 'guid'], axis=1, inplace=True)
    corr_matrix = dataframe_for_corr.corr()
    print(corr_matrix)
    print(sn.heatmap(corr_matrix))

def get_grid_for_vectorizer():
    y_val = val_df['label']
    # max_features_list = [1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
    max_features_list = [20000, 40000, 60000, 80000, 100000, 120000, 140000, 160000]

 
    recall_11 = []
    precision_11 = []
    f1_11 = []
    recall_12 = []
    precision_12 = []
    f1_12 = []
    recall_13 = []
    precision_13 = []
    f1_13 = []
    ngram11 = pd.DataFrame()
    ngram12 = pd.DataFrame()
    ngram13 = pd.DataFrame()
    ngram11['max_features'] = max_features_list
    ngram12['max_features'] = max_features_list
    ngram13['max_features'] = max_features_list

    for i in max_features_list:
        print(f"{i} max features")
        print("ngrams 1, 1")
        y_pred = vectorize(CountVectorizer(max_features=i, ngram_range=(1,1)), train_df, val_df)
        get_classification_report(y_val, y_pred)
        recall_11.append(recall_score(y_val, y_pred, average='macro'))
        precision_11.append(precision_score(y_val, y_pred, average='macro'))
        f1_11.append(f1_score(y_val, y_pred, average='macro'))
        
        print("ngrams 1, 2")
        y_pred = vectorize(CountVectorizer(max_features=i, ngram_range=(1,2)), train_df, val_df)
        get_classification_report(y_val, y_pred)
        recall_12.append(recall_score(y_val, y_pred, average='macro'))
        precision_12.append(precision_score(y_val, y_pred, average='macro'))
        f1_12.append(f1_score(y_val, y_pred, average='macro'))    

        print("ngrams 1, 3")
        y_pred = vectorize(CountVectorizer(max_features=i, ngram_range=(1,3)), train_df, val_df)
        get_classification_report(y_val, y_pred)
        recall_13.append(recall_score(y_val, y_pred, average='macro'))
        precision_13.append(precision_score(y_val, y_pred, average='macro'))
        f1_13.append(f1_score(y_val, y_pred, average='macro'))
    ngram11['recall'] = recall_11
    ngram12['recall'] = recall_12
    ngram13['recall'] = recall_13

    ngram11['precision'] = precision_11
    ngram12['precision'] = precision_12
    ngram13['precision'] = precision_13

    ngram11['f1 score'] = f1_11
    ngram12['f1 score'] = f1_12
    ngram13['f1 score'] = f1_13

    print('ngrams = (1,1)')
    print(ngram11)
    print('ngrams = (1,2)')
    print(ngram12)
    print('ngrams = (1,3)')
    print(ngram13)


    # ngrams = (1,1)
    # max_features    recall  precision  f1 score
    # 0          1000  0.353907   0.422829  0.353061
    # 1          1500  0.378696   0.480780  0.389003
    # 2          2000  0.394235   0.498808  0.407762
    # 3          2500  0.386661   0.464475  0.395956
    # 4          3000  0.395621   0.472302  0.408615
    # 5          3500  0.391389   0.450991  0.402868
    # 6          4000  0.395184   0.458481  0.408238
    # 7          4500  0.398516   0.459608  0.410061
    # 8          5000  0.403117   0.452271  0.413827
    # ngrams = (1,2)
    # max_features    recall  precision  f1 score
    # 0          1000  0.349225   0.430400  0.349119
    # 1          1500  0.376029   0.486845  0.386146
    # 2          2000  0.374178   0.435928  0.380729
    # 3          2500  0.386895   0.438375  0.394220
    # 4          3000  0.393723   0.445310  0.402578
    # 5          3500  0.388774   0.432330  0.396576
    # 6          4000  0.402765   0.439050  0.409822
    # 7          4500  0.397930   0.444518  0.407704
    # 8          5000  0.400460   0.439486  0.408899
    # ngrams = (1,3)
    # max_features    recall  precision  f1 score
    # 0          1000  0.350289   0.407213  0.349299
    # 1          1500  0.372852   0.476816  0.381397
    # 2          2000  0.382654   0.454444  0.392289
    # 3          2500  0.382856   0.432587  0.389588
    # 4          3000  0.397387   0.454157  0.406531
    # 5          3500  0.389065   0.435522  0.397006
    # 6          4000  0.406177   0.456683  0.417564
    # 7          4500  0.408248   0.447513  0.416734
    # 8          5000  0.404490   0.440648  0.412214


    # ngrams = (1,1)
    #    max_features    recall  precision  f1 score
    # 0         20000  0.374433   0.427330  0.386291
    # 1         40000  0.374199   0.435340  0.387950
    # 2         60000  0.367911   0.434070  0.380485
    # 3         80000  0.372217   0.438371  0.384420
    # 4        100000  0.368525   0.436945  0.380082
    # 5        120000  0.368586   0.436806  0.380162
    # 6        140000  0.370287   0.438360  0.381561
    # 7        160000  0.373155   0.440839  0.384364
    # ngrams = (1,2)
    #    max_features    recall  precision  f1 score
    # 0         20000  0.401002   0.455549  0.413690
    # 1         40000  0.397924   0.444217  0.410483
    # 2         60000  0.396964   0.471114  0.414180
    # 3         80000  0.396332   0.463063  0.411778
    # 4        100000  0.402213   0.479819  0.419759
    # 5        120000  0.404209   0.495292  0.423732
    # 6        140000  0.400250   0.476251  0.415782
    # 7        160000  0.401277   0.483351  0.418014
    # ngrams = (1,3)
    #    max_features    recall  precision  f1 score
    # 0         20000  0.400787   0.448780  0.411675
    # 1         40000  0.401102   0.465694  0.417268
    # 2         60000  0.390821   0.438916  0.402186
    # 3         80000  0.399752   0.474545  0.416355
    # 4        100000  0.397631   0.475447  0.414111
    # 5        120000  0.407106   0.479699  0.423702
    # 6        140000  0.400883   0.481597  0.417393
    # 7        160000  0.400733   0.475716  0.416247

    
# it somehow overfits over my train and validation set. 
# [0.74003085 0.73738657 0.73908646]
def smote_kfold_vectorizer():
    X = CountVectorizer(ngram_range=(1,2)).fit_transform(train_df['combined'])
    y = train_df['label']
    X_resampled, y_resampled = SMOTE(random_state=42, sampling_strategy='auto').fit_resample(X, y)
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    cross_val_results = cross_val_score(LogisticRegression(max_iter=1000, random_state=42), X_resampled, y_resampled, cv=kf, scoring='f1_macro')
    print(cross_val_results)


def smote_split_vectorizer():
    vectorizer = CountVectorizer(ngram_range=(1,2))
    X = vectorizer.fit_transform(train_df['combined'])
    y = train_df['label']
#     Train_test_split corrupts my data ???
#     X_train, y_train, X_val, y_val = train_test_split(X, y, random_state=42, shuffle=True, test_size=0.2)
    index_list = np.arange(X.shape[0])
    np.random.shuffle(index_list)
    switch_idx = int(0.8 * len(index_list))
    train_idx = index_list[:switch_idx]
    val_idx = index_list[switch_idx:]
    X_train, y_train, X_val, y_val = X[train_idx], y[train_idx], X[val_idx], y[val_idx]

    X_resampled, y_resampled = SMOTE(random_state=42, sampling_strategy='not majority').fit_resample(X_train, y_train)
    classifier = LogisticRegression(max_iter=1000, random_state=42).fit(X_resampled, y_resampled)
    y_pred = classifier.predict(X_val)
    get_classification_report(y_val, y_pred)
    # Output with SMOTE('not majority')

    #               precision    recall  f1-score   support

    #            0       0.11      0.14      0.13       546
    #            1       0.16      0.24      0.19       284
    #            2       0.61      0.65      0.63      5334
    #            3       0.72      0.65      0.69      6071

    #     accuracy                           0.62     12235
    #    macro avg       0.40      0.42      0.41     12235
    # weighted avg       0.63      0.62      0.63     12235



    # Output without SMOTE

    #               precision    recall  f1-score   support

    #            0       0.32      0.04      0.07       554
    #            1       0.35      0.08      0.13       270
    #            2       0.63      0.71      0.67      5407
    #            3       0.71      0.72      0.72      6004

    #     accuracy                           0.67     12235
    #    macro avg       0.50      0.39      0.40     12235
    # weighted avg       0.65      0.67      0.65     12235

    # Output with SMOTE({0:2500, 1:2500})

    #               precision    recall  f1-score   support

    #            0       0.18      0.05      0.08       514
    #            1       0.24      0.19      0.21       291
    #            2       0.64      0.67      0.66      5440
    #            3       0.71      0.73      0.72      5990

    #     accuracy                           0.66     12235
    #    macro avg       0.44      0.41      0.41     12235
    # weighted avg       0.64      0.66      0.65     12235


    # Output with SMOTE(0:3000, 1:3000)

    #               precision    recall  f1-score   support

    #            0       0.17      0.10      0.13       554
    #            1       0.18      0.19      0.18       278
    #            2       0.63      0.67      0.65      5351
    #            3       0.73      0.71      0.72      6052

    #     accuracy                           0.65     12235
    #    macro avg       0.43      0.42      0.42     12235
    # weighted avg       0.65      0.65      0.65     12235


# tried to also truncate the validation score, but it turns out that the f1score is lower(not by much)
# the version in which I don't truncate the test data gives the same f1score of 0.67688. It means that the truncated values didn't really matter in the end
def evaluate_truncated():
    X = train_df['combined']
    y = train_df['label']
    index_list = np.arange(X.shape[0])
    np.random.shuffle(index_list)
    switch_idx = int(0.8 * len(index_list))
    train_idx = index_list[:switch_idx]
    val_idx = index_list[switch_idx:]
    X_train, y_train, X_val, y_val = X[train_idx], y[train_idx], X[val_idx], y[val_idx]

    vectorizer = CountVectorizer(token_pattern="\w+", ngram_range=(1,2))
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_val_vectorized = vectorizer.transform(X_val)
    row_indices, column_indices = X_train_vectorized.nonzero()
    unique_counts = np.bincount(column_indices)
    valid_indices = np.where(unique_counts[column_indices] > 1)[0]
    valid_col_ind = column_indices[valid_indices]
    valid_row_ind = row_indices[valid_indices]
    valid_data = X_train_vectorized.data[valid_indices] 
    X_train_truncated = csr_array((valid_data, (valid_row_ind, valid_col_ind)), shape = X_train_vectorized.shape) 

    len_majority = len(y_train[y_train == 2])
    len_minority = int(0.3 * len_majority)
    X_resampled, y_resampled = SMOTE(random_state=42, sampling_strategy={0: len_minority, 1: len_minority}).fit_resample(X_train_truncated, y_train)
    classifier = LogisticRegression(max_iter=1000, random_state=42)
    classifier.fit(X_resampled, y_resampled)
    y_pred = classifier.predict(X_val_vectorized)
    get_classification_report(y_val, y_pred)


def get_grid_for_linear_svc():
    X = train_df['combined']
    y = train_df['label']
    index_list = np.arange(X.shape[0])
    np.random.shuffle(index_list)
    switch_idx = int(0.8 * len(index_list))
    train_idx = index_list[:switch_idx]
    val_idx = index_list[switch_idx:]
    X_train, y_train, X_val, y_val = X[train_idx], y[train_idx], X[val_idx], y[val_idx]

    # vectorizer = TfidfVectorizer(token_pattern="\w+", min_df=2, max_df=0.35, ngram_range=(1,2))
    vectorizer = CountVectorizer(token_pattern="\w+", min_df=2, max_df=0.35, ngram_range=(1,2))
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_val_vectorized = vectorizer.transform(X_val)

    len_majority = len(y_train[y_train == 2])
    len_minority = int(0.3 * len_majority)
    X_resampled03, y_resampled03 = SMOTE(random_state=42, sampling_strategy={0: len_minority, 1: len_minority}).fit_resample(X_train_vectorized, y_train)
    len_minority = int(0.5 * len_majority)
    X_resampled05, y_resampled05 = SMOTE(random_state=42, sampling_strategy={0: len_minority, 1: len_minority}).fit_resample(X_train_vectorized, y_train)
    len_minority = int(0.8 * len_majority)
    X_resampled08, y_resampled08 = SMOTE(random_state=42, sampling_strategy={0: len_minority, 1: len_minority}).fit_resample(X_train_vectorized, y_train)

    C = [0.1, 1, 10, 100, 1000]
    C_data = [0.1, 0.1, 0.1, 1, 1, 1, 10, 10, 10, 100, 100, 100, 1000, 1000, 1000]



    f1 = []
    f1_smote03 = []
    f1_smote05 = []
    f1_smote08 = []
    my_data = pd.DataFrame()
    my_data['C'] = C_data

    for i in C:
        print(f'C = {i}')
        print('max_iter=300')
        classifier = LinearSVC(max_iter=300, C=i, random_state=42)

        classifier.fit(X_train_vectorized, y_train)
        y_pred = classifier.predict(X_val_vectorized)
        f1.append(f1_score(y_val, y_pred, average='macro'))

        classifier.fit(X_resampled03, y_resampled03)
        y_pred = classifier.predict(X_val_vectorized)
        f1_smote03.append(f1_score(y_val, y_pred, average='macro'))

        classifier.fit(X_resampled05, y_resampled05)
        y_pred = classifier.predict(X_val_vectorized)
        f1_smote05.append(f1_score(y_val, y_pred, average='macro'))

        classifier.fit(X_resampled08, y_resampled08)
        y_pred = classifier.predict(X_val_vectorized)
        f1_smote08.append(f1_score(y_val, y_pred, average='macro'))


        print('max_iter=500')
        classifier = LinearSVC(max_iter=500, C=i, random_state=42)

        classifier.fit(X_train_vectorized, y_train)
        y_pred = classifier.predict(X_val_vectorized)
        f1.append(f1_score(y_val, y_pred, average='macro'))

        classifier.fit(X_resampled03, y_resampled03)
        y_pred = classifier.predict(X_val_vectorized)
        f1_smote03.append(f1_score(y_val, y_pred, average='macro'))

        classifier.fit(X_resampled05, y_resampled05)
        y_pred = classifier.predict(X_val_vectorized)
        f1_smote05.append(f1_score(y_val, y_pred, average='macro'))

        classifier.fit(X_resampled08, y_resampled08)
        y_pred = classifier.predict(X_val_vectorized)
        f1_smote08.append(f1_score(y_val, y_pred, average='macro'))

        print('max_iter=1000')
        classifier = LinearSVC(max_iter=1000, C=i, random_state=42)

        classifier.fit(X_train_vectorized, y_train)
        y_pred = classifier.predict(X_val_vectorized)
        f1.append(f1_score(y_val, y_pred, average='macro'))

        classifier.fit(X_resampled03, y_resampled03)
        y_pred = classifier.predict(X_val_vectorized)
        f1_smote03.append(f1_score(y_val, y_pred, average='macro'))

        classifier.fit(X_resampled05, y_resampled05)
        y_pred = classifier.predict(X_val_vectorized)
        f1_smote05.append(f1_score(y_val, y_pred, average='macro'))

        classifier.fit(X_resampled08, y_resampled08)
        y_pred = classifier.predict(X_val_vectorized)
        f1_smote08.append(f1_score(y_val, y_pred, average='macro'))


    my_data['f1 score'] = f1
    my_data['f1 score SMOTE 03'] = f1_smote03
    my_data['f1 score SMOTE 05'] = f1_smote05
    my_data['f1 score SMOTE 08'] = f1_smote08
    print('F1 scores for C (LinearSVC)')
    print(my_data)

    
    # first: max_iter=300
    # second: max_iter=500
    # third: max_iter=1000

    # F1 scores for C (LinearSVC)
    #          C  f1 score  f1 score SMOTE 03  f1 score SMOTE 05  f1 score SMOTE 08
    # 0      0.1  0.239062           0.410665           0.411497           0.406122
    # 1      0.1  0.239062           0.410665           0.411497           0.406122
    # 2      0.1  0.239147           0.410665           0.411497           0.406122
    # 3      1.0  0.238437           0.398708           0.400484           0.393146
    # 4      1.0  0.238364           0.398803           0.400351           0.393020
    # 5      1.0  0.238608           0.398751           0.400331           0.393247
    # 6     10.0  0.237798           0.390940           0.391226           0.383718
    # 7     10.0  0.237242           0.391125           0.389882           0.384207
    # 8     10.0  0.237558           0.392139           0.387138           0.383749
    # 9    100.0  0.238594           0.392698           0.387623           0.384519
    # 10   100.0  0.238353           0.382285           0.386067           0.383433
    # 11   100.0  0.237794           0.383623           0.385774           0.384248
    # 12  1000.0  0.236673           0.386959           0.383845           0.378507
    # 13  1000.0  0.237531           0.377160           0.383731           0.380405
    # 14  1000.0  0.238510           0.373649           0.383875           0.378231


