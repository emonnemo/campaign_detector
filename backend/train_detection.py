import ast
import csv
import numpy as np
import pandas as pd
import pickle as pkl
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from preprocess import Preprocessor


word_lists = []
hashtag_lists = []
THRESHOLD = 10

def _parse_bytes(field):
    """ Convert string represented in Python byte-string literal syntax into a
    decoded character string. Other field types returned unchanged.
    """
    result = field
    try:
        result = ast.literal_eval(field)
    finally:
        return result.decode() if isinstance(result, bytes) else field

def preprocess(text):
    text = _parse_bytes(text)
    preprocessor = Preprocessor()
    tokens = preprocessor.tokenize(text)
    tokens = preprocessor.normalize(tokens)
    tokens = preprocessor.remove_stopwords(tokens)
    tokens = preprocessor.stem(tokens)
    hashtag = preprocessor.hashtag
    return tokens, hashtag

def save_model(model, file):
    output = open('models/%s' % file, 'wb')
    pkl.dump(model, output)
    output.close()

def load_model(file):
    input = open('models/%s' % file, 'rb')
    return pkl.load(input)

def extract_features(df, train=False):
    features = []
    targets = []
    if train:
        train_hashtags = []
        preprocessed_rows = []
        bag_of_words = {}

    for index, row in df.iterrows():
        if index == (200 if train else 10000):
            break
        tokens, hashtag = preprocess(row['Teks'])
        targets.append(row.get('Label', []))

        if train:
            train_hashtags.append(hashtag)
            preprocessed_rows.append(tokens)

            for item in hashtag:
                if item not in hashtag_lists:
                    hashtag_lists.append(item)

            for token in tokens:
                if token in bag_of_words:
                    bag_of_words[token] += 1
                else:
                    bag_of_words[token] = 1
        else:
            feature = [1 if word in tokens else 0 for word in word_lists]
            hashtag_feature = [1 if item in hashtag else 0 for item in hashtag_lists]
            total_feature = np.append(feature, hashtag_feature)
            features.append(total_feature)

    if train:
        for word in bag_of_words:
            if bag_of_words[word] >= THRESHOLD:
                word_lists.append(word)
        
        save_model(hashtag_lists, 'hashtag_lists.pkl')
        save_model(word_lists, 'word_lists.pkl')
        print (hashtag_lists)
        print (word_lists)

        # extract feature for train data
        for index, row in enumerate(preprocessed_rows):
            feature = [1 if word in row else 0 for word in word_lists]
            hashtag_feature = [1 if item in train_hashtags[index] else 0 for item in hashtag_lists]
            total_feature = np.append(feature, hashtag_feature)
            features.append(total_feature)

    features = np.array(features)
    targets = np.array(targets)
    return features, targets

def svc_param_selection(features, targets, nfolds):
    Cs = [0.001, 0.01, 0.1, 1, 10]
    gammas = [0.001, 0.01, 0.1, 1]
    # testing
    print ('--Finding best parameters for SVM--')
    param_grid = {'C': Cs, 'gamma' : gammas}
    grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=nfolds)
    grid_search.fit(features, targets)
    print (grid_search.best_params_)
    print (grid_search.grid_scores_)
    print (grid_search.cv_results_)
    return grid_search.best_params_

if __name__ == '__main__':
    df = pd.read_csv('/Users/andikakusuma/Documents/Kuliah/NLP/Tubes_text/campaign_detection/backend/corpus/data_latih.csv', sep=';', skiprows=0, encoding='utf-8')
    # split data
    data_train, data_test = train_test_split(df, test_size=0.1, random_state=1)
    # data_train, data_validation = train_test_split(data_train, test_size=0.1, random_state=1)
    data_train = data_train.reset_index(drop=True)
    # data_validation = data_validation.reset_index(drop=True)
    data_test = data_test.reset_index(drop=True)

    train_features, train_targets = extract_features(data_train, True)
    print (train_features.shape, train_targets.shape)

    hashtag_lists = load_model('hashtag_lists.pkl')
    word_lists = load_model('word_lists.pkl')

    test_features, test_targets = extract_features(data_test)
    print (test_features.shape, test_targets.shape)

    # Train using train_features svm
    svc_param_selection(train_features, train_targets, 10)
    # best for now is C=10, gamma=0.1
    model = SVC(C=10, gamma=0.1).fit(train_features, train_targets)
    save_model(model, 'classifier.pkl')
    print (model.score(train_features, train_targets))
    # print (model.score(validation_features, validation_targets))
    print (model.score(test_features, test_targets))

