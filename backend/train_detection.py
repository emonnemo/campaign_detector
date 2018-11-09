import ast
import csv
import numpy as np
import pandas as pd
import pickle as pkl
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from preprocess import Preprocessor

preprocessor = Preprocessor()
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
    tokens = preprocessor.tokenize(text)
    # tokens = preprocessor.normalize_once(tokens) # 1.27.7 min
    tokens = preprocessor.normalize(tokens) # 1.25.7 min
    tokens = preprocessor.remove_stopwords(tokens)
    tokens = preprocessor.stem(tokens)
    hashtag = preprocessor.hashtag
    return tokens, hashtag

def extract_feature(text, hashtag_lists, word_lists):
    tokens, hashtag = preprocess(text)
    feature = [1 if word in tokens else 0 for word in word_lists]
    hashtag_feature = [1 if item in hashtag else 0 for item in hashtag_lists]
    total_feature = np.append(feature, hashtag_feature)
    return total_feature

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
    save_model(train_features, 'train_features.pkl')
    save_model(train_targets, 'train_targets.pkl')
    print (train_features.shape, train_targets.shape)

    hashtag_lists = load_model('hashtag_lists.pkl')
    word_lists = load_model('word_lists.pkl')

    test_features, test_targets = extract_features(data_test)
    save_model(test_features, 'test_features.pkl')
    save_model(test_targets, 'test_targets.pkl')
    print (test_features.shape, test_targets.shape)

    train_features = load_model('train_features.pkl')
    train_targets = load_model('train_targets.pkl')
    test_features = load_model('test_features.pkl')
    test_targets = load_model('test_targets.pkl')
    # for feat in train_features:
    #     print (feat.shape)=
    train_targets[:664][-1] = 0
    model = DecisionTreeClassifier().fit(train_features, train_targets)
    save_model(model, 'dtl.pkl')
    print (model.score(train_features, train_targets))
    print (model.score(test_features, test_targets))

    # Train using train_features svm
    # try:
    #     svc_param_selection(train_features, train_targets, 10)
    # except:
    #     pass
    # best for now is C=10, gamma=0.01
    model = SVC(C=10, gamma=0.01).fit(train_features, train_targets)
    save_model(model, 'svc.pkl')
    print (model.score(train_features, train_targets))
    # print (model.score(validation_features, validation_targets))
    print (model.score(test_features, test_targets))

    model = MLPClassifier().fit(train_features, train_targets)
    save_model(model, 'mlp.pkl')
    print (model.score(train_features, train_targets))
    # print (model.score(validation_features, validation_targets))
    print (model.score(test_features, test_targets))

