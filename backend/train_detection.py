import ast
import csv
import numpy as np
import pandas as pd
import pickle as pkl
from sklearn.svm import SVC
from preprocess import Preprocessor

THRESHOLD = 2

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
        hashtag_lists = []
        word_lists = []

    for index, row in data_validation.iterrows():
        if index >= 50:
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

        # extract feature for train data
        for index, row in enumerate(preprocessed_rows):
            feature = [1 if word in row else 0 for word in word_lists]
            hashtag_feature = [1 if item in train_hashtags[index] else 0 for item in hashtag_lists]
            total_feature = np.append(feature, hashtag_feature)
            features.append(total_feature)

    features = np.array(features)
    targets = np.array(targets)
    return features, targets

if __name__ == '__main__':
    df = pd.read_csv('/Users/andikakusuma/Documents/Kuliah/NLP/Tubes_text/campaign_detection/backend/corpus/data_latih.csv', sep=';', skiprows=0, encoding='utf-8')
    # randomize data
    # df = df.sample(frac=1).reset_index(drop=True)

    count_train = int(0.8 * len(df))
    count_validation = int(0.1 * len(df))
    count_test = len(df) - count_train - count_validation
    data_train = df[:count_train]
    data_validation = df[count_train:count_train + count_validation].reset_index(drop=True)
    data_test = df[count_train + count_validation:].reset_index(drop=True)

    # train_features = []
    # train_targets = []
    validation_features = []
    validation_targets = []
    test_features = []
    test_targets = []

    # preprocessed_rows = []
    # train_hashtags = []
    # bag_of_words = {}
    # hashtag_lists = []
    # word_lists = []

    train_features, train_targets = extract_features(data_train, True)

    hashtag_lists = load_model('hashtag_lists.pkl')
    word_lists = load_model('word_lists.pkl')

    validation_features, validation_targets = extract_features(data_validation, True)
    print (validation_features.shape, validation_targets.shape)

    test_features, test_targets = extract_features(data_test, True)
    print (test_features.shape, test_targets.shape)

    # Train using train_features
    model = SVC().fit(train_features, train_targets)
    save_model(model, 'classifier.pkl')
    print (model.score(validation_features, validation_targets))
    print (model.score(test_features, test_targets))

