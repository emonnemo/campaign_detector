from django.conf import settings
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
import pickle as pkl

ABSOLUTE_BACKEND_PATH = settings.ABSOLUTE_BACKEND_PATH
NUMBERS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
VOWEL = ['a', 'i', 'u', 'e', 'o']

if __name__ == '__main__':
    spelling = {}
    letter = {}

    with open('dicts/number_dict.txt', 'r') as file:
        lines = file.read().splitlines()
        for line in lines:
            content = line.split()
            spelling[content[0]] = content[1]
            letter[content[0]] = content[2]
        number_dict = {}
        number_dict['spelling'] = spelling
        number_dict['letter'] = letter
        output = open('models/number_dict.pkl', 'wb')
        pkl.dump(number_dict, output)
    
    df = pd.read_csv('%s/backend/corpus/number_corpus.csv' % ABSOLUTE_BACKEND_PATH, sep=',', skiprows=0)
    
    # feature extraction
    features = []
    labels = []

    for idx, row in df.iterrows():
        data = row['data']
        target = row['target']

        labels.append(target)
        for index, char in enumerate(data):
            if char in NUMBERS:
                # feature <number, charbefore, is_first, is_last, is_last_vowel, is_number_vowel>
                feature = [int(char), ord(data[index-1]), index == 0, index == len(data) - 1, \
                    (number_dict['letter'][str(data[index-1])] in VOWEL) if data[index-1] in NUMBERS else data[index-1] in VOWEL, \
                    number_dict['letter'][char] in VOWEL]
                features.append(feature)
                break
        

    features = np.array(features)
    labels = np.array(labels)
    model = DecisionTreeClassifier().fit(features, labels)

    output = open('models/number_model.pkl', 'wb')
    pkl.dump(model, output)
    