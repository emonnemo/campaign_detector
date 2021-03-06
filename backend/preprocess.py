from django.conf import settings
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import copy
import numpy as np
import pickle as pkl
import re

NUMBERS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
VOWEL = ['a', 'i', 'u', 'e', 'o']
USERNAME_EXCEPTIONS = ['@jokowi', '@prabowo', '@sandiuno']
PUNCTUATIONS = [',', '.', '?', ';', '!', ':', '“']
ABSOLUTE_BACKEND_PATH = settings.ABSOLUTE_BACKEND_PATH


class Preprocessor():

    def __init__(self, file='corpus/normalized_text.txt'):
        '''
        Load normalized word pairs.
        The expected file format is:
        [word] [normalized_word]
        [word] [normalized_word]
        ...
        '''
        self.dictionary = {}
        with open('%s/%s' % (ABSOLUTE_BACKEND_PATH, file), 'r') as file:
            lines = file.read().splitlines()
            for line in lines:
                content = line.split()
                self.dictionary[content[0]] = content[1]

        # load model
        with open('%s/models/number_model.pkl' % ABSOLUTE_BACKEND_PATH, 'rb') as file:
            self.model = pkl.load(file)

        with open('%s/dicts/number_dict.pkl' % ABSOLUTE_BACKEND_PATH, 'rb') as file:
            self.number_dict = pkl.load(file)

    def normalize_to_lower(self):
        self.tokens = [word.lower() for word in self.tokens]
        return self

    def normalize_punctuations(self):
        results = []
        for token in self.tokens:
            last_index = 0
            for idx, char in enumerate(token):
                if char in PUNCTUATIONS:
                    if False if idx == 0 else token[idx - 1] in NUMBERS and False if idx == len(token) - 1 else token[idx + 1] in NUMBERS:
                        pass
                    else:
                        results.append(token[last_index:idx])
                        last_index = idx + 1
            results.append(token[last_index:])
        self.tokens = results
        return self

    def normalize_remove_empty(self):
        self.tokens = list(filter(lambda str: str != '', self.tokens))
        return self
    
    def normalize_username(self):
        dropped_tokens = []
        for token in self.tokens:
            if token.startswith('@') and token not in USERNAME_EXCEPTIONS:
                dropped_tokens.append(token)
        for token in dropped_tokens:
            self.tokens.remove(token)
        return self

    def normalize_hashtag(self):
        self.hashtag = []
        for token in self.tokens:
            if token.startswith('#'):
                self.hashtag.append(token)
        for token in self.hashtag:
            self.tokens.remove(token)
        return self

    def normalize_link(self):
        dropped_tokens = []
        for token in self.tokens:
            if 'http' in token:
                dropped_tokens.append(token)
        for token in dropped_tokens:
            self.tokens.remove(token)
        return self

    def normalize_repeated_chars(self):
        result = []
        for index, token in enumerate(self.tokens):
            new_token = ''
            num = 0
            last_char = ''
            for char in token:
                if char == last_char:
                    num += 1
                else:
                    if num == 2:
                        new_token += last_char
                    new_token += char
                    num = 1
                    last_char = char
            result.append(new_token)
        self.tokens = result
        return self

    def normalize_slang(self):
        '''
        Normalize list of tokens using our own dictionary, egs.
        gw -> saya, nga -> tidak, dst
        Returning list of normalized tokens.
        '''
        result = copy.deepcopy(self.tokens)
        for index, token in enumerate(self.tokens):
            if token in self.dictionary:
                result[index] = self.dictionary[token]
        self.tokens = result
        return self

    def normalize_number(self):
        '''
        '''

        result = copy.deepcopy(self.tokens)

        # find the number in the string
        for idx, token in enumerate(self.tokens):
            if not any(char in [',', '.'] for char in token) and not all(char in NUMBERS for char in token):
                features = []
                number_occurence = []
                is_contain_number = False
                for index, char in enumerate(token):
                    if char in NUMBERS:
                        is_contain_number = True
                        number_occurence.append(index)
                        # feature <number, charbefore, index, is_last, is_last_char_vowel, is_number_vowel>
                        feature = [int(char), ord(token[index-1]), index == 0, index == len(token) - 1, \
                            (self.number_dict['letter'][str(token[index-1])] in VOWEL) if token[index-1] in NUMBERS else token[index-1] in VOWEL, self.number_dict['letter'][char] in VOWEL]
                        # feature = [int(char), ord(token[index - 1]), 1 if index == 0 else 0, 1 if index == len(token) - 1 else 0, \
                        #     1 if token[index-1] in VOWEL else 0, 1 if number_dict['letter'][char] in VOWEL else 0]
                        features.append(feature)
                if is_contain_number:
                    features = np.array(features)
                    prediction = self.model.predict(features)
                    new_token = ''
                    old_index = -1
                    for i, index in enumerate(number_occurence):
                        new_token += token[old_index+1:index]
                        number = features[i][0]
                        target = prediction[i]
                        if target == 0:
                            tmp = new_token
                            for _ in range(number - 1):
                                new_token += tmp
                        elif target == 1:
                            new_token += self.number_dict['spelling'][str(number)]
                        elif target == 2:
                            new_token += self.number_dict['letter'][str(number)]
                        old_index = index
                    new_token += token[index+1:]
                    result[idx] = new_token

        return self

    def normalize(self, tokens):
        self.tokens = tokens
        # PIPELINE
        # normalize_username -> normalize_hashtag -> normalize_link -> ...
        self.normalize_to_lower().normalize_link().normalize_punctuations()\
            .normalize_remove_empty().normalize_username().normalize_hashtag()\
            .normalize_repeated_chars().normalize_slang().normalize_number()

        return self.tokens

    def normalize_once(self, tokens):
        self.hashtag = []

        results = []
        for token in tokens:
            new_token = token.lower() # lower case
            if 'http' not in token: # remove link
                new_tokens = [] # hold separate token from punctuations
                last_index = 0
                for idx, char in enumerate(new_token):
                    if char in PUNCTUATIONS:
                        if False if idx == 0 else token[idx - 1] in NUMBERS and False if idx == len(new_token) - 1 else new_token[idx + 1] in NUMBERS:
                            pass
                        else:
                            if new_token[last_index:idx] != '':
                                new_tokens.append(new_token[last_index:idx])
                            last_index = idx + 1
                if new_token[last_index:] != '':
                    new_tokens.append(new_token[last_index:])
                for new_token in new_tokens:
                    if not (new_token.startswith('@') and new_token not in USERNAME_EXCEPTIONS):
                        if new_token.startswith('#'):
                            self.hashtag.append(new_token)
                        else:
                            remove_rep_token = ''
                            num = 0
                            last_char = ''
                            for char in new_token:
                                if char == last_char:
                                    num += 1
                                else:
                                    if num == 2:
                                        remove_rep_token += last_char
                                    remove_rep_token += char
                                    num = 1
                                    last_char = char
                            
                            if remove_rep_token in self.dictionary:
                                remove_rep_token = self.dictionary[remove_rep_token]

                            
                            if not any(char in [',', '.'] for char in remove_rep_token) and not all(char in NUMBERS for char in remove_rep_token):
                                features = []
                                number_occurence = []
                                is_contain_number = False
                                for index, char in enumerate(remove_rep_token):
                                    if char in NUMBERS:
                                        is_contain_number = True
                                        number_occurence.append(index)
                                        # feature <number, charbefore, index, is_last, is_last_char_vowel, is_number_vowel>
                                        feature = [int(char), ord(remove_rep_token[index-1]), index == 0, index == len(remove_rep_token) - 1, \
                                            (self.number_dict['letter'][str(remove_rep_token[index-1])] in VOWEL) if remove_rep_token[index-1] in NUMBERS else remove_rep_token[index-1] in VOWEL, self.number_dict['letter'][char] in VOWEL]
                                        # feature = [int(char), ord(token[index - 1]), 1 if index == 0 else 0, 1 if index == len(token) - 1 else 0, \
                                        #     1 if token[index-1] in VOWEL else 0, 1 if number_dict['letter'][char] in VOWEL else 0]
                                        features.append(feature)
                                if is_contain_number:
                                    features = np.array(features)
                                    prediction = self.model.predict(features)
                                    new_token = ''
                                    old_index = -1
                                    for i, index in enumerate(number_occurence):
                                        new_token += token[old_index+1:index]
                                        number = features[i][0]
                                        target = prediction[i]
                                        if target == 0:
                                            tmp = new_token
                                            for _ in range(number - 1):
                                                new_token += tmp
                                        elif target == 1:
                                            new_token += self.number_dict['spelling'][str(number)]
                                        elif target == 2:
                                            new_token += self.number_dict['letter'][str(number)]
                                        old_index = index
                                    new_token += token[index+1:]
                                else:
                                    new_token = remove_rep_token
                            results.append(new_token)
        return results

    def remove_stopwords(self, tokens):
        return [word for word in tokens if not word in stopwords.words('indonesian')]

    def stem(self, tokens):
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
        self.tokens = [stemmer.stem(word) for word in tokens]
        self.normalize_remove_empty()
        return self.tokens

    def tokenize(self, text):
        return re.split('\n+| +\n+| +\r+| +|\n+|\r+', text)
        # return re.split('\n+| +\n+| +\r+|[\x80-\xff] +| +[\x80-\xff]|[\x80-\xff]| +|\n+|\r+', text)