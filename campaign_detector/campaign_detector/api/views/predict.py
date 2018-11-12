import logging
import pickle as pkl
import requests

import sys
# Add the backend folder path to the sys.path list
sys.path.append('../../campaign_detection/backend')
from collections import Counter
from train_detection import extract_feature
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView


ABSOLUTE_BACKEND_PATH = '/Users/andikakusuma/Documents/Kuliah/NLP/Tubes_text/campaign_detection/backend'
logger = logging.getLogger(__name__)


class PredictView(APIView):

    def post(self, request, format=None):

        model1 = pkl.load(open('%s/models/mlp.pkl' % ABSOLUTE_BACKEND_PATH, 'rb'))
        model2 = pkl.load(open('%s/models/dtl.pkl' % ABSOLUTE_BACKEND_PATH, 'rb'))
        model3 = pkl.load(open('%s/models/svc.pkl' % ABSOLUTE_BACKEND_PATH, 'rb'))

        word_lists = pkl.load(open('%s/models/word_lists.pkl' % ABSOLUTE_BACKEND_PATH, 'rb'))
        hashtag_lists = pkl.load(open('%s/models/hashtag_lists.pkl' % ABSOLUTE_BACKEND_PATH, 'rb'))
        tweet = request.data.get('tweet', '')
        feature = extract_feature(tweet, hashtag_lists, word_lists)

        all_predictions = []
        all_predictions.append(model1.predict([feature])[0])
        all_predictions.append(model2.predict([feature])[0])
        all_predictions.append(model3.predict([feature])[0])
        counts = Counter(all_predictions)
        prediction, _ = counts.most_common(1)[0]

        return Response({'tweet': tweet, 'prediction': prediction}, status=status.HTTP_200_OK)

    def get(self, request, format=None):
        query = request.GET.get('query', '')
        response = requests.get('http://localhost:8000/api/v1/scrap/?query=%s' % query)
        tweets = response.json().get('tweets')
        predictions = []
        for tweet in tweets:
            model1 = pkl.load(open('%s/models/mlp.pkl' % ABSOLUTE_BACKEND_PATH, 'rb'))
            model2 = pkl.load(open('%s/models/dtl.pkl' % ABSOLUTE_BACKEND_PATH, 'rb'))
            model3 = pkl.load(open('%s/models/svc.pkl' % ABSOLUTE_BACKEND_PATH, 'rb'))

            word_lists = pkl.load(open('%s/models/word_lists.pkl' % ABSOLUTE_BACKEND_PATH, 'rb'))
            hashtag_lists = pkl.load(open('%s/models/hashtag_lists.pkl' % ABSOLUTE_BACKEND_PATH, 'rb'))
            # tweet = request.data.get('tweet', '')
            feature = extract_feature(tweet['text'], hashtag_lists, word_lists)

            all_predictions = []
            all_predictions.append(model1.predict([feature])[0])
            all_predictions.append(model2.predict([feature])[0])
            all_predictions.append(model3.predict([feature])[0])
            counts = Counter(all_predictions)
            prediction, _ = counts.most_common(1)[0]
            predictions.append(prediction)

        return Response({'tweets': tweets, 'predictions': predictions})

        