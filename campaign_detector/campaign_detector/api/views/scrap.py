import csv
import json
import tweepy

from django.conf import settings
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

consumer_key = settings.CONSUMER_KEY
consumer_secret = settings.CONSUMER_SECRET
access_token = settings.ACCESS_TOKEN
access_token_secret = settings.ACCESS_TOKEN_SECRET

auth = tweepy.auth.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

class ScrapView(APIView):

    def get(self, request, format=None):
        num_tweets = request.data.get('count', 10)
        query = request.GET.get('query')
        print (query)
        tweets = []
        for tweet in tweepy.Cursor(api.search, q="%s -filter:retweets -filter:media" % query, count= num_tweets, geocode='-7.4962531,110.2159931,600km', tweet_mode='extended').items(num_tweets): # , geocode=''
            teks = tweet.full_text.encode('utf-8')
            lokasi = None if not tweet.place else tweet.place.name
            waktu = tweet.created_at
            user = tweet.user.screen_name
            label = ''
            tweets.append({
                'text': teks,
                'time': waktu,
                'username': user
            })

        return Response({'tweets': tweets}, status=status.HTTP_200_OK)
