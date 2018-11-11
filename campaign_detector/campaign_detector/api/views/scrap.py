import csv
import json
import tweepy

from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

consumer_key = '97A4nowwbKg6gg5Cps5HRkoG4'
consumer_secret = 'v2x4suxa7xBmsmCD9ySBJXTzt9Ot4Kf8iaCD73YQFp7paIcZ5H'
access_token = '341808782-UH985JOeFwAoDYQsm2TGYgt6Ekq7h9iInwlrzVha'
access_token_secret = 'dIqQ6edncv0HBmfT6GzOyaX163WT6xXK1aAFQ6hx9fJtP'

auth = tweepy.auth.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

class ScrapView(APIView):

    def get(self, request, format=None):
        num_tweets = request.data.get('count', 10)
        tweets = []
        for tweet in tweepy.Cursor(api.search, q="jokowi -filter:retweets -filter:media", count= num_tweets, geocode='-7.4962531,110.2159931,600km', tweet_mode='extended').items(num_tweets): # , geocode=''
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
