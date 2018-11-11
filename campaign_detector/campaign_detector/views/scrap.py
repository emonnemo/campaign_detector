import requests
from django.template.loader import get_template
from django.http import HttpResponse
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView


class ScrapView(APIView):

    def get(self, request, format=None):

        template = get_template('scrap.html')
        html = template.render({'text_before': '', })
        return HttpResponse(html)

    def post(self, request, format=None):
        response = requests.get('http://localhost:8000/api/v1/predict/')
        tweets = response.json().get('tweets')
        predictions = response.json().get('predictions1')
        predicted_data = []
        for idx in range(len(tweets) - 1):
            predicted_data.append({
                'tweet' : tweets[idx],
                'username': 'dummy',
                'time': 'dummy',
                'prediction': predictions[idx][0],
            })

        template = get_template('scrap.html')
        html = template.render({'predicted_data': predicted_data, })
        return HttpResponse(html)