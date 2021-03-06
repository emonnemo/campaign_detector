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
        results = ['Non-kampanye', 'Kubu 01', 'Kubu 02']
        query = request.data.get('query', '')
        response = requests.get('http://localhost:8000/api/v1/predict/?query=%s' % query)
        tweets = response.json().get('tweets')
        predictions = response.json().get('predictions')
        predicted_data = []
        for idx in range(len(tweets)):
            predicted_data.append({
                'tweet' : tweets[idx]['text'],
                'username': tweets[idx]['username'],
                'time': tweets[idx]['time'],
                'prediction': results[predictions[idx]],
            })

        template = get_template('scrap.html')
        html = template.render({'predicted_data': predicted_data, })
        return HttpResponse(html)