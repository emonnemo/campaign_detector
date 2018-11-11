import requests
from django.template.loader import get_template
from django.http import HttpResponse
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView


class HomeView(APIView):

    def get(self, request, format=None):

        template = get_template('home.html')
        html = template.render({'text_before': '', })
        return HttpResponse(html)

    def post(self, request, format=None):
        results = ['Non-kampanye', 'Kubu 01', 'Kubu 02']
        text = request.data['input-text']
        response = requests.post('http://localhost:8000/api/v1/predict/', data={'tweet': text})
        prediction = response.json().get('prediction1')[0]

        template = get_template('home.html')
        html = template.render({'result': results[prediction], 'text_before': text, })
        return HttpResponse(html)