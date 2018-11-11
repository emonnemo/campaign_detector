from django.template.loader import get_template
from django.http import HttpResponse
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView


class HomeView(APIView):

    def get(self, request, format=None):
        name = 'test'

        template = get_template('home.html')
        html = template.render({'name': name, })
        return HttpResponse(html)

    def post(self, request, format=None):
        name = 'non-kampanye'

        template = get_template('home.html')
        html = template.render({'result': name, })
        return HttpResponse(html)