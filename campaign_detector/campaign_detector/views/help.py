import requests
from django.template.loader import get_template
from django.http import HttpResponse
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView


class HelpView(APIView):

    def get(self, request, format=None):

        template = get_template('help.html')
        html = template.render()
        return HttpResponse(html)