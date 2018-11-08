import logging

from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView


logger = logging.getLogger(__name__)


class PredictView(APIView):

    def get(self, request, format=None):
        tweet = request.data.get('tweet', '')

        return Response({'message': 'error'}, status=status.HTTP_200_OK)
