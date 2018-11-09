from django.contrib.auth import views
from django.urls import path
from campaign_detector.api.views import predict
from campaign_detector.api.views import scrap

urlpatterns = [
    path('predict/', predict.PredictView.as_view(), name='predict'),
    path('scrap/', scrap.ScrapView.as_view(), name='scrap'),
]
