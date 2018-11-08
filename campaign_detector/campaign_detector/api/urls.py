from django.contrib.auth import views
from django.urls import path
from campaign_detector.api.views import predict

urlpatterns = [
    path('predict/', predict.PredictView.as_view(), name='predict'),
]
