from django.urls import path
from .views import spam_detection_api

urlpatterns = [
    path('predict/', spam_detection_api, name='predict'),
]
