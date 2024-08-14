from django.urls import path
from .views import index



urlpatterns = [
    path("", index, name="index"),
    path("model/<str:groupId>", index),
]
