"""ClusterCast URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path("", views.cluster_group, name="ClusterCast-home"),
    path("cluster-run/", views.cluster_run, name="ClusterCast-cluster-run"),
    path("cluster-group/", views.cluster_group, name="ClusterCast-cluster-group"),
    path("cluster-group/<int:id>/", views.cluster_group_detail, name="cluster_group_detail"),
    path("cluster/<int:group_id>/<int:cluster_id>/", views.cluster_detail, name="cluster_detail"),
    path("cluster/<int:group_id>/<int:cluster_id>/train-models/", views.train_new_models, name="cluster_train"),
    path("forcast/", views.forcast, name="ClusterCast-forcast"),
    path("forcast/<int:forcast_id>/<str:prediction_start_date>/<str:prediction_end_date>/", views.forcast_detail, name="forcast_detail"),
    path("predict/<int:prediction_id>/", views.prediction_detail, name="prediction_detail"),
    path("get_prediction_vis_files/<int:prediction_id>/", views.get_prediction_vis_files, name="get_prediction_vis_files"),
    path("get_models/<model_ids>/", views.get_models, name="get_models"),
]