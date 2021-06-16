from django.urls import path
from django.conf.urls import url, include

from . import views

# 웹 사용위한 path 추가 노민성 정종현
urlpatterns = [
    path('', views.start, name="start"),
    path('home/', views.home, name="home"),
    path('chattrain', views.chattrain, name="chattrain"),
    path('home/chatanswer', views.chatanswer, name="chatanswer"),
]
