from django.contrib import admin
from django.urls import path
from home import views

urlpatterns = [

    path('', views.index, name = 'home'),
    path('binary', views.predict_image_binary, name='binary'),
    path('multiclass', views.predict_image_multiclass, name='multiclass'),
   # path('binary', views.predict_image, name='predict_image'),
    #path('multiclass', views.predict_image_2, name='predict_image')
    #path('awais',views.awais, name = 'awais'),
    #path('contact',views.contact, name = 'contact')
]
