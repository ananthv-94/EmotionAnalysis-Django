
from django.contrib import admin
from django.urls import path
from emo import views
from django.views.decorators.csrf import csrf_exempt


urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.Base.as_view(), name='base'),
    path('emotion/', views.Emotion.as_view(), name='emotion'),
    path('video/', views.video, name = 'video'),
    path('rearrange/', views.Rearrange.as_view(), name = 'rearrange'),
    path('restapi_move/', csrf_exempt(views.Restapi_move.as_view()), name = 'restapi_move'),
    path('move_image/', views.Move_images.as_view(), name = 'move_image'),
    path('train/', views.Train_model.as_view(), name = 'train'),
    path('restapi_predict/', csrf_exempt(views.Restapi_predict.as_view()), name = 'restapi_predict'),
]
