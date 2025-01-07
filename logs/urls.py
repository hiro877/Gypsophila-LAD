from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),  # トップページ
    path('upload/', views.upload_log, name='upload_log'),  # ログアップロードページ
    path('future_app1/', views.future_app1, name='future_app1'),  # 将来のアプリケーション1
    path('future_app2/', views.future_app2, name='future_app2'),  # 将来のアプリケーション2
    path('about/', views.about, name='about'),  # Aboutページ
    path('parse_result/', views.parse_result_view, name='parse_result'),
]

