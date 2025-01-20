from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),  # トップページ
    path('upload/', views.upload_log, name='upload_log'),  # ログアップロードページ
    path('anomaly_detection/upload/', views.anomaly_detection_upload, name='anomaly_detection_upload'),  # 新しいアップロードページ
    path('anomaly_detection/', views.anomaly_detection, name='anomaly_detection'),
    path('future_app2/', views.future_app2, name='future_app2'),  # 将来のアプリケーション2
    path('about/', views.about, name='about'),  # Aboutページ
    path('parse_result/', views.parse_result_view, name='parse_result'),
    # 学習とテストの追加
    path('train/', views.train, name='train'),
    path('test/', views.test, name='test'),
]

