# log_ad_app/asgi.py

import os
from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack
import logs.routing  # WebSocket用のルーティング設定

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "log_ad_app.settings")

# DjangoのASGIアプリケーションを取得
django_asgi_app = get_asgi_application()

application = ProtocolTypeRouter({
    "http": django_asgi_app,  # HTTPリクエストは通常のDjangoアプリケーションで処理
    "websocket": AuthMiddlewareStack(
        URLRouter(
            logs.routing.websocket_urlpatterns  # /ws/progress/などのWebSocketルート
        )
    ),
})
