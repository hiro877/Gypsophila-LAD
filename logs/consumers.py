# logs/consumers.py
import json
from channels.generic.websocket import AsyncWebsocketConsumer

class ProgressConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.group_name = "training_progress"
        # グループに参加
        await self.channel_layer.group_add(
            self.group_name,
            self.channel_name
        )
        await self.accept()

    async def disconnect(self, close_code):
        # グループから退出
        await self.channel_layer.group_discard(
            self.group_name,
            self.channel_name
        )

    # グループからのメッセージ受信時のハンドラ
    async def training_progress(self, event):
        message = event["message"]
        await self.send(text_data=json.dumps({
            "message": message
        }))
