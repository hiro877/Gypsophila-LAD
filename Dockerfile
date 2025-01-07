# ベースイメージ
FROM python:3.10-slim

# 作業ディレクトリを設定
WORKDIR /app

# 必要なファイルをコピー
COPY requirements.txt ./

# Pythonパッケージをインストール
RUN pip install --no-cache-dir -r requirements.txt

# プロジェクトのコードをコピー
COPY . .

# ポートを公開
EXPOSE 8000

# アプリケーションの起動コマンド
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]

