# ベースイメージ
FROM python:3.10-slim

# 作業ディレクトリを設定
WORKDIR /app

# 必要なシステムパッケージをインストール
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# PyTorchのインストール（GPU対応とCPU対応を分岐）
# 環境変数 USE_GPU によって制御
ARG USE_GPU=false
RUN if [ "$USE_GPU" = "true" ]; then \
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118; \
    else \
        pip install torch torchvision torchaudio; \
    fi

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

