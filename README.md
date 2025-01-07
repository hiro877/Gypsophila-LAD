# Log Application

This repository contains a Django-based web application for uploading and parsing logs. The application is containerized using Docker for easy setup and deployment.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Getting Started](#getting-started)
    - [Clone the Repository](#1-clone-the-repository)
    - [Build and Start the Docker Containers](#2-build-and-start-the-docker-containers)
    - [Access the Application](#3-access-the-application)
    - [Stop the Application](#4-stop-the-application)
3. [Detailed Steps](#detailed-steps)
4. [Notes](#notes)
5. [Troubleshooting](#troubleshooting)
6. [日本語版 (Japanese Version)](#日本語版-japanese-version)

---

## Prerequisites

Before starting, ensure you have the following installed on your system:

- [Docker](https://www.docker.com/)
- [Docker Compose](https://docs.docker.com/compose/)

---

## Getting Started

Follow these steps to set up and run the application:

### 1. Clone the Repository

```bash
git clone <repository-url>
cd <repository-directory>
```

### 2. Build and Start the Docker Containers

Run the following command to build and start the containers:

```bash
docker-compose up --build
```

### 3. Access the Application

Once the containers are up and running, the application will be available at:

```
http://localhost:8000
```

### 4. Stop the Application

To stop the containers, press `Ctrl+C` or run the following command in a separate terminal:

```bash
docker-compose down
```

---

## Detailed Steps

1. **Build the Docker Images**:
   If you want to ensure everything is built cleanly, use the following command:

   ```bash
   docker-compose build
   ```

2. **Start the Containers in Detached Mode**:
   Run the containers in the background:

   ```bash
   docker-compose up -d
   ```

3. **Check Running Containers**:
   Verify that the containers are running:

   ```bash
   docker ps
   ```

4. **View Logs**:
   To debug or monitor, view the container logs:

   ```bash
   docker-compose logs
   ```

5. **Restart the Containers**:
   If needed, restart the application:

   ```bash
   docker-compose restart
   ```

6. **Stop and Remove Containers**:
   To clean up, stop and remove containers and associated networks:

   ```bash
   docker-compose down --volumes
   ```

---

## Notes

- Ensure that you have the correct permissions to run Docker commands (e.g., use `sudo` if required).
- The application uses SQLite as the default database for simplicity. You can configure other databases by modifying the `DATABASES` setting in `settings.py`.

---

## Troubleshooting

1. **Port Conflicts**:
   Ensure no other services are running on port `8000`.

2. **Docker Daemon**:
   Verify Docker is running:
   ```bash
   docker info
   ```

3. **Environment Variables**:
   Update the `.env` file if required and ensure it’s correctly formatted.

4. **Permission Issues**:
   If you encounter file permission errors, use:
   ```bash
   sudo docker-compose up --build
   ```

Feel free to open an issue if you face any difficulties.

---

## 日本語版 (Japanese Version)

### ログアプリケーション

このリポジトリには、ログのアップロードと解析を行うDjangoベースのWebアプリケーションが含まれています。このアプリケーションは、Dockerを使用してコンテナ化されており、簡単にセットアップしてデプロイできます。

---

### 必要条件

始める前に、以下がインストールされていることを確認してください：

- [Docker](https://www.docker.com/)
- [Docker Compose](https://docs.docker.com/compose/)

---

### 始め方

以下の手順に従って、アプリケーションをセットアップし実行します：

#### 1. リポジトリをクローン

```bash
git clone <repository-url>
cd <repository-directory>
```

#### 2. Dockerコンテナをビルドして起動

以下のコマンドを実行して、コンテナをビルドおよび起動します：

```bash
docker-compose up --build
```

#### 3. アプリケーションにアクセス

コンテナが起動すると、アプリケーションは以下のURLで利用可能になります：

```
http://localhost:8000
```

#### 4. アプリケーションを停止

コンテナを停止するには、`Ctrl+C`を押すか、別のターミナルで以下のコマンドを実行してください：

```bash
docker-compose down
```

---

### 詳細な手順

1. **Dockerイメージのビルド**:
   必要に応じてクリーンビルドを実行してください：
   ```bash
   docker-compose build
   ```

2. **バックグラウンドでコンテナを起動**:
   アプリケーションをバックグラウンドで実行する場合：
   ```bash
   docker-compose up -d
   ```

3. **実行中のコンテナを確認**:
   実行中のコンテナを確認するには：
   ```bash
   docker ps
   ```

4. **ログの表示**:
   デバッグやモニタリングを行うには：
   ```bash
   docker-compose logs
   ```

5. **コンテナの再起動**:
   アプリケーションを再起動する場合：
   ```bash
   docker-compose restart
   ```

6. **コンテナと関連データの削除**:
   コンテナと関連データを削除するには：
   ```bash
   docker-compose down --volumes
   ```

---

### メモ

- Dockerコマンドを実行するための権限を確認してください（必要に応じて`sudo`を使用してください）。
- デフォルトではSQLiteを使用しています。他のデータベースを使用する場合は、`settings.py` 内の `DATABASES` 設定を更新してください。

---

### トラブルシューティング

1. **ポート競合**:
   ポート `8000` で他のサービスが動作していないことを確認してください。

2. **Dockerデーモンの確認**:
   Dockerが動作していることを確認するには：
   ```bash
   docker info
   ```

3. **環境変数の確認**:
   `.env` ファイルを適切に更新し、フォーマットが正しいことを確認してください。

4. **権限エラー**:
   ファイル権限のエラーが発生した場合は：
   ```bash
   sudo docker-compose up --build
   ```

---




