#  Under Construction / 工事中 

This project is currently under development. For updates, follow our X account: [@tattoqq9](https://x.com/tattoqq9)

このプロジェクトは現在開発中です。進捗についてはXアカウント[@tattoqq9](https://x.com/tattoqq9)をご確認ください。

---

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

### 🛨 工事中 🛨

このプロジェクトは現在開発中です。進捗についてはXアカウント[@your_account](https://x.com/your_account)をご確認ください。

---

### ログアプリケーション

このリポジトリには、ログのアップロードと解析を行うDjangoベースのWebアプリケーションが含まれています。このアプリケーションは、Dockerを使用してコンテナ化されており、簡単にセットアップしてデプロイできます。

