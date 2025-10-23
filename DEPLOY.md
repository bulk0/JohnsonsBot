# Инструкция по деплою в Yandex Cloud

## Предварительные требования

1. Установите [Yandex Cloud CLI](https://cloud.yandex.ru/docs/cli/quickstart)
2. Установите [Docker](https://docs.docker.com/get-docker/)
3. Создайте аккаунт в [Yandex Cloud](https://cloud.yandex.ru/)

## Шаги по деплою

### 1. Настройка Yandex Cloud

```bash
# Инициализация Yandex Cloud CLI
yc init

# Создание Container Registry
yc container registry create --name johnsons-bot-registry

# Получение ID реестра
yc container registry list
```

### 2. Настройка секретов

```bash
# Создание секрета для токена бота
yc lockbox secret create --name telegram-bot-token \
    --description "Telegram Bot Token" \
    --payload TELEGRAM_BOT_TOKEN=your_bot_token_here
```

### 3. Сборка и загрузка Docker образа

```bash
# Получение Docker credentials
yc container registry configure-docker

# Сборка образа
docker build -t cr.yandex/<registry-id>/johnsons-bot:latest .

# Загрузка образа в реестр
docker push cr.yandex/<registry-id>/johnsons-bot:latest
```

### 4. Деплой контейнера

```bash
# Замените YOUR_REGISTRY_ID в serverless.yaml на ваш ID реестра
# Затем выполните:
yc serverless container create --name johnsons-bot
yc serverless container revision deploy \
    --container-name johnsons-bot \
    --image cr.yandex/<registry-id>/johnsons-bot:latest \
    --cores 1 \
    --memory 512M \
    --concurrency 1 \
    --execution-timeout 30s \
    --service-account-id <service-account-id>
```

### 5. Настройка Webhook

Бот работает через webhook (Telegram отправляет обновления на ваш контейнер).

#### 5.1. Получить URL контейнера

```bash
yc serverless container get johnsons-bot
```

Скопируйте URL (например: `https://xxxxx.containers.yandexcloud.net`)

#### 5.2. Добавить секреты в GitHub

1. GitHub → **Settings** → **Secrets and variables** → **Actions**
2. Добавьте секреты:
   - **`WEBHOOK_URL`** = URL контейнера (без слеша в конце)
   - **`TELEGRAM_BOT_TOKEN`** = токен бота от @BotFather
   - **`YC_SERVICE_ACCOUNT_SECRET_KEY`** = JSON ключ сервисного аккаунта
   - **`YC_FOLDER_ID`** = ID каталога
   - **`YC_REGISTRY`** = ID Container Registry
   - **`YC_SERVICE_ACCOUNT_ID`** = ID сервисного аккаунта

#### 5.3. Запустить GitHub Actions

После добавления секретов push в main запустит автоматический деплой:

```bash
git push origin main
```

Или запустите вручную: **GitHub Actions** → **Run workflow**

#### 5.4. Проверить webhook

```bash
# Проверить настройку webhook
curl "https://api.telegram.org/bot<YOUR_TOKEN>/getWebhookInfo"

# Должен показать ваш URL контейнера
```

### 6. Проверка работы

```bash
# Просмотр логов в реальном времени
yc serverless container logs johnsons-bot --follow

# Просмотр статуса контейнера
yc serverless container list

# Список ревизий
yc serverless container revision list --container-name johnsons-bot
```

**Тестирование:** Откройте Telegram и отправьте боту `/start`

## Асинхронная обработка задач (YMQ + Worker)

### RU

1. Создайте S3‑бакет для артефактов (вход/выход):
   - Название: `johnsons-bot-artifacts`
   - Папки: `uploads/<job_id>/`, `results/<job_id>/`
2. Создайте очередь YMQ (SQS‑совместимая):
   - Основная: `johnsons-bot-jobs`
   - DLQ: `johnsons-bot-jobs-dlq`
   - Настройте ретраи/backoff по требованиям
3. Сервисный аккаунт (SA):
   - Роли: `storage.editor` (на бакет), `ymq.writer` (для webhook), `ymq.reader` (для worker)
4. Переменные окружения (Actions/Lockbox):
   - `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `S3_ENDPOINT=storage.yandexcloud.net`, `S3_REGION=ru-central1`
   - `YMQ_QUEUE_URL` (URL очереди), при использовании SQS API — совместимые ключи
5. Вебхук‑контейнер:
   - Загружает `.sav` в S3: `uploads/<job_id>/source.sav`
   - Публикует задачу в YMQ (payload с параметрами)
6. Воркер (Cloud Function с триггером YMQ):
   - Читает сообщение, скачивает файл из S3, считает, пишет результаты в S3 и шлёт прогресс/итог в Telegram

### EN

1. Create S3 bucket for artifacts (input/output):
   - Name: `johnsons-bot-artifacts`
   - Folders: `uploads/<job_id>/`, `results/<job_id>/`
2. Create YMQ queue (SQS‑compatible):
   - Main: `johnsons-bot-jobs`
   - DLQ: `johnsons-bot-jobs-dlq`
   - Configure retries/backoff as needed
3. Service Account (SA):
   - Roles: `storage.editor` (bucket), `ymq.writer` (webhook), `ymq.reader` (worker)
4. Environment variables (Actions/Lockbox):
   - `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `S3_ENDPOINT=storage.yandexcloud.net`, `S3_REGION=ru-central1`
   - `YMQ_QUEUE_URL` (queue URL), for SQS API provide compatible credentials
5. Webhook container:
   - Uploads `.sav` to S3: `uploads/<job_id>/source.sav`
   - Publishes job to YMQ (payload with parameters)
6. Worker (Cloud Function with YMQ trigger):
   - Consumes message, downloads file from S3, computes, writes results to S3 and sends progress/final output to Telegram

## Важные замечания

1. **Ресурсы**: В `serverless.yaml` указано 512MB памяти и 1 ядро. Эти значения можно изменить в зависимости от нагрузки.

2. **Логирование**: Логи отправляются в Cloud Logging. Вы можете просматривать их через веб-консоль или CLI.

3. **Масштабирование**: Параметр `concurrency: 1` означает, что будет обрабатываться только один запрос одновременно. При необходимости это значение можно увеличить.

4. **Timeout**: Установлен таймаут в 30 секунд. Если ваши операции требуют больше времени, увеличьте значение `execution-timeout`.

## Локальная разработка

Для локальной разработки бот автоматически использует **polling** режим:

```bash
# Создайте .env файл (не коммитится в git)
echo "TELEGRAM_BOT_TOKEN=your_token_here" > .env

# Установите зависимости
pip install -r requirements.txt

# Запустите бота локально
python bot.py

# В логах увидите: "Starting bot in polling mode (local development)"
```

**Важно:** Не устанавливайте `WEBHOOK_URL` в `.env` для локальной разработки!

## Обновление бота

Через GitHub Actions (рекомендуется):

```bash
git add .
git commit -m "feat: ваши изменения"
git push origin main
```

Или вручную через CLI:

```bash
docker build -t cr.yandex/<registry-id>/johnsons-bot:latest .
docker push cr.yandex/<registry-id>/johnsons-bot:latest
yc serverless container revision deploy --container-name johnsons-bot \
    --image cr.yandex/<registry-id>/johnsons-bot:latest
```

## Мониторинг и обслуживание

- Используйте Cloud Logging для просмотра логов
- Настройте алерты в Cloud Monitoring
- Регулярно проверяйте использование ресурсов

## Оценка стоимости

Yandex Cloud Serverless Containers тарифицируется по времени выполнения и использованным ресурсам. Рекомендуется:

1. Установить бюджетные алерты
2. Регулярно проверять использование ресурсов
3. Оптимизировать параметры контейнера по мере необходимости

## Поддержка

При возникновении проблем:

1. Проверьте логи в Cloud Logging
2. Убедитесь, что все секреты настроены правильно
3. Проверьте квоты и лимиты вашего аккаунта
