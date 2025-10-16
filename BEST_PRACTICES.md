# Best Practices для Production

Чек-лист лучших практик для работы с приложением в облаке Yandex Cloud.

---

## ✅ Что уже настроено

- [x] **Защита ветки main** - требуется Pull Request для изменений
- [x] **Секреты в .gitignore** - `.env` не попадает в репозиторий
- [x] **CI/CD pipeline** - автоматический деплой через GitHub Actions
- [x] **Переменные окружения** - `TELEGRAM_BOT_TOKEN` передаётся через secrets
- [x] **Версионирование образов** - используется git commit SHA
- [x] **Автоматические тесты** - pytest запускается перед деплоем

---

## 🔐 1. Безопасность

### GitHub Secrets (настроено)
Все секреты хранятся в GitHub Settings → Secrets:
- `TELEGRAM_BOT_TOKEN` - токен Telegram бота
- `YC_SERVICE_ACCOUNT_SECRET_KEY` - ключ сервисного аккаунта
- `YC_FOLDER_ID` - ID каталога Yandex Cloud
- `YC_REGISTRY` - ID Container Registry
- `YC_SERVICE_ACCOUNT_ID` - ID сервисного аккаунта

### Проверка безопасности
```bash
# Убедитесь, что секреты не в коде
git grep -i "token\|secret\|password" | grep -v ".gitignore"

# Проверьте файлы, которые будут закоммичены
git status
```

### Рекомендации
- [ ] Регулярно ротируйте токены (раз в 3-6 месяцев)
- [ ] Используйте разные токены для dev/prod
- [ ] Ограничьте права сервисного аккаунта минимально необходимыми

---

## 📊 2. Мониторинг и логирование

### Просмотр логов контейнера

```bash
# Real-time логи
yc serverless container logs johnsons-bot --follow

# Логи за период
yc logging read \
  --folder-id=$(yc config get folder-id) \
  --filter='resource.type="serverless.container" AND resource.id="<container-id>"' \
  --since="2025-10-16T00:00:00Z"
```

### Метрики в Yandex Cloud Console

**Что отслеживать:**
- 📈 Количество запросов в секунду
- ⏱️ Среднее время ответа (latency)
- ❌ Количество ошибок (HTTP 5xx)
- 💾 Использование памяти
- 🔄 CPU utilization

**Где смотреть:**
[console.cloud.yandex.ru/folders/{folder-id}/serverless-containers](https://console.cloud.yandex.ru)

### Настройка алертов

1. **Monitoring** → **Alerts** → **Create Alert**
2. Создайте алерты на:
   - Количество ошибок > 10 за 5 минут
   - Использование памяти > 80%
   - Время ответа > 10 секунд

---

## 💰 3. Контроль стоимости

### Настройка бюджетных алертов

1. **Yandex Cloud Console** → **Billing** → **Budgets**
2. Создайте бюджет с лимитом (например, 1000₽/месяц)
3. Настройте уведомления:
   - 50% бюджета
   - 80% бюджета
   - 100% бюджета

### Оптимизация ресурсов

**Текущие настройки:**
```yaml
revision-memory: 512Mb        # Память контейнера
revision-cores: 1             # Количество ядер
revision-concurrency: 1       # Одновременных запросов
revision-execution-timeout: 30s  # Таймаут выполнения
```

**Рекомендации:**
- Если бот обрабатывает мало запросов - уменьшите memory до 256Mb
- Увеличьте concurrency до 2-5 для экономии на холодных стартах
- Используйте минимально необходимый timeout

### Мониторинг стоимости

```bash
# Просмотр детализации расходов
yc billing service list

# Экспорт в CSV для анализа
yc billing service list-billable-object-bindings \
  --format csv > billing.csv
```

---

## 🧪 4. Тестирование

### Локальное тестирование

```bash
# Установка зависимостей
pip install -r requirements.txt
pip install pytest pytest-cov

# Запуск всех тестов
pytest tests/ -v

# Запуск с покрытием
pytest tests/ --cov=. --cov-report=html

# Проверка конкретного файла
pytest tests/test_basic.py -v
```

### CI/CD тесты

Тесты автоматически запускаются в GitHub Actions перед каждым деплоем.

**Рекомендации:**
- [ ] Добавьте интеграционные тесты
- [ ] Настройте проверку code coverage (минимум 70%)
- [ ] Добавьте pre-commit hooks для локальной проверки

---

## 🔄 5. Версионирование и откат

### Просмотр ревизий контейнера

```bash
# Список всех ревизий
yc serverless container revision list \
  --container-name johnsons-bot

# Детали конкретной ревизии
yc serverless container revision get <revision-id>
```

### Откат на предыдущую версию

```bash
# Найдите ID нужной ревизии
yc serverless container revision list --container-name johnsons-bot

# Откатите на конкретную ревизию
yc serverless container rollback \
  --name johnsons-bot \
  --revision-id <previous-revision-id>
```

### Стратегия версионирования

**Текущий подход:** Используется git commit SHA как тег образа
- ✅ Уникальность каждой версии
- ✅ Прослеживаемость изменений
- ✅ Легко найти соответствующий код

**Теги Docker образов:**
```
cr.yandex/<registry-id>/johnsons-bot:<git-sha>
```

---

## 🚨 6. Аварийное восстановление

### Backup данных

**Что бэкапить:**
- Код: уже в Git ✅
- Конфигурация: в GitHub Secrets ✅
- Данные пользователей: если есть БД (настройте автобэкапы)

### Disaster Recovery Plan

1. **Полная потеря контейнера:**
   ```bash
   # Пересоздать контейнер
   yc serverless container create --name johnsons-bot
   
   # Запустить workflow деплоя вручную
   # GitHub Actions → Run workflow
   ```

2. **Проблемы с образом:**
   ```bash
   # Откат на предыдущую ревизию
   yc serverless container rollback --name johnsons-bot
   ```

3. **Проблемы с токеном бота:**
   - Получите новый токен у @BotFather
   - Обновите секрет `TELEGRAM_BOT_TOKEN` в GitHub
   - Задеплойте заново

### Контакты для экстренных случаев

- **Yandex Cloud Support:** [support@cloud.yandex.ru](mailto:support@cloud.yandex.ru)
- **Telegram Support:** [@BotSupport](https://t.me/BotSupport)

---

## 📈 7. Масштабирование

### Горизонтальное масштабирование

Serverless Containers автоматически масштабируется:
```yaml
revision-concurrency: 1  # Увеличьте для обработки нескольких запросов
```

**Рекомендации по concurrency:**
- `1` - для задач с большим использованием памяти
- `2-5` - для обычных ботов
- `10+` - для легковесных stateless приложений

### Мониторинг производительности

```bash
# Проверка активных инстансов
yc serverless container revision get <revision-id>

# Метрики запросов
yc monitoring metric-data \
  --folder-id $(yc config get folder-id) \
  --from "2025-10-16T00:00:00Z"
```

---

## 📝 8. Документация

### Обязательная документация

- [x] `README.md` - описание проекта
- [x] `DEPLOY.md` - инструкция по деплою
- [x] `BEST_PRACTICES.md` - этот документ
- [ ] `API.md` - описание команд бота (если нужно)
- [ ] `TROUBLESHOOTING.md` - решение частых проблем

### Документирование изменений

**Используйте Conventional Commits:**
```bash
git commit -m "feat: добавить команду /stats"
git commit -m "fix: исправить ошибку при загрузке файла"
git commit -m "docs: обновить README"
```

**Создавайте релизы в GitHub:**
- Settings → Releases → Create new release
- Используйте семантическое версионирование: v1.0.0, v1.1.0, v2.0.0

---

## 🔧 9. Регулярное обслуживание

### Еженедельно
- [ ] Проверить логи на ошибки
- [ ] Проверить использование ресурсов
- [ ] Проверить расходы в биллинге

### Ежемесячно
- [ ] Обновить зависимости Python
- [ ] Проверить security alerts от GitHub
- [ ] Проанализировать метрики использования

### Ежеквартально
- [ ] Ротация токенов и ключей
- [ ] Аудит прав доступа
- [ ] Проверка backup'ов
- [ ] Обновление документации

### Обновление зависимостей

```bash
# Проверка устаревших пакетов
pip list --outdated

# Обновление requirements.txt
pip freeze > requirements.txt

# Проверка безопасности
pip install safety
safety check
```

---

## 🎯 10. KPI и метрики успеха

### Технические метрики
- **Uptime:** > 99.9%
- **Response time:** < 2 секунды
- **Error rate:** < 1%
- **Test coverage:** > 70%

### Бизнес-метрики
- Количество активных пользователей
- Количество обработанных файлов
- Средняя длительность сессии
- Retention rate

### Мониторинг в Telegram

Добавьте команду для администратора:
```python
@admin_only
async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Показать статистику бота"""
    # Количество пользователей, файлов, ошибок и т.д.
```

---

## 📞 Полезные ссылки

- **Yandex Cloud Console:** [console.cloud.yandex.ru](https://console.cloud.yandex.ru)
- **Документация Serverless Containers:** [cloud.yandex.ru/docs/serverless-containers](https://cloud.yandex.ru/docs/serverless-containers)
- **Telegram Bot API:** [core.telegram.org/bots/api](https://core.telegram.org/bots/api)
- **GitHub Actions:** [docs.github.com/actions](https://docs.github.com/en/actions)

---

## ✨ Следующие шаги

1. **Сейчас:**
   - [ ] Добавить секрет `TELEGRAM_BOT_TOKEN` в GitHub (если еще не добавлен)
   - [ ] Протестировать новый workflow с тестами
   - [ ] Настроить алерты в Yandex Cloud

2. **В ближайшее время:**
   - [ ] Настроить бюджетные алерты
   - [ ] Добавить больше тестов
   - [ ] Документировать команды бота

3. **Долгосрочно:**
   - [ ] Настроить staging окружение
   - [ ] Добавить метрики использования
   - [ ] Реализовать graceful shutdown

---

**Дата последнего обновления:** 16 октября 2025

