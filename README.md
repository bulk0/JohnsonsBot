# Johnson's Relative Weights Calculator Bot

![CI/CD](https://github.com/bulk0/JohnsonsBot/actions/workflows/main.yml/badge.svg?branch=main)
[![codecov](https://codecov.io/gh/bulk0/JohnsonsBot/branch/main/graph/badge.svg)](https://codecov.io/gh/bulk0/JohnsonsBot)
![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

Telegram бот для автоматизированного расчета относительных весов Джонсона на основе SPSS данных.

## 🌟 Возможности

- Прием и обработка SPSS файлов (.sav)
- Автоматическая валидация данных
- Расчет весов Джонсона для:
  - Общей выборки
  - Групповой анализ
  - Анализ по брендам
- Поддержка различных форматов вывода (Excel, CSV)
- Обработка пропущенных значений
- Интерактивный процесс выбора переменных

## 🌿 Ветки разработки

- `main` - основная ветка, содержит стабильный код
- `develop` - ветка для разработки и тестирования новых функций

## 🚀 Установка

1. Клонируйте репозиторий:
```bash
git clone https://github.com/YOUR_USERNAME/JohnsonsBot.git
cd JohnsonsBot
```

2. Создайте виртуальное окружение:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate  # Windows
```

3. Установите зависимости:
```bash
pip install -r requirements.txt
```

4. Создайте файл .env и добавьте токен бота:
```bash
TELEGRAM_BOT_TOKEN=your_bot_token_here
```

## 🔧 Использование

1. Запустите бота:
```bash
python bot.py
```

2. Отправьте боту SPSS файл (.sav)
3. Следуйте инструкциям бота для выбора:
   - Зависимых переменных
   - Независимых переменных
   - Переменных для группировки (опционально)

## 📊 Методология

Бот использует несколько подходов к обработке данных:
- MICE (Multiple Imputation by Chained Equations)
- Гибридный подход с базовой линией
- Простая импутация средними значениями

Подробное описание методологии доступно в файле Multiple Imputations Readme.md

## 🛠 Технологии

- Python 3.11
- python-telegram-bot
- pandas
- numpy
- pyreadstat
- scikit-learn

## 📝 Лицензия

[MIT](LICENSE)

## 👥 Авторы

- [@BaukoshaWorks](https://t.me/BaukoshaWorks) - Разработка и поддержка

## 🤝 Участие в проекте

Мы приветствуем ваш вклад в проект! Пожалуйста:

1. Форкните репозиторий
2. Создайте ветку для ваших изменений
3. Внесите изменения
4. Отправьте пул-реквест

## 📞 Поддержка

Если у вас возникли вопросы или проблемы:
- Создайте Issue в репозитории
- Свяжитесь с [@BaukoshaWorks](https://t.me/BaukoshaWorks) в Telegram
