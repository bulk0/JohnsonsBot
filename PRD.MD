# Johnson's Bot MVP - Product Requirements Document

-------------------

## ENGLISH VERSION

## 1. Product Overview

### 1.1 Problem Statement
Marketing research managers need an efficient and user-friendly tool to perform Johnson's relative weights calculations on their SPSS data. Currently, this process might be time-consuming and require specific technical knowledge.

### 1.2 Solution
A Telegram bot that automates the Johnson's relative weights calculation process, handles data validation, and guides users through the necessary data transformations.

### 1.3 Target Users
- Primary users: Marketing research managers
- User characteristics: Familiar with SPSS data formats and marketing research methodology

## 2. Features and Functionality

### 2.1 Core Features (MVP)

#### Data Input
- Accept SPSS database files (.sav format) through Telegram
- Interactive data structure validation
- Guided data transformation process if needed

#### Analysis Configuration
- Option to select analysis scope:
  - Total base analysis
  - Group-based analysis
- Option to configure brand-specific calculations:
  - Single brand analysis
  - Multiple brands analysis

#### Calculations
- Perform Johnson's relative weights calculations
- Support for different data structures
- Automatic data transformation when needed

#### Output Generation
- Generate results in multiple formats:
  - Excel (.xlsx) format for weights
  - CSV format for weights
  - CSV format for processed data

### 2.2 User Interaction Flow

1. **Initial Contact**
   - User starts interaction with the bot
   - Bot provides welcome message and instructions

2. **Data Upload**
   - User uploads SPSS file
   - Bot validates file format and structure

3. **Data Structure Validation**
   - Bot checks if data matches required structure
   - If structure doesn't match:
     - Bot initiates interactive dialogue
     - Guides user through structure clarification
     - Performs necessary transformations

4. **Analysis Configuration**
   - Bot prompts user to select analysis scope
   - User specifies if group-based analysis is needed
   - User indicates brand analysis preferences

5. **Processing and Results**
   - Bot performs calculations
   - Generates output files
   - Delivers results to user

## 3. Technical Requirements

### 3.1 Platform
- Telegram Bot API integration
- Python-based backend
- SPSS file processing capabilities

### 3.2 Performance Requirements
- Support for standard SPSS file sizes
- Reasonable calculation time for typical datasets

### 3.3 Error Handling
- Interactive error resolution through user dialogue
- Clear error messages and guidance
- Data validation at each step

## 4. User Experience

### 4.1 Bot Communication
- Clear, professional communication style
- Step-by-step guidance
- Interactive problem resolution

### 4.2 Data Validation
- Immediate feedback on file uploads
- Clear explanation of data structure requirements
- Interactive correction process

### 4.3 Results Delivery
- Organized delivery of output files
- Clear labeling and documentation
- Option to regenerate results if needed

## 5. MVP Success Criteria

### 5.1 Functional Success
- Successful processing of standard SPSS files
- Accurate Johnson's weights calculations
- Correct output file generation

### 5.2 User Success
- Users can successfully upload and process files
- Users can understand and respond to bot prompts
- Users receive expected calculation results

### 5.3 Technical Success
- Stable bot operation
- Reliable file processing
- Accurate data transformations

## 6. Future Considerations (Post-MVP)

### 6.1 Potential Enhancements
- Support for additional file formats
- Advanced analysis options
- Batch processing capabilities
- Result visualization features
- Custom output format options

### 6.2 Scalability
- Support for larger datasets
- Multiple simultaneous users
- Enhanced processing speed

## 7. Implementation Timeline

### 7.1 MVP Development Phases
1. Core Bot Development
2. File Processing Implementation
3. Calculation Engine Integration
4. User Interface Development
5. Testing and Validation
6. Initial Deployment

### 7.2 MVP Launch
- Initial user testing
- Feedback collection
- Performance monitoring
- Iterative improvements

## 8. Technical Dependencies

### 8.1 Required Components
- Python environment
- Telegram Bot API
- SPSS file processing libraries
- Data analysis libraries
- File generation utilities

### 8.2 Integration Requirements
- Telegram platform integration
- File storage system
- Calculation engine
- Output generation system

-------------------

## РУССКАЯ ВЕРСИЯ

## 1. Обзор продукта

### 1.1 Описание проблемы
Менеджерам маркетинговых исследований необходим эффективный и удобный инструмент для расчета относительных весов Джонсона на основе их данных SPSS. В настоящее время этот процесс может быть трудоемким и требовать специальных технических знаний.

### 1.2 Решение
Telegram-бот, который автоматизирует процесс расчета относительных весов Джонсона, выполняет проверку данных и помогает пользователям в необходимых преобразованиях данных.

### 1.3 Целевые пользователи
- Основные пользователи: Менеджеры маркетинговых исследований
- Характеристики пользователей: Знакомы с форматами данных SPSS и методологией маркетинговых исследований

## 2. Функции и возможности

### 2.1 Основные функции (MVP)

#### Ввод данных
- Прием файлов базы данных SPSS (формат .sav) через Telegram
- Интерактивная проверка структуры данных
- Сопровождаемый процесс трансформации данных при необходимости

#### Настройка анализа
- Возможность выбора области анализа:
  - Анализ по всей базе
  - Анализ по группам
- Возможность настройки расчетов по брендам:
  - Анализ одного бренда
  - Анализ нескольких брендов

#### Вычисления
- Выполнение расчетов относительных весов Джонсона
- Поддержка различных структур данных
- Автоматическая трансформация данных при необходимости

#### Генерация результатов
- Генерация результатов в нескольких форматах:
  - Формат Excel (.xlsx) для весов
  - Формат CSV для весов
  - Формат CSV для обработанных данных

### 2.2 Процесс взаимодействия с пользователем

1. **Начальный контакт**
   - Пользователь начинает взаимодействие с ботом
   - Бот предоставляет приветственное сообщение и инструкции

2. **Загрузка данных**
   - Пользователь загружает файл SPSS
   - Бот проверяет формат и структуру файла

3. **Проверка структуры данных**
   - Бот проверяет соответствие данных требуемой структуре
   - Если структура не соответствует:
     - Бот начинает интерактивный диалог
     - Помогает пользователю прояснить структуру
     - Выполняет необходимые преобразования

4. **Настройка анализа**
   - Бот предлагает выбрать область анализа
   - Пользователь указывает необходимость анализа по группам
   - Пользователь указывает предпочтения по анализу брендов

5. **Обработка и результаты**
   - Бот выполняет расчеты
   - Генерирует выходные файлы
   - Доставляет результаты пользователю

## 3. Технические требования

### 3.1 Платформа
- Интеграция с Telegram Bot API
- Бэкенд на Python
- Возможности обработки файлов SPSS

### 3.2 Требования к производительности
- Поддержка стандартных размеров файлов SPSS
- Разумное время расчета для типичных наборов данных

### 3.3 Обработка ошибок
- Интерактивное разрешение ошибок через диалог с пользователем
- Четкие сообщения об ошибках и руководство
- Проверка данных на каждом этапе

## 4. Пользовательский опыт

### 4.1 Коммуникация с ботом
- Четкий, профессиональный стиль общения
- Пошаговое руководство
- Интерактивное решение проблем

### 4.2 Проверка данных
- Мгновенная обратная связь при загрузке файлов
- Четкое объяснение требований к структуре данных
- Интерактивный процесс исправления

### 4.3 Доставка результатов
- Организованная доставка выходных файлов
- Четкая маркировка и документация
- Возможность повторной генерации результатов при необходимости

## 5. Критерии успеха MVP

### 5.1 Функциональный успех
- Успешная обработка стандартных файлов SPSS
- Точные расчеты весов Джонсона
- Корректная генерация выходных файлов

### 5.2 Пользовательский успех
- Пользователи могут успешно загружать и обрабатывать файлы
- Пользователи понимают подсказки бота и могут на них реагировать
- Пользователи получают ожидаемые результаты расчетов

### 5.3 Технический успех
- Стабильная работа бота
- Надежная обработка файлов
- Точные преобразования данных

## 6. Будущие перспективы (после MVP)

### 6.1 Потенциальные улучшения
- Поддержка дополнительных форматов файлов
- Расширенные опции анализа
- Возможности пакетной обработки
- Функции визуализации результатов
- Настраиваемые форматы вывода

### 6.2 Масштабируемость
- Поддержка больших наборов данных
- Множество одновременных пользователей
- Повышенная скорость обработки

## 7. График реализации

### 7.1 Фазы разработки MVP
1. Разработка ядра бота
2. Реализация обработки файлов
3. Интеграция модуля расчетов
4. Разработка пользовательского интерфейса
5. Тестирование и валидация
6. Начальное развертывание

### 7.2 Запуск MVP
- Начальное пользовательское тестирование
- Сбор обратной связи
- Мониторинг производительности
- Итеративные улучшения

## 8. Технические зависимости

### 8.1 Необходимые компоненты
- Среда Python
- Telegram Bot API
- Библиотеки обработки файлов SPSS
- Библиотеки анализа данных
- Утилиты генерации файлов

### 8.2 Требования к интеграции
- Интеграция с платформой Telegram
- Система хранения файлов
- Модуль расчетов
- Система генерации выходных данных