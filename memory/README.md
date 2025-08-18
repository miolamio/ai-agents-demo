# CrewAI + Mem0 Integration Demo

Система планирования путешествий с персистентной памятью, демонстрирующая интеграцию CrewAI с Mem0 для создания персонализированного AI-агента.

## 🚀 Возможности

- **Персистентная память**: Использует Mem0 для сохранения предпочтений пользователей между сессиями
- **Интеллектуальное планирование**: CrewAI агент для создания персонализированных маршрутов
- **Поиск в реальном времени**: Интеграция с Serper Dev для актуальной информации
- **Автоматическая конфигурация**: Загрузка настроек из .env файла
- **Демо режим**: Работает без API ключей для демонстрации структуры

## 📦 Установка

1. **Клонируйте репозиторий и перейдите в папку:**
   ```bash
   cd memory/
   ```

2. **Установите зависимости:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Настройте переменные окружения:**
   ```bash
   cp env.example .env
   ```

4. **Заполните .env файл своими API ключами:**
   - [Mem0 Platform](https://platform.mem0.ai/) - для персистентной памяти
   - [OpenAI](https://platform.openai.com/) - для AI модели
   - [Serper Dev](https://serper.dev/) - для поиска в интернете

## 🎯 Использование

### Базовый запуск:
```bash
python crewai-mem0.py
```

### Программное использование:
```python
from crewai_mem0 import plan_trip, store_user_preferences

# Сохранение предпочтений пользователя
conversation = [
    {"role": "user", "content": "Я предпочитаю пляжный отдых"},
    {"role": "user", "content": "Люблю апартаменты через Airbnb"}
]
store_user_preferences("user_123", conversation)

# Планирование поездки с учетом предпочтений
result = plan_trip("Бали, Индонезия", "user_123")
print(result)
```

## 🏗️ Архитектура

### Компоненты:

1. **Mem0 Client** - управление персистентной памятью
2. **CrewAI Agent** - интеллектуальный агент планирования
3. **SerperDev Tool** - инструмент поиска в интернете
4. **Environment Management** - управление конфигурацией

### Процесс работы:

```
Пользователь → Сохранение предпочтений → Mem0
                        ↓
Запрос планирования → CrewAI Agent → Получение предпочтений
                        ↓
SerperDev поиск → Создание персонализированного плана
```

## 🔧 Конфигурация

### Переменные окружения (.env):

```env
# Обязательные
MEM0_API_KEY=your-mem0-api-key
OPENAI_API_KEY=your-openai-api-key
SERPER_API_KEY=your-serper-api-key

# Опциональные
OPENAI_MODEL=gpt-4o-mini
LOG_LEVEL=INFO
MAX_TOKENS=2000
```

### Поддерживаемые модели OpenAI:
- `gpt-4o-mini` (по умолчанию)
- `gpt-4o`
- `gpt-4-turbo`
- `gpt-3.5-turbo`

## 📝 Примеры использования

### Сохранение предпочтений:
```python
preferences = [
    {"role": "user", "content": "Я вегетарианец"},
    {"role": "user", "content": "Предпочитаю экологичные варианты"},
    {"role": "user", "content": "Бюджет до $100/день"}
]
store_user_preferences("eco_traveler", preferences)
```

### Планирование с предпочтениями:
```python
# Агент автоматически учтет сохраненные предпочтения
result = plan_trip("Коста-Рика", "eco_traveler")
```

## 🛠️ Разработка

### Структура проекта:
```
memory/
├── crewai-mem0.py      # Основной код
├── requirements.txt     # Зависимости
├── env.example         # Пример конфигурации
└── README.md           # Документация
```

### Добавление новых функций:

1. **Новый агент:**
   ```python
   def create_custom_agent():
       return Agent(
           role="Ваша роль",
           goal="Ваша цель",
           backstory="История агента",
           tools=[ваши_инструменты]
       )
   ```

2. **Новая задача:**
   ```python
   def create_custom_task(agent):
       return Task(
           description="Описание задачи",
           expected_output="Ожидаемый результат",
           agent=agent
       )
   ```

## 🔍 Отладка

### Проверка статуса API ключей:
Скрипт автоматически показывает статус всех API ключей при запуске.

### Логи:
Установите `LOG_LEVEL=DEBUG` в .env для детального логирования.

### Демо режим:
Скрипт работает без API ключей в демонстрационном режиме.

## 📚 Ресурсы

- [CrewAI Documentation](https://docs.crewai.com/)
- [Mem0 Documentation](https://docs.mem0.ai/)
- [Serper Dev API](https://serper.dev/api)
- [OpenAI API](https://platform.openai.com/docs)

## 🤝 Вклад

Приветствуются pull requests и issues! Пожалуйста:

1. Создайте issue для обсуждения изменений
2. Форкните репозиторий
3. Создайте feature branch
4. Добавьте тесты для новой функциональности
5. Создайте pull request

## 📄 Лицензия

MIT License - см. файл LICENSE для деталей.

