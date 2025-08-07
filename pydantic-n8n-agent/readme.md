# 🎯 Финальная настройка проекта

## 📁 Структура проекта

```
web-research-agent/
├── web_research_agent.py     # Основной модуль агента
├── examples.py               # Примеры использования
├── web_interface.py          # Веб-интерфейс (Streamlit)
├── requirements.txt          # Зависимости
├── .env.example             # Пример файла с переменными окружения
├── .env                     # Ваши API ключи (создайте сами)
└── README.md                # Документация
```

## 📄 Файл .env.example

Создайте файл `.env.example` со следующим содержимым:

```env
# OpenAI API Configuration
OPENAI_API_KEY=your-openai-api-key-here

# Tavily API Configuration
TAVILY_API_KEY=your-tavily-api-key-here

# Optional: Model Configuration
# OPENAI_MODEL=gpt-4o-mini
# SEARCH_DEPTH=basic
# MAX_SEARCH_RESULTS=10
```

## 📦 Файл requirements.txt

```txt
# Core dependencies
pydantic-ai>=0.0.49
pydantic>=2.0.0
tavily-python>=0.5.0
python-dotenv>=1.0.0

# For web interface (optional)
streamlit>=1.29.0
pandas>=2.0.0
plotly>=5.17.0

# For async support
asyncio
aiohttp>=3.9.0

# Development dependencies (optional)
pytest>=7.4.0
pytest-asyncio>=0.21.0
black>=23.0.0
mypy>=1.5.0
```

## 🚀 Быстрый старт (3 шага)

### Шаг 1: Установка

```bash
# Клонируйте репозиторий или создайте папку проекта
mkdir web-research-agent
cd web-research-agent

# Создайте виртуальное окружение
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Установите зависимости
pip install -r requirements.txt
```

### Шаг 2: Настройка API ключей

```bash
# Скопируйте пример конфигурации
cp .env.example .env

# Отредактируйте .env и добавьте ваши ключи
nano .env  # или используйте любой текстовый редактор
```

### Шаг 3: Запуск

**Вариант A: Командная строка**
```python
python examples.py  # Запустить примеры
```

**Вариант B: Python скрипт**
```python
from web_research_agent import WebResearchAssistant

assistant = WebResearchAssistant(
    openai_api_key="your-key",
    tavily_api_key="your-key"
)

result = assistant.process_message_sync("Ваш запрос")
print(result.model_dump_json(indent=2))
```

**Вариант C: Веб-интерфейс**
```bash
streamlit run web_interface.py
# Откройте http://localhost:8501 в браузере
```

## 🔄 Сравнительная таблица: n8n vs Pydantic AI

| Функция | n8n | Pydantic AI |
|---------|-----|-------------|
| **Визуальный интерфейс** | ✅ Drag-and-drop | ❌ Код (но есть Streamlit UI) |
| **Типизация** | ❌ Слабая | ✅ Строгая (Pydantic) |
| **Производительность** | 🟡 Средняя | ✅ Высокая |
| **Масштабируемость** | 🟡 Ограниченная | ✅ Отличная |
| **Кастомизация** | 🟡 Ограниченная | ✅ Полная |
| **Развертывание** | 🟡 Требует n8n сервер | ✅ Любой Python хостинг |
| **Тестирование** | ❌ Сложное | ✅ Unit тесты |
| **Версионирование** | ❌ Сложное | ✅ Git |
| **Цена** | 💰 Платный для команд | ✅ Бесплатно (оплата только API) |

## 🎨 Дополнительные возможности

### 1. Добавление новых моделей

```python
# Claude
research_agent = Agent(
    'anthropic:claude-3-opus-20240229',
    # ...
)

# Gemini
research_agent = Agent(
    'google-gla:gemini-1.5-pro',
    # ...
)
```

### 2. Кастомные инструменты

```python
@research_agent.tool
async def analyze_sentiment(ctx: RunContext[AgentDependencies], text: str) -> Dict:
    """Анализ тональности текста"""
    # Ваша логика
    return {"sentiment": "positive", "score": 0.85}
```

### 3. Интеграция с базой данных

```python
import sqlite3

class AgentDependencies(BaseModel):
    tavily_client: TavilyClient
    db_connection: sqlite3.Connection
    
    class Config:
        arbitrary_types_allowed = True

# Сохранение результатов в БД
@research_agent.tool
async def save_to_db(ctx: RunContext[AgentDependencies], data: Dict) -> bool:
    cursor = ctx.deps.db_connection.cursor()
    # SQL запросы
    return True
```

### 4. Webhook интеграция

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class ResearchRequest(BaseModel):
    query: str
    webhook_url: Optional[str] = None

@app.post("/research")
async def research_endpoint(request: ResearchRequest):
    assistant = WebResearchAssistant(...)
    result = await assistant.process_message(request.query)
    
    # Отправка результата на webhook
    if request.webhook_url:
        async with aiohttp.ClientSession() as session:
            await session.post(request.webhook_url, json=result.model_dump())
    
    return result.model_dump()
```

## 🔍 Отладка и мониторинг

### Включение логирования

```python
import logging
from pydantic_ai import set_debug_mode

# Включить debug режим Pydantic AI
set_debug_mode(True)

# Настройка логирования
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('research_agent.log'),
        logging.StreamHandler()
    ]
)
```

### Мониторинг использования токенов

```python
result = agent.run_sync(query)
usage = result.usage()
print(f"Токены запроса: {usage.request_tokens}")
print(f"Токены ответа: {usage.response_tokens}")
print(f"Всего токенов: {usage.total_tokens}")
print(f"Примерная стоимость: ${usage.total_tokens * 0.00002:.4f}")
```

## 📈 Производительность

### Оптимизация для больших нагрузок

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class OptimizedResearchAssistant(WebResearchAssistant):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.executor = ThreadPoolExecutor(max_workers=5)
    
    async def batch_process(self, queries: List[str]) -> List[ResearchOutput]:
        """Обработка множественных запросов параллельно"""
        tasks = [self.process_message(q) for q in queries]
        return await asyncio.gather(*tasks)
```

## 🛡️ Безопасность

### Рекомендации по безопасности

1. **Никогда не коммитьте .env файл в Git:**
```bash
echo ".env" >> .gitignore
```

2. **Используйте переменные окружения на продакшене:**
```python
import os
from pathlib import Path

# Загрузка только в development
if Path('.env').exists():
    load_dotenv()

# На продакшене используйте системные переменные
openai_key = os.environ['OPENAI_API_KEY']
```

3. **Ограничьте rate limits:**
```python
from asyncio import Semaphore

class RateLimitedAssistant(WebResearchAssistant):
    def __init__(self, *args, max_concurrent=3, **kwargs):
        super().__init__(*args, **kwargs)
        self.semaphore = Semaphore(max_concurrent)
    
    async def process_message(self, query: str) -> ResearchOutput:
        async with self.semaphore:
            return await super().process_message(query)
```

## 📚 Дополнительные ресурсы

- 📖 [Pydantic AI Cookbook](https://ai.pydantic.dev/cookbook/)
- 🎓 [Pydantic AI Examples](https://github.com/pydantic/pydantic-ai/tree/main/examples)
- 💬 [Pydantic Discord](https://discord.gg/pydantic)
- 🐛 [Issue Tracker](https://github.com/pydantic/pydantic-ai/issues)

## ✅ Чек-лист готовности

- [ ] Python 3.9+ установлен
- [ ] Виртуальное окружение создано
- [ ] Зависимости установлены
- [ ] API ключи получены и настроены
- [ ] Базовый пример работает
- [ ] Веб-интерфейс запускается (опционально)
- [ ] Логирование настроено (опционально)

## 🎉 Поздравляем!

Вы успешно перенесли n8n workflow в Pydantic AI! Теперь у вас есть:

- ✅ **Полный контроль** над кодом
- ✅ **Типобезопасность** благодаря Pydantic
- ✅ **Масштабируемость** для production
- ✅ **Тестируемость** с unit тестами
- ✅ **Гибкость** для кастомизации

Happy researching! 🚀