# 🚀 Быстрая настройка Observability примеров

Пошаговое руководство для запуска всех примеров мониторинга и наблюдаемости LLM агентов.

## 📋 Предварительные требования

- Python 3.8+
- Docker и Docker Compose (для LangFuse)
- API ключи от провайдеров LLM

## 🔧 1. Настройка переменных окружения

```bash
# Скопируйте шаблон
cp env.example .env

# Отредактируйте файл .env
nano .env
```

### Минимальные ключи для начала:

| Сервис | Переменная | Где получить |
|--------|------------|--------------|
| **LangSmith** | `LANGSMITH_API_KEY` | https://smith.langchain.com |
| **OpenAI** | `OPENAI_API_KEY` | https://platform.openai.com/api-keys |
| **Anthropic** | `ANTHROPIC_API_KEY` | https://console.anthropic.com |

## 🐳 2. Запуск LangFuse (опционально)

```bash
# Запустить LangFuse локально
docker-compose -f docker-compose.langfuse.yml up -d

# Проверить статус
docker-compose -f docker-compose.langfuse.yml ps

# Первичная настройка
open http://localhost:3000
```

### Настройка LangFuse:
1. Создайте аккаунт администратора
2. Создайте новый проект 
3. Получите Public Key и Secret Key
4. Добавьте ключи в `.env`:
   ```bash
   LANGFUSE_PUBLIC_KEY=pk-lf-ваш-ключ
   LANGFUSE_SECRET_KEY=sk-lf-ваш-ключ
   ```

## 📦 3. Установка зависимостей

```bash
pip install -r requirements.txt
```

## 🎯 4. Тестирование примеров

### Example 1: LangSmith трассировка
```bash
# Установите зависимости (согласно официальной документации)
pip install -U langchain langchain-openai

# Настройте переменные окружения
export LANGSMITH_TRACING=true
export LANGSMITH_ENDPOINT=https://api.smith.langchain.com
export LANGSMITH_API_KEY=your-api-key-here
export LANGSMITH_PROJECT=your-project-name
export OPENAI_API_KEY=your-openai-api-key

# Запустите пример
python example1_langsmith_tracing.py
```
✅ **Требует:** `LANGSMITH_API_KEY` (или legacy `LANGCHAIN_API_KEY`)

### Example 2: DSPy оптимизация
```bash
python example2_dspy_optimization.py
```
✅ **Требует:** `OPENAI_API_KEY` (или любой LLM API)

### Example 3: LangFuse мониторинг
```bash
python example3_langfuse_monitoring.py
```
✅ **Требует:** LangFuse запущен + ключи настроены

### Example 4: LiteLLM контроль затрат
```bash
python example4_litellm_cost_control.py
```
✅ **Требует:** `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`

## 🔍 5. Проверка результатов

### LangSmith
- Откройте https://smith.langchain.com
- Найдите ваши трассировки в проекте

### LangFuse  
- Откройте http://localhost:3000
- Проверьте Dashboard с метриками

### Терминал
- Все примеры выводят статистику в консоль
- Проверьте логи на ошибки подключения

## 🚨 Troubleshooting

### Проблема: "API key not found"
```bash
# Проверьте переменные окружения
env | grep -E "(OPENAI|LANGSMITH|LANGCHAIN|ANTHROPIC|LANGFUSE)"

# Загрузите .env в текущую сессию
export $(cat .env | xargs)

# Проверьте подключение к LangSmith
python -c "
from langsmith import Client
client = Client()
print('✅ Подключение к LangSmith успешно!')
print(f'Проект: {client.info}')
"
```

### Проблема: LangFuse не запускается
```bash
# Проверьте порты
netstat -tulpn | grep 3000

# Просмотрите логи
docker-compose -f docker-compose.langfuse.yml logs

# Перезапустите
docker-compose -f docker-compose.langfuse.yml restart
```

### Проблема: Ошибки импорта
```bash
# Переустановите зависимости
pip install -r requirements.txt --upgrade

# Проверьте версию Python
python --version  # Должна быть 3.8+
```

## 📊 Дополнительные возможности

### Мониторинг бюджета
```bash
# Установите лимит в .env
echo "DAILY_BUDGET=5.0" >> .env

# Запустите с контролем затрат
python example4_litellm_cost_control.py
```

### Batch обработка
```python
# Пример массовой обработки с скидками
from example4_litellm_cost_control import CostControlAgent

agent = CostControlAgent(daily_budget=10.0)
results = agent.batch_process([
    "Вопрос 1",
    "Вопрос 2", 
    "Вопрос 3"
])
```

### Экспорт метрик
```bash
# Выгрузка данных из LangFuse
curl -H "Authorization: Bearer $LANGFUSE_SECRET_KEY" \
     "http://localhost:3000/api/public/traces"
```

## ✅ Готово!

После выполнения всех шагов у вас будет полнофункциональная система наблюдаемости для LLM агентов с:

- 📊 Трассировкой запросов (LangSmith/LangFuse)
- 🎯 Автоматической оптимизацией (DSPy)  
- 💰 Контролем затрат (LiteLLM)
- 📈 Визуализацией метрик
- 🔄 Кэшированием и маршрутизацией

Начните с example1 и постепенно добавляйте другие инструменты по мере необходимости!
