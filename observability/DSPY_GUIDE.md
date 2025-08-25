# 🚀 DSPy + OpenAI - Руководство по программированию языковых моделей

## 🎯 Что это?

**`example2b_dspy_openai.py`** - это рабочий пример DSPy с реальной интеграцией OpenAI, демонстрирующий декларативное программирование языковых моделей.

### ✨ Возможности:

1. **🤖 Простые ответы** - быстрые и краткие ответы (SimpleQA)
2. **🧠 Рассуждения** - пошаговый Chain-of-Thought анализ (ReasoningQA)  
3. **🔍 Проверка фактов** - ответы с указанием достоверности (FactCheckQA)
4. **📊 Автоматическая оценка** - метрики качества ответов
5. **⚡ Оптимизация** - базовая оптимизация промптов через BootstrapFewShot

## 🚀 Быстрый старт

### 1. **Установка зависимостей**
```bash
pip install dspy-ai openai python-dotenv
```

### 2. **Настройка API ключей**
```bash
# Добавьте в .env файл:
echo "OPENAI_API_KEY=ваш-openai-ключ" >> .env
```

### 3. **Запуск примера**
```bash
python example2b_dspy_openai.py
```

## 🛠️ Архитектура DSPy

### **1. Сигнатуры (Signatures)**
Определяют входы и выходы задач:

```python
class SimpleQA(dspy.Signature):
    """Отвечает на вопросы коротко и по существу"""
    
    question = dspy.InputField(desc="Вопрос пользователя")
    answer = dspy.OutputField(desc="Краткий и точный ответ")
```

### **2. Модули (Modules)**
Содержат логику обработки:

```python
class SmartQAAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.simple_qa = dspy.Predict(SimpleQA)
        self.reasoning_qa = dspy.ChainOfThought(ReasoningQA)
    
    def forward(self, question: str, mode: str = "simple"):
        if mode == "simple":
            return self.simple_qa(question=question)
        elif mode == "reasoning":
            return self.reasoning_qa(question=question)
```

### **3. Метрики оценки**
Автоматически оценивают качество:

```python
def evaluate_answer_quality(example, prediction, trace=None) -> float:
    # Оценка по длине, релевантности, структуре
    score = 0.0
    # ... логика оценки
    return min(score, 1.0)
```

## 📊 Результаты тестирования

### **🤖 Простой режим (среднее время: 2.0с)**
```
Q: Что такое искусственный интеллект?
A: Искусственный интеллект (ИИ) — это область компьютерных наук, 
   занимающаяся созданием систем, способных выполнять задачи, 
   требующие человеческого интеллекта...
Оценка: 0.77
```

### **🧠 Режим с рассуждениями (среднее время: 7.0с)**
```
Q: Что такое искусственный интеллект?
Рассуждения: Искусственный интеллект (ИИ) — это область компьютерных 
наук... Основные моменты:

1. **Определение**: Искусственный интеллект - это способность машин...
2. **Типы ИИ**: 
   - **Узкий ИИ** (или слабый ИИ) — системы...
   - **Общий ИИ** (или сильный ИИ) — гипотетическая система...
3. **Методы ИИ**: ИИ включает в себя такие технологии...
4. **Применение**: ИИ находит применение в различных областях...

Итоговый ответ: Искусственный интеллект — это область компьютерных 
наук, занимающаяся созданием систем...
Оценка: 0.60
```

### **📈 Статистика производительности:**
- **Простые ответы**: 0.75 среднее качество, 2.0с среднее время
- **С рассуждениями**: 0.53 среднее качество, 7.0с среднее время
- **Trade-off**: Рассуждения занимают в 3.5 раза больше времени

## 🎮 Режимы работы

### **Режим 1: Простое Q&A**
```bash
python example2b_dspy_openai.py
# Выберите: 1

📋 Вопрос 1: Что такое искусственный интеллект?
🤖 Простой ответ (2.7с): [Краткий ответ]
🧠 Рассуждения (5.8с): [Пошаговый анализ]
📊 Оценки: Простой=0.77 | Рассуждения=0.60
```

### **Режим 2: Демонстрация оптимизации**
- Создание обучающего набора данных
- Тестирование неоптимизированной модели
- Оптимизация через BootstrapFewShot
- Сравнение результатов до и после

### **Режим 3: Комбинированный**
- Оба режима последовательно

## ⚙️ Сигнатуры DSPy

### **1. SimpleQA - Быстрые ответы**
```python
class SimpleQA(dspy.Signature):
    """Отвечает на вопросы коротко и по существу"""
    question = dspy.InputField(desc="Вопрос пользователя")
    answer = dspy.OutputField(desc="Краткий и точный ответ")
```

**Использование**: `dspy.Predict(SimpleQA)`
**Время**: ~2 секунды
**Качество**: Высокое для простых вопросов

### **2. ReasoningQA - С рассуждениями**
```python
class ReasoningQA(dspy.Signature):
    """Отвечает на вопросы с пошаговыми рассуждениями"""
    question = dspy.InputField(desc="Вопрос пользователя") 
    reasoning = dspy.OutputField(desc="Пошаговые рассуждения")
    answer = dspy.OutputField(desc="Финальный ответ на основе рассуждений")
```

**Использование**: `dspy.ChainOfThought(ReasoningQA)`
**Время**: ~7 секунд
**Качество**: Глубокий анализ, структурированные рассуждения

### **3. FactCheckQA - Проверка фактов**
```python
class FactCheckQA(dspy.Signature):
    """Проверяет факты и дает обоснованные ответы"""
    question = dspy.InputField(desc="Вопрос пользователя")
    confidence = dspy.OutputField(desc="Уровень уверенности (высокий/средний/низкий)")
    sources = dspy.OutputField(desc="Возможные источники информации")
    answer = dspy.OutputField(desc="Ответ с указанием степени достоверности")
```

**Использование**: `dspy.ChainOfThought(FactCheckQA)`
**Специализация**: Оценка достоверности информации

## 🔧 Кастомизация и расширение

### **Создание новых сигнатур:**
```python
class CodeReviewQA(dspy.Signature):
    """Анализирует код и дает рекомендации"""
    code = dspy.InputField(desc="Код для анализа")
    language = dspy.InputField(desc="Язык программирования")
    issues = dspy.OutputField(desc="Найденные проблемы")
    suggestions = dspy.OutputField(desc="Рекомендации по улучшению")
    rating = dspy.OutputField(desc="Оценка качества кода от 1 до 10")
```

### **Настройка модели:**
```python
# Изменение модели OpenAI
lm = dspy.LM(
    model='openai/gpt-4o',  # Более мощная модель
    api_key=os.getenv("OPENAI_API_KEY"),
    max_tokens=2000,
    temperature=0.3  # Меньше креативности
)
```

### **Создание собственных метрик:**
```python
def evaluate_code_quality(example, prediction, trace=None) -> float:
    """Оценивает качество анализа кода"""
    if not hasattr(prediction, 'rating'):
        return 0.0
    
    try:
        rating = int(prediction.rating)
        # Проверяем адекватность оценки
        if 1 <= rating <= 10:
            return 0.8
    except:
        pass
    
    return 0.3
```

## 🚀 Оптимизация промптов

### **BootstrapFewShot**
```python
optimizer = dspy.BootstrapFewShot(
    metric=evaluate_answer_quality,
    max_bootstrapped_demos=3,  # Количество примеров
    max_labeled_demos=2        # Количество меток
)

optimized_model = optimizer.compile(qa_agent, trainset=training_data)
```

**Что происходит:**
1. DSPy анализирует обучающие данные
2. Автоматически создает few-shot примеры
3. Оптимизирует промпты для лучшей производительности
4. Возвращает скомпилированную модель

## 💡 Примеры использования

### **В исследовательских задачах:**
```python
research_agent = SmartQAAgent()

# Быстрый поиск фактов
quick_fact = research_agent(
    question="Когда была основана компания OpenAI?",
    mode="simple"
)

# Глубокий анализ
deep_analysis = research_agent(
    question="Как влияет ИИ на рынок труда?",
    mode="reasoning"
)
```

### **В образовательных целях:**
```python
# Объяснение сложных концепций
explanation = research_agent(
    question="Объясни квантовые вычисления простыми словами",
    mode="reasoning"
)
```

### **В проверке фактов:**
```python
fact_check = research_agent(
    question="Правда ли, что Python самый популярный язык программирования?",
    mode="fact_check"
)
```

## 🔍 Отладка и мониторинг

### **Инспекция промптов:**
```python
# Просмотр последних промптов
dspy.inspect_history(n=3)

# Анализ структуры промпта
print("System prompt:", agent.simple_qa.signature)
print("Few-shot examples:", agent.simple_qa.demos)
```

### **Профилирование производительности:**
```python
import time

start_time = time.time()
result = qa_agent(question="Тестовый вопрос", mode="simple")
end_time = time.time()

print(f"Время выполнения: {end_time - start_time:.2f}с")
print(f"Длина ответа: {len(result.answer)} символов")
```

## 📈 Интеграция в проекты

### **REST API с FastAPI:**
```python
from fastapi import FastAPI
from example2b_dspy_openai import SmartQAAgent

app = FastAPI()
qa_agent = SmartQAAgent()

@app.post("/ask")
async def ask_question(question: str, mode: str = "simple"):
    result = qa_agent(question=question, mode=mode)
    return {
        "question": question,
        "answer": result.answer,
        "reasoning": getattr(result, 'reasoning', None),
        "mode": mode
    }
```

### **Telegram бот:**
```python
import telebot
from example2b_dspy_openai import SmartQAAgent

bot = telebot.TeleBot("YOUR_BOT_TOKEN")
qa_agent = SmartQAAgent()

@bot.message_handler(commands=['ask'])
def handle_question(message):
    question = message.text[5:]  # Remove '/ask '
    result = qa_agent(question=question, mode="simple")
    bot.reply_to(message, result.answer)
```

## 🎯 Лучшие практики

### **1. Выбор режима:**
- **Simple**: Для быстрых фактических вопросов
- **Reasoning**: Для сложных аналитических задач
- **Fact_check**: Для проверки достоверности

### **2. Оптимизация производительности:**
- Используйте `gpt-4o-mini` для быстрых задач
- Переходите на `gpt-4o` для сложных рассуждений
- Кэшируйте часто используемые ответы

### **3. Оценка качества:**
- Создавайте специфичные метрики для ваших задач
- Используйте A/B тестирование разных подходов
- Регулярно обновляйте обучающие данные

---

🎉 **DSPy превращает программирование ИИ в декларативный процесс!**

Вместо мучительного создания промптов вы описываете желаемое поведение через сигнатуры, а DSPy автоматически оптимизирует промпты для достижения лучших результатов! 🚀
