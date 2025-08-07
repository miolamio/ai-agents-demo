# sentiment_agent.py
import os
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from typing import Literal

# Убедитесь, что ваш API-ключ OpenAI установлен как переменная окружения
# export OPENAI_API_KEY='sk-...'
if "OPENAI_API_KEY" not in os.environ:
    raise ValueError("Необходимо установить переменную окружения OPENAI_API_KEY")

# Шаг 1: Определяем схему ответа (наш "контракт" с LLM)
class SentimentResult(BaseModel):
    """Структурированный результат анализа тональности отзыва."""
    sentiment: Literal["positive", "neutral", "negative"] = Field(
        description="Общая тональность отзыва: позитивная, нейтральная или негативная."
    )
    keywords: list[str] = Field(
        description="Список ключевых тем или продуктов, упомянутых в отзыве."
    )
    rating_suggestion: int = Field(
        description="Предлагаемая оценка в звездах от 1 до 5 на основе текста.",
        ge=1,  # Добавляем валидацию: "больше или равно 1"
        le=5   # и "меньше или равно 5"
    )

# Шаг 2: Создаем экземпляр агента
# Мы используем быструю и недорогую модель для этой задачи
sentiment_agent = Agent(
    "openai:gpt-4o-mini",
    output_type=SentimentResult,  # Вот здесь и происходит магия!
    system_prompt="Ты — полезный ассистент, который анализирует отзывы клиентов и извлекает информацию в структурированном виде."
)

# Шаг 3: Запускаем агент с пользовательским вводом
customer_review = (
    "Очень доволен новым ноутбуком! Экран просто потрясающий, и батарея держит весь день. "
    "Правда, доставка немного задержалась, но в целом впечатления отличные. "
    "Думаю, это твердая пятерка."
)

print(f"Анализируем отзыв: '{customer_review}'")
print("-" * 20)

# Используем run_sync для синхронного выполнения
result = sentiment_agent.run_sync(customer_review)

# Шаг 4: Работаем со структурированным результатом
if result.output:
    print("Анализ завершен. Результат:")
    # result.output - это экземпляр нашего класса SentimentResult!
    print(f"  Тональность: {result.output.sentiment}")
    print(f"  Ключевые слова: {result.output.keywords}")
    print(f"  Предложенный рейтинг: {result.output.rating_suggestion}")

    # Мы можем быть уверены, что типы данных корректны
    assert isinstance(result.output.rating_suggestion, int)
    assert result.output.sentiment in ["positive", "neutral", "negative"]
else:
    print("Не удалось получить структурированный результат.")
    print("Ошибки:", result.errors)

import logfire

# Шаг 0: Добавляем Logfire
logfire.configure(token="pylf_v1_us_DQN6NX83mbX9hcJz6yWWpRdgVFS47xxHJmKRzPhjfgsp") # Настраиваем Logfire
logfire.instrument_pydantic_ai() # Включаем автоматическую трассировку Pydantic!

# Запустим агент с валидными данными
try:
    result = sentiment_agent.run_sync(customer_review)
    if result.data:
        logfire.info("Успешный анализ: {sentiment}", sentiment=result.data.sentiment)
except Exception as e:
    logfire.error("Ошибка при анализе: {e}", e=e)

# Попробуем создать модель с невалидными данными, чтобы увидеть ошибку в Logfire
try:
    invalid_model = SentimentResult(sentiment="very-positive", keywords="", rating_suggestion=10)
except Exception as e:
    # Эта ошибка будет автоматически залогирована благодаря instrument_pydantic
    print("\nПерехвачена ожидаемая ошибка для демонстрации в Logfire.")