# weather_agent.py
import os
from pydantic import BaseModel, Field
from pydantic_ai import Agent
import random

if "OPENAI_API_KEY" not in os.environ:
    raise ValueError("Необходимо установить переменную окружения OPENAI_API_KEY")

# Шаг 1: Определяем инструмент. Это обычная Python-функция.
# Ключевую роль играет docstring!
def get_current_weather(city: str) -> str:
    """
    Возвращает текущую погоду для указанного города.
    
    :param city: Название города, например, 'Москва' или 'Париж'.
    :return: Строка с описанием погоды.
    """
    print(f"--- [ИНСТРУМЕНТ ВЫЗВАН] Получение погоды для города: {city} ---")
    # В реальном приложении здесь был бы вызов API
    # Мы же просто вернем случайные данные для демонстрации
    conditions = ["солнечно", "облачно", "дождь", "снег"]
    temperature = random.randint(-10, 30)
    return f"В городе {city} сейчас {random.choice(conditions)}, температура {temperature}°C."

# Шаг 2: Создаем агента и регистрируем наш инструмент
# Обратите внимание, что мы убрали result_type, так как теперь агент
# будет возвращать обычный текстовый ответ, синтезированный после вызова инструмента.
weather_agent = Agent(
    "openai:gpt-4o-mini",
    tools=[get_current_weather] # Передаем функцию в виде списка
)

# Шаг 3: Запускаем агент с задачей, которая требует использования инструмента
user_prompt = "Привет! Подскажи, какая сейчас погода в Лондоне?"
result = weather_agent.run_sync(user_prompt)

print("-" * 20)
print("Финальный ответ агента:")
print(result.output)