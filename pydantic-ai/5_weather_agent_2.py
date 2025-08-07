# weather_agent_v2.py
import os
from pydantic_ai import Agent, RunContext
from dataclasses import dataclass
from datetime import date

if "OPENAI_API_KEY" not in os.environ:
    raise ValueError("Необходимо установить переменную окружения OPENAI_API_KEY")

# Шаг 1: Создаем "зависимость" - класс, который инкапсулирует логику
@dataclass
class WeatherAPIClient:
    """Клиент для мок-API погоды."""
    api_key: str

    def get_weather(self, city: str) -> str:
        # Проверяем, что "ключ" был передан
        if not self.api_key.startswith("wx_"):
            return "Ошибка: неверный API ключ."
        print(f"--- [API КЛИЕНТ] Вызов API погоды для города: {city} с ключом {self.api_key[:5]}... ---")
        return f"Погода в {city} отличная!"

# Шаг 2: Создаем агента, указывая тип зависимости
# Мы также добавим динамический системный промпт
agent = Agent(
    "openai:gpt-4o-mini",
    deps_type=WeatherAPIClient # Указываем, какой тип зависимости ожидает наш агент
)

# Шаг 3: Рефакторим наш инструмент для использования зависимости
# Он теперь принимает 'ctx: RunContext' первым аргументом
@agent.tool
async def get_current_weather(ctx: RunContext, city: str) -> str:
    """Возвращает текущую погоду для указанного города, используя API клиент."""
    # Доступ к нашему клиенту осуществляется через ctx.deps
    weather_client = ctx.deps
    return weather_client.get_weather(city)

# Шаг 4: Добавляем динамический системный промпт
# Эта функция будет вызываться каждый раз при запуске агента
@agent.system_prompt
async def get_current_date(ctx: RunContext) -> str:
    """Добавляет текущую дату в системный промпт."""
    return f"Текущая дата: {date.today()}. Всегда учитывай ее в своих ответах."

# Шаг 5: Запускаем агент, "внедряя" экземпляр зависимости
async def main():
    # Создаем экземпляр нашей зависимости
    weather_client_instance = WeatherAPIClient(api_key="wx_12345_secret")
    
    user_prompt = "Какая погода в Берлине?"
    
    # Передаем зависимость через аргумент `deps`
    result = await agent.run(user_prompt, deps=weather_client_instance)
    
    print("-" * 20)
    print("Финальный ответ агента:")
    print(result.output)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())