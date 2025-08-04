"""
Демонстрация Pydantic AI: Структурированный анализатор погоды
Показывает, как использовать Pydantic AI для получения типизированных ответов
"""

# 1. Импорты
from pydantic_ai import Agent, RunContext
from pydantic import BaseModel
import requests
from dotenv import load_dotenv
import os

# 2. Определение Pydantic моделей
class WeatherAnalysis(BaseModel):
    """Структура для анализа погоды"""
    city: str
    temperature: float
    conditions: str
    recommendation: str
    clothing_advice: list[str]

# 3. Создание функции для получения погоды
def get_weather(city: str) -> dict:
    """Получает данные о погоде через OpenWeather API"""
    api_key = os.getenv("OPENWEATHER_API_KEY")
    
    if not api_key:
        # Если нет API ключа, возвращаем тестовые данные
        return {
            "temperature": -5.0,
            "description": "снег"
        }
    
    try:
        # URL для запроса к OpenWeather API
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {
            "q": city,
            "appid": api_key,
            "units": "metric",
            "lang": "ru"
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        return {
            "temperature": data["main"]["temp"],
            "description": data["weather"][0]["description"]
        }
        
    except Exception as e:
        print(f"⚠️ Ошибка при получении данных о погоде: {e}")
        print("Использую тестовые данные...")
        # Возвращаем тестовые данные в случае ошибки
        return {
            "temperature": -5.0,
            "description": "снег"
        }

def main():
    # Загрузка переменных окружения
    load_dotenv()
    
    # Проверка наличия ключа OpenAI
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Ошибка: Не найден OPENAI_API_KEY в файле .env")
        print("Создайте файл .env и добавьте ваш ключ OpenAI")
        exit(1)
    
    print("=== Pydantic AI Demo: Анализатор погоды ===\n")
    
    try:
        # 4. Создание агента
        # Инициализация Agent с model='openai:gpt-3.5-turbo' (обновлено для новой версии API)
        agent = Agent(
            model='openai:gpt-3.5-turbo',
            output_type=WeatherAnalysis,
            system_prompt="""
            Ты - эксперт по анализу погоды. На основе данных о температуре и условиях погоды 
            в городе, ты должен дать рекомендации и советы по одежде.
            
            Всегда отвечай на русском языке. Будь дружелюбным и полезным.
            """,
        )
        
        # Добавление инструмента
        @agent.tool
        def weather_tool(ctx: RunContext, city: str) -> dict:
            """Получает данные о погоде для указанного города"""
            return get_weather(city)
        
        # 5. Основная логика
        city = "Берлин"
        print(f"Получаю данные о погоде в городе {city}...")
        
        # Запрос анализа погоды
        result = agent.run_sync(
            f"Получи данные о погоде в городе {city} и сделай подробный анализ с рекомендациями по одежде"
        )
        
        # Вывод структурированного результата
        print(f"\nАнализ погоды:")
        print(f"- Город: {result.output.city}")
        print(f"- Температура: {result.output.temperature}°C")
        print(f"- Условия: {result.output.conditions}")
        print(f"- Рекомендация: {result.output.recommendation}")
        print(f"- Что надеть:")
        for item in result.output.clothing_advice:
            print(f"  • {item}")
        
    except Exception as e:
        print(f"❌ Ошибка при работе с Pydantic AI: {e}")
        print("Убедитесь, что:")
        print("1. Установлены все зависимости: pip install -r requirements.txt")
        print("2. Правильно указан OPENAI_API_KEY в файле .env")
        print("3. (Опционально) Добавлен OPENWEATHER_API_KEY для реальных данных о погоде")
        print("4. Есть доступ к интернету")

if __name__ == "__main__":
    main()