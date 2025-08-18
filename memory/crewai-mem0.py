"""
CrewAI + Mem0 Integration Demo
Система планирования путешествий с персистентной памятью
"""

import os
from dotenv import load_dotenv
from mem0 import MemoryClient
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool

# Загрузка переменных окружения из .env файла
load_dotenv()

# Конфигурация API ключей из переменных окружения
MEM0_API_KEY = os.getenv("MEM0_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2000"))

# Установка переменных окружения
if MEM0_API_KEY:
    os.environ["MEM0_API_KEY"] = MEM0_API_KEY
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
if SERPER_API_KEY:
    os.environ["SERPER_API_KEY"] = SERPER_API_KEY

# Инициализация Mem0 клиента
client = None
if MEM0_API_KEY:
    try:
        client = MemoryClient()
        print("✅ Mem0 клиент успешно инициализирован")
    except Exception as e:
        print(f"❌ Ошибка инициализации Mem0 клиента: {e}")
else:
    print("⚠️  MEM0_API_KEY не найден в переменных окружения")

def store_user_preferences(user_id: str, conversation: list):
    """Сохранение пользовательских предпочтений из истории разговора"""
    if client:
        try:
            client.add(conversation, user_id=user_id)
            print(f"✅ Предпочтения сохранены для пользователя {user_id}")
        except Exception as e:
            print(f"❌ Ошибка сохранения предпочтений: {e}")
    else:
        print(f"⚠️  Демо режим: Сохранение предпочтений для пользователя {user_id}")
        for msg in conversation:
            print(f"  {msg['role']}: {msg['content']}")

def get_user_preferences(user_id: str):
    """Получение сохраненных предпочтений пользователя"""
    if client:
        try:
            preferences = client.get_all(user_id=user_id)
            print(f"✅ Получены предпочтения для пользователя {user_id}: {len(preferences)} записей")
            return preferences
        except Exception as e:
            print(f"❌ Ошибка получения предпочтений: {e}")
            return []
    else:
        print(f"⚠️  Демо режим: Получение предпочтений для пользователя {user_id}")
        return []

def create_travel_agent():
    """Создание агента планирования путешествий с возможностями поиска"""
    tools = []
    
    # Добавляем инструмент поиска, если доступен SERPER_API_KEY
    if SERPER_API_KEY:
        try:
            search_tool = SerperDevTool()
            tools.append(search_tool)
            print("✅ Инструмент поиска SerperDev добавлен")
        except Exception as e:
            print(f"❌ Ошибка инициализации SerperDev: {e}")
    else:
        print("⚠️  SERPER_API_KEY не найден, агент будет работать без поиска")

    return Agent(
        role="Персонализированный агент планирования путешествий",
        goal="Планировать персонализированные туристические маршруты",
        backstory="""Вы опытный планировщик путешествий, известный своим 
        скрупулезным вниманием к деталям и способностью создавать уникальные 
        персонализированные маршруты на основе предпочтений клиентов.""",
        allow_delegation=False,
        memory=True,
        tools=tools if tools else None,
        verbose=True
    )

def create_planning_task(agent, destination: str, user_preferences: str = ""):
    """Создание задачи планирования путешествия"""
    preferences_context = f"\n\nПредпочтения пользователя: {user_preferences}" if user_preferences else ""
    
    return Task(
        description=f"""Найдите места для проживания, питания и посещения в {destination}.
        Учтите следующие требования:
        - Предоставьте детальную информацию о рекомендуемых местах
        - Учитывайте бюджетные варианты и премиум опции
        - Включите культурные достопримечательности и развлечения
        - Добавьте практические советы по транспорту{preferences_context}""",
        expected_output=f"Подробный список мест для проживания, питания и посещения в {destination} с описаниями и рекомендациями.",
        agent=agent,
    )

def setup_crew(agents: list, tasks: list, user_id: str):
    """Настройка команды с интеграцией памяти Mem0"""
    return Crew(
        agents=agents,
        tasks=tasks,
        process=Process.sequential,
        memory=True,
        memory_config={
            "provider": "mem0",
            "config": {"user_id": user_id},
        },
        verbose=True
    )

def plan_trip(destination: str, user_id: str):
    """Главная функция планирования путешествия"""
    print(f"🌍 Начинаем планирование поездки в {destination} для пользователя {user_id}")
    
    # Получение сохраненных предпочтений пользователя
    user_preferences = get_user_preferences(user_id)
    preferences_text = ""
    if user_preferences:
        preferences_text = "\n".join([pref.get('content', '') for pref in user_preferences])
    
    # Создание агента
    travel_agent = create_travel_agent()
    
    # Создание задачи с учетом предпочтений
    planning_task = create_planning_task(travel_agent, destination, preferences_text)
    
    # Настройка команды
    crew = setup_crew([travel_agent], [planning_task], user_id)
    
    # Выполнение и возврат результатов
    print("🚀 Запуск команды планирования...")
    try:
        result = crew.kickoff()
        print("✅ Планирование завершено!")
        return result
    except Exception as e:
        print(f"❌ Ошибка при планировании: {e}")
        return f"Не удалось запланировать поездку в {destination}. Ошибка: {e}"

def demo_conversation_storage():
    """Демонстрация сохранения разговора с пользовательскими предпочтениями"""
    print("💾 Демонстрация сохранения предпочтений пользователя...")
    
    # Пример разговора с предпочтениями
    messages = [
        {
            "role": "user",
            "content": "Привет! Я планирую отпуск и мне нужен совет.",
        },
        {
            "role": "assistant", 
            "content": "Привет! Я буду рад помочь с планированием отпуска. Какой тип места назначения вы предпочитаете?",
        },
        {
            "role": "user", 
            "content": "Я больше люблю пляжный отдых, чем горы."
        },
        {
            "role": "assistant",
            "content": "Интересно. Вы предпочитаете отели или апартаменты?",
        },
        {
            "role": "user", 
            "content": "Мне больше нравятся апартаменты через Airbnb."
        },
        {
            "role": "user",
            "content": "Также я вегетарианец и предпочитаю экологичные варианты."
        }
    ]
    
    # Сохранение предпочтений
    store_user_preferences("travel_user_1", messages)
    return messages

if __name__ == "__main__":
    print("🎯 CrewAI + Mem0 Integration Demo")
    print("=" * 50)
    
    # Демонстрация сохранения предпочтений
    demo_conversation_storage()
    print("\n" + "=" * 50)
    
    # Планирование поездки
    destination = "Бали, Индонезия"
    user_id = "travel_user_1"
    
    result = plan_trip(destination, user_id)
    print(f"\n📋 Результат планирования:\n{result}")
    
    print("\n" + "=" * 50)
    print("🔧 Для полной функциональности:")
    print("1. Установите зависимости: pip install -r requirements.txt")
    print("2. Скопируйте env.example в .env: cp env.example .env")
    print("3. Получите API ключи и заполните .env файл:")
    print("   - Mem0 Platform: https://platform.mem0.ai/")
    print("   - OpenAI: https://platform.openai.com/")
    print("   - Serper Dev: https://serper.dev/")
    print("4. Запустите снова: python crewai-mem0.py")
    
    # Показываем статус API ключей
    print("\n📋 Статус API ключей:")
    print(f"   MEM0_API_KEY: {'✅ Установлен' if MEM0_API_KEY else '❌ Не найден'}")
    print(f"   OPENAI_API_KEY: {'✅ Установлен' if OPENAI_API_KEY else '❌ Не найден'}")
    print(f"   SERPER_API_KEY: {'✅ Установлен' if SERPER_API_KEY else '❌ Не найден'}")
    print(f"   Модель OpenAI: {OPENAI_MODEL}")
    print(f"   Максимум токенов: {MAX_TOKENS}")
