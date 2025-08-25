#!/usr/bin/env python3
"""
Пример 1b: Интеграция LangSmith с LangChain
Демонстрирует автоматическую трассировку при использовании LangChain

Основано на официальной документации LangSmith:
https://smith.langchain.com/
"""

import os
from datetime import datetime

# Загрузка переменных окружения из .env файла
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # Если dotenv не установлен, используем системные переменные

def demo_with_langchain():
    """Демонстрация автоматической трассировки с LangChain"""
    
    print("=== Пример 1b: LangChain + LangSmith интеграция ===\n")
    
    # Проверяем переменные окружения
    required_vars = {
        "LANGSMITH_API_KEY": os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY"),
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "LANGSMITH_TRACING": os.getenv("LANGSMITH_TRACING", "false"),
        "LANGSMITH_PROJECT": os.getenv("LANGSMITH_PROJECT") or os.getenv("LANGCHAIN_PROJECT", "default")
    }
    
    print("🔍 Проверка переменных окружения:")
    for var, value in required_vars.items():
        status = "✅" if value and value != "false" else "❌"
        display_value = value[:20] + "..." if value and len(value) > 20 else value or "не установлена"
        print(f"  {status} {var}: {display_value}")
    
    if not required_vars["LANGSMITH_API_KEY"]:
        print("\n⚠️  Для работы с реальным LangSmith установите:")
        print("export LANGSMITH_API_KEY='your-api-key'")
        print("export LANGSMITH_TRACING=true")
        print("export LANGSMITH_PROJECT='your-project-name'")
        print("\nИспользуется демо-режим...\n")
        demo_without_real_api()
        return
    
    try:
        # Попытка импорта LangChain
        from langchain_openai import ChatOpenAI
        
        print(f"\n🚀 Создание LLM с трассировкой...")
        print(f"📊 Проект: {required_vars['LANGSMITH_PROJECT']}")
        print(f"🔗 Endpoint: {os.getenv('LANGSMITH_ENDPOINT', 'https://api.smith.langchain.com')}")
        
        # Создаем модель - трассировка включается автоматически
        llm = ChatOpenAI()
        
        # Тестовые запросы
        test_prompts = [
            "Hello, world!",
            "Explain machine learning in simple terms",
            "What is the capital of France?"
        ]
        
        print(f"\n📝 Выполнение {len(test_prompts)} запросов с автоматической трассировкой:")
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n{i}. Запрос: {prompt}")
            
            try:
                response = llm.invoke(prompt)
                print(f"   Ответ: {response.content[:100]}...")
                print(f"   ✅ Трассировка отправлена в LangSmith")
                
            except Exception as e:
                print(f"   ❌ Ошибка: {e}")
        
        print(f"\n🎉 Готово! Проверьте трассировки в LangSmith:")
        print(f"🔗 https://smith.langchain.com/")
        
    except ImportError as e:
        print(f"\n❌ Ошибка импорта: {e}")
        print("Для работы с LangChain установите:")
        print("pip install langchain langchain-openai")
        demo_without_real_api()
        
    except Exception as e:
        print(f"\n❌ Ошибка выполнения: {e}")
        demo_without_real_api()


def demo_without_real_api():
    """Демонстрация концепций без реальных API вызовов"""
    
    print("\n🎭 ДЕМО-РЕЖИМ: Симуляция LangChain + LangSmith")
    print("-" * 50)
    
    # Эмулируем создание трассировок
    demo_traces = [
        {
            "id": "trace_001",
            "name": "ChatOpenAI.invoke",
            "input": "Hello, world!",
            "output": "Hello! How can I help you today?",
            "start_time": datetime.now(),
            "duration_ms": 850,
            "tokens": {"input": 3, "output": 8, "total": 11},
            "cost": 0.00022
        },
        {
            "id": "trace_002", 
            "name": "ChatOpenAI.invoke",
            "input": "Explain machine learning",
            "output": "Machine learning is a subset of artificial intelligence...",
            "start_time": datetime.now(),
            "duration_ms": 1200,
            "tokens": {"input": 4, "output": 45, "total": 49},
            "cost": 0.00098
        },
        {
            "id": "trace_003",
            "name": "ChatOpenAI.invoke", 
            "input": "What is the capital of France?",
            "output": "The capital of France is Paris.",
            "start_time": datetime.now(),
            "duration_ms": 600,
            "tokens": {"input": 8, "output": 7, "total": 15},
            "cost": 0.00030
        }
    ]
    
    print("📊 Симуляция автоматических трассировок:")
    
    total_cost = 0
    total_tokens = 0
    
    for trace in demo_traces:
        print(f"\n🔍 Трассировка: {trace['id']}")
        print(f"   📝 Запрос: {trace['input']}")
        print(f"   🤖 Ответ: {trace['output'][:50]}...")
        print(f"   ⏱️  Время: {trace['duration_ms']}ms")
        print(f"   🎯 Токены: {trace['tokens']['total']} ({trace['tokens']['input']}→{trace['tokens']['output']})")
        print(f"   💰 Стоимость: ${trace['cost']:.5f}")
        
        total_cost += trace['cost']
        total_tokens += trace['tokens']['total']
    
    print(f"\n📈 Общая статистика:")
    print(f"   💰 Общая стоимость: ${total_cost:.5f}")
    print(f"   🎯 Общее количество токенов: {total_tokens}")
    print(f"   📊 Средняя стоимость за токен: ${total_cost/total_tokens:.7f}")
    
    print(f"\n✨ В реальном LangSmith вы увидите:")
    print(f"   📊 Временные диаграммы выполнения")
    print(f"   🔍 Детальные логи каждого шага")
    print(f"   💰 Анализ затрат по моделям")
    print(f"   📈 Метрики производительности")
    print(f"   🐛 Отладочную информацию")


def show_integration_examples():
    """Показывает примеры интеграции из официальной документации"""
    
    print("\n📚 Примеры кода из официальной документации LangSmith:")
    print("-" * 60)
    
    print("\n1️⃣ Настройка переменных окружения:")
    print("""
export LANGSMITH_TRACING="true"
export LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
export LANGSMITH_API_KEY="your-api-key"
export LANGSMITH_PROJECT="pr-artistic-gastropod-72"
export OPENAI_API_KEY="your-openai-api-key"
""")
    
    print("2️⃣ Простой пример с ChatOpenAI:")
    print("""
from langchain_openai import ChatOpenAI

llm = ChatOpenAI()
llm.invoke("Hello, world!")
# Трассировка создается автоматически!
""")
    
    print("3️⃣ Установка зависимостей:")
    print("""
pip install -U langchain langchain-openai
""")
    
    print("4️⃣ Проверка подключения:")
    print("""
from langsmith import Client
client = Client()
print("Connected to LangSmith!")
""")


if __name__ == "__main__":
    demo_with_langchain()
    show_integration_examples()
    
    print(f"\n🔗 Дополнительные ресурсы:")
    print(f"   📖 Conceptual Guide: https://docs.smith.langchain.com/concepts")
    print(f"   🎓 LangSmith Academy: https://academy.langchain.com/")
    print(f"   🛠️  Интеграции: https://docs.smith.langchain.com/tracing")
