#!/usr/bin/env python3
"""
Инструкция по настройке OpenRouter для RAG агентов
"""

import os
from dotenv import load_dotenv

def show_openrouter_setup():
    """Показывает инструкции по настройке OpenRouter"""
    print("🔧 Настройка OpenRouter для RAG агентов")
    print("=" * 50)
    
    print("\n📋 Пошаговая инструкция:")
    print("1. Зарегистрируйтесь на https://openrouter.ai/")
    print("2. Создайте API ключ в разделе 'Keys'")
    print("3. Добавьте кредиты на счет (минимум $5)")
    print("4. Скопируйте API ключ")
    
    print("\n💰 Стоимость (примерно):")
    print("• Claude 3.5 Sonnet: $3 за 1M токенов входа, $15 за 1M токенов выхода")
    print("• GPT-4o Mini: $0.15 за 1M токенов входа, $0.60 за 1M токенов выхода")
    print("• Llama 3.1 8B: $0.05 за 1M токенов (бесплатно до лимита)")
    print("• Эмбеддинги: $0.10 за 1M токенов")
    
    print("\n🔑 Настройка API ключа:")
    print("Вариант 1 - Через .env файл:")
    print("   echo 'OPENROUTER_API_KEY=your_actual_key_here' > .env")
    
    print("\nВариант 2 - Через переменную окружения:")
    print("   export OPENROUTER_API_KEY='your_actual_key_here'")
    
    print("\n📝 Доступные модели в нашем проекте:")
    models = [
        ("anthropic/claude-3.5-sonnet", "Высокое качество, хорош для сложных задач"),
        ("openai/gpt-4o-mini", "Быстрая и дешевая модель OpenAI"),
        ("meta-llama/llama-3.1-8b-instruct", "Open source, часто бесплатная"),
        ("google/gemini-flash-1.5", "Быстрая модель Google"),
        ("openai/text-embedding-ada-002", "Эмбеддинги OpenAI")
    ]
    
    for model, description in models:
        print(f"   • {model}")
        print(f"     {description}")
    
    print("\n🧪 Тестирование настройки:")
    print("После настройки запустите:")
    print("   python rag_agent_examples.py models")

def check_current_setup():
    """Проверяет текущую настройку"""
    print("\n🔍 Проверка текущей настройки:")
    print("=" * 35)
    
    load_dotenv()
    
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if openrouter_key and openrouter_key != "your_openrouter_api_key_here":
        print("✅ OPENROUTER_API_KEY установлен")
        print(f"   Ключ: {openrouter_key[:12]}...{openrouter_key[-4:]}")
    else:
        print("❌ OPENROUTER_API_KEY не установлен или содержит placeholder")
    
    if openai_key and openai_key != "your_openai_api_key_here":
        print("✅ OPENAI_API_KEY установлен (fallback)")
        print(f"   Ключ: {openai_key[:12]}...{openai_key[-4:]}")
    else:
        print("❌ OPENAI_API_KEY не установлен или содержит placeholder")
    
    if not openrouter_key and not openai_key:
        print("\n⚠️  Ни один API ключ не настроен!")
        print("Настройте хотя бы один для работы системы.")
    elif openrouter_key:
        print("\n🎯 Рекомендация: Используйте OpenRouter")
        print("   • Больше моделей на выбор")
        print("   • Часто дешевле OpenAI")
        print("   • Доступ к open source моделям")
    else:
        print("\n🎯 Используется OpenAI (fallback)")

def test_connection():
    """Тестирует подключение к API"""
    print("\n🧪 Тестирование подключения:")
    print("=" * 30)
    
    load_dotenv()
    
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    
    if not openrouter_key or openrouter_key == "your_openrouter_api_key_here":
        print("❌ OPENROUTER_API_KEY не настроен для тестирования")
        return
    
    try:
        from rag_agent_examples import ChatOpenRouter
        
        print("🔗 Тестируем подключение к OpenRouter...")
        
        # Тестируем простой запрос
        llm = ChatOpenRouter(
            model_name="openai/gpt-4o-mini",  # Самая дешевая модель
            temperature=0,
            max_tokens=50,
            request_timeout=10
        )
        
        response = llm.invoke("Скажи 'Привет' на русском языке").content
        print(f"✅ Подключение успешно!")
        print(f"   Ответ: {response}")
        
    except Exception as e:
        print(f"❌ Ошибка подключения: {e}")
        print("Проверьте:")
        print("   • Корректность API ключа")
        print("   • Наличие кредитов на счету")
        print("   • Интернет соединение")

def main():
    """Главная функция настройки"""
    show_openrouter_setup()
    check_current_setup()
    
    # Спрашиваем пользователя о тестировании
    try:
        test_input = input("\n🤔 Хотите протестировать подключение? (y/n): ").strip().lower()
        if test_input in ['y', 'yes', 'да', 'д']:
            test_connection()
    except KeyboardInterrupt:
        print("\n👋 Настройка прервана")

if __name__ == "__main__":
    main()
