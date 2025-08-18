#!/usr/bin/env python3
"""
Быстрая демонстрация LangChain + Mem0 интеграции
Простой пример для тестирования функциональности
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from mem0 import MemoryClient

# Загрузка переменных окружения
load_dotenv()

def quick_demo():
    """Быстрая демонстрация основной функциональности"""
    print("🚀 Быстрая демо LangChain + Mem0")
    print("=" * 40)
    
    # Проверка API ключей
    openai_key = os.getenv("OPENAI_API_KEY")
    mem0_key = os.getenv("MEM0_API_KEY")
    
    print(f"OpenAI API: {'✅' if openai_key else '❌'}")
    print(f"Mem0 API: {'✅' if mem0_key else '❌'}")
    
    if not openai_key:
        print("\n❌ Требуется OPENAI_API_KEY для работы")
        return
    
    # Инициализация компонентов
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    
    mem0_client = None
    if mem0_key:
        try:
            # Инициализируем клиент с явным указанием API ключа
            mem0_client = MemoryClient(api_key=mem0_key)
            print("✅ Mem0 клиент инициализирован")
        except Exception as e:
            print(f"⚠️ Mem0 недоступен: {e}")
    
    # Простой диалог
    user_id = "demo_user"
    messages = [
        "Привет! Меня зовут Анна, я учитель математики",
        "Мне 32 года, живу в Москве", 
        "Люблю читать книги по психологии",
        "Планирую изучать Python для работы"
    ]
    
    print(f"\n💬 Диалог с пользователем {user_id}:")
    print("-" * 40)
    
    for msg in messages:
        print(f"👤 Пользователь: {msg}")
        
        # Ответ от LLM
        response = llm.invoke(f"Ответь дружелюбно на сообщение: {msg}")
        print(f"🤖 Помощник: {response.content}")
        
        # Сохранение в Mem0
        if mem0_client:
            try:
                conversation = [
                    {"role": "user", "content": msg},
                    {"role": "assistant", "content": response.content}
                ]
                mem0_client.add(messages=conversation, user_id=user_id)
                print("💾 Сохранено в память")
            except Exception as e:
                print(f"⚠️ Ошибка сохранения: {e}")
        
        print("-" * 30)
    
    # Показ сохраненных воспоминаний
    if mem0_client:
        try:
            memories = mem0_client.get_all(user_id=user_id)
            if memories:
                print(f"\n🧠 Сохраненные воспоминания ({len(memories)} шт.):")
                for i, memory in enumerate(memories[:5], 1):
                    if memory and isinstance(memory, dict):
                        content = memory.get('memory', memory.get('text', memory.get('content', '')))
                        print(f"{i}. {content}")
            else:
                print("\n🧠 Пока нет сохраненных воспоминаний")
        except Exception as e:
            print(f"❌ Ошибка получения воспоминаний: {e}")
    
    print(f"\n✅ Демонстрация завершена!")

if __name__ == "__main__":
    quick_demo()
