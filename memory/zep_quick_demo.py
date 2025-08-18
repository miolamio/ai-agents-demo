#!/usr/bin/env python3
"""
Быстрая демонстрация Zep Cloud Memory
Простой пример для тестирования основной функциональности Zep
"""

import os
import asyncio
import uuid
from dotenv import load_dotenv

# Zep imports
from zep_cloud.client import AsyncZep
from zep_cloud import Message

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

# Загрузка переменных окружения
load_dotenv()

async def quick_zep_demo():
    """Быстрая демонстрация основной функциональности Zep"""
    print("🚀 Быстрая демо Zep Cloud Memory")
    print("=" * 40)
    
    # Проверка API ключей
    zep_key = os.getenv("ZEP_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    print(f"Zep API: {'✅' if zep_key else '❌'}")
    print(f"OpenAI API: {'✅' if openai_key else '❌'}")
    
    if not zep_key:
        print("\n❌ Требуется ZEP_API_KEY для работы")
        print("Получите ключ на: https://www.getzep.com/")
        return
        
    if not openai_key:
        print("\n❌ Требуется OPENAI_API_KEY для работы")
        return
    
    # Инициализация клиентов
    try:
        zep = AsyncZep(api_key=zep_key)
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
        print("✅ Клиенты инициализированы")
    except Exception as e:
        print(f"❌ Ошибка инициализации: {e}")
        return
    
    # Создание пользователя и потока
    user_id = "demo_user_" + str(uuid.uuid4())[:8]
    thread_id = str(uuid.uuid4())
    
    print(f"\n👤 User ID: {user_id}")
    print(f"🔗 Thread ID: {thread_id}")
    
    # ВАЖНО: Сначала создаем пользователя и поток в Zep
    try:
        print("🔧 Создание пользователя в Zep...")
        await zep.user.add(
            user_id=user_id,
            first_name="Дмитрий",
            last_name="Тестовый"
        )
        print("✅ Пользователь создан")
        
        print("🔧 Создание потока в Zep...")
        await zep.thread.create(
            thread_id=thread_id,
            user_id=user_id
        )
        print("✅ Поток создан")
        
    except Exception as e:
        print(f"❌ Ошибка создания пользователя/потока: {e}")
        return
    
    # Тестовые сообщения
    conversations = [
        {
            "user": "Привет! Меня зовут Дмитрий, я разработчик из Санкт-Петербурга.",
            "context": "Знакомство с пользователем"
        },
        {
            "user": "Мне 30 лет, работаю в стартапе, разрабатываю мобильные приложения.",
            "context": "Информация о работе"
        },
        {
            "user": "Увлекаюсь искусственным интеллектом и машинным обучением.",
            "context": "Хобби и интересы"
        },
        {
            "user": "У меня есть собака породы лабрадор по кличке Рекс.",
            "context": "Личная информация"
        }
    ]
    
    print(f"\n💬 Проведение диалога:")
    print("-" * 40)
    
    # Проведение диалога с сохранением в Zep
    for i, conv in enumerate(conversations, 1):
        user_message = conv["user"]
        print(f"\n{i}. 👤 Пользователь: {user_message}")
        
        # Получение ответа от LLM
        try:
            response = await llm.ainvoke([HumanMessage(content=user_message)])
            ai_response = response.content
            print(f"   🤖 Помощник: {ai_response}")
            
            # Сохранение в Zep
            messages_to_save = [
                Message(
                    role="user",
                    name="Дмитрий",
                    content=user_message
                ),
                Message(
                    role="assistant", 
                    content=ai_response
                )
            ]
            
            await zep.thread.add_messages(
                thread_id=thread_id,
                messages=messages_to_save,
            )
            print("   💾 Сохранено в Zep")
            
        except Exception as e:
            print(f"   ❌ Ошибка: {e}")
        
        # Пауза между сообщениями
        await asyncio.sleep(0.5)
    
    # Тестирование памяти
    print(f"\n🧠 Тестирование памяти:")
    print("-" * 40)
    
    try:
        # Небольшая пауза для обработки данных в Zep
        print("⏳ Ожидание обработки данных в Zep...")
        await asyncio.sleep(2)
        
        # Получение контекста пользователя
        memory = await zep.thread.get_user_context(thread_id=thread_id)
        
        print("📋 Контекст из Zep:")
        if memory and memory.context:
            print(memory.context)
        else:
            print("Контекст пока пуст. Возможно, данные еще обрабатываются.")
        
        # Тестовые вопросы (только если есть контекст)
        if memory and memory.context:
            test_questions = [
                "Как меня зовут и откуда я?",
                "Где я работаю и что разрабатываю?", 
                "Чем я увлекаюсь?",
                "Есть ли у меня питомцы?"
            ]
            
            print(f"\n❓ Тестовые вопросы:")
            for i, question in enumerate(test_questions, 1):
                print(f"\n{i}. Вопрос: {question}")
                
                # Формирование промпта с контекстом
                prompt = f"""Ответь на вопрос пользователя, используя информацию из контекста.
                
Контекст пользователя:
{memory.context}

Вопрос: {question}

Ответ:"""
                
                try:
                    response = await llm.ainvoke([HumanMessage(content=prompt)])
                    print(f"   Ответ: {response.content}")
                except Exception as e:
                    print(f"   ❌ Ошибка получения ответа: {e}")
        else:
            print("\n⏳ Пропуск тестовых вопросов - контекст еще не готов")
        
        # Поиск фактов (только если есть контекст)
        if memory and memory.context:
            print(f"\n🔍 Тестирование поиска фактов:")
            search_queries = ["работа", "питомец", "хобби"]
            
            for query in search_queries:
                try:
                    facts = await zep.graph.search(
                        user_id=user_id,
                        query=query,
                        limit=3
                    )
                    print(f"\n🔎 Поиск '{query}':")
                    if facts:
                        for item in facts:
                            # Пытаемся получить факт или другую информацию
                            if hasattr(item, 'fact'):
                                print(f"   • Факт: {item.fact}")
                            elif hasattr(item, 'summary'):
                                print(f"   • Узел: {item.summary}")
                            elif hasattr(item, 'content'):
                                print(f"   • Контент: {item.content}")
                            else:
                                print(f"   • Результат: {item}")
                    else:
                        print(f"   📭 Результаты по запросу '{query}' не найдены")
                except Exception as e:
                    print(f"   ❌ Ошибка поиска '{query}': {e}")
        else:
            print(f"\n⏳ Пропуск поиска фактов - данные еще обрабатываются")
                
    except Exception as e:
        print(f"❌ Ошибка тестирования памяти: {e}")
    
    print(f"\n✅ Демонстрация завершена!")

# Функция для очистки тестовых данных
async def cleanup_demo_data():
    """Очистка тестовых данных (опционально)"""
    print("🧹 Очистка тестовых данных...")
    # В реальном приложении здесь можно добавить логику очистки
    print("✅ Очистка завершена")

if __name__ == "__main__":
    print("Выберите действие:")
    print("1. Запустить демо")
    print("2. Очистить тестовые данные")
    
    choice = input("Введите номер (1 или 2): ").strip()
    
    if choice == "1":
        asyncio.run(quick_zep_demo())
    elif choice == "2":
        asyncio.run(cleanup_demo_data())
    else:
        print("Запуск демо по умолчанию...")
        asyncio.run(quick_zep_demo())
