#!/usr/bin/env python3
"""
Zep + LangGraph Memory Integration Demo
Реализация персонализированного агента с долговременной памятью

Основано на документации: https://help.getzep.com/ecosystem/langgraph-memory

Возможности:
- Персистентная память через Zep Cloud
- Контекстно-зависимые ответы
- Поиск фактов и сущностей
- Автоматическое извлечение и обновление фактов
- Интеграция с LangGraph для управления состоянием
"""

import os
import asyncio
import uuid
from typing import TypedDict, Annotated
from dotenv import load_dotenv

# Zep imports
from zep_cloud.client import AsyncZep
from zep_cloud import Message

# LangChain imports  
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, trim_messages
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# LangGraph imports
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.prebuilt import ToolNode

# Загрузка переменных окружения
load_dotenv()

# Проверка API ключей
def check_api_keys():
    """Проверка наличия необходимых API ключей"""
    zep_key = os.getenv('ZEP_API_KEY')
    openai_key = os.getenv('OPENAI_API_KEY')
    
    print("🔑 Проверка API ключей:")
    print(f"Zep API: {'✅' if zep_key else '❌'}")
    print(f"OpenAI API: {'✅' if openai_key else '❌'}")
    
    if not zep_key:
        raise ValueError("❌ Требуется ZEP_API_KEY. Получите ключ на https://www.getzep.com/")
    if not openai_key:
        raise ValueError("❌ Требуется OPENAI_API_KEY. Получите ключ на https://platform.openai.com/")
    
    return zep_key, openai_key

# Инициализация клиентов
zep_key, openai_key = check_api_keys()
zep = AsyncZep(api_key=zep_key)

# Определение состояния агента
class State(TypedDict):
    """Состояние агента LangGraph"""
    messages: Annotated[list, add_messages]
    first_name: str
    last_name: str
    thread_id: str
    user_name: str

# Инструменты для поиска в Zep
@tool
async def search_facts(state: State, query: str, limit: int = 5) -> list[str]:
    """Поиск фактов во всех разговорах с пользователем.
    
    Args:
        state (State): Состояние агента.
        query (str): Поисковый запрос.
        limit (int): Количество результатов. По умолчанию 5.
        
    Returns:
        list: Список фактов, соответствующих запросу.
    """
    try:
        edges = await zep.graph.search(
            user_id=state["user_name"], 
            query=query, 
            limit=limit
        )
        # Извлекаем факты из результатов поиска
        facts = []
        for item in edges:
            if hasattr(item, 'fact'):
                facts.append(item.fact)
            elif hasattr(item, 'content'):
                facts.append(item.content)
        return facts
    except Exception as e:
        print(f"⚠️ Ошибка поиска фактов: {e}")
        return []

@tool  
async def search_nodes(state: State, query: str, limit: int = 5) -> list[str]:
    """Поиск узлов во всех разговорах с пользователем.
    
    Args:
        state (State): Состояние агента.
        query (str): Поисковый запрос.
        limit (int): Количество результатов. По умолчанию 5.
        
    Returns:
        list: Список резюме узлов, соответствующих запросу.
    """
    try:
        nodes = await zep.graph.search(
            user_id=state["user_name"], 
            query=query, 
            limit=limit
        )
        # Извлекаем резюме из результатов поиска
        summaries = []
        for item in nodes:
            if hasattr(item, 'summary'):
                summaries.append(item.summary)
            elif hasattr(item, 'fact'):
                summaries.append(item.fact)
            elif hasattr(item, 'content'):
                summaries.append(item.content)
        return summaries
    except Exception as e:
        print(f"⚠️ Ошибка поиска узлов: {e}")
        return []

# Настройка инструментов и LLM
tools = [search_facts, search_nodes]
tool_node = ToolNode(tools)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tools)

# Основная функция чат-бота
async def chatbot(state: State):
    """Основная функция чат-бота с интеграцией Zep"""
    try:
        # Получение контекста из Zep
        memory = await zep.thread.get_user_context(state["thread_id"])
        
        # Создание системного сообщения с контекстом
        system_message = SystemMessage(
            content=f"""Ты - дружелюбный и отзывчивый помощник с долговременной памятью. 
            Используй информацию о пользователе из предыдущих разговоров для персонализации ответов.
            Будь внимательным, эмпатичным и полезным.
            
            Контекст пользователя:
            {memory.context}"""
        )
        
        # Подготовка сообщений для LLM
        messages = [system_message] + state["messages"]
        
        # Получение ответа от LLM
        response = await llm.ainvoke(messages)
        
        # Сохранение нового диалога в Zep
        if state["messages"]:
            messages_to_save = [
                Message(
                    role="user",
                    name=f"{state['first_name']} {state['last_name']}",
                    content=state["messages"][-1].content,
                ),
                Message(role="assistant", content=response.content),
            ]
            
            await zep.thread.add_messages(
                thread_id=state["thread_id"],
                messages=messages_to_save,
            )
        
        return {"messages": [response]}
        
    except Exception as e:
        print(f"❌ Ошибка в chatbot: {e}")
        error_response = AIMessage(content="Извините, произошла ошибка. Попробуйте еще раз.")
        return {"messages": [error_response]}

# Функция маршрутизации
def should_continue(state: State):
    """Определяет, нужно ли использовать инструменты"""
    messages = state["messages"]
    last_message = messages[-1]
    
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    return END

# Создание графа LangGraph
def create_graph():
    """Создание графа LangGraph с интеграцией Zep"""
    workflow = StateGraph(State)
    
    # Добавление узлов
    workflow.add_node("chatbot", chatbot)
    workflow.add_node("tools", tool_node)
    
    # Настройка маршрутов
    workflow.set_entry_point("chatbot")
    workflow.add_conditional_edges("chatbot", should_continue)
    workflow.add_edge("tools", "chatbot")
    
    # Компиляция с checkpointer
    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)

# Функция для вызова графа
async def graph_invoke(
    message: str, 
    first_name: str, 
    last_name: str, 
    thread_id: str, 
    ai_response_only: bool = True
):
    """Вызов графа с сообщением пользователя"""
    graph = create_graph()
    user_name = f"{first_name}_{last_name}_{thread_id[:8]}"
    
    try:
        result = await graph.ainvoke(
            {
                "messages": [HumanMessage(content=message)],
                "first_name": first_name,
                "last_name": last_name,
                "thread_id": thread_id,
                "user_name": user_name,
            },
            config={"configurable": {"thread_id": thread_id}},
        )
        
        if ai_response_only:
            return result["messages"][-1].content
        else:
            return result["messages"]
            
    except Exception as e:
        print(f"❌ Ошибка выполнения графа: {e}")
        return "Извините, произошла ошибка при обработке вашего сообщения."

# Функция для извлечения сообщений
def extract_messages(result):
    """Извлечение сообщений из результата"""
    messages = []
    for msg in result["messages"]:
        if hasattr(msg, 'content'):
            role = "assistant" if isinstance(msg, AIMessage) else "user"
            messages.append(f"{role}: {msg.content}")
    return "\n".join(messages)

# Функция демонстрации
async def demo_conversation():
    """Демонстрация разговора с персистентной памятью"""
    print("🚀 Демонстрация Zep + LangGraph Memory")
    print("=" * 50)
    
    # Генерация уникальных идентификаторов
    first_name = "Анна"
    last_name = "Петрова" 
    thread_id = str(uuid.uuid4())
    user_name = f"{first_name}_{last_name}_{thread_id[:8]}"
    
    print(f"👤 Пользователь: {first_name} {last_name}")
    print(f"🔗 Thread ID: {thread_id}")
    print(f"🆔 User Name: {user_name}")
    
    # ВАЖНО: Сначала создаем пользователя и поток в Zep
    try:
        print("\n🔧 Создание пользователя в Zep...")
        await zep.user.add(
            user_id=user_name,
            first_name=first_name,
            last_name=last_name
        )
        print("✅ Пользователь создан")
        
        print("🔧 Создание потока в Zep...")
        await zep.thread.create(
            thread_id=thread_id,
            user_id=user_name
        )
        print("✅ Поток создан")
        
    except Exception as e:
        print(f"❌ Ошибка создания пользователя/потока: {e}")
        return
        
    print("-" * 50)
    
    # Серия сообщений для демонстрации
    messages = [
        "Привет! Меня зовут Анна, я программист из Москвы.",
        "Мне 28 лет, работаю в IT-компании уже 5 лет.",
        "Увлекаюсь машинным обучением и изучаю Python.",
        "У меня есть кот по имени Мурзик, он очень игривый.",
        "Планирую в следующем году изучать LangChain подробнее."
    ]
    
    # Проведение диалога
    for i, message in enumerate(messages, 1):
        print(f"\n💬 Сообщение {i}:")
        print(f"👤 Анна: {message}")
        
        response = await graph_invoke(message, first_name, last_name, thread_id)
        print(f"🤖 Помощник: {response}")
        
        # Небольшая пауза между сообщениями
        await asyncio.sleep(0.5)
    
    print("\n" + "="*50)
    print("📊 Тестирование памяти агента")
    print("="*50)
    
    # Тестовые вопросы для проверки памяти
    test_questions = [
        "Как меня зовут и где я работаю?",
        "Сколько мне лет и какой у меня опыт?",
        "Что я изучаю и чем увлекаюсь?", 
        "Расскажи про моего кота.",
        "Какие у меня планы на следующий год?"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n❓ Тест {i}: {question}")
        response = await graph_invoke(question, first_name, last_name, thread_id)
        print(f"🤖 Ответ: {response}")
        await asyncio.sleep(0.5)
    
    # Показ контекста из Zep
    print("\n" + "="*50)
    print("🧠 Контекст из Zep Memory")
    print("="*50)
    
    try:
        print("⏳ Ожидание обработки данных в Zep...")
        await asyncio.sleep(3)
        
        memory = await zep.thread.get_user_context(thread_id=thread_id)
        if memory and memory.context:
            print(memory.context)
        else:
            print("Контекст пока пуст. Данные могут еще обрабатываться в Zep.")
    except Exception as e:
        print(f"❌ Ошибка получения контекста: {e}")
    
    print(f"\n✅ Демонстрация завершена!")

# Интерактивный режим
async def interactive_mode():
    """Интерактивный режим для общения с агентом"""
    print("🎮 Интерактивный режим Zep + LangGraph")
    print("Введите 'quit' для выхода")
    print("="*50)
    
    # Настройка пользователя
    first_name = input("👤 Введите ваше имя: ").strip() or "Пользователь"
    last_name = input("👤 Введите вашу фамилию: ").strip() or "Тестовый"
    thread_id = str(uuid.uuid4())
    user_name = f"{first_name}_{last_name}_{thread_id[:8]}"
    
    print(f"\n🔗 Thread ID: {thread_id}")
    print(f"🆔 User Name: {user_name}")
    
    # Создание пользователя и потока в Zep
    try:
        print("🔧 Создание пользователя и потока...")
        await zep.user.add(
            user_id=user_name,
            first_name=first_name,
            last_name=last_name
        )
        await zep.thread.create(
            thread_id=thread_id,
            user_id=user_name
        )
        print("✅ Пользователь и поток созданы")
    except Exception as e:
        print(f"❌ Ошибка создания: {e}")
        return
    
    print("💬 Начинаем диалог...\n")
    
    while True:
        try:
            user_input = input(f"{first_name}: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'выход']:
                print("👋 До свидания!")
                break
            
            if not user_input:
                continue
                
            response = await graph_invoke(user_input, first_name, last_name, thread_id)
            print(f"🤖 Помощник: {response}\n")
            
        except KeyboardInterrupt:
            print("\n👋 До свидания!")
            break
        except Exception as e:
            print(f"❌ Ошибка: {e}")

# Основная функция
async def main():
    """Главная функция"""
    print("🔧 Инициализация Zep + LangGraph Demo")
    
    try:
        check_api_keys()
        print("✅ API ключи проверены")
        
        # Выбор режима
        print("\nВыберите режим:")
        print("1. Автоматическая демонстрация")
        print("2. Интерактивный режим")
        
        choice = input("Введите номер (1 или 2): ").strip()
        
        if choice == "1":
            await demo_conversation()
        elif choice == "2":
            await interactive_mode()
        else:
            print("❌ Неверный выбор. Запуск демонстрации...")
            await demo_conversation()
            
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")

if __name__ == "__main__":
    asyncio.run(main())
