"""
Демонстрация LangChain: Простой чат-бот с памятью
Этот пример показывает базовую работу с LangChain и RunnableWithMessageHistory
Обновлено для совместимости с LangChain 0.3+ (больше не используются deprecated API)
"""

# 1. Импорты
from langchain_openai import ChatOpenAI
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os
import time
from typing import List

# 2. Загрузка переменных окружения
load_dotenv()

# Проверка наличия ключа
if not os.getenv("OPENAI_API_KEY"):
    print("❌ Ошибка: Не найден OPENAI_API_KEY в файле .env")
    print("Создайте файл .env и добавьте ваш ключ OpenAI")
    exit(1)

# 3. Создание in-memory класса для истории сообщений
class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    """In memory implementation of chat message history."""
    
    messages: List[BaseMessage] = Field(default_factory=list)
    
    def add_messages(self, messages: List[BaseMessage]) -> None:
        """Add a list of messages to the store"""
        self.messages.extend(messages)
    
    def clear(self) -> None:
        self.messages = []

# Хранилище для истории сессий
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryHistory()
    return store[session_id]

def main():
    print("=== LangChain Demo: Чат-бот с памятью ===\n")
    
    try:
        # 4. Инициализация модели и создание chain с памятью
        # Создание ChatOpenAI с model="gpt-3.5-turbo", temperature=0.7
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7
        )
        
        # Создание prompt template с поддержкой истории
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Ты дружелюбный помощник. Отвечай на русском языке и помни предыдущие сообщения."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ])
        
        # Создание runnable chain
        runnable = prompt | llm
        
        # Создание RunnableWithMessageHistory
        conversation = RunnableWithMessageHistory(
            runnable,
            get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )
        
        # 4. Основной цикл
        # Приветственное сообщение
        print("Бот: Привет! Я чат-бот с памятью. Могу запоминать наш разговор.")
        print()
        
        # Цикл с тремя примерами вопросов
        questions = [
            "Меня зовут Иван, я учусь программированию",
            "Какие языки программирования ты рекомендуешь для начинающих?",
            "А ты помнишь, как меня зовут?"
        ]
        
        # Используем фиксированный session_id для демонстрации
        config = {"configurable": {"session_id": "demo_session"}}
        
        for question in questions:
            print(f"Вы: {question}")
            
            # Получение ответа от чат-бота
            result = conversation.invoke(
                {"input": question},
                config=config
            )
            print(f"Бот: {result.content}")
            print()
            
            # Пауза между вопросами для наглядности
            time.sleep(1)
        
        # 5. Вывод истории разговора из памяти
        print("История разговора:")
        print("=" * 50)
        session_history = get_session_history("demo_session")
        for message in session_history.messages:
            if hasattr(message, 'content'):
                message_type = "Human" if message.__class__.__name__ == "HumanMessage" else "AI"
                print(f"{message_type}: {message.content}")
        
    except Exception as e:
        print(f"❌ Ошибка при работе с LangChain: {e}")
        print("Убедитесь, что:")
        print("1. Установлены все зависимости: pip install -r requirements.txt")
        print("2. Правильно указан OPENAI_API_KEY в файле .env")
        print("3. Есть доступ к интернету для обращения к OpenAI API")

if __name__ == "__main__":
    main()