"""
LangChain + Mem0 Integration Demo
Персональный помощник с долговременной памятью
"""

import os
from datetime import datetime
from typing import List, Dict, Any
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables import RunnablePassthrough
from mem0 import MemoryClient

# Загрузка переменных окружения
load_dotenv()

# Конфигурация
MEM0_API_KEY = os.getenv("MEM0_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2000"))

# Установка переменных окружения
if MEM0_API_KEY:
    os.environ["MEM0_API_KEY"] = MEM0_API_KEY
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

class PersonalAssistant:
    """Персональный помощник с долговременной памятью"""
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.setup_components()
        
    def search_memories(self, query: str, limit: int = 5) -> List[Dict]:
        """Поиск воспоминаний по запросу"""
        if not self.mem0_client:
            return []
            
        try:
            results = self.mem0_client.search(
                query=query, 
                user_id=self.user_id,
                limit=limit,
                output_format="v1.1"  # Используем новый формат v1.1
            )
            
            # Обработка нового формата v1.1
            if isinstance(results, dict) and 'results' in results:
                return results['results']
            return results if results else []
        except Exception as e:
            print(f"❌ Ошибка поиска воспоминаний: {e}")
            return []
        
    def setup_components(self):
        """Инициализация компонентов"""
        # Инициализация Mem0 клиента
        self.mem0_client = None
        if MEM0_API_KEY:
            try:
                # Инициализируем клиент с явным указанием API ключа
                self.mem0_client = MemoryClient(api_key=MEM0_API_KEY)
                print("✅ Mem0 клиент успешно инициализирован")
            except Exception as e:
                print(f"❌ Ошибка инициализации Mem0: {e}")
        else:
            print("⚠️  MEM0_API_KEY не найден - работа в демо режиме")
            
        # Инициализация OpenAI
        self.llm = None
        if OPENAI_API_KEY:
            try:
                self.llm = ChatOpenAI(
                    model=OPENAI_MODEL,
                    max_tokens=MAX_TOKENS,
                    temperature=0.7
                )
                print(f"✅ OpenAI модель {OPENAI_MODEL} инициализирована")
            except Exception as e:
                print(f"❌ Ошибка инициализации OpenAI: {e}")
        else:
            print("⚠️  OPENAI_API_KEY не найден")
            
        # Краткосрочная память для текущего разговора
        self.session_history = InMemoryChatMessageHistory()
        
        self.setup_conversation_chain()
        
    def setup_conversation_chain(self):
        """Настройка цепочки разговора"""
        if not self.llm:
            return
            
        # Системный промпт с инструкциями
        system_prompt = """Ты персональный помощник пользователя. Твоя задача:

1. Вести дружелюбный и полезный разговор
2. Запоминать важные факты о пользователе (интересы, предпочтения, планы, личная информация)
3. Использовать сохраненные воспоминания для персонализации ответов
4. Задавать уточняющие вопросы для лучшего понимания пользователя

ВАЖНЫЕ ФАКТЫ ДЛЯ ЗАПОМИНАНИЯ:
- Личная информация (имя, возраст, профессия, семейное положение)
- Интересы и хобби
- Предпочтения (еда, музыка, фильмы, книги)
- Цели и планы
- Важные даты и события
- Контакты и связи
- Проблемы и заботы

Контекст из долговременной памяти: {memory_context}

Отвечай естественно, используя информацию из памяти когда это уместно."""

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])
        
        # Создаем цепочку с современным подходом
        self.chain = self.prompt | self.llm
        
        # Настраиваем runnable with message history
        def get_session_history(session_id: str) -> BaseChatMessageHistory:
            return self.session_history
            
        self.conversation = RunnableWithMessageHistory(
            self.chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )
    
    def get_long_term_memory(self) -> str:
        """Получение долговременных воспоминаний о пользователе"""
        if not self.mem0_client:
            return "Долговременная память недоступна (демо режим)"
            
        try:
            memories = self.mem0_client.get_all(
                user_id=self.user_id,
                output_format="v1.1"  # Используем новый формат v1.1
            )
            
            # Обработка нового формата v1.1
            if isinstance(memories, dict) and 'results' in memories:
                memory_list = memories['results']
            else:
                memory_list = memories if memories else []
                
            if not memory_list:
                return "Пока нет сохраненных воспоминаний"
                
            memory_text = "Что я помню о пользователе:\n"
            for i, memory in enumerate(memory_list[:10], 1):  # Последние 10 воспоминаний
                # Проверяем, что memory не None
                if not memory or not isinstance(memory, dict):
                    continue
                    
                # Обрабатываем новый формат данных v1.1
                content = memory.get('memory', memory.get('text', memory.get('content', '')))
                
                # Безопасное извлечение категории
                metadata = memory.get('metadata', {})
                if metadata and isinstance(metadata, dict):
                    category = metadata.get('category', '')
                else:
                    category = ''
                    
                if category:
                    memory_text += f"{i}. [{category}] {content}\n"
                else:
                    memory_text += f"{i}. {content}\n"
                
            return memory_text
            
        except Exception as e:
            print(f"❌ Ошибка получения воспоминаний: {e}")
            return "Ошибка доступа к долговременной памяти"
    
    def save_to_long_term_memory(self, conversation_history: List[Dict]):
        """Сохранение важных фактов в долговременную память"""
        if not self.mem0_client:
            print("⚠️  Демо режим: воспоминания не сохраняются")
            return
            
        try:
            # Сохраняем с метаданными для лучшей организации
            result = self.mem0_client.add(
                messages=conversation_history, 
                user_id=self.user_id,
                output_format="v1.1",  # Используем новый формат v1.1
                metadata={
                    "category": "conversation",
                    "timestamp": datetime.now().isoformat(),
                    "source": "langchain_assistant"
                }
            )
            print("💾 Воспоминания сохранены в долговременную память")
            
        except Exception as e:
            print(f"❌ Ошибка сохранения воспоминаний: {e}")
    
    def chat(self, user_input: str) -> str:
        """Основная функция чата"""
        if not self.llm:
            return "❌ OpenAI не настроен. Проверьте API ключ."
            
        try:
            # Получаем контекст из долговременной памяти
            # Сначала пробуем поиск по релевантности
            relevant_memories = self.search_memories(user_input, limit=3)
            if relevant_memories:
                memory_context = "Релевантные воспоминания:\n"
                for i, memory in enumerate(relevant_memories, 1):
                    if memory and isinstance(memory, dict):
                        content = memory.get('memory', memory.get('text', ''))
                        memory_context += f"{i}. {content}\n"
            else:
                # Fallback к общему контексту памяти
                memory_context = self.get_long_term_memory()
            
            # Генерируем ответ с новым API
            response = self.conversation.invoke(
                {
                    "input": user_input,
                    "memory_context": memory_context
                },
                config={"configurable": {"session_id": self.user_id}}
            )
            
            # Извлекаем текст ответа
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Сохраняем обмен в долговременную память
            conversation_exchange = [
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": response_text}
            ]
            self.save_to_long_term_memory(conversation_exchange)
            
            return response_text
            
        except Exception as e:
            return f"❌ Ошибка обработки сообщения: {e}"
    
    def show_memories(self):
        """Показать все сохраненные воспоминания"""
        print(f"\n🧠 Воспоминания о пользователе {self.user_id}:")
        print("=" * 50)
        
        if not self.mem0_client:
            print("⚠️  Долговременная память недоступна (демо режим)")
            return
            
        try:
            memories = self.mem0_client.get_all(
                user_id=self.user_id,
                output_format="v1.1"  # Используем новый формат v1.1
            )
            
            # Обработка нового формата v1.1 - данные в словаре с ключом 'results'
            if isinstance(memories, dict):
                if 'results' in memories:
                    memory_list = memories['results']
                else:
                    memory_list = [memories]  # Если это единичное воспоминание
            else:
                memory_list = memories if memories else []
            
            if not memory_list:
                print("📝 Пока нет сохраненных воспоминаний")
                return
                
            for i, memory in enumerate(memory_list, 1):
                # Проверяем, что memory не None
                if not memory or not isinstance(memory, dict):
                    continue
                    
                # Обрабатываем новый формат данных v1.1
                content = memory.get('memory', memory.get('text', memory.get('content', '')))
                created = memory.get('created_at', 'Неизвестно')
                
                # Безопасное извлечение метаданных
                metadata = memory.get('metadata', {})
                if metadata and isinstance(metadata, dict):
                    category = metadata.get('category', '')
                    source = metadata.get('source', '')
                else:
                    category = ''
                    source = ''
                
                if category:
                    print(f"{i}. [{category}] {content}")
                else:
                    print(f"{i}. {content}")
                    
                print(f"   Создано: {created}")
                if source:
                    print(f"   Источник: {source}")
                print("-" * 30)
                
        except Exception as e:
            print(f"❌ Ошибка получения воспоминаний: {e}")
    
    def clear_memories(self):
        """Очистить все воспоминания (осторожно!)"""
        if not self.mem0_client:
            print("⚠️  Долговременная память недоступна")
            return
            
        try:
            # Получаем все воспоминания
            memories = self.mem0_client.get_all(
                user_id=self.user_id,
                output_format="v1.1"  # Используем новый формат v1.1
            )
            
            # Обработка нового формата v1.1
            if isinstance(memories, dict) and 'results' in memories:
                memory_list = memories['results']
            else:
                memory_list = memories if memories else []
            
            # Удаляем каждое воспоминание
            for memory in memory_list:
                if memory and isinstance(memory, dict):
                    memory_id = memory.get('id')
                    if memory_id:
                        self.mem0_client.delete(memory_id)
                    
            print(f"🗑️  Все воспоминания пользователя {self.user_id} удалены")
            
        except Exception as e:
            print(f"❌ Ошибка очистки воспоминаний: {e}")

def interactive_chat():
    """Интерактивный чат с помощником"""
    print("🤖 Персональный помощник с памятью")
    print("=" * 50)
    
    # Запрос ID пользователя
    user_id = input("👤 Введите ваш ID (или нажмите Enter для demo_user): ").strip()
    if not user_id:
        user_id = "demo_user"
    
    # Создание помощника
    assistant = PersonalAssistant(user_id)
    
    print(f"\n🎯 Чат с персональным помощником начат для пользователя: {user_id}")
    print("💡 Команды:")
    print("   /memories - показать воспоминания")
    print("   /search <запрос> - поиск в воспоминаниях")
    print("   /clear - очистить воспоминания") 
    print("   /quit - выйти")
    print("-" * 50)
    
    while True:
        try:
            user_input = input(f"\n{user_id}: ").strip()
            
            if not user_input:
                continue
                
            # Обработка команд
            if user_input.lower() == '/quit':
                print("👋 До свидания!")
                break
            elif user_input.lower() == '/memories':
                assistant.show_memories()
                continue
            elif user_input.lower().startswith('/search '):
                search_query = user_input[8:].strip()  # Убираем '/search '
                if search_query:
                    print(f"🔍 Поиск воспоминаний по запросу: '{search_query}'")
                    results = assistant.search_memories(search_query)
                    if results:
                        print("Найденные воспоминания:")
                        for i, memory in enumerate(results, 1):
                            if memory and isinstance(memory, dict):
                                content = memory.get('memory', memory.get('text', ''))
                                print(f"{i}. {content}")
                    else:
                        print("Ничего не найдено")
                else:
                    print("❌ Укажите запрос для поиска: /search <ваш запрос>")
                continue
            elif user_input.lower() == '/clear':
                confirm = input("⚠️  Вы уверены? Это удалит ВСЕ воспоминания (y/N): ")
                if confirm.lower() == 'y':
                    assistant.clear_memories()
                continue
            
            # Обычный чат
            print("🤖 Помощник: ", end="", flush=True)
            response = assistant.chat(user_input)
            print(response)
            
        except KeyboardInterrupt:
            print("\n👋 Чат завершен по прерыванию")
            break
        except Exception as e:
            print(f"❌ Ошибка: {e}")

def demo_conversation():
    """Демонстрация работы с предустановленным диалогом"""
    print("🎬 Демонстрация работы помощника")
    print("=" * 50)
    
    assistant = PersonalAssistant("demo_user")
    
    # Примеры сообщений для демонстрации
    demo_messages = [
        "Привет! Меня зовут Алексей, мне 28 лет",
        "Я работаю программистом в IT компании",
        "Мое хобби - фотография и путешествия",
        "Планирую поехать в Японию в следующем году",
        "Люблю читать научную фантастику",
        "У меня есть кот по имени Барсик"
    ]
    
    print("Симуляция разговора:")
    for message in demo_messages:
        print(f"\n👤 Пользователь: {message}")
        response = assistant.chat(message)
        print(f"🤖 Помощник: {response}")
        
    print(f"\n{'='*50}")
    print("Демонстрация завершена. Воспоминания сохранены!")
    
    # Показываем сохраненные воспоминания
    assistant.show_memories()

if __name__ == "__main__":
    print("🧠 LangChain + Mem0 Integration Demo")
    print("=" * 60)
    
    # Проверка статуса API ключей
    print("📋 Статус компонентов:")
    print(f"   MEM0_API_KEY: {'✅ Установлен' if MEM0_API_KEY else '❌ Не найден'}")
    print(f"   OPENAI_API_KEY: {'✅ Установлен' if OPENAI_API_KEY else '❌ Не найден'}")
    print(f"   Модель OpenAI: {OPENAI_MODEL}")
    print(f"   Максимум токенов: {MAX_TOKENS}")
    
    if not OPENAI_API_KEY:
        print("\n⚠️  Для работы требуется OPENAI_API_KEY!")
        print("1. Скопируйте env.example в .env")
        print("2. Добавьте ваши API ключи в .env файл")
        exit(1)
    
    print("\n" + "=" * 60)
    
    # Выбор режима
    mode = input("Выберите режим:\n1. Интерактивный чат\n2. Демо разговор\nВаш выбор (1/2): ").strip()
    
    if mode == "2":
        demo_conversation()
    else:
        interactive_chat()
