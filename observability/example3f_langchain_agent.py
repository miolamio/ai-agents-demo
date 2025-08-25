#!/usr/bin/env python3
"""
Пример 3f: LangChain агент с инструментами + LangFuse мониторинг

Демонстрирует:
- LangChain агент с пользовательскими инструментами (tools)
- Инструмент для определения точного времени
- Интерактивный чат через терминал
- Полную интеграцию с LangFuse для мониторинга
- Трассировку tool calls и reasoning
"""

import os
import time
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

# Загрузка переменных окружения
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Переменные окружения загружены")
except ImportError:
    print("⚠️  python-dotenv не установлен")

# Проверяем доступность всех необходимых библиотек
try:
    from langfuse import observe, get_client
    from langfuse.langchain import CallbackHandler
    print("✅ LangFuse SDK доступен")
    LANGFUSE_AVAILABLE = True
except ImportError:
    print("❌ LangFuse SDK не установлен")
    print("Установите: pip install langfuse")
    LANGFUSE_AVAILABLE = False

try:
    from langchain_openai import ChatOpenAI
    from langchain.agents import create_openai_functions_agent, AgentExecutor
    from langchain.tools import tool
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
    print("✅ LangChain SDK доступен")
    LANGCHAIN_AVAILABLE = True
except ImportError:
    print("❌ LangChain не установлен")
    print("Установите: pip install langchain langchain-openai")
    LANGCHAIN_AVAILABLE = False

if not LANGCHAIN_AVAILABLE or not LANGFUSE_AVAILABLE:
    print("❌ Необходимые библиотеки недоступны")
    exit(1)


# Определяем инструменты для агента
@tool
def get_current_time(timezone_name: str = "UTC") -> str:
    """
    Определяет точное текущее время.
    
    Args:
        timezone_name: Название часового пояса (например: "UTC", "Europe/Moscow", "America/New_York")
        
    Returns:
        str: Текущее время в указанном часовом поясе
    """
    try:
        from zoneinfo import ZoneInfo
    except ImportError:
        # Fallback для Python < 3.9
        import pytz
        
        if timezone_name == "UTC":
            tz = pytz.UTC
        elif timezone_name == "Europe/Moscow":
            tz = pytz.timezone('Europe/Moscow')
        elif timezone_name == "America/New_York":
            tz = pytz.timezone('America/New_York')
        else:
            tz = pytz.UTC
            
        now = datetime.now(tz)
        return f"Текущее время в {timezone_name}: {now.strftime('%Y-%m-%d %H:%M:%S %Z')}"
    
    try:
        if timezone_name == "UTC":
            now = datetime.now(timezone.utc)
        else:
            # Поддерживаем основные часовые пояса
            timezone_map = {
                "Europe/Moscow": ZoneInfo("Europe/Moscow"),
                "America/New_York": ZoneInfo("America/New_York"),
                "Asia/Tokyo": ZoneInfo("Asia/Tokyo"),
                "Europe/London": ZoneInfo("Europe/London"),
            }
            tz = timezone_map.get(timezone_name, timezone.utc)
            now = datetime.now(tz)
        
        return f"Текущее время в {timezone_name}: {now.strftime('%Y-%m-%d %H:%M:%S %Z')}"
    except Exception as e:
        # Fallback на UTC
        now = datetime.now(timezone.utc)
        return f"Текущее время (UTC): {now.strftime('%Y-%m-%d %H:%M:%S %Z')} (не удалось получить {timezone_name}: {e})"


@tool
def get_date_info(date_offset: int = 0) -> str:
    """
    Получает информацию о дате.
    
    Args:
        date_offset: Смещение дней от текущей даты (0 = сегодня, 1 = завтра, -1 = вчера)
        
    Returns:
        str: Информация о дате включая день недели
    """
    from datetime import timedelta
    
    target_date = datetime.now() + timedelta(days=date_offset)
    
    # Названия дней недели на русском
    weekdays = [
        "понедельник", "вторник", "среда", "четверг", 
        "пятница", "суббота", "воскресенье"
    ]
    
    # Названия месяцев на русском
    months = [
        "января", "февраля", "марта", "апреля", "мая", "июня",
        "июля", "августа", "сентября", "октября", "ноября", "декабря"
    ]
    
    weekday = weekdays[target_date.weekday()]
    month = months[target_date.month - 1]
    
    if date_offset == 0:
        prefix = "Сегодня"
    elif date_offset == 1:
        prefix = "Завтра"
    elif date_offset == -1:
        prefix = "Вчера"
    else:
        prefix = f"Через {date_offset} дней" if date_offset > 0 else f"{abs(date_offset)} дней назад"
    
    return f"{prefix}: {target_date.day} {month} {target_date.year} года, {weekday}"


@tool
def calculate_time_difference(time1: str, time2: str) -> str:
    """
    Вычисляет разность между двумя временными метками.
    
    Args:
        time1: Первое время в формате "HH:MM" или "YYYY-MM-DD HH:MM"
        time2: Второе время в аналогичном формате
        
    Returns:
        str: Разность во времени
    """
    try:
        # Пытаемся распарсить разные форматы времени
        formats = ["%H:%M", "%Y-%m-%d %H:%M", "%H:%M:%S"]
        
        parsed_time1 = None
        parsed_time2 = None
        
        for fmt in formats:
            try:
                if len(time1.split()) == 1 and ":" in time1:
                    # Только время, добавляем сегодняшнюю дату
                    today = datetime.now().strftime("%Y-%m-%d")
                    parsed_time1 = datetime.strptime(f"{today} {time1}", "%Y-%m-%d %H:%M")
                else:
                    parsed_time1 = datetime.strptime(time1, fmt)
                break
            except ValueError:
                continue
                
        for fmt in formats:
            try:
                if len(time2.split()) == 1 and ":" in time2:
                    # Только время, добавляем сегодняшнюю дату
                    today = datetime.now().strftime("%Y-%m-%d")
                    parsed_time2 = datetime.strptime(f"{today} {time2}", "%Y-%m-%d %H:%M")
                else:
                    parsed_time2 = datetime.strptime(time2, fmt)
                break
            except ValueError:
                continue
        
        if not parsed_time1 or not parsed_time2:
            return "Ошибка: не удалось распарсить время. Используйте формат HH:MM или YYYY-MM-DD HH:MM"
        
        diff = abs(parsed_time2 - parsed_time1)
        hours, remainder = divmod(diff.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        
        return f"Разность времени: {int(hours)} часов {int(minutes)} минут {int(seconds)} секунд"
        
    except Exception as e:
        return f"Ошибка при вычислении разности времени: {e}"


class TimeAgentChat:
    """Умный агент с инструментами времени и LangFuse мониторингом"""
    
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.conversation_history = []
        
        # Получаем клиенты
        self.langfuse = get_client()
        self.langfuse_handler = CallbackHandler()
        
        # Проверяем конфигурацию
        self._check_configuration()
        
        # Инициализируем LLM
        self.llm = ChatOpenAI(
            model=model,
            temperature=0.7,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Определяем инструменты
        self.tools = [get_current_time, get_date_info, calculate_time_difference]
        
        # Создаем промпт для агента
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """Ты - полезный помощник, который может определять время и работать с датами.

У тебя есть следующие инструменты:
1. get_current_time - для получения точного текущего времени в любом часовом поясе
2. get_date_info - для получения информации о датах (сегодня, завтра, вчера и т.д.)
3. calculate_time_difference - для вычисления разности между временными метками

Используй эти инструменты когда пользователь спрашивает про время, дату или просит вычислить временные интервалы.
Отвечай на русском языке, будь дружелюбным и полезным.

Примеры запросов, которые требуют использования инструментов:
- "Который час?"
- "Какое сейчас время в Москве?"
- "Какое сегодня число?"
- "Какой завтра день недели?"
- "Сколько времени между 14:30 и 16:45?"

Если пользователь задает обычные вопросы не связанные со временем, отвечай обычно без использования инструментов."""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # Создаем агента
        self.agent = create_openai_functions_agent(
            llm=self.llm,
            tools=self.tools, 
            prompt=self.prompt
        )
        
        # Создаем executor
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            return_intermediate_steps=True,
            handle_parsing_errors=True
        )
        
        print(f"🤖 Агент готов (модель: {model})")
        print(f"🔧 Доступные инструменты: {[tool.name for tool in self.tools]}")
    
    def _check_configuration(self):
        """Проверяет конфигурацию API ключей"""
        
        print(f"\n🔧 Конфигурация:")
        
        # OpenAI
        openai_key = os.getenv("OPENAI_API_KEY")
        print(f"  OpenAI API Key: {'✅ установлен' if openai_key else '❌ не установлен'}")
        
        # LangFuse  
        langfuse_host = os.getenv("LANGFUSE_HOST", "не установлен")
        langfuse_key = os.getenv("LANGFUSE_PUBLIC_KEY", "не установлен")
        
        print(f"  LangFuse Host: {langfuse_host}")
        print(f"  LangFuse Key: {'✅ установлен' if langfuse_key != 'не установлен' else '❌ не установлен'}")
        
        if not openai_key:
            print("\n⚠️  ВНИМАНИЕ: OpenAI API ключ не настроен!")
            print("   Добавьте в .env: OPENAI_API_KEY=ваш-ключ")
            return False
            
        return True
    
    @observe(name="agent_chat_session")
    def start_chat(self, user_id: str = "terminal_user", session_id: Optional[str] = None) -> None:
        """
        Запускает интерактивный чат с агентом
        
        Args:
            user_id: ID пользователя
            session_id: ID сессии (если None, генерируется автоматически)
        """
        
        if session_id is None:
            session_id = f"langchain_session_{int(time.time())}"
        
        # Обновляем трассировку
        self.langfuse.update_current_trace(
            name="langchain_agent_chat",
            user_id=user_id,
            session_id=session_id,
            tags=["langchain", "agent", "tools", "interactive"],
            metadata={
                "agent_model": self.model,
                "available_tools": [tool.name for tool in self.tools],
                "started_at": datetime.now().isoformat()
            }
        )
        
        print(f"\n🎯 Чат с LangChain агентом начат!")
        print(f"   Сессия: {session_id}")
        print(f"   Агент умеет определять время и работать с датами")
        print(f"   Введите 'quit', 'exit' или 'стоп' для завершения")
        print("=" * 70)
        
        chat_history = []
        message_count = 0
        total_tool_calls = 0
        
        try:
            while True:
                # Получаем ввод пользователя
                user_input = input(f"\n👤 Вы: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'стоп', 'выход']:
                    print("👋 До свидания!")
                    break
                
                if not user_input:
                    print("⚠️  Пустое сообщение, попробуйте еще раз")
                    continue
                
                # Обрабатываем сообщение через агента
                response_data = self._process_agent_message(
                    user_input=user_input,
                    chat_history=chat_history,
                    user_id=user_id,
                    session_id=session_id
                )
                
                # Показываем ответ
                print(f"🤖 Агент: {response_data['response']}")
                
                # Показываем информацию об использованных инструментах
                if response_data.get('tools_used'):
                    print(f"   🔧 Использованы инструменты: {', '.join(response_data['tools_used'])}")
                
                # Краткая статистика
                print(f"   📊 Время: {response_data.get('response_time', 0.0):.2f}с | "
                      f"Шагов: {response_data.get('steps', 0)}")
                
                # Обновляем историю
                chat_history.extend([
                    HumanMessage(content=user_input),
                    AIMessage(content=response_data['response'])
                ])
                
                message_count += 1
                total_tool_calls += len(response_data.get('tools_used', []))
        
        except KeyboardInterrupt:
            print("\n\n👋 Чат прерван пользователем")
        
        # Обновляем финальную статистику
        self.langfuse.update_current_trace(
            output={
                "chat_summary": {
                    "total_messages": message_count,
                    "total_tool_calls": total_tool_calls,
                    "session_duration": f"{message_count * 30}+ секунд",  # Примерная оценка
                    "final_history_length": len(chat_history)
                }
            },
            tags=["completed", f"messages-{message_count}", f"tools-{total_tool_calls}"]
        )
        
        print(f"\n📊 Итоги чата:")
        print(f"   Сообщений обработано: {message_count}")
        print(f"   Вызовов инструментов: {total_tool_calls}")
        print(f"   История диалога: {len(chat_history)} записей")
    
    @observe(name="process_agent_message")
    def _process_agent_message(self, user_input: str, chat_history: List, user_id: str, session_id: str) -> Dict[str, Any]:
        """Обрабатывает сообщение через LangChain агента"""
        
        start_time = time.time()
        
        # Обновляем span
        self.langfuse.update_current_span(
            input={
                "user_message": user_input,
                "chat_history_length": len(chat_history),
                "user_id": user_id,
                "session_id": session_id
            },
            metadata={
                "message_length": len(user_input),
                "timestamp": datetime.now().isoformat()
            }
        )
        
        try:
            # Вызываем агента с LangFuse callback
            result = self.agent_executor.invoke(
                {
                    "input": user_input,
                    "chat_history": chat_history
                },
                config={
                    "callbacks": [self.langfuse_handler],
                    "metadata": {
                        "langfuse_user_id": user_id,
                        "langfuse_session_id": session_id,
                        "langfuse_tags": ["agent-execution", "tool-usage"]
                    }
                }
            )
            
            response_time = time.time() - start_time
            
            # Анализируем результат
            agent_response = result.get('output', 'Извините, не смог обработать запрос')
            intermediate_steps = result.get('intermediate_steps', [])
            
            # Определяем использованные инструменты
            tools_used = []
            for step in intermediate_steps:
                if len(step) >= 2:
                    action = step[0]
                    if hasattr(action, 'tool'):
                        tools_used.append(action.tool)
            
            # Убираем дубликаты
            tools_used = list(set(tools_used))
            
            response_data = {
                "response": agent_response,
                "response_time": response_time,
                "steps": len(intermediate_steps),
                "tools_used": tools_used,
                "success": True
            }
            
            # Обновляем span с результатом
            self.langfuse.update_current_span(
                output=response_data,
                metadata={
                    "response_length": len(agent_response),
                    "intermediate_steps_count": len(intermediate_steps),
                    "tools_used_count": len(tools_used)
                }
            )
            
            return response_data
            
        except Exception as e:
            response_time = time.time() - start_time
            
            error_response = {
                "response": f"Извините, произошла ошибка: {str(e)}",
                "response_time": response_time,
                "steps": 0,
                "tools_used": [],
                "success": False,
                "error": str(e)
            }
            
            self.langfuse.update_current_span(
                output=error_response,
                metadata={"error": True}
            )
            
            return error_response


@observe(name="demo_time_agent")
def demo_time_agent():
    """Демонстрация агента с автоматическими запросами"""
    
    print("\n🎯 Демонстрация LangChain агента с инструментами времени")
    print("=" * 70)
    
    agent = TimeAgentChat(model="gpt-4o-mini")
    
    langfuse = get_client()
    langfuse.update_current_trace(
        name="time_agent_demo",
        user_id="demo_user",
        session_id="demo_session",
        tags=["demo", "automated", "langchain", "tools"]
    )
    
    # Тестовые запросы
    demo_questions = [
        "Привет! Который сейчас час?",
        "Какое время в Москве?",
        "Какое сегодня число и день недели?",
        "Сколько времени между 14:30 и 16:45?",
        "Что ты умеешь делать?"
    ]
    
    chat_history = []
    
    for i, question in enumerate(demo_questions, 1):
        print(f"\n👤 Вопрос {i}: {question}")
        
        # Обрабатываем через агента
        response_data = agent._process_agent_message(
            user_input=question,
            chat_history=chat_history,
            user_id="demo_user",
            session_id="demo_session"
        )
        
        print(f"🤖 Агент: {response_data['response']}")
        
        if response_data.get('tools_used'):
            print(f"   🔧 Использованы инструменты: {', '.join(response_data['tools_used'])}")
        
        print(f"   📊 Время: {response_data['response_time']:.2f}с | "
              f"Шагов: {response_data.get('steps', 0)}")
        
        # Обновляем историю
        chat_history.extend([
            HumanMessage(content=question),
            AIMessage(content=response_data['response'])
        ])
        
        # Небольшая пауза
        time.sleep(1)
    
    print(f"\n📊 Демонстрация завершена!")
    print(f"   Вопросов обработано: {len(demo_questions)}")


def main():
    """Главная функция"""
    
    print("=== LangChain Agent + Tools + LangFuse Monitoring ===")
    
    if not LANGCHAIN_AVAILABLE or not LANGFUSE_AVAILABLE:
        print("❌ Необходимые библиотеки недоступны")
        return
    
    # Проверяем конфигурацию
    openai_key = os.getenv("OPENAI_API_KEY")
    langfuse_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    
    if not openai_key:
        print("⚠️  OpenAI API ключ не настроен")
        print("   Добавьте в .env: OPENAI_API_KEY=ваш-ключ")
        return
    
    if not langfuse_key:
        print("⚠️  LangFuse ключи не настроены")
        print("   Трассировки будут созданы локально")
    
    try:
        # Выбор режима
        print(f"\n🎯 Выберите режим:")
        print("   1. Интерактивный чат с агентом")
        print("   2. Автоматическая демонстрация")
        print("   3. Оба режима")
        
        choice = input("\nВведите номер (1-3) или Enter для интерактивного чата: ").strip()
        
        if choice in ['1', '']:
            # Интерактивный чат
            agent = TimeAgentChat(model="gpt-4o-mini")
            agent.start_chat(user_id="interactive_user")
            
        elif choice == '2':
            # Только демонстрация
            demo_time_agent()
            
        elif choice == '3':
            # Оба режима
            demo_time_agent()
            
            print(f"\n" + "="*70)
            input("Нажмите Enter для перехода к интерактивному чату...")
            
            agent = TimeAgentChat(model="gpt-4o-mini")
            agent.start_chat(user_id="interactive_user")
        
        else:
            print("⚠️  Неверный выбор, запускаю интерактивный чат")
            agent = TimeAgentChat(model="gpt-4o-mini")
            agent.start_chat(user_id="interactive_user")
        
        # Отправляем все данные в LangFuse
        langfuse = get_client()
        langfuse.flush()
        print(f"\n📤 Все трассировки отправлены в LangFuse")
        
        # Информация о просмотре результатов
        langfuse_host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        print(f"\n🌐 Проверьте результаты в LangFuse:")
        print(f"   {langfuse_host}")
        print(f"\n✅ Демонстрация завершена!")
        
    except KeyboardInterrupt:
        print(f"\n👋 Программа завершена пользователем")
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
