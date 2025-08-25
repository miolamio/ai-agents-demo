#!/usr/bin/env python3
"""
Пример 3e: Чат с OpenAI + LangFuse мониторинг

Демонстрирует:
- Реальную интеграцию OpenAI API с LangFuse v3
- Автоматическую трассировку LLM вызовов
- Мониторинг токенов, стоимости и производительности
- Интерактивный чат с сохранением истории
"""

import os
import time
from datetime import datetime
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
    from langfuse.openai import openai  # Специальная интеграция LangFuse с OpenAI
    print("✅ LangFuse SDK доступен")
    LANGFUSE_AVAILABLE = True
except ImportError:
    print("❌ LangFuse SDK не установлен")
    print("Установите: pip install langfuse")
    LANGFUSE_AVAILABLE = False
    exit(1)

try:
    # Проверяем что OpenAI доступен через LangFuse интеграцию
    openai.OpenAI()
    print("✅ OpenAI интеграция доступна")
    OPENAI_AVAILABLE = True
except Exception as e:
    print(f"❌ OpenAI не доступен: {e}")
    print("Установите: pip install openai")
    OPENAI_AVAILABLE = False


class IntelligentChatBot:
    """Умный чат-бот с полным мониторингом через LangFuse"""
    
    def __init__(self, name: str = "ChatBot", model: str = "gpt-4o-mini"):
        self.name = name
        self.model = model
        self.conversation_history = []
        
        # Получаем клиенты
        self.langfuse = get_client()
        
        # Настройка OpenAI через LangFuse интеграцию
        self.openai_client = openai.OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        print(f"🤖 Чат-бот '{name}' готов (модель: {model})")
        
        # Проверяем конфигурацию
        self._check_configuration()
    
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
    
    @observe(name="chat_conversation")
    def start_conversation(self, user_id: str = "demo_user", session_id: Optional[str] = None) -> None:
        """
        Запускает интерактивную беседу с пользователем
        
        Args:
            user_id: ID пользователя
            session_id: ID сессии (если None, генерируется автоматически)
        """
        
        if session_id is None:
            session_id = f"session_{int(time.time())}"
        
        # Обновляем трассировку беседы
        self.langfuse.update_current_trace(
            name="interactive_chat_session",
            user_id=user_id,
            session_id=session_id,
            tags=["interactive", "chat", "openai", self.model],
            metadata={
                "bot_name": self.name,
                "model": self.model,
                "started_at": datetime.now().isoformat()
            }
        )
        
        print(f"\n💬 Начинаем беседу! (Сессия: {session_id})")
        print(f"   Напишите 'quit', 'exit' или 'стоп' для завершения")
        print("=" * 60)
        
        total_messages = 0
        total_tokens = 0
        total_cost = 0.0
        
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
                
                # Обрабатываем сообщение
                response_data = self._process_message(user_input, user_id, session_id)
                
                # Показываем ответ
                print(f"🤖 {self.name}: {response_data['response']}")
                
                # Обновляем статистику
                total_messages += 1
                total_tokens += response_data.get('total_tokens', 0)
                total_cost += response_data.get('cost_estimate', 0.0)
                
                # Показываем краткую статистику
                print(f"   📊 Токены: {response_data.get('total_tokens', 0)} | "
                      f"Стоимость: ${response_data.get('cost_estimate', 0.0):.4f} | "
                      f"Время: {response_data.get('response_time', 0.0):.2f}с")
        
        except KeyboardInterrupt:
            print("\n\n👋 Беседа прервана пользователем")
        
        # Обновляем финальную статистику беседы
        self.langfuse.update_current_trace(
            output={
                "conversation_summary": {
                    "total_messages": total_messages,
                    "total_tokens": total_tokens,
                    "total_cost": total_cost,
                    "average_tokens_per_message": total_tokens / max(total_messages, 1),
                    "conversation_length": len(self.conversation_history)
                }
            },
            tags=["completed", f"messages-{total_messages}", f"tokens-{total_tokens}"]
        )
        
        print(f"\n📊 Итоги беседы:")
        print(f"   Сообщений: {total_messages}")
        print(f"   Всего токенов: {total_tokens}")
        print(f"   Общая стоимость: ${total_cost:.4f}")
        print(f"   История сохранена: {len(self.conversation_history)} сообщений")
    
    @observe(name="process_user_message")
    def _process_message(self, user_input: str, user_id: str, session_id: str) -> Dict[str, Any]:
        """Обрабатывает одно сообщение пользователя"""
        
        start_time = time.time()
        
        # Обновляем span для обработки сообщения
        self.langfuse.update_current_span(
            input={
                "user_message": user_input,
                "user_id": user_id,
                "session_id": session_id,
                "conversation_length": len(self.conversation_history)
            },
            metadata={
                "message_length": len(user_input),
                "timestamp": datetime.now().isoformat()
            }
        )
        
        # Добавляем сообщение пользователя в историю
        self.conversation_history.append({
            "role": "user", 
            "content": user_input
        })
        
        # Получаем ответ от OpenAI
        llm_response = self._call_openai_api(self.conversation_history)
        
        # Добавляем ответ бота в историю
        assistant_message = llm_response["response"]
        self.conversation_history.append({
            "role": "assistant",
            "content": assistant_message
        })
        
        response_time = time.time() - start_time
        
        # Результат обработки
        result = {
            "response": assistant_message,
            "response_time": response_time,
            "input_tokens": llm_response.get("input_tokens", 0),
            "output_tokens": llm_response.get("output_tokens", 0),
            "total_tokens": llm_response.get("total_tokens", 0),
            "cost_estimate": llm_response.get("cost_estimate", 0.0),
            "model_used": llm_response.get("model", self.model)
        }
        
        # Обновляем span с результатом
        self.langfuse.update_current_span(
            output=result,
            metadata={
                "response_length": len(assistant_message),
                "conversation_turn": len(self.conversation_history) // 2
            }
        )
        
        return result
    
    @observe(name="openai_api_call")
    def _call_openai_api(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Вызывает OpenAI API с полным мониторингом"""
        
        if not OPENAI_AVAILABLE:
            # Заглушка для демонстрации
            return {
                "response": "Извините, OpenAI API недоступен. Это демо-ответ.",
                "input_tokens": 20,
                "output_tokens": 15,
                "total_tokens": 35,
                "cost_estimate": 0.0001,
                "model": "demo-model"
            }
        
        # Обновляем span для LLM вызова
        self.langfuse.update_current_span(
            input={
                "messages": messages,
                "model": self.model,
                "conversation_length": len(messages)
            },
            metadata={
                "api_provider": "openai",
                "model_family": "gpt"
            }
        )
        
        try:
            # Вызов OpenAI API через LangFuse интеграцию
            # Это автоматически создаст generation в LangFuse
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=1000,
                # LangFuse автоматически перехватит этот вызов
                name="chat-response",  # Имя для LangFuse generation
            )
            
            # Извлекаем данные из ответа
            assistant_message = response.choices[0].message.content
            usage = response.usage
            
            # Примерный расчет стоимости (актуальные цены могут отличаться)
            cost_per_input_token = 0.00015 / 1000   # gpt-4o-mini input
            cost_per_output_token = 0.0006 / 1000   # gpt-4o-mini output
            
            cost_estimate = (
                usage.prompt_tokens * cost_per_input_token +
                usage.completion_tokens * cost_per_output_token
            )
            
            result = {
                "response": assistant_message,
                "input_tokens": usage.prompt_tokens,
                "output_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
                "cost_estimate": cost_estimate,
                "model": self.model
            }
            
            # Обновляем span с результатом
            self.langfuse.update_current_span(
                output=result,
                metadata={
                    "finish_reason": response.choices[0].finish_reason,
                    "response_id": response.id
                }
            )
            
            return result
            
        except Exception as e:
            # Обрабатываем ошибки API
            error_result = {
                "response": f"Извините, произошла ошибка: {str(e)}",
                "input_tokens": 0,
                "output_tokens": 0, 
                "total_tokens": 0,
                "cost_estimate": 0.0,
                "model": self.model,
                "error": str(e)
            }
            
            self.langfuse.update_current_span(
                output=error_result,
                metadata={"error": True}
            )
            
            return error_result
    
    @observe(name="get_conversation_summary")
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Получает сводку по текущей беседе"""
        
        if not self.conversation_history:
            return {"message": "Беседа еще не начата"}
        
        user_messages = [msg for msg in self.conversation_history if msg["role"] == "user"]
        assistant_messages = [msg for msg in self.conversation_history if msg["role"] == "assistant"]
        
        summary = {
            "total_exchanges": len(user_messages),
            "total_messages": len(self.conversation_history),
            "average_user_message_length": sum(len(msg["content"]) for msg in user_messages) / len(user_messages),
            "average_assistant_message_length": sum(len(msg["content"]) for msg in assistant_messages) / len(assistant_messages),
            "conversation_start": datetime.now().isoformat() if self.conversation_history else None,
            "last_messages": self.conversation_history[-4:] if len(self.conversation_history) >= 4 else self.conversation_history
        }
        
        self.langfuse.update_current_span(
            input={"conversation_length": len(self.conversation_history)},
            output=summary
        )
        
        return summary


@observe(name="demo_automated_conversation")
def demo_automated_conversation():
    """Демонстрация автоматизированной беседы"""
    
    print("\n🎯 Демонстрация автоматизированной беседы")
    print("=" * 60)
    
    bot = IntelligentChatBot("DemoBot", model="gpt-4o-mini")
    
    # Предустановленные сообщения для демо
    demo_messages = [
        "Привет! Расскажи о себе",
        "Что ты знаешь о машинном обучении?",
        "Можешь объяснить, что такое LLM?",
        "Спасибо за интересную беседу!"
    ]
    
    langfuse = get_client()
    langfuse.update_current_trace(
        name="automated_demo_conversation",
        user_id="demo_user_auto",
        session_id="auto_demo_session",
        tags=["demo", "automated", "showcase"]
    )
    
    total_tokens = 0
    total_cost = 0.0
    
    for i, message in enumerate(demo_messages, 1):
        print(f"\n👤 Пользователь: {message}")
        
        # Обрабатываем сообщение
        response_data = bot._process_message(
            user_input=message,
            user_id="demo_user_auto", 
            session_id="auto_demo_session"
        )
        
        print(f"🤖 {bot.name}: {response_data['response']}")
        print(f"   📊 Токены: {response_data['total_tokens']} | "
              f"Стоимость: ${response_data['cost_estimate']:.4f} | "
              f"Время: {response_data['response_time']:.2f}с")
        
        total_tokens += response_data['total_tokens']
        total_cost += response_data['cost_estimate']
        
        # Небольшая пауза между сообщениями
        time.sleep(1)
    
    # Получаем сводку
    summary = bot.get_conversation_summary()
    
    print(f"\n📊 Итоги автоматической демонстрации:")
    print(f"   Обменов сообщениями: {summary['total_exchanges']}")
    print(f"   Всего токенов: {total_tokens}")
    print(f"   Общая стоимость: ${total_cost:.4f}")
    print(f"   Средняя длина ответа: {summary['average_assistant_message_length']:.0f} символов")


def main():
    """Главная функция"""
    
    print("=== OpenAI Chat + LangFuse Monitoring ===")
    
    if not LANGFUSE_AVAILABLE:
        print("❌ LangFuse SDK недоступен")
        return
    
    langfuse = get_client()
    
    print(f"\n🔧 Проверка конфигурации...")
    openai_key = os.getenv("OPENAI_API_KEY")
    langfuse_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    
    if not openai_key:
        print("⚠️  OpenAI API ключ не настроен")
        print("   Демонстрация будет работать с заглушками")
    
    if not langfuse_key:
        print("⚠️  LangFuse ключи не настроены")
        print("   Трассировки будут созданы локально")
    
    try:
        # Выбор режима
        print(f"\n🎯 Выберите режим:")
        print("   1. Интерактивный чат")
        print("   2. Автоматическая демонстрация")
        print("   3. Оба режима")
        
        choice = input("\nВведите номер (1-3) или Enter для демо: ").strip()
        
        if choice in ['1', '']:
            # Интерактивный чат
            bot = IntelligentChatBot("ChatGPT-Assistant", model="gpt-4o-mini")
            bot.start_conversation(user_id="interactive_user")
            
        elif choice == '2':
            # Только демонстрация
            demo_automated_conversation()
            
        elif choice == '3':
            # Оба режима
            demo_automated_conversation()
            
            print(f"\n" + "="*60)
            input("Нажмите Enter для перехода к интерактивному чату...")
            
            bot = IntelligentChatBot("ChatGPT-Assistant", model="gpt-4o-mini")
            bot.start_conversation(user_id="interactive_user")
        
        else:
            print("⚠️  Неверный выбор, запускаю демонстрацию")
            demo_automated_conversation()
        
        # Отправляем все данные в LangFuse
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
