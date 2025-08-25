#!/usr/bin/env python3
"""
Пример 3d: Финальная интеграция с LangFuse v3

Демонстрирует:
- Правильное использование LangFuse v3 SDK
- Декоратор @observe для автоматической трассировки
- Реальную интеграцию с настоящим LangFuse сервером
- Иерархические трассировки и spans
"""

import os
import time
from datetime import datetime
from typing import List, Dict, Any

# Загрузка переменных окружения
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Переменные окружения загружены")
except ImportError:
    print("⚠️  python-dotenv не установлен")

# Проверим доступность LangFuse SDK
try:
    from langfuse import observe, get_client
    print("✅ LangFuse v3 SDK доступен")
    LANGFUSE_AVAILABLE = True
except ImportError:
    print("❌ LangFuse SDK не установлен")
    print("Установите: pip install langfuse")
    LANGFUSE_AVAILABLE = False
    exit(1)


class SmartAgent:
    """Умный агент с полной интеграцией LangFuse v3"""
    
    def __init__(self, name: str = "SmartAgent"):
        self.name = name
        
        # Получаем глобальный клиент LangFuse
        self.langfuse = get_client()
        
        print(f"🤖 Агент '{name}' инициализирован")
        print(f"🔗 LangFuse клиент готов")
    
    @observe(name="process_user_request")
    def process_user_request(self, user_query: str, user_id: str, session_id: str) -> Dict[str, Any]:
        """
        Обрабатывает запрос пользователя с полной трассировкой
        
        Args:
            user_query: Запрос пользователя
            user_id: ID пользователя  
            session_id: ID сессии
            
        Returns:
            dict: Результат обработки
        """
        
        # Обновляем текущую трассировку с информацией о пользователе
        self.langfuse.update_current_trace(
            user_id=user_id,
            session_id=session_id,
            tags=["user-request", "production"],
            metadata={
                "agent_name": self.name,
                "query_length": len(user_query),
                "timestamp": datetime.now().isoformat()
            }
        )
        
        print(f"📝 Обрабатываю запрос: '{user_query[:50]}...'")
        
        # Шаг 1: Анализ запроса
        analysis = self._analyze_query(user_query)
        
        # Шаг 2: Генерация ответа
        response = self._generate_response(analysis)
        
        # Шаг 3: Постобработка
        final_result = self._postprocess_result(response)
        
        # Обновляем трассировку с финальным результатом
        self.langfuse.update_current_trace(
            output={
                "response": final_result["response"],
                "confidence": final_result["confidence"],
                "processing_time": final_result["processing_time"]
            },
            tags=["completed", f"confidence-{final_result['confidence']}"]
        )
        
        return final_result
    
    @observe(name="analyze_query")
    def _analyze_query(self, query: str) -> Dict[str, Any]:
        """Анализирует пользовательский запрос"""
        
        print(f"  🔍 Анализирую запрос...")
        
        # Обновляем текущий span
        self.langfuse.update_current_span(
            input={"query": query, "length": len(query)},
            metadata={"step": "analysis", "model": "analysis-v1"}
        )
        
        # Симулируем анализ
        time.sleep(0.1)
        
        # Простой анализ намерений
        intent = "question" if "?" in query else "statement"
        confidence = 0.9 if "?" in query else 0.7
        entities = query.split()[:3]  # Первые 3 слова как сущности
        
        result = {
            "intent": intent,
            "confidence": confidence,
            "entities": entities,
            "language": "ru" if any(ord(c) > 127 for c in query) else "en",
            "complexity": "simple" if len(query.split()) < 5 else "complex"
        }
        
        # Обновляем span с результатом
        self.langfuse.update_current_span(
            output=result,
            metadata={"entities_count": len(entities)}
        )
        
        print(f"    ✅ Анализ завершен: {intent} ({confidence})")
        return result
    
    @observe(name="generate_response")
    def _generate_response(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Генерирует ответ на основе анализа"""
        
        print(f"  🎯 Генерирую ответ...")
        
        # Обновляем span
        self.langfuse.update_current_span(
            input=analysis,
            metadata={"step": "generation", "model": "response-generator-v2"}
        )
        
        # Симулируем генерацию ответа
        time.sleep(0.2)
        
        intent = analysis["intent"]
        entities = analysis["entities"]
        
        # Генерируем ответ в зависимости от намерения
        if intent == "question":
            response = f"Отвечаю на ваш вопрос о {', '.join(entities[:2])}"
        else:
            response = f"Понял ваше сообщение о {', '.join(entities[:2])}"
        
        # Добавляем дополнительную информацию для сложных запросов
        if analysis["complexity"] == "complex":
            response += ". Это интересный и сложный вопрос, требующий детального рассмотрения."
        
        result = {
            "response": response,
            "model_used": "response-generator-v2",
            "temperature": 0.7,
            "tokens_used": len(response.split()) + len(' '.join(entities)),
            "generation_time": 0.2
        }
        
        # Обновляем span
        self.langfuse.update_current_span(
            output=result,
            metadata={
                "response_length": len(response),
                "tokens": result["tokens_used"]
            }
        )
        
        print(f"    ✅ Ответ сгенерирован ({result['tokens_used']} токенов)")
        return result
    
    @observe(name="postprocess_result")
    def _postprocess_result(self, generation: Dict[str, Any]) -> Dict[str, Any]:
        """Постобработка результата"""
        
        print(f"  🔧 Постобработка...")
        
        # Обновляем span
        self.langfuse.update_current_span(
            input=generation,
            metadata={"step": "postprocessing"}
        )
        
        # Симулируем постобработку
        time.sleep(0.05)
        
        # Расчет уверенности и финальная обработка
        confidence_score = 0.95 if generation["tokens_used"] > 10 else 0.8
        processing_time = generation.get("generation_time", 0) + 0.05
        
        result = {
            "response": generation["response"].strip(),
            "confidence": confidence_score,
            "processing_time": processing_time,
            "tokens_used": generation["tokens_used"],
            "safety_check": "passed",
            "content_filter": "clean",
            "final_review": "approved"
        }
        
        # Обновляем span
        self.langfuse.update_current_span(
            output=result,
            metadata={"confidence_score": confidence_score}
        )
        
        print(f"    ✅ Постобработка завершена (уверенность: {confidence_score})")
        return result


@observe(name="conversation_simulation")
def simulate_conversation(agent: SmartAgent, messages: List[str], user_id: str, session_id: str) -> List[Dict[str, Any]]:
    """Симулирует диалог с пользователем"""
    
    langfuse = get_client()
    
    # Обновляем трассировку диалога
    langfuse.update_current_trace(
        name="multi-turn-conversation",
        user_id=user_id,
        session_id=session_id,
        tags=["conversation", "simulation", "multi-turn"],
        metadata={
            "total_messages": len(messages),
            "conversation_type": "simulation"
        }
    )
    
    results = []
    total_tokens = 0
    total_time = 0
    
    for i, message in enumerate(messages, 1):
        print(f"\n💬 Сообщение {i}/{len(messages)}: {message}")
        
        # Обрабатываем каждое сообщение
        result = agent.process_user_request(
            user_query=message,
            user_id=user_id,
            session_id=session_id
        )
        
        results.append(result)
        total_tokens += result["tokens_used"]
        total_time += result["processing_time"]
        
        print(f"🤖 Ответ: {result['response']}")
        print(f"📊 Токены: {result['tokens_used']}, Время: {result['processing_time']:.2f}с")
    
    # Обновляем финальную статистику диалога
    langfuse.update_current_trace(
        output={
            "messages_processed": len(messages),
            "total_tokens": total_tokens,
            "total_time": total_time,
            "average_confidence": sum(r["confidence"] for r in results) / len(results),
            "conversation_summary": f"Обработано {len(messages)} сообщений"
        },
        tags=["completed", f"messages-{len(messages)}", f"tokens-{total_tokens}"]
    )
    
    return results


def demo_basic_functionality():
    """Демонстрация базовой функциональности"""
    
    print("\n🎯 Демонстрация 1: Базовая функциональность")
    print("=" * 60)
    
    agent = SmartAgent("DemoAgent")
    
    # Простой запрос
    result = agent.process_user_request(
        user_query="Как работает машинное обучение?",
        user_id="demo_user_1",
        session_id="demo_session_1"
    )
    
    print(f"\n📋 Результат:")
    print(f"  Ответ: {result['response']}")
    print(f"  Уверенность: {result['confidence']}")
    print(f"  Время обработки: {result['processing_time']:.2f}с")
    print(f"  Токены: {result['tokens_used']}")


def demo_conversation():
    """Демонстрация диалога"""
    
    print("\n🎯 Демонстрация 2: Многооборотный диалог")
    print("=" * 60)
    
    agent = SmartAgent("ConversationAgent")
    
    # Диалог
    messages = [
        "Привет! Расскажи о LangFuse",
        "Как он помогает в мониторинге LLM?", 
        "Спасибо за объяснение!"
    ]
    
    results = simulate_conversation(
        agent=agent,
        messages=messages,
        user_id="demo_user_2", 
        session_id="demo_session_2"
    )
    
    print(f"\n📊 Статистика диалога:")
    print(f"  Сообщений обработано: {len(results)}")
    print(f"  Общее время: {sum(r['processing_time'] for r in results):.2f}с")
    print(f"  Общие токены: {sum(r['tokens_used'] for r in results)}")
    print(f"  Средняя уверенность: {sum(r['confidence'] for r in results) / len(results):.2f}")


def main():
    """Главная функция демонстрации"""
    
    print("=== LangFuse v3 SDK - Финальная демонстрация ===")
    
    if not LANGFUSE_AVAILABLE:
        print("❌ LangFuse SDK недоступен")
        return
    
    # Проверяем конфигурацию
    langfuse = get_client()
    
    print(f"\n🔧 Конфигурация:")
    host = os.getenv("LANGFUSE_HOST", "не установлен")
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "не установлен")
    
    print(f"  Host: {host}")
    print(f"  Public Key: {'✅ установлен' if public_key != 'не установлен' else '❌ не установлен'}")
    
    if public_key == "не установлен":
        print("\n⚠️  API ключи не настроены, но демонстрация продолжится")
        print("   (трассировки будут созданы, но не отправлены)")
    
    # Запускаем демонстрации
    try:
        demo_basic_functionality()
        demo_conversation()
        
        # Принудительно отправляем все данные
        langfuse.flush()
        print(f"\n📤 Все трассировки отправлены в LangFuse")
        
        print(f"\n🌐 Проверьте результаты в веб-интерфейсе:")
        if host != "не установлен":
            print(f"   {host}")
        else:
            print(f"   https://cloud.langfuse.com (если используется облачная версия)")
        
        print(f"\n✅ Демонстрация завершена успешно!")
        
    except Exception as e:
        print(f"❌ Ошибка в демонстрации: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
