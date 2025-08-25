#!/usr/bin/env python3
"""
Пример 1: Базовая трассировка с LangSmith

Демонстрирует:
- Инициализацию LangSmith клиента
- Создание трассировки для простого агента
- Логирование входных и выходных данных
- Настройку метаданных для отладки
"""

import os
from datetime import datetime
from langsmith import Client
from langsmith.run_trees import RunTree

# Загрузка переменных окружения из .env файла
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Переменные окружения загружены из .env файла")
except ImportError:
    print("⚠️  python-dotenv не установлен. Используются системные переменные окружения.")
    print("Для автоматической загрузки .env установите: pip install python-dotenv")


class SimpleAgent:
    """Простой агент для демонстрации трассировки"""
    
    def __init__(self, name="SimpleAgent"):
        # Инициализация LangSmith клиента
        self.ls_client = Client(
            api_url=os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"),
            api_key=os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY", "demo-key")
        )
        self.name = name
        
    def process_query(self, query, user_id=None):
        """
        Обрабатывает пользовательский запрос с трассировкой
        
        Args:
            query (str): Пользовательский запрос
            user_id (str, optional): ID пользователя для группировки
            
        Returns:
            dict: Результат обработки запроса
        """
        
        # Создаем основной run с RunTree
        parent_run = RunTree(
            name=f"{self.name}_process_query",
            run_type="chain",
            inputs={
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "agent": self.name
            },
            tags=["demo", "simple-agent", "production"],
            metadata={
                "user_id": user_id,
                "version": "1.0",
                "environment": "demo"
            }
        )
        parent_run.post()
        
        try:
            # Эмулируем обработку запроса агентом
            steps = []
            
            # Шаг 1: Анализ запроса
            analysis_result = self._analyze_query(query, parent_run)
            steps.append(analysis_result)
            
            # Шаг 2: Генерация ответа
            response_result = self._generate_response(analysis_result, parent_run)
            steps.append(response_result)
            
            # Финальный результат
            result = {
                "query": query,
                "response": response_result["response"],
                "steps": steps,
                "total_tokens": sum(step.get("tokens", 0) for step in steps),
                "success": True
            }
            
            # Завершаем основной run
            parent_run.end(outputs=result)
            parent_run.patch()
            
            return result
            
        except Exception as e:
            # В случае ошибки логируем её в LangSmith
            parent_run.end(error=f"Ошибка обработки запроса: {str(e)}")
            parent_run.patch()
            raise
    
    def _analyze_query(self, query, parent_run):
        """Анализ пользовательского запроса"""
        
        # Создаем дочерний run для шага анализа
        analysis_run = parent_run.create_child(
            name="analyze_query",
            run_type="tool",
            inputs={"query": query},
        )
        analysis_run.post()
        
        # Эмулируем анализ
        result = {
            "intent": "question" if "?" in query else "statement",
            "keywords": query.split(),
            "complexity": "simple" if len(query.split()) < 10 else "complex",
            "tokens": len(query.split())
        }
        
        analysis_run.end(outputs=result)
        analysis_run.patch()
        
        return result
    
    def _generate_response(self, analysis, parent_run):
        """Генерация ответа на основе анализа"""
        
        # Создаем дочерний run для генерации
        generation_run = parent_run.create_child(
            name="generate_response",
            run_type="llm",
            inputs={
                "analysis": analysis,
                "model": "demo-model-v1"
            }
        )
        generation_run.post()
        
        # Эмулируем генерацию ответа
        if analysis["intent"] == "question":
            response = f"Отвечаю на ваш вопрос о {', '.join(analysis['keywords'])}"
        else:
            response = f"Понял ваше утверждение о {', '.join(analysis['keywords'])}"
        
        result = {
            "response": response,
            "model_used": "demo-model-v1",
            "tokens": len(response.split()) + 10,  # +10 для служебных токенов
            "confidence": 0.95
        }
        
        generation_run.end(outputs=result)
        generation_run.patch()
        
        return result


def main():
    """Демонстрация использования агента с трассировкой"""
    
    print("=== Пример 1: Базовая трассировка с LangSmith ===\n")
    
    # Создаем агента
    agent = SimpleAgent("DemoAgent")
    
    # Тестовые запросы
    test_queries = [
        "Как работает машинное обучение?",
        "Python - отличный язык программирования",
        "Что такое нейронные сети и как они устроены?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"Запрос {i}: {query}")
        
        try:
            result = agent.process_query(query, user_id=f"user_{i}")
            print(f"Ответ: {result['response']}")
            print(f"Токенов использовано: {result['total_tokens']}")
            print(f"Шагов выполнено: {len(result['steps'])}")
            print("-" * 50)
            
        except Exception as e:
            print(f"Ошибка при обработке запроса: {e}")
    
    print("\nТрассировка завершена. Данные отправлены в LangSmith.")
    print("Для просмотра трассировок откройте https://smith.langchain.com")


def check_environment_variables():
    """Проверяет и отображает статус переменных окружения"""
    
    print("\n🔍 Проверка переменных окружения:")
    print("-" * 50)
    
    # Основные переменные для LangSmith
    env_vars = {
        "LANGSMITH_API_KEY": os.getenv("LANGSMITH_API_KEY"),
        "LANGSMITH_TRACING": os.getenv("LANGSMITH_TRACING"),
        "LANGSMITH_PROJECT": os.getenv("LANGSMITH_PROJECT"),
        "LANGSMITH_ENDPOINT": os.getenv("LANGSMITH_ENDPOINT"),
        # Legacy переменные
        "LANGCHAIN_API_KEY": os.getenv("LANGCHAIN_API_KEY"),
        "LANGCHAIN_TRACING_V2": os.getenv("LANGCHAIN_TRACING_V2"),
        "LANGCHAIN_PROJECT": os.getenv("LANGCHAIN_PROJECT")
    }
    
    for var, value in env_vars.items():
        if value:
            # Маскируем API ключи для безопасности
            if "API_KEY" in var and len(value) > 10:
                display_value = value[:8] + "..." + value[-4:]
            else:
                display_value = value
            print(f"✅ {var}: {display_value}")
        else:
            print(f"❌ {var}: не установлена")
    
    # Определяем активный API ключ
    api_key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
    
    if not api_key:
        print("\n⚠️  API ключ LangSmith не найден!")
        print("\n📋 Для настройки создайте файл .env:")
        print("cp env.example .env")
        print("\n🔧 Затем отредактируйте .env и добавьте:")
        print("LANGSMITH_API_KEY=your-api-key-here")
        print("LANGSMITH_TRACING=true")
        print("LANGSMITH_PROJECT=your-project-name")
        print("\n🔗 Получить API ключ: https://smith.langchain.com")
        print("\nДля демонстрации будет использован тестовый ключ.\n")
        return False
    else:
        project = os.getenv("LANGSMITH_PROJECT") or os.getenv("LANGCHAIN_PROJECT", "default")
        endpoint = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
        print(f"\n✅ Конфигурация готова!")
        print(f"📊 Проект: {project}")
        print(f"🔗 Endpoint: {endpoint}")
        return True


if __name__ == "__main__":
    # Проверяем переменные окружения
    env_ready = check_environment_variables()
    
    if env_ready:
        print("\n🚀 Запуск с реальным API...")
    else:
        print("🎭 Запуск в демо-режиме...")
    
    main()