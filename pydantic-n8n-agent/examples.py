"""
examples.py - Примеры использования Web Research Agent

Этот файл содержит различные примеры использования агента
для быстрого старта и понимания возможностей.
"""

import asyncio
import json
from typing import List
from web_research_agent import WebResearchAssistant, ResearchOutput
import os
from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv()

# ============================================================================
# Примеры использования
# ============================================================================

def example_basic_search():
    """Пример базового поиска"""
    print("=" * 60)
    print("ПРИМЕР 1: Базовый поиск")
    print("=" * 60)
    
    # Создаем ассистента
    assistant = WebResearchAssistant(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        tavily_api_key=os.getenv("TAVILY_API_KEY")
    )
    
    # Выполняем поиск
    result = assistant.process_message_sync(
        "What are the main benefits of meditation for mental health?"
    )
    
    # Выводим результаты
    print(f"\n📝 Резюме: {result.summary}")
    print(f"\n🎯 Основная тема: {result.categories.main_topic}")
    print(f"📊 Уровень уверенности: {result.confidence_level}")
    print(f"💭 Тональность: {result.sentiment}")
    
    print("\n🔍 Ключевые находки:")
    for i, finding in enumerate(result.key_findings[:3], 1):
        print(f"\n{i}. {finding.title}")
        print(f"   Релевантность: {'⭐' * finding.relevance_score}")
        print(f"   {finding.description[:150]}...")
        print(f"   Источник: {finding.source}")
    
    return result


def example_conversation_with_memory():
    """Пример диалога с сохранением контекста"""
    print("\n" + "=" * 60)
    print("ПРИМЕР 2: Диалог с памятью")
    print("=" * 60)
    
    assistant = WebResearchAssistant(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        tavily_api_key=os.getenv("TAVILY_API_KEY")
    )
    
    # Первый запрос
    print("\n👤 Запрос 1: Что такое квантовые компьютеры?")
    result1 = assistant.process_message_sync(
        "Что такое квантовые компьютеры и как они работают?"
    )
    print(f"🤖 Ответ: {result1.summary}")
    
    # Второй запрос с учетом контекста
    print("\n👤 Запрос 2: Какие компании лидируют в этой области?")
    result2 = assistant.process_message_sync(
        "Какие компании лидируют в этой области?"
    )
    print(f"🤖 Ответ: {result2.summary}")
    
    # Третий запрос
    print("\n👤 Запрос 3: Какие практические применения уже существуют?")
    result3 = assistant.process_message_sync(
        "Какие практические применения уже существуют?"
    )
    print(f"🤖 Ответ: {result3.summary}")
    
    # Показываем извлеченные организации
    all_orgs = set()
    for result in [result1, result2, result3]:
        all_orgs.update(result.entities.organizations)
    
    print("\n🏢 Упомянутые организации:")
    for org in sorted(all_orgs):
        print(f"  • {org}")


async def example_async_multiple_searches():
    """Пример асинхронного выполнения нескольких поисков"""
    print("\n" + "=" * 60)
    print("ПРИМЕР 3: Асинхронные множественные поиски")
    print("=" * 60)
    
    assistant = WebResearchAssistant(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        tavily_api_key=os.getenv("TAVILY_API_KEY")
    )
    
    queries = [
        "Latest breakthroughs in cancer research 2024",
        "Climate change impact on ocean ecosystems",
        "Advances in renewable energy storage technology"
    ]
    
    print("\n🚀 Запускаем параллельные поиски...")
    
    # Создаем задачи для параллельного выполнения
    tasks = []
    for query in queries:
        # Создаем новый ассистент для каждого запроса (чтобы не смешивать контексты)
        new_assistant = WebResearchAssistant(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            tavily_api_key=os.getenv("TAVILY_API_KEY")
        )
        tasks.append(new_assistant.process_message(query))
    
    # Выполняем все задачи параллельно
    results = await asyncio.gather(*tasks)
    
    # Выводим результаты
    for query, result in zip(queries, results):
        print(f"\n📌 Запрос: {query}")
        print(f"   Резюме: {result.summary[:200]}...")
        print(f"   Найдено ключевых пунктов: {len(result.key_findings)}")
        print(f"   Уверенность: {result.confidence_level}")


def example_export_to_json():
    """Пример экспорта результатов в JSON файл"""
    print("\n" + "=" * 60)
    print("ПРИМЕР 4: Экспорт в JSON")
    print("=" * 60)
    
    assistant = WebResearchAssistant(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        tavily_api_key=os.getenv("TAVILY_API_KEY")
    )
    
    # Выполняем исследование
    query = "Artificial Intelligence trends and predictions for 2025"
    print(f"\n🔍 Исследуем: {query}")
    
    result = assistant.process_message_sync(query)
    
    # Экспортируем в JSON файл
    output_file = "research_output.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result.model_dump(), f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Результаты сохранены в {output_file}")
    
    # Показываем структуру файла
    print("\n📄 Структура сохраненного файла:")
    data = result.model_dump()
    for key in data.keys():
        if isinstance(data[key], list):
            print(f"  • {key}: [{len(data[key])} элементов]")
        elif isinstance(data[key], dict):
            print(f"  • {key}: {{{len(data[key])} полей}}")
        else:
            print(f"  • {key}: {type(data[key]).__name__}")


def example_domain_specific_research():
    """Пример исследования в конкретной области"""
    print("\n" + "=" * 60)
    print("ПРИМЕР 5: Исследование в конкретной области")
    print("=" * 60)
    
    assistant = WebResearchAssistant(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        tavily_api_key=os.getenv("TAVILY_API_KEY")
    )
    
    # Технический запрос
    tech_query = """
    Compare the performance and features of the latest 
    JavaScript frameworks: React 19, Vue 3.4, and Angular 17. 
    Focus on bundle size, performance benchmarks, and developer experience.
    """
    
    print(f"\n💻 Технический анализ...")
    result = assistant.process_message_sync(tech_query)
    
    # Анализируем результаты
    print(f"\n📊 Анализ результатов:")
    print(f"  • Основная тема: {result.categories.main_topic}")
    print(f"  • Подтемы: {', '.join(result.categories.subtopics[:5])}")
    
    # Группируем находки по релевантности
    high_relevance = [f for f in result.key_findings if f.relevance_score >= 8]
    medium_relevance = [f for f in result.key_findings if 5 <= f.relevance_score < 8]
    
    print(f"\n⭐ Высокая релевантность ({len(high_relevance)} находок):")
    for finding in high_relevance[:3]:
        print(f"  • {finding.title}")
    
    print(f"\n📍 Средняя релевантность ({len(medium_relevance)} находок):")
    for finding in medium_relevance[:2]:
        print(f"  • {finding.title}")
    
    # Предложения для дальнейшего исследования
    print(f"\n💡 Рекомендации для углубленного изучения:")
    for i, query in enumerate(result.additional_queries[:3], 1):
        print(f"  {i}. {query}")


def example_analyze_sentiment():
    """Пример анализа тональности новостей"""
    print("\n" + "=" * 60)
    print("ПРИМЕР 6: Анализ тональности")
    print("=" * 60)
    
    assistant = WebResearchAssistant(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        tavily_api_key=os.getenv("TAVILY_API_KEY")
    )
    
    topics = [
        "Electric vehicle market growth 2024",
        "Global economic recession risks",
        "Breakthrough in Alzheimer's treatment"
    ]
    
    sentiments = {"positive": 0, "neutral": 0, "negative": 0}
    
    print("\n📰 Анализируем новости по темам:")
    
    for topic in topics:
        result = assistant.process_message_sync(topic)
        sentiments[result.sentiment] += 1
        
        emoji = {"positive": "😊", "neutral": "😐", "negative": "😟"}[result.sentiment]
        print(f"\n  {emoji} {topic}")
        print(f"     Тональность: {result.sentiment}")
        print(f"     Уверенность: {result.confidence_level}")
    
    # Общая статистика
    print("\n📈 Общая статистика тональности:")
    total = sum(sentiments.values())
    for sentiment, count in sentiments.items():
        percentage = (count / total) * 100
        bar = "█" * int(percentage / 5)
        print(f"  {sentiment:8}: {bar} {percentage:.0f}%")


# ============================================================================
# Тестовые функции
# ============================================================================

def test_structured_output():
    """Тест проверки структурированного вывода"""
    print("\n" + "=" * 60)
    print("ТЕСТ: Проверка структуры вывода")
    print("=" * 60)
    
    assistant = WebResearchAssistant(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        tavily_api_key=os.getenv("TAVILY_API_KEY")
    )
    
    result = assistant.process_message_sync("Python programming best practices")
    
    # Проверяем наличие всех полей
    checks = {
        "query": isinstance(result.query, str),
        "timestamp": isinstance(result.timestamp, str),
        "summary": isinstance(result.summary, str) and len(result.summary) > 0,
        "key_findings": isinstance(result.key_findings, list) and len(result.key_findings) > 0,
        "categories": result.categories is not None,
        "entities": result.entities is not None,
        "sentiment": result.sentiment in ["positive", "neutral", "negative"],
        "confidence_level": result.confidence_level in ["high", "medium", "low"],
        "additional_queries": isinstance(result.additional_queries, list)
    }
    
    print("\n✔️ Проверка структуры:")
    for field, is_valid in checks.items():
        status = "✅" if is_valid else "❌"
        print(f"  {status} {field}: {'Корректно' if is_valid else 'Ошибка'}")
    
    # Проверяем валидацию Pydantic
    try:
        json_str = result.model_dump_json()
        print(f"\n✅ JSON валидация пройдена")
        print(f"   Размер JSON: {len(json_str)} символов")
    except Exception as e:
        print(f"\n❌ Ошибка JSON валидации: {e}")
    
    return all(checks.values())


# ============================================================================
# Главная функция для запуска всех примеров
# ============================================================================

def run_all_examples():
    """Запустить все примеры"""
    print("\n" + "🚀 ЗАПУСК ВСЕХ ПРИМЕРОВ " + "🚀")
    print("=" * 60)
    
    # Проверяем наличие API ключей
    if not os.getenv("OPENAI_API_KEY") or not os.getenv("TAVILY_API_KEY"):
        print("❌ Ошибка: Установите OPENAI_API_KEY и TAVILY_API_KEY в файле .env")
        return
    
    # Запускаем синхронные примеры
    example_basic_search()
    example_conversation_with_memory()
    example_export_to_json()
    example_domain_specific_research()
    example_analyze_sentiment()
    
    # Запускаем асинхронный пример
    print("\n" + "=" * 60)
    asyncio.run(example_async_multiple_searches())
    
    # Запускаем тесты
    test_structured_output()
    
    print("\n" + "=" * 60)
    print("✅ ВСЕ ПРИМЕРЫ ВЫПОЛНЕНЫ УСПЕШНО!")
    print("=" * 60)


if __name__ == "__main__":
    # Запускаем все примеры
    #run_all_examples()
    
    # Или запустите конкретный пример:
    example_basic_search()
    # example_conversation_with_memory()
    # asyncio.run(example_async_multiple_searches())