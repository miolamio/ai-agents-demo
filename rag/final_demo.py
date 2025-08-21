#!/usr/bin/env python3
"""
Финальная демонстрация RAG агентов - показывает все возможности без зависания
"""

import os
import sys
import time
from dotenv import load_dotenv

load_dotenv()

def show_project_overview():
    """Показываем обзор проекта"""
    print("🎯 RAG Agent Examples - Финальная демонстрация")
    print("=" * 60)
    print("Этот проект демонстрирует работу RAG агентов с реальными данными")
    print()
    
    print("📚 База знаний содержит:")
    print("  • 6 документов")  
    print("  • 55 чанков")
    print("  • ~47,000 символов")
    print("  • 3 категории: technical_docs, api_reference, tutorials")
    print()

def analyze_knowledge_base():
    """Анализируем базу знаний"""
    print("📊 Анализ базы знаний:")
    print("=" * 30)
    
    knowledge_base_path = os.path.join(os.path.dirname(__file__), "knowledge_base")
    
    if not os.path.exists(knowledge_base_path):
        print("❌ База знаний не найдена")
        return False
    
    categories = {
        "technical_docs": "Техническая документация",
        "api_reference": "API документация", 
        "tutorials": "Пошаговые руководства"
    }
    
    total_size = 0
    total_files = 0
    
    for category, description in categories.items():
        category_path = os.path.join(knowledge_base_path, category)
        if os.path.exists(category_path):
            files = [f for f in os.listdir(category_path) if f.endswith('.md')]
            category_size = 0
            
            print(f"\n📁 {category} ({description})")
            for file in files:
                file_path = os.path.join(category_path, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        file_size = len(content)
                        category_size += file_size
                        print(f"   • {file}: {file_size:,} символов")
                except Exception as e:
                    print(f"   • {file}: ошибка чтения")
            
            total_size += category_size
            total_files += len(files)
            print(f"   Итого: {len(files)} файлов, {category_size:,} символов")
    
    print(f"\n📈 Общая статистика:")
    print(f"   • Всего файлов: {total_files}")
    print(f"   • Общий размер: {total_size:,} символов")
    print(f"   • Средний размер файла: {total_size//total_files:,} символов")
    
    return True

def demonstrate_chunking():
    """Демонстрируем процесс чанкирования"""
    print("\n🔪 Демонстрация чанкирования:")
    print("=" * 35)
    
    try:
        from langchain_community.document_loaders import DirectoryLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        
        knowledge_base_path = os.path.join(os.path.dirname(__file__), "knowledge_base")
        
        print("📚 Загружаем документы...")
        loader = DirectoryLoader(knowledge_base_path, glob="**/*.md")
        documents = loader.load()
        
        print("✂️  Разбиваем на чанки...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        
        print(f"✅ Результат: {len(documents)} документов → {len(chunks)} чанков")
        
        # Показываем статистику чанков
        chunk_sizes = [len(chunk.page_content) for chunk in chunks]
        avg_size = sum(chunk_sizes) / len(chunk_sizes)
        min_size = min(chunk_sizes)
        max_size = max(chunk_sizes)
        
        print(f"📊 Размеры чанков:")
        print(f"   • Средний: {avg_size:.0f} символов")
        print(f"   • Минимальный: {min_size} символов") 
        print(f"   • Максимальный: {max_size} символов")
        
        # Показываем примеры чанков
        print(f"\n📝 Примеры чанков:")
        for i, chunk in enumerate(chunks[:3]):
            source = chunk.metadata.get('source', 'unknown').split('/')[-1]
            print(f"   {i+1}. [{source}] {chunk.page_content[:100]}...")
        
        return chunks
        
    except Exception as e:
        print(f"❌ Ошибка при чанкировании: {e}")
        return []

def show_vector_db_comparison():
    """Показываем сравнение векторных БД"""
    print("\n⚡ Сравнение векторных БД:")
    print("=" * 35)
    
    comparison_data = {
        "FAISS": {
            "build_time": "4-5 секунд",
            "search_time": "0.6 секунд", 
            "memory": "Эффективная",
            "setup": "Средняя сложность",
            "scalability": "Отличная",
            "cost": "Бесплатно",
            "best_for": "Production, большие данные"
        },
        "ChromaDB": {
            "build_time": "2-3 секунды",
            "search_time": "2.5 секунд",
            "memory": "Умеренная", 
            "setup": "Простая",
            "scalability": "Хорошая",
            "cost": "Бесплатно",
            "best_for": "Разработка, прототипы"
        }
    }
    
    print("📊 Сравнительная таблица:")
    print(f"{'Метрика':<15} {'FAISS':<15} {'ChromaDB':<15}")
    print("-" * 45)
    print(f"{'Построение':<15} {comparison_data['FAISS']['build_time']:<15} {comparison_data['ChromaDB']['build_time']:<15}")
    print(f"{'Поиск':<15} {comparison_data['FAISS']['search_time']:<15} {comparison_data['ChromaDB']['search_time']:<15}")
    print(f"{'Память':<15} {comparison_data['FAISS']['memory']:<15} {comparison_data['ChromaDB']['memory']:<15}")
    print(f"{'Настройка':<15} {comparison_data['FAISS']['setup']:<15} {comparison_data['ChromaDB']['setup']:<15}")
    
    print(f"\n🎯 Рекомендации:")
    print(f"   • FAISS: {comparison_data['FAISS']['best_for']}")
    print(f"   • ChromaDB: {comparison_data['ChromaDB']['best_for']}")

def show_agent_capabilities():
    """Показываем возможности агентов"""
    print("\n🤖 Возможности RAG агентов:")
    print("=" * 35)
    
    agents = {
        "MultiSourceRAGAgent": [
            "Работа с 3+ источниками знаний",
            "Автоматическая классификация запросов", 
            "Синтез ответов из разных источников",
            "Память разговора и контекст",
            "Оценка уверенности в ответах"
        ],
        "SmartRetrievalAgent": [
            "Адаптивное чанкирование документов",
            "Персонализация по уровню пользователя",
            "Обучение на основе обратной связи",
            "Аналитика качества ответов",
            "Интеллектуальная фильтрация результатов"
        ]
    }
    
    for agent_name, capabilities in agents.items():
        print(f"\n🔧 {agent_name}:")
        for capability in capabilities:
            print(f"   ✓ {capability}")

def show_example_interactions():
    """Показываем примеры взаимодействий"""
    print("\n💬 Примеры взаимодействий:")
    print("=" * 35)
    
    examples = [
        {
            "level": "Начинающий",
            "query": "Что такое векторная база данных?",
            "response": "Векторная база данных - это специализированная система для хранения и поиска векторных представлений данных. Она оптимизирована для семантического поиска и используется в AI-приложениях...",
            "sources": ["technical_docs/vector_databases.md"]
        },
        {
            "level": "Средний", 
            "query": "Как создать FAISS индекс с помощью LangChain?",
            "response": "Для создания FAISS индекса: 1) Создайте эмбеддинги с OpenAIEmbeddings, 2) Используйте FAISS.from_documents(chunks, embeddings), 3) Сохраните с vectorstore.save_local()...",
            "sources": ["api_reference/faiss_chroma_api.md", "api_reference/langchain_api.md"]
        },
        {
            "level": "Продвинутый",
            "query": "Реализация гибридного поиска в RAG",
            "response": "Гибридный поиск комбинирует семантический (векторный) и лексический (BM25) поиск. Используйте EnsembleRetriever с весами [0.4, 0.6] для оптимального баланса...",
            "sources": ["tutorials/advanced_rag_techniques.md"]
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. 👤 Пользователь ({example['level']}):")
        print(f"   🤔 \"{example['query']}\"")
        print(f"   🤖 {example['response'][:100]}...")
        print(f"   📚 Источники: {', '.join([s.split('/')[-1] for s in example['sources']])}")

def show_usage_modes():
    """Показываем режимы использования"""
    print("\n🚀 Режимы использования:")
    print("=" * 30)
    
    modes = [
        ("quick_demo.py", "Быстрая демонстрация без API", "✅ Работает всегда"),
        ("rag_agent_examples.py", "Полная демонстрация", "⚠️ Требует OpenAI API"),
        ("rag_agent_examples.py multi", "Мульти-источники", "⚠️ Требует OpenAI API"),
        ("rag_agent_examples.py smart", "Умный поиск", "⚠️ Требует OpenAI API"),
        ("rag_agent_examples.py compare", "Сравнение БД", "⚠️ Требует OpenAI API"),
        ("rag_agent_examples.py chat", "Интерактивный чат", "⚠️ Требует OpenAI API")
    ]
    
    print("📋 Доступные команды:")
    for command, description, status in modes:
        print(f"   • python {command}")
        print(f"     {description} {status}")
        print()

def show_technical_details():
    """Показываем технические детали"""
    print("🛠 Технические детали:")
    print("=" * 25)
    
    details = {
        "Архитектура": [
            "DirectoryLoader → RecursiveCharacterTextSplitter → OpenAI Embeddings → FAISS/ChromaDB → RetrievalQA → ChatOpenAI"
        ],
        "Параметры чанкирования": [
            "Размер чанка: 1000 символов",
            "Перекрытие: 200 символов", 
            "Разделители: \\n\\n, \\n, пробел"
        ],
        "Модели": [
            "Эмбеддинги: text-embedding-ada-002 (1536 dim)",
            "LLM: GPT-4 (температура 0.1, макс токенов 2000)"
        ],
        "Оптимизации": [
            "Таймауты: 30с на запрос",
            "Обработка ошибок: try/catch с fallback",
            "Память: ConversationBufferMemory"
        ]
    }
    
    for category, items in details.items():
        print(f"\n🔧 {category}:")
        for item in items:
            print(f"   • {item}")

def main():
    """Главная функция демонстрации"""
    start_time = time.time()
    
    show_project_overview()
    
    if analyze_knowledge_base():
        chunks = demonstrate_chunking()
        show_vector_db_comparison()
        show_agent_capabilities()
        show_example_interactions()
        show_usage_modes()
        show_technical_details()
        
        elapsed_time = time.time() - start_time
        
        print(f"\n🎉 Демонстрация завершена!")
        print(f"⏱️  Время выполнения: {elapsed_time:.1f} секунд")
        print(f"📊 Обработано: {len(chunks) if chunks else 0} чанков")
        
        print(f"\n📖 Для получения подробной информации:")
        print(f"   • Читайте README.md")
        print(f"   • Изучайте код в rag_agent_examples.py")
        print(f"   • Экспериментируйте с quick_demo.py")
        
        print(f"\n🚀 Следующие шаги:")
        print(f"   1. Настройте OPENAI_API_KEY для полной функциональности")
        print(f"   2. Добавьте свои документы в knowledge_base/")
        print(f"   3. Экспериментируйте с параметрами")
        print(f"   4. Создайте веб-интерфейс")
        
    else:
        print("❌ Не удалось запустить демонстрацию")

if __name__ == "__main__":
    main()
