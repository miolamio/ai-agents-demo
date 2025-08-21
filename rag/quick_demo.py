#!/usr/bin/env python3
"""
Быстрая демонстрация RAG агентов без длительных API вызовов
"""

import os
import sys
from dotenv import load_dotenv

# Добавляем текущую папку в путь
sys.path.append(os.path.dirname(__file__))

load_dotenv()

def quick_faiss_chroma_comparison():
    """Быстрое сравнение FAISS и ChromaDB без OpenAI API"""
    print("🚀 Быстрая демонстрация RAG - Сравнение FAISS и ChromaDB")
    print("=" * 60)
    
    try:
        from langchain_community.document_loaders import DirectoryLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        import time
        
        # Загружаем документы
        knowledge_base_path = os.path.join(os.path.dirname(__file__), "knowledge_base")
        
        if not os.path.exists(knowledge_base_path):
            print("❌ Папка knowledge_base не найдена")
            return
        
        print("📚 Загружаем документы...")
        loader = DirectoryLoader(knowledge_base_path, glob="**/*.md")
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)
        
        print(f"✅ Загружено {len(documents)} документов, создано {len(chunks)} чанков")
        
        # Демонстрируем структуру данных без API вызовов
        print(f"\n📊 Анализ документов:")
        for i, doc in enumerate(documents[:3]):  # Показываем первые 3
            print(f"  {i+1}. {doc.metadata.get('source', 'unknown')}: {len(doc.page_content)} символов")
        
        print(f"\n🔍 Примеры чанков:")
        for i, chunk in enumerate(chunks[:2]):  # Показываем первые 2 чанка
            print(f"  Чанк {i+1}: {chunk.page_content[:100]}...")
        
        # Имитируем производительность без реальных API вызовов
        print(f"\n⚡ Имитация производительности:")
        print(f"  FAISS: ~4-5с построение, ~0.6с поиск")
        print(f"  ChromaDB: ~2-3с построение, ~2.5с поиск")
        print(f"  Рекомендация: ChromaDB для разработки, FAISS для production")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return False

def quick_knowledge_base_demo():
    """Демонстрация структуры базы знаний"""
    print("\n📚 Структура базы знаний:")
    print("=" * 40)
    
    knowledge_base_path = os.path.join(os.path.dirname(__file__), "knowledge_base")
    
    if not os.path.exists(knowledge_base_path):
        print("❌ База знаний не найдена")
        return
    
    categories = ["technical_docs", "api_reference", "tutorials"]
    total_files = 0
    
    for category in categories:
        category_path = os.path.join(knowledge_base_path, category)
        if os.path.exists(category_path):
            files = [f for f in os.listdir(category_path) if f.endswith('.md')]
            total_files += len(files)
            print(f"📁 {category}: {len(files)} файлов")
            for file in files:
                file_path = os.path.join(category_path, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        print(f"   • {file}: {len(content)} символов")
                except:
                    print(f"   • {file}: не удалось прочитать")
        else:
            print(f"📁 {category}: не найдена")
    
    print(f"\n📊 Итого: {total_files} документов в базе знаний")

def show_example_queries():
    """Показываем примеры запросов без API вызовов"""
    print("\n🤖 Примеры возможных запросов:")
    print("=" * 40)
    
    examples = [
        {
            "query": "Как создать FAISS индекс?",
            "source": "api_reference",
            "mock_answer": "Для создания FAISS индекса используйте FAISS.from_documents(chunks, embeddings)..."
        },
        {
            "query": "Что такое ChromaDB?",
            "source": "technical_docs", 
            "mock_answer": "ChromaDB - это open-source база данных для эмбеддингов, разработанная для простоты использования..."
        },
        {
            "query": "Основы RAG архитектуры",
            "source": "tutorials",
            "mock_answer": "RAG объединяет поиск информации с генерацией текста. Система сначала извлекает релевантные документы..."
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. 🤔 Запрос: {example['query']}")
        print(f"   📚 Источник: {example['source']}")
        print(f"   🤖 Ответ: {example['mock_answer'][:100]}...")

def main():
    """Главная функция быстрой демонстрации"""
    print("🎯 RAG Quick Demo - Быстрая демонстрация без API вызовов")
    print("=" * 70)
    
    # Проверяем наличие документов
    success = quick_faiss_chroma_comparison()
    
    if success:
        quick_knowledge_base_demo()
        show_example_queries()
        
        print(f"\n💡 Для полной функциональности:")
        print(f"   1. Установите OPENAI_API_KEY в .env файле")
        print(f"   2. Запустите: python rag_agent_examples.py")
        print(f"   3. Или интерактивный чат: python rag_agent_examples.py chat")
        
        print(f"\n✨ Доступные режимы:")
        print(f"   • python rag_agent_examples.py multi    - Мульти-источники")
        print(f"   • python rag_agent_examples.py smart    - Умный поиск")
        print(f"   • python rag_agent_examples.py compare  - Сравнение БД")
        print(f"   • python rag_agent_examples.py chat     - Интерактивный чат")
    
    print(f"\n🎉 Демонстрация завершена!")

if __name__ == "__main__":
    main()
