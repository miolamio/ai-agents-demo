#!/usr/bin/env python3
"""
Тестирование Pinecone интеграции
"""

import os
from dotenv import load_dotenv

load_dotenv()

def test_pinecone_connection():
    """Тестируем подключение к Pinecone"""
    print("🧪 Тестирование Pinecone подключения")
    print("=" * 40)
    
    # Проверяем наличие ключей
    api_key = os.getenv("PINECONE_API_KEY")
    host = os.getenv("PINECONE_HOST")
    
    if not api_key or api_key == "your_pinecone_api_key_here":
        print("❌ PINECONE_API_KEY не установлен")
        print("Установите реальный API ключ в .env файле")
        return False
    
    if not host:
        print("❌ PINECONE_HOST не установлен")
        return False
    
    print(f"✅ API ключ: {api_key[:12]}...{api_key[-4:]}")
    print(f"✅ Host: {host}")
    
    try:
        from pinecone import Pinecone
        
        print("\n🔗 Инициализируем Pinecone клиент...")
        pc = Pinecone(api_key=api_key)
        
        print("📋 Получаем список индексов...")
        indexes = pc.list_indexes()
        
        if hasattr(indexes, 'indexes'):
            index_names = [idx.name for idx in indexes.indexes]
        else:
            index_names = [str(idx) for idx in indexes]
        
        print(f"📊 Найдено индексов: {len(index_names)}")
        for name in index_names:
            print(f"   • {name}")
        
        # Пробуем подключиться к индексу через host
        print(f"\n🎯 Подключаемся к индексу через host...")
        try:
            index = pc.Index(host=host)
            stats = index.describe_index_stats()
            
            print(f"✅ Подключение успешно!")
            print(f"   📊 Векторов: {stats.total_vector_count}")
            print(f"   📏 Размерность: {stats.dimension}")
            print(f"   📈 Заполненность: {stats.index_fullness}")
            
            return True
            
        except Exception as e:
            print(f"❌ Ошибка подключения к индексу: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Ошибка инициализации Pinecone: {e}")
        return False

def test_pinecone_with_langchain():
    """Тестируем интеграцию Pinecone с LangChain"""
    print("\n🔗 Тестирование LangChain + Pinecone")
    print("=" * 40)
    
    if not test_pinecone_connection():
        return False
    
    try:
        from langchain_pinecone import PineconeVectorStore
        from langchain_openai import OpenAIEmbeddings
        from langchain.schema import Document
        from pinecone import Pinecone
        
        # Создаем тестовые документы
        test_docs = [
            Document(
                page_content="FAISS - это библиотека для поиска векторов",
                metadata={"source": "test1.md", "type": "definition"}
            ),
            Document(
                page_content="ChromaDB - это база данных для эмбеддингов",
                metadata={"source": "test2.md", "type": "definition"}
            )
        ]
        
        print("📚 Создали тестовые документы")
        
        # Инициализируем компоненты
        api_key = os.getenv("PINECONE_API_KEY")
        host = os.getenv("PINECONE_HOST")
        
        pc = Pinecone(api_key=api_key)
        index = pc.Index(host=host)
        # Используем Cohere эмбеддинги с размерностью 1024 для совместимости с индексом
        from langchain_cohere import CohereEmbeddings
        embeddings = CohereEmbeddings(model="embed-multilingual-v3.0")
        
        print("🔗 Создаем векторное хранилище...")
        vectorstore = PineconeVectorStore(
            index=index,
            embedding=embeddings,
            text_key="text",
            namespace="test"
        )
        
        # Добавляем документы
        print("📝 Добавляем тестовые документы...")
        texts = [doc.page_content for doc in test_docs]
        metadatas = [doc.metadata for doc in test_docs]
        vectorstore.add_texts(texts, metadatas)
        
        print("🔍 Тестируем поиск...")
        results = vectorstore.similarity_search("библиотека поиска", k=1)
        
        if results:
            print(f"✅ Поиск работает! Найдено: {len(results)} результатов")
            print(f"   📄 Результат: {results[0].page_content}")
            return True
        else:
            print("❌ Поиск не вернул результатов")
            return False
            
    except Exception as e:
        print(f"❌ Ошибка тестирования LangChain: {e}")
        return False

def main():
    """Главная функция тестирования"""
    print("🧪 Тестирование интеграции Pinecone")
    print("=" * 50)
    
    # Базовое тестирование
    connection_ok = test_pinecone_connection()
    
    if connection_ok:
        # Тестирование с LangChain
        langchain_ok = test_pinecone_with_langchain()
        
        if langchain_ok:
            print("\n🎉 Все тесты пройдены успешно!")
            print("✅ Pinecone готов к использованию в vector_db_comparison.py")
        else:
            print("\n⚠️  Базовое подключение работает, но есть проблемы с LangChain")
    else:
        print("\n❌ Базовое подключение не работает")
        print("Проверьте API ключ и host URL")

if __name__ == "__main__":
    main()
