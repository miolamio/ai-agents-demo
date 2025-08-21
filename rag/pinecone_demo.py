#!/usr/bin/env python3
"""
Специальная демонстрация работы с Pinecone
"""

import os
import time
from dotenv import load_dotenv

load_dotenv()

def demo_pinecone_only():
    """Демонстрация только Pinecone"""
    print("🌲 Pinecone Demo - Специальная демонстрация")
    print("=" * 50)
    
    # Проверяем ключи
    pinecone_key = os.getenv("PINECONE_API_KEY")
    pinecone_host = os.getenv("PINECONE_HOST")
    cohere_key = os.getenv("COHERE_API_KEY")
    
    if not pinecone_key or not pinecone_host or not cohere_key:
        print("❌ Не все ключи настроены:")
        print(f"   PINECONE_API_KEY: {'✅' if pinecone_key else '❌'}")
        print(f"   PINECONE_HOST: {'✅' if pinecone_host else '❌'}")
        print(f"   COHERE_API_KEY: {'✅' if cohere_key else '❌'}")
        return
    
    try:
        from pinecone import Pinecone
        from langchain_pinecone import PineconeVectorStore
        from langchain_cohere import CohereEmbeddings
        from langchain.schema import Document
        
        print("🔗 Инициализация Pinecone...")
        pc = Pinecone(api_key=pinecone_key)
        
        # Получаем информацию об индексах
        indexes = pc.list_indexes()
        index_names = [idx.name for idx in indexes.indexes] if hasattr(indexes, 'indexes') else []
        print(f"📋 Доступные индексы: {index_names}")
        
        # Подключаемся через host URL
        print(f"🔗 Подключаемся к индексу через host URL...")
        index = pc.Index(host=pinecone_host)
        
        # Получаем статистику
        stats = index.describe_index_stats()
        print(f"📊 Статистика индекса:")
        print(f"   • Векторов: {stats.total_vector_count}")
        print(f"   • Размерность: {stats.dimension}")
        print(f"   • Заполненность: {stats.index_fullness}")
        print(f"   • Namespaces: {list(stats.namespaces.keys()) if stats.namespaces else []}")
        
        # Создаем эмбеддинги
        print(f"🧠 Инициализация Cohere эмбеддингов...")
        embeddings = CohereEmbeddings(model="embed-multilingual-v3.0")
        
        # Создаем векторное хранилище
        print(f"📦 Создание векторного хранилища...")
        vectorstore = PineconeVectorStore(
            index=index,
            embedding=embeddings,
            text_key="text",
            namespace="demo"  # Используем отдельный namespace для демо
        )
        
        # Добавляем тестовые документы
        test_docs = [
            "FAISS - это библиотека Facebook для поиска векторов",
            "ChromaDB - open source база данных для эмбеддингов", 
            "Pinecone - облачная векторная база данных",
            "RAG объединяет поиск и генерацию текста"
        ]
        
        print(f"📝 Добавляем тестовые документы...")
        vectorstore.add_texts(
            texts=test_docs,
            metadatas=[{"source": f"demo_{i}.md"} for i in range(len(test_docs))]
        )
        
        # Тестируем поиск
        print(f"🔍 Тестируем поиск...")
        search_queries = [
            "векторная база данных",
            "библиотека поиска",
            "RAG система"
        ]
        
        for query in search_queries:
            start_time = time.time()
            results = vectorstore.similarity_search(query, k=2)
            search_time = time.time() - start_time
            
            print(f"\n🔎 Запрос: '{query}'")
            print(f"   ⏱️  Время поиска: {search_time:.3f}с")
            print(f"   📄 Найдено результатов: {len(results)}")
            
            for i, result in enumerate(results):
                print(f"   {i+1}. {result.page_content}")
        
        # Финальная статистика
        final_stats = index.describe_index_stats()
        print(f"\n📊 Финальная статистика:")
        print(f"   • Общее количество векторов: {final_stats.total_vector_count}")
        print(f"   • Namespaces: {list(final_stats.namespaces.keys()) if final_stats.namespaces else []}")
        
        print(f"\n🎉 Pinecone демонстрация завершена успешно!")
        
    except Exception as e:
        print(f"❌ Ошибка в Pinecone демо: {e}")

if __name__ == "__main__":
    demo_pinecone_only()
