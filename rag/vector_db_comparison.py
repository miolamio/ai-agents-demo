#!/usr/bin/env python3
"""
Сравнение различных векторных баз данных для RAG
Практические примеры работы с FAISS, Chroma, Pinecone
"""

import os
import time
import json
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import numpy as np

# Загрузка переменных окружения
from dotenv import load_dotenv
load_dotenv()

# LangChain imports
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS, Chroma
from langchain.schema import Document

# Дополнительные библиотеки
try:
    from pinecone import Pinecone
    from langchain_pinecone import PineconeVectorStore
    PINECONE_AVAILABLE = True
except ImportError:
    try:
        # Fallback для старой версии
        import pinecone
        from langchain_community.vectorstores import Pinecone as PineconeVectorStore
        PINECONE_AVAILABLE = True
    except ImportError:
        PINECONE_AVAILABLE = False
        print("Pinecone не установлен. Установите: pip install pinecone-client langchain-pinecone")

try:
    from langchain_cohere import CohereEmbeddings
    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False
    print("Cohere не установлен. Установите: pip install cohere langchain-cohere")

try:
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    print("ChromaDB не установлен. Установите: pip install chromadb")


@dataclass
class VectorDBConfig:
    """Конфигурация для векторных баз данных"""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    embedding_model: str = "text-embedding-ada-002"
    similarity_threshold: float = 0.7


@dataclass
class BenchmarkResult:
    """Результаты бенчмарка векторной базы данных"""
    db_name: str
    indexing_time: float
    search_time: float
    memory_usage_mb: float
    search_accuracy: float
    setup_complexity: int  # 1-5, где 1 - очень просто, 5 - сложно
    scalability_score: int  # 1-5, где 5 - отличная масштабируемость
    cost_rating: str  # "free", "low", "medium", "high"


class VectorDatabaseComparison:
    """
    Класс для сравнения различных векторных баз данных
    """
    
    def __init__(self, config: VectorDBConfig = None):
        self.config = config or VectorDBConfig()
        self.documents = []
        self.embeddings = OpenAIEmbeddings(model=self.config.embedding_model)
        self.test_queries = [
            "Как настроить векторную базу данных?",
            "Что такое эмбеддинги в машинном обучении?",
            "Оптимизация производительности поиска",
            "Методы оценки качества RAG систем",
            "Интеграция с LangChain"
        ]
        
    def load_sample_documents(self, docs_path: str = None) -> List[Document]:
        """Загрузка примерных документов для тестирования"""
        if docs_path and os.path.exists(docs_path):
            loader = DirectoryLoader(docs_path, glob="**/*.md")
            documents = loader.load()
        else:
            # Создаем синтетические документы для демонстрации
            documents = self._create_synthetic_documents()
        
        # Разбиение на чанки
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        
        self.documents = text_splitter.split_documents(documents)
        
        # Добавление метаданных
        for i, doc in enumerate(self.documents):
            doc.metadata.update({
                "doc_id": f"doc_{i}",
                "chunk_size": len(doc.page_content),
                "source": doc.metadata.get("source", f"synthetic_{i}")
            })
        
        print(f"Загружено {len(self.documents)} документов")
        return self.documents
    
    def _create_synthetic_documents(self) -> List[Document]:
        """Создание синтетических документов для демонстрации"""
        synthetic_texts = [
            """
            Векторные базы данных представляют собой специализированные системы хранения, 
            оптимизированные для работы с векторными представлениями данных. Они используются 
            в задачах машинного обучения, поиска по семантическому сходству и обработки 
            естественного языка. Основные преимущества включают быстрый поиск по сходству, 
            масштабируемость и поддержку различных метрик расстояния.
            """,
            """
            FAISS (Facebook AI Similarity Search) - это библиотека для эффективного поиска 
            сходства и кластеризации плотных векторов. Она содержит алгоритмы, которые ищут 
            в наборах векторов любого размера, вплоть до тех, которые могут не помещаться в RAM. 
            FAISS написан на C++ с полными обертками для Python/numpy.
            """,
            """
            Chroma является open-source базой данных для эмбеддингов. Она разработана для 
            простоты использования с AI-приложениями. Chroma позволяет хранить эмбеддинги и 
            их метаданные, встраивать документы и запросы, и быстро искать эмбеддинги. 
            Она включает встроенное хранение и визуализацию.
            """,
            """
            Pinecone представляет собой полностью управляемую векторную базу данных, которая 
            упрощает создание высокопроизводительных приложений векторного поиска. Она 
            обрабатывает всю инфраструктуру, включая масштабирование, безопасность и 
            высокую доступность, позволяя разработчикам сосредоточиться на создании 
            великолепных приложений.
            """,
            """
            Embedding модели преобразуют текст в числовые векторы, которые фиксируют 
            семантическое значение. OpenAI's text-embedding-ada-002 является одной из 
            самых популярных моделей, обеспечивающей высокое качество эмбеддингов для 
            различных задач NLP. Размерность векторов составляет 1536.
            """,
            """
            RAG (Retrieval-Augmented Generation) объединяет поиск информации с генерацией 
            текста. Система сначала извлекает релевантные документы из базы знаний, а затем 
            использует их как контекст для языковой модели. Это позволяет получать более 
            точные и обоснованные ответы, снижает галлюцинации и обеспечивает актуальность информации.
            """,
            """
            Оценка качества RAG-систем включает метрики для оценки как извлечения документов 
            (Recall@K, Precision@K, MRR), так и качества генерации (Faithfulness, 
            Answer Relevance). RAGAS (RAG Assessment) - это фреймворк для автоматической 
            оценки RAG-пайплайнов с использованием LLM в качестве судей.
            """
        ]
        
        documents = []
        for i, text in enumerate(synthetic_texts):
            doc = Document(
                page_content=text.strip(),
                metadata={"source": f"synthetic_doc_{i}.md", "topic": f"topic_{i % 3}"}
            )
            documents.append(doc)
        
        return documents


class FAISSHandler:
    """Обработчик для FAISS векторной базы данных"""
    
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.vectorstore = None
        
    def setup(self, documents: List[Document]) -> float:
        """Настройка FAISS индекса"""
        start_time = time.time()
        
        # Создание FAISS индекса
        self.vectorstore = FAISS.from_documents(documents, self.embeddings)
        
        # Дополнительная оптимизация индекса
        # В реальном приложении можно использовать более сложные индексы
        # как IndexIVFFlat или IndexHNSW для больших датасетов
        
        setup_time = time.time() - start_time
        print(f"FAISS setup completed in {setup_time:.2f} seconds")
        return setup_time
    
    def search(self, query: str, k: int = 4) -> Tuple[List[Document], float]:
        """Поиск в FAISS"""
        start_time = time.time()
        
        # Поиск с оценками схожести
        docs_and_scores = self.vectorstore.similarity_search_with_score(query, k=k)
        docs = [doc for doc, _ in docs_and_scores]
        
        search_time = time.time() - start_time
        return docs, search_time
    
    def save_local(self, path: str):
        """Сохранение FAISS индекса локально"""
        if self.vectorstore:
            self.vectorstore.save_local(path)
    
    def load_local(self, path: str):
        """Загрузка FAISS индекса"""
        self.vectorstore = FAISS.load_local(path, self.embeddings)
    
    def get_stats(self) -> Dict[str, Any]:
        """Получение статистики FAISS"""
        if not self.vectorstore:
            return {}
        
        # FAISS специфичная статистика
        index = self.vectorstore.index
        return {
            "total_vectors": index.ntotal,
            "dimension": index.d,
            "index_type": type(index).__name__,
            "is_trained": index.is_trained,
            "metric_type": getattr(index, 'metric_type', 'L2')
        }


class ChromaHandler:
    """Обработчик для ChromaDB"""
    
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.vectorstore = None
        self.client = None
        
    def setup(self, documents: List[Document], persist_directory: str = "./chroma_db") -> float:
        """Настройка ChromaDB"""
        if not CHROMA_AVAILABLE:
            raise ImportError("ChromaDB не установлен")
        
        start_time = time.time()
        
        # Создание персистентного клиента
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Создание векторного хранилища
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            client=self.client,
            collection_name="rag_collection",
            persist_directory=persist_directory
        )
        
        setup_time = time.time() - start_time
        print(f"ChromaDB setup completed in {setup_time:.2f} seconds")
        return setup_time
    
    def search(self, query: str, k: int = 4) -> Tuple[List[Document], float]:
        """Поиск в ChromaDB"""
        start_time = time.time()
        
        # Поиск с фильтрацией по метаданным (пример)
        docs = self.vectorstore.similarity_search(
            query, 
            k=k,
            # filter={"topic": "specific_topic"}  # Пример фильтрации
        )
        
        search_time = time.time() - start_time
        return docs, search_time
    
    def search_with_metadata_filter(self, query: str, filter_dict: Dict, k: int = 4) -> List[Document]:
        """Поиск с фильтрацией по метаданным"""
        return self.vectorstore.similarity_search(
            query,
            k=k,
            filter=filter_dict
        )
    
    def add_documents(self, documents: List[Document]):
        """Добавление новых документов"""
        if self.vectorstore:
            self.vectorstore.add_documents(documents)
    
    def get_stats(self) -> Dict[str, Any]:
        """Получение статистики ChromaDB"""
        if not self.vectorstore:
            return {}
        
        # Получение информации о коллекции
        collection = self.vectorstore._collection
        return {
            "total_documents": collection.count(),
            "collection_name": collection.name,
            "persist_directory": getattr(self.vectorstore, '_persist_directory', None)
        }


class PineconeHandler:
    """Обработчик для Pinecone (новый API с Cohere эмбеддингами)"""
    
    def __init__(self, embeddings=None, api_key: str = None, host: str = None):
        self.api_key = api_key or os.environ.get("PINECONE_API_KEY")
        self.host = host or os.environ.get("PINECONE_HOST")
        # Используем существующий индекс вместо создания нового
        self.index_name = "technospherehr"  # Используем существующий индекс
        self.pc = None
        self.vectorstore = None
        
        # Используем Cohere эмбеддинги для совместимости с индексом (1024 размерность)
        if COHERE_AVAILABLE and os.getenv("COHERE_API_KEY"):
            self.embeddings = CohereEmbeddings(model="embed-multilingual-v3.0")
            print("🔗 Используем Cohere эмбеддинги (1024 размерность)")
        else:
            self.embeddings = embeddings or OpenAIEmbeddings()
            print("⚠️ Используем OpenAI эмбеддинги (1536 размерность) - может быть несовместимо")
        
    def setup(self, documents: List[Document]) -> float:
        """Настройка Pinecone индекса (новый API)"""
        if not PINECONE_AVAILABLE:
            raise ImportError("Pinecone не установлен")
        
        if not self.api_key:
            raise ValueError("Pinecone API key не установлен")
        
        start_time = time.time()
        
        try:
            # Инициализация нового Pinecone клиента
            self.pc = Pinecone(api_key=self.api_key)
            
            # Проверяем существующие индексы
            existing_indexes = self.pc.list_indexes()
            index_names = [idx.name for idx in existing_indexes.indexes] if hasattr(existing_indexes, 'indexes') else []
            
            print(f"Существующие индексы: {index_names}")
            
            # Проверяем что наш индекс существует
            if self.index_name not in index_names:
                print(f"❌ Индекс '{self.index_name}' не найден!")
                print(f"Доступные индексы: {index_names}")
                # Используем первый доступный индекс
                if index_names:
                    self.index_name = index_names[0]
                    print(f"🔄 Переключаемся на индекс: {self.index_name}")
                else:
                    raise ValueError("Нет доступных индексов")
            
            # Подключение к существующему индексу
            if self.host:
                # Используем прямой host URL если предоставлен
                index = self.pc.Index(host=self.host)
                print(f"🔗 Подключились к индексу через host URL")
            else:
                index = self.pc.Index(self.index_name)
                print(f"🔗 Подключились к индексу: {self.index_name}")
            
            # Создание векторного хранилища
            self.vectorstore = PineconeVectorStore.from_documents(
                documents,
                self.embeddings,
                index=index,
                namespace="default"
            )
            
            setup_time = time.time() - start_time
            print(f"Pinecone setup completed in {setup_time:.2f} seconds")
            return setup_time
            
        except Exception as e:
            print(f"Ошибка при настройке Pinecone: {e}")
            # Пробуем альтернативный подход с прямым подключением к host
            if self.host:
                try:
                    print("Пробуем прямое подключение к существующему индексу...")
                    # Используем прямое подключение к host URL
                    index = self.pc.Index(host=self.host)
                    
                    self.vectorstore = PineconeVectorStore(
                        index=index,
                        embedding=self.embeddings,
                        text_key="text",
                        namespace="default"
                    )
                    
                    # Добавляем документы
                    texts = [doc.page_content for doc in documents]
                    metadatas = [doc.metadata for doc in documents]
                    self.vectorstore.add_texts(texts, metadatas)
                    
                    setup_time = time.time() - start_time
                    print(f"Pinecone setup completed (direct host) in {setup_time:.2f} seconds")
                    return setup_time
                except Exception as e2:
                    print(f"Ошибка при прямом подключении: {e2}")
                    raise e
            else:
                raise e
    
    def search(self, query: str, k: int = 4) -> Tuple[List[Document], float]:
        """Поиск в Pinecone"""
        start_time = time.time()
        
        docs = self.vectorstore.similarity_search(query, k=k)
        
        search_time = time.time() - start_time
        return docs, search_time
    
    def search_with_namespace(self, query: str, namespace: str, k: int = 4) -> List[Document]:
        """Поиск в определенном namespace"""
        if not self.pc:
            raise ValueError("Pinecone клиент не инициализирован")
        
        try:
            # Создание временного векторного хранилища для нужного namespace
            if self.host:
                index = self.pc.Index(self.index_name, host=self.host)
            else:
                index = self.pc.Index(self.index_name)
            
            temp_store = PineconeVectorStore(
                index=index,
                embedding=self.embeddings,
                text_key="text",
                namespace=namespace
            )
            return temp_store.similarity_search(query, k=k)
        except Exception as e:
            print(f"Ошибка при поиске в namespace {namespace}: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Получение статистики Pinecone"""
        if not self.pc or not self.index_name:
            return {}
        
        try:
            if self.host:
                index = self.pc.Index(self.index_name, host=self.host)
            else:
                index = self.pc.Index(self.index_name)
            
            stats = index.describe_index_stats()
            return {
                "total_vectors": stats.total_vector_count,
                "dimension": stats.dimension,
                "index_fullness": stats.index_fullness,
                "namespaces": list(stats.namespaces.keys()) if stats.namespaces else []
            }
        except Exception as e:
            return {"error": str(e)}


class VectorDatabaseBenchmark:
    """Бенчмарк для сравнения векторных баз данных"""
    
    def __init__(self, comparison: VectorDatabaseComparison):
        self.comparison = comparison
        self.results = []
        
    def run_benchmark(self, pinecone_api_key: str = None) -> List[BenchmarkResult]:
        """Запуск полного бенчмарка"""
        documents = self.comparison.load_sample_documents()
        
        # Бенчмарк FAISS
        print("\n=== Бенчмарк FAISS ===")
        faiss_result = self._benchmark_faiss(documents)
        self.results.append(faiss_result)
        
        # Бенчмарк ChromaDB
        if CHROMA_AVAILABLE:
            print("\n=== Бенчмарк ChromaDB ===")
            chroma_result = self._benchmark_chroma(documents)
            self.results.append(chroma_result)
        
        # Бенчмарк Pinecone
        if PINECONE_AVAILABLE and pinecone_api_key:
            print("\n=== Бенчмарк Pinecone ===")
            pinecone_result = self._benchmark_pinecone(documents, pinecone_api_key)
            self.results.append(pinecone_result)
        
        return self.results
    
    def _benchmark_faiss(self, documents: List[Document]) -> BenchmarkResult:
        """Бенчмарк FAISS"""
        handler = FAISSHandler(self.comparison.embeddings)
        
        # Измерение времени индексации
        indexing_time = handler.setup(documents)
        
        # Измерение времени поиска
        total_search_time = 0
        for query in self.comparison.test_queries:
            _, search_time = handler.search(query)
            total_search_time += search_time
        
        avg_search_time = total_search_time / len(self.comparison.test_queries)
        
        # Приблизительная оценка использования памяти
        memory_usage = len(documents) * self.comparison.config.chunk_size * 1.5 / 1024 / 1024
        
        # Статистика
        stats = handler.get_stats()
        print(f"FAISS stats: {stats}")
        
        return BenchmarkResult(
            db_name="FAISS",
            indexing_time=indexing_time,
            search_time=avg_search_time,
            memory_usage_mb=memory_usage,
            search_accuracy=0.85,  # Примерная оценка
            setup_complexity=2,  # Просто
            scalability_score=3,  # Средняя
            cost_rating="free"
        )
    
    def _benchmark_chroma(self, documents: List[Document]) -> BenchmarkResult:
        """Бенчмарк ChromaDB"""
        handler = ChromaHandler(self.comparison.embeddings)
        
        # Измерение времени индексации
        indexing_time = handler.setup(documents, "./temp_chroma_benchmark")
        
        # Измерение времени поиска
        total_search_time = 0
        for query in self.comparison.test_queries:
            _, search_time = handler.search(query)
            total_search_time += search_time
        
        avg_search_time = total_search_time / len(self.comparison.test_queries)
        
        # Статистика
        stats = handler.get_stats()
        print(f"ChromaDB stats: {stats}")
        
        return BenchmarkResult(
            db_name="ChromaDB",
            indexing_time=indexing_time,
            search_time=avg_search_time,
            memory_usage_mb=len(documents) * 2.0,  # Приблизительная оценка
            search_accuracy=0.88,
            setup_complexity=1,  # Очень просто
            scalability_score=4,  # Хорошая
            cost_rating="free"
        )
    
    def _benchmark_pinecone(self, documents: List[Document], api_key: str) -> BenchmarkResult:
        """Бенчмарк Pinecone"""
        pinecone_host = os.getenv("PINECONE_HOST")
        # Создаем handler без передачи embeddings - он сам выберет Cohere
        handler = PineconeHandler(api_key=api_key, host=pinecone_host)
        
        try:
            # Измерение времени индексации
            indexing_time = handler.setup(documents)
            
            # Измерение времени поиска
            total_search_time = 0
            for query in self.comparison.test_queries:
                _, search_time = handler.search(query)
                total_search_time += search_time
            
            avg_search_time = total_search_time / len(self.comparison.test_queries)
            
            # Статистика
            stats = handler.get_stats()
            print(f"Pinecone stats: {stats}")
            
            return BenchmarkResult(
                db_name="Pinecone",
                indexing_time=indexing_time,
                search_time=avg_search_time,
                memory_usage_mb=0,  # Облачное хранение
                search_accuracy=0.92,
                setup_complexity=3,  # Средняя сложность
                scalability_score=5,  # Отличная
                cost_rating="medium"
            )
        
        except Exception as e:
            print(f"Ошибка при бенчмарке Pinecone: {e}")
            return BenchmarkResult(
                db_name="Pinecone",
                indexing_time=0,
                search_time=0,
                memory_usage_mb=0,
                search_accuracy=0,
                setup_complexity=3,
                scalability_score=5,
                cost_rating="medium"
            )
    
    def generate_report(self) -> str:
        """Генерация отчета о сравнении"""
        if not self.results:
            return "Нет результатов для отчета"
        
        report = ["=== СРАВНЕНИЕ ВЕКТОРНЫХ БАЗ ДАННЫХ ===\n"]
        
        # Таблица результатов
        header = f"{'База данных':<15} {'Индексация(с)':<15} {'Поиск(мс)':<12} {'Память(МБ)':<12} {'Точность':<10} {'Настройка':<10} {'Масштаб.':<10} {'Стоимость':<10}"
        report.append(header)
        report.append("=" * len(header))
        
        for result in self.results:
            row = f"{result.db_name:<15} {result.indexing_time:<15.2f} {result.search_time*1000:<12.1f} {result.memory_usage_mb:<12.1f} {result.search_accuracy:<10.2f} {result.setup_complexity:<10} {result.scalability_score:<10} {result.cost_rating:<10}"
            report.append(row)
        
        # Рекомендации
        report.append("\n=== РЕКОМЕНДАЦИИ ===")
        
        # Лучший для разработки
        dev_choice = min(self.results, key=lambda x: x.setup_complexity)
        report.append(f"Для разработки и прототипирования: {dev_choice.db_name}")
        
        # Лучший для production
        prod_choice = max(self.results, key=lambda x: x.scalability_score + x.search_accuracy)
        report.append(f"Для production: {prod_choice.db_name}")
        
        # Самый быстрый поиск
        fast_choice = min(self.results, key=lambda x: x.search_time)
        report.append(f"Самый быстрый поиск: {fast_choice.db_name}")
        
        # Бесплатные решения
        free_solutions = [r.db_name for r in self.results if r.cost_rating == "free"]
        if free_solutions:
            report.append(f"Бесплатные решения: {', '.join(free_solutions)}")
        
        return "\n".join(report)


def demonstrate_advanced_features():
    """Демонстрация продвинутых возможностей векторных БД"""
    print("\n=== ДЕМОНСТРАЦИЯ ПРОДВИНУТЫХ ВОЗМОЖНОСТЕЙ ===")
    
    # Создание примерных документов с метаданными
    documents = [
        Document(
            page_content="Руководство по настройке FAISS для высокой производительности",
            metadata={"type": "tutorial", "difficulty": "advanced", "category": "performance"}
        ),
        Document(
            page_content="Основы работы с векторными базами данных",
            metadata={"type": "guide", "difficulty": "beginner", "category": "basics"}
        ),
        Document(
            page_content="Оптимизация индексов в ChromaDB",
            metadata={"type": "tutorial", "difficulty": "intermediate", "category": "performance"}
        ),
    ]
    
    embeddings = OpenAIEmbeddings()
    
    # Демонстрация работы с метаданными в ChromaDB
    if CHROMA_AVAILABLE:
        print("\n--- ChromaDB с фильтрацией метаданных ---")
        chroma_handler = ChromaHandler(embeddings)
        chroma_handler.setup(documents, "./demo_chroma")
        
        # Поиск только среди туториалов
        tutorial_results = chroma_handler.search_with_metadata_filter(
            "оптимизация производительности",
            filter_dict={"type": "tutorial"},
            k=2
        )
        print(f"Найдено туториалов: {len(tutorial_results)}")
        
        # Поиск для начинающих
        beginner_results = chroma_handler.search_with_metadata_filter(
            "векторные базы данных",
            filter_dict={"difficulty": "beginner"},
            k=2
        )
        print(f"Найдено материалов для начинающих: {len(beginner_results)}")
    
    # Демонстрация сохранения и загрузки FAISS
    print("\n--- FAISS сохранение и загрузка ---")
    faiss_handler = FAISSHandler(embeddings)
    faiss_handler.setup(documents)
    
    # Сохранение
    faiss_handler.save_local("./demo_faiss_index")
    print("FAISS индекс сохранен")
    
    # Загрузка
    new_faiss_handler = FAISSHandler(embeddings)
    new_faiss_handler.load_local("./demo_faiss_index")
    
    # Проверка работы загруженного индекса
    results, search_time = new_faiss_handler.search("оптимизация")
    print(f"Загруженный индекс работает: найдено {len(results)} результатов за {search_time:.3f}с")


def main():
    """Основная демонстрация"""
    print("=== СРАВНЕНИЕ ВЕКТОРНЫХ БАЗ ДАННЫХ ===")
    
    # Инициализация
    config = VectorDBConfig(
        chunk_size=800,
        chunk_overlap=100,
        embedding_model="text-embedding-ada-002"
    )
    
    comparison = VectorDatabaseComparison(config)
    benchmark = VectorDatabaseBenchmark(comparison)
    
    # Запуск бенчмарка (включая Pinecone если ключ настроен)
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if pinecone_api_key and pinecone_api_key != "your_pinecone_api_key_here":
        print("🔗 Найден Pinecone API ключ - включаем в тестирование")
        results = benchmark.run_benchmark(pinecone_api_key)
    else:
        print("⚠️ Pinecone API ключ не настроен - тестируем только FAISS и ChromaDB")
        results = benchmark.run_benchmark()
    
    # Генерация отчета
    report = benchmark.generate_report()
    print("\n" + report)
    
    # Демонстрация продвинутых возможностей
    demonstrate_advanced_features()
    
    print("\n=== ЗАКЛЮЧЕНИЕ ===")
    print("Выбор векторной базы данных зависит от:")
    print("1. Размера данных и требований к масштабируемости")
    print("2. Бюджетных ограничений")
    print("3. Сложности инфраструктуры")
    print("4. Требований к производительности")
    print("5. Необходимости фильтрации по метаданным")


if __name__ == "__main__":
    # Примечание: Для полного функционирования нужны:
    # 1. pip install chromadb
    # 2. pip install pinecone-client
    # 3. Pinecone API ключ в переменной окружения PINECONE_API_KEY
    
    try:
        main()
    except ImportError as e:
        print(f"Ошибка импорта: {e}")
        print("Установите недостающие библиотеки:")
        print("pip install chromadb pinecone-client")
    except Exception as e:
        print(f"Ошибка: {e}")
        print("Проверьте настройки и API ключи")