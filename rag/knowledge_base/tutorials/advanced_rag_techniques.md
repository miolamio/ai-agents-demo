# Продвинутые техники RAG

## Введение

После освоения базовых принципов RAG, пора изучить продвинутые техники, которые значительно улучшат качество вашей системы. В этом руководстве мы рассмотрим современные подходы к оптимизации RAG.

## 1. Гибридный поиск (Hybrid Search)

### Теория
Гибридный поиск комбинирует:
- **Семантический поиск** (векторный) - понимает смысл
- **Лексический поиск** (BM25) - ищет точные совпадения

### Реализация
```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

# Создаем BM25 ретривер
bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 3

# Создаем векторный ретривер
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Комбинируем с разными весами
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.4, 0.6]  # Больше вес векторному поиску
)

# Используем в QA цепочке
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=ensemble_retriever,
    return_source_documents=True
)
```

### Оптимизация весов
```python
def test_hybrid_weights(questions, ground_truth):
    """Тестирование разных весов для гибридного поиска"""
    weight_combinations = [
        [0.2, 0.8],  # Больше векторного
        [0.4, 0.6],  # Умеренно больше векторного
        [0.5, 0.5],  # Равные веса
        [0.6, 0.4],  # Больше BM25
        [0.8, 0.2]   # Намного больше BM25
    ]
    
    best_score = 0
    best_weights = None
    
    for weights in weight_combinations:
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=weights
        )
        
        score = evaluate_retriever(ensemble_retriever, questions, ground_truth)
        print(f"Веса {weights}: оценка {score:.3f}")
        
        if score > best_score:
            best_score = score
            best_weights = weights
    
    print(f"Лучшие веса: {best_weights} с оценкой {best_score:.3f}")
    return best_weights
```

## 2. Переранжирование (Reranking)

### Кросс-энкодерное переранжирование
```python
from sentence_transformers import CrossEncoder

class RerankerRetriever:
    def __init__(self, base_retriever, reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.base_retriever = base_retriever
        self.reranker = CrossEncoder(reranker_model)
    
    def get_relevant_documents(self, query, k=4):
        # Получаем больше документов для переранжирования
        initial_docs = self.base_retriever.get_relevant_documents(query, k=k*2)
        
        if not initial_docs:
            return []
        
        # Подготавливаем пары (запрос, документ)
        pairs = [[query, doc.page_content] for doc in initial_docs]
        
        # Получаем оценки релевантности
        scores = self.reranker.predict(pairs)
        
        # Сортируем по убыванию оценки
        doc_scores = list(zip(initial_docs, scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Возвращаем топ-k документов
        return [doc for doc, score in doc_scores[:k]]

# Использование
reranker_retriever = RerankerRetriever(ensemble_retriever)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=reranker_retriever,
    return_source_documents=True
)
```

### LLM-based переранжирование
```python
from langchain.prompts import PromptTemplate

class LLMReranker:
    def __init__(self, llm):
        self.llm = llm
        self.prompt = PromptTemplate(
            template="""Оцени релевантность документа для данного запроса по шкале от 0 до 10.

Запрос: {query}

Документ: {document}

Оценка (только число от 0 до 10):""",
            input_variables=["query", "document"]
        )
    
    def rerank(self, query, documents, top_k=4):
        scored_docs = []
        
        for doc in documents:
            prompt_text = self.prompt.format(query=query, document=doc.page_content[:500])
            
            try:
                score_text = self.llm.predict(prompt_text).strip()
                score = float(score_text)
                scored_docs.append((doc, score))
            except:
                # Если не удалось получить оценку, ставим среднюю
                scored_docs.append((doc, 5.0))
        
        # Сортируем по убыванию оценки
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, score in scored_docs[:top_k]]
```

## 3. Многоэтапный поиск (Multi-step Retrieval)

### HyDE (Hypothetical Document Embeddings)
```python
class HyDERetriever:
    def __init__(self, vectorstore, llm):
        self.vectorstore = vectorstore
        self.llm = llm
        self.hyde_prompt = PromptTemplate(
            template="""Напиши короткий параграф, который мог бы ответить на этот вопрос:

Вопрос: {question}

Параграф:""",
            input_variables=["question"]
        )
    
    def get_relevant_documents(self, query, k=4):
        # Генерируем гипотетический документ
        hypothetical_doc = self.llm.predict(
            self.hyde_prompt.format(question=query)
        )
        
        # Ищем по гипотетическому документу
        docs = self.vectorstore.similarity_search(hypothetical_doc, k=k)
        
        return docs

# Использование
hyde_retriever = HyDERetriever(vectorstore, llm)
```

### Мульти-запросный поиск
```python
class MultiQueryRetriever:
    def __init__(self, vectorstore, llm):
        self.vectorstore = vectorstore
        self.llm = llm
        self.query_prompt = PromptTemplate(
            template="""Ты - AI ассистент. Твоя задача - сгенерировать 3 различные версии данного пользовательского вопроса для поиска в векторной базе данных. Предоставь эти альтернативные вопросы, разделенные новыми строками.

Оригинальный вопрос: {question}

Альтернативные вопросы:""",
            input_variables=["question"]
        )
    
    def get_relevant_documents(self, query, k=4):
        # Генерируем альтернативные запросы
        alternative_queries = self.llm.predict(
            self.query_prompt.format(question=query)
        ).strip().split('\n')
        
        all_docs = []
        queries = [query] + [q.strip() for q in alternative_queries if q.strip()]
        
        # Ищем по каждому запросу
        for q in queries:
            docs = self.vectorstore.similarity_search(q, k=k//len(queries) + 1)
            all_docs.extend(docs)
        
        # Удаляем дубликаты
        unique_docs = []
        seen_content = set()
        
        for doc in all_docs:
            if doc.page_content not in seen_content:
                unique_docs.append(doc)
                seen_content.add(doc.page_content)
        
        return unique_docs[:k]
```

## 4. Адаптивное чанкирование

### Семантическое чанкирование
```python
from sentence_transformers import SentenceTransformer
import numpy as np

class SemanticChunker:
    def __init__(self, model_name="all-MiniLM-L6-v2", threshold=0.5):
        self.model = SentenceTransformer(model_name)
        self.threshold = threshold
    
    def split_text(self, text):
        sentences = text.split('. ')
        if len(sentences) < 2:
            return [text]
        
        # Получаем эмбеддинги для каждого предложения
        embeddings = self.model.encode(sentences)
        
        chunks = []
        current_chunk = [sentences[0]]
        
        for i in range(1, len(sentences)):
            # Вычисляем схожесть с предыдущим предложением
            similarity = np.dot(embeddings[i-1], embeddings[i]) / (
                np.linalg.norm(embeddings[i-1]) * np.linalg.norm(embeddings[i])
            )
            
            if similarity > self.threshold:
                current_chunk.append(sentences[i])
            else:
                # Начинаем новый чанк
                chunks.append('. '.join(current_chunk) + '.')
                current_chunk = [sentences[i]]
        
        # Добавляем последний чанк
        if current_chunk:
            chunks.append('. '.join(current_chunk) + '.')
        
        return chunks

# Использование
semantic_chunker = SemanticChunker(threshold=0.3)

def create_semantic_chunks(documents):
    all_chunks = []
    
    for doc in documents:
        chunks_text = semantic_chunker.split_text(doc.page_content)
        
        for chunk_text in chunks_text:
            chunk = Document(
                page_content=chunk_text,
                metadata=doc.metadata.copy()
            )
            all_chunks.append(chunk)
    
    return all_chunks

semantic_chunks = create_semantic_chunks(documents)
```

### Контекстно-осведомленное чанкирование
```python
class ContextAwareChunker:
    def __init__(self, chunk_size=1000, overlap=200):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def split_documents(self, documents):
        chunks = []
        
        for doc in documents:
            text = doc.page_content
            
            # Находим заголовки
            lines = text.split('\n')
            headers = []
            
            for i, line in enumerate(lines):
                if line.startswith('#'):
                    headers.append((i, line.strip()))
            
            # Создаем чанки с учетом структуры
            current_pos = 0
            
            while current_pos < len(text):
                chunk_end = min(current_pos + self.chunk_size, len(text))
                
                # Ищем ближайший заголовок для контекста
                context_header = ""
                for header_pos, header_text in headers:
                    header_char_pos = text.find(header_text)
                    if header_char_pos <= current_pos:
                        context_header = header_text
                
                chunk_text = text[current_pos:chunk_end]
                
                # Добавляем контекст заголовка
                if context_header and not chunk_text.startswith('#'):
                    chunk_text = f"Контекст: {context_header}\n\n{chunk_text}"
                
                chunk = Document(
                    page_content=chunk_text,
                    metadata={
                        **doc.metadata,
                        'chunk_id': len(chunks),
                        'context_header': context_header
                    }
                )
                chunks.append(chunk)
                
                current_pos += self.chunk_size - self.overlap
        
        return chunks
```

## 5. Фильтрация и постобработка

### Фильтрация по релевантности
```python
class RelevanceFilter:
    def __init__(self, threshold=0.7):
        self.threshold = threshold
    
    def filter_documents(self, query, documents_with_scores):
        filtered = []
        
        for doc, score in documents_with_scores:
            # Для FAISS score - это расстояние (меньше = лучше)
            # Преобразуем в схожесть
            similarity = 1 / (1 + score)
            
            if similarity >= self.threshold:
                filtered.append((doc, similarity))
        
        return filtered

# Интеграция с поиском
def search_with_relevance_filter(vectorstore, query, k=4, threshold=0.7):
    # Получаем больше документов для фильтрации
    docs_with_scores = vectorstore.similarity_search_with_score(query, k=k*2)
    
    # Фильтруем по релевантности
    relevance_filter = RelevanceFilter(threshold)
    filtered_docs = relevance_filter.filter_documents(query, docs_with_scores)
    
    # Возвращаем только документы
    return [doc for doc, score in filtered_docs[:k]]
```

### Дедупликация документов
```python
from difflib import SequenceMatcher

class DocumentDeduplicator:
    def __init__(self, similarity_threshold=0.8):
        self.similarity_threshold = similarity_threshold
    
    def deduplicate(self, documents):
        unique_docs = []
        
        for doc in documents:
            is_duplicate = False
            
            for unique_doc in unique_docs:
                similarity = SequenceMatcher(
                    None, 
                    doc.page_content, 
                    unique_doc.page_content
                ).ratio()
                
                if similarity > self.similarity_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_docs.append(doc)
        
        return unique_docs
```

## 6. Кеширование и оптимизация

### Кеширование эмбеддингов
```python
import pickle
import hashlib
from pathlib import Path

class EmbeddingCache:
    def __init__(self, cache_dir="./embedding_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def _get_cache_path(self, text):
        # Создаем хеш от текста
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return self.cache_dir / f"{text_hash}.pkl"
    
    def get_embedding(self, text, embedding_func):
        cache_path = self._get_cache_path(text)
        
        # Проверяем кеш
        if cache_path.exists():
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        
        # Вычисляем и кешируем
        embedding = embedding_func(text)
        
        with open(cache_path, 'wb') as f:
            pickle.dump(embedding, f)
        
        return embedding

# Использование
cache = EmbeddingCache()

def cached_embed_query(text):
    return cache.get_embedding(text, embeddings.embed_query)
```

### Асинхронная обработка
```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class AsyncRAGSystem:
    def __init__(self, qa_chain):
        self.qa_chain = qa_chain
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    async def process_query(self, query):
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor, 
            self.qa_chain, 
            {"query": query}
        )
        return result
    
    async def process_multiple_queries(self, queries):
        tasks = [self.process_query(query) for query in queries]
        results = await asyncio.gather(*tasks)
        return results

# Использование
async_rag = AsyncRAGSystem(qa_chain)

# Пример асинхронной обработки
async def main():
    queries = [
        "Что такое Python?",
        "Как создать функцию?",
        "Что такое списки?"
    ]
    
    results = await async_rag.process_multiple_queries(queries)
    
    for query, result in zip(queries, results):
        print(f"Q: {query}")
        print(f"A: {result['result'][:100]}...")
        print("-" * 50)

# asyncio.run(main())
```

## 7. Мониторинг и метрики

### Продвинутые метрики
```python
class RAGMetrics:
    def __init__(self):
        self.queries_processed = 0
        self.total_processing_time = 0
        self.retrieval_scores = []
        self.generation_scores = []
    
    def log_query(self, query, result, processing_time, retrieval_score=None):
        self.queries_processed += 1
        self.total_processing_time += processing_time
        
        if retrieval_score:
            self.retrieval_scores.append(retrieval_score)
    
    def get_stats(self):
        avg_time = self.total_processing_time / max(1, self.queries_processed)
        avg_retrieval_score = sum(self.retrieval_scores) / max(1, len(self.retrieval_scores))
        
        return {
            "total_queries": self.queries_processed,
            "avg_processing_time": avg_time,
            "avg_retrieval_score": avg_retrieval_score,
            "queries_per_minute": 60 / avg_time if avg_time > 0 else 0
        }

# Интеграция метрик
metrics = RAGMetrics()

def ask_with_metrics(question):
    start_time = time.time()
    
    result = qa_chain({"query": question})
    
    processing_time = time.time() - start_time
    
    # Простая оценка качества поиска
    retrieval_score = len(result['source_documents']) / 4.0  # Нормализуем к 1.0
    
    metrics.log_query(question, result, processing_time, retrieval_score)
    
    return result

# Использование
result = ask_with_metrics("Как работают переменные в Python?")
stats = metrics.get_stats()
print(f"Статистика системы: {stats}")
```

## Заключение

Продвинутые техники RAG позволяют:

1. **Улучшить качество поиска** - гибридный поиск, переранжирование
2. **Повысить релевантность** - мульти-запросы, HyDE
3. **Оптимизировать производительность** - кеширование, асинхронность
4. **Обеспечить мониторинг** - метрики, логирование

### Рекомендации по применению:

- Начинайте с простых техник и постепенно усложняйте
- Всегда измеряйте влияние каждого улучшения
- Учитывайте trade-off между качеством и производительностью
- Адаптируйте техники под ваши специфические данные

Эти техники помогут создать production-ready RAG систему высокого качества!
