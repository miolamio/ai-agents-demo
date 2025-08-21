# FAISS и ChromaDB API Reference

## FAISS (Facebook AI Similarity Search)

### Основные классы

#### IndexFlatL2
Простейший индекс с точным поиском.

```python
import faiss
import numpy as np

# Создание индекса
dimension = 1536  # Размерность векторов OpenAI
index = faiss.IndexFlatL2(dimension)

# Добавление векторов
vectors = np.random.random((1000, dimension)).astype('float32')
index.add(vectors)

# Поиск
query_vector = np.random.random((1, dimension)).astype('float32')
distances, indices = index.search(query_vector, k=5)
```

#### IndexIVFFlat
Индекс с кластеризацией для ускорения поиска.

```python
# Создание индекса с кластерами
nlist = 100  # Количество кластеров
quantizer = faiss.IndexFlatL2(dimension)
index = faiss.IndexIVFFlat(quantizer, dimension, nlist)

# Обучение индекса
index.train(vectors)
index.add(vectors)

# Настройка поиска
index.nprobe = 10  # Количество кластеров для поиска
distances, indices = index.search(query_vector, k=5)
```

#### IndexHNSW
Граф-индекс для быстрого приближенного поиска.

```python
# Создание HNSW индекса
index = faiss.IndexHNSWFlat(dimension, 32)  # 32 - количество соединений

# Настройка параметров
index.hnsw.efConstruction = 40  # Параметр построения
index.hnsw.efSearch = 16  # Параметр поиска

index.add(vectors)
distances, indices = index.search(query_vector, k=5)
```

### LangChain интеграция

```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

# Создание из документов
vectorstore = FAISS.from_documents(documents, embeddings)

# Создание из текстов
texts = ["текст 1", "текст 2", "текст 3"]
vectorstore = FAISS.from_texts(texts, embeddings)

# Поиск
results = vectorstore.similarity_search("запрос", k=4)
results_with_scores = vectorstore.similarity_search_with_score("запрос", k=4)

# Сохранение и загрузка
vectorstore.save_local("./faiss_index")
new_vectorstore = FAISS.load_local("./faiss_index", embeddings)
```

### Оптимизация FAISS

#### Квантизация
```python
# Создание индекса с квантизацией
index = faiss.IndexIVFPQ(quantizer, dimension, nlist, 8, 8)
# 8 - количество бит на субвектор, 8 - количество субвекторов

index.train(vectors)
index.add(vectors)
```

#### GPU ускорение
```python
# Перенос на GPU
gpu_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, index)

# Поиск на GPU
distances, indices = gpu_index.search(query_vector, k=5)
```

## ChromaDB

### Основные операции

#### Создание клиента
```python
import chromadb
from chromadb.config import Settings

# Персистентный клиент
client = chromadb.PersistentClient(path="./chroma_db")

# Клиент в памяти
client = chromadb.Client()

# HTTP клиент
client = chromadb.HttpClient(host="localhost", port=8000)
```

#### Работа с коллекциями
```python
# Создание коллекции
collection = client.create_collection(
    name="my_collection",
    metadata={"description": "Моя коллекция документов"}
)

# Получение существующей коллекции
collection = client.get_collection(name="my_collection")

# Список всех коллекций
collections = client.list_collections()

# Удаление коллекции
client.delete_collection(name="my_collection")
```

#### Добавление данных
```python
# Добавление документов
collection.add(
    documents=["Документ 1", "Документ 2", "Документ 3"],
    metadatas=[
        {"source": "file1.pdf", "page": 1},
        {"source": "file1.pdf", "page": 2},
        {"source": "file2.pdf", "page": 1}
    ],
    ids=["doc1", "doc2", "doc3"]
)

# Добавление с эмбеддингами
collection.add(
    documents=["Документ 1"],
    embeddings=[[0.1, 0.2, 0.3, ...]],  # Ваши эмбеддинги
    ids=["doc1"]
)
```

#### Поиск
```python
# Поиск по запросу
results = collection.query(
    query_texts=["Мой запрос"],
    n_results=5,
    where={"source": "file1.pdf"}  # Фильтр по метаданным
)

# Поиск по эмбеддингу
results = collection.query(
    query_embeddings=[[0.1, 0.2, 0.3, ...]],
    n_results=5
)

# Поиск с включением различных полей
results = collection.query(
    query_texts=["запрос"],
    n_results=5,
    include=["documents", "metadatas", "distances"]
)
```

### LangChain интеграция

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

# Создание из документов
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory="./chroma_db",
    collection_name="my_collection"
)

# Поиск
results = vectorstore.similarity_search("запрос", k=4)

# Поиск с фильтрацией
results = vectorstore.similarity_search(
    "запрос",
    k=4,
    filter={"source": "specific_file.pdf"}
)

# Добавление документов
vectorstore.add_documents(new_documents)

# Обновление документа
vectorstore.update_document("doc_id", new_document)

# Удаление документов
vectorstore.delete(["doc_id1", "doc_id2"])
```

### Продвинутые возможности ChromaDB

#### Кастомные функции расстояния
```python
collection = client.create_collection(
    name="custom_distance",
    metadata={"hnsw:space": "cosine"}  # cosine, l2, ip
)
```

#### Фильтрация по метаданным
```python
# Различные операторы фильтрации
results = collection.query(
    query_texts=["запрос"],
    where={
        "$and": [
            {"source": {"$eq": "file1.pdf"}},
            {"page": {"$gt": 5}}
        ]
    }
)

# Операторы: $eq, $ne, $gt, $gte, $lt, $lte, $in, $nin
results = collection.query(
    query_texts=["запрос"],
    where={"category": {"$in": ["tech", "science"]}}
)
```

#### Получение статистики
```python
# Количество документов
count = collection.count()

# Получение всех документов
all_docs = collection.get()

# Получение конкретных документов
specific_docs = collection.get(
    ids=["doc1", "doc2"],
    include=["documents", "metadatas"]
)
```

## Сравнение производительности

### FAISS vs ChromaDB

| Критерий | FAISS | ChromaDB |
|----------|-------|----------|
| Скорость поиска | Очень высокая | Высокая |
| Память | Эффективная | Умеренная |
| Простота использования | Средняя | Высокая |
| Персистентность | Ручная | Автоматическая |
| Фильтрация метаданных | Ограниченная | Богатая |
| Масштабируемость | Отличная | Хорошая |

### Рекомендации по выбору

**Используйте FAISS если:**
- Нужна максимальная производительность
- Работаете с большими объемами данных (>1M векторов)
- Готовы управлять персистентностью вручную
- Не требуется сложная фильтрация по метаданным

**Используйте ChromaDB если:**
- Нужна простота разработки
- Важна фильтрация по метаданным
- Требуется автоматическая персистентность
- Работаете с малыми/средними объемами данных (<1M векторов)

## Примеры оптимизации

### FAISS оптимизация
```python
# Для больших датасетов
nlist = int(4 * math.sqrt(len(vectors)))  # Эвристика для количества кластеров
index = faiss.IndexIVFFlat(quantizer, dimension, nlist)

# Настройка поиска
index.nprobe = min(nlist, 32)  # Баланс точности и скорости
```

### ChromaDB оптимизация
```python
# Настройка HNSW параметров
collection = client.create_collection(
    name="optimized",
    metadata={
        "hnsw:construction_ef": 200,  # Качество построения
        "hnsw:M": 16,  # Количество соединений
        "hnsw:space": "cosine"
    }
)
```
