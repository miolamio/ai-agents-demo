# LangChain API Reference для RAG

## Основные классы

### DocumentLoader
Базовый класс для загрузки документов различных форматов.

```python
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

# Загрузка PDF файлов
pdf_loader = PyPDFLoader("document.pdf")
documents = pdf_loader.load()

# Загрузка всех файлов из папки
dir_loader = DirectoryLoader("./docs", glob="**/*.md")
documents = dir_loader.load()
```

### TextSplitter
Разбивает документы на чанки для лучшего поиска.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]
)

chunks = splitter.split_documents(documents)
```

#### Параметры TextSplitter
- `chunk_size` (int): Максимальный размер чанка в символах
- `chunk_overlap` (int): Перекрытие между чанками
- `separators` (List[str]): Разделители в порядке приоритета

### Embeddings
Преобразует текст в векторные представления.

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    openai_api_key="your-api-key"
)

# Получение эмбеддинга для текста
vector = embeddings.embed_query("Пример текста")

# Получение эмбеддингов для списка документов
vectors = embeddings.embed_documents(["текст 1", "текст 2"])
```

#### Параметры OpenAIEmbeddings
- `model` (str): Модель для создания эмбеддингов
- `openai_api_key` (str): API ключ OpenAI
- `chunk_size` (int): Размер батча для обработки

### VectorStore
Хранилище векторов с возможностью поиска по сходству.

#### FAISS
```python
from langchain_community.vectorstores import FAISS

# Создание индекса из документов
vectorstore = FAISS.from_documents(documents, embeddings)

# Поиск похожих документов
results = vectorstore.similarity_search("запрос", k=4)

# Поиск с оценками схожести
results_with_scores = vectorstore.similarity_search_with_score("запрос", k=4)

# Сохранение и загрузка индекса
vectorstore.save_local("./faiss_index")
vectorstore = FAISS.load_local("./faiss_index", embeddings)
```

#### ChromaDB
```python
from langchain_community.vectorstores import Chroma

# Создание персистентного хранилища
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory="./chroma_db",
    collection_name="my_collection"
)

# Поиск с фильтрацией по метаданным
results = vectorstore.similarity_search(
    "запрос",
    k=4,
    filter={"source": "specific_document.pdf"}
)

# Добавление новых документов
vectorstore.add_documents(new_documents)
```

### RetrievalQA
Цепочка для вопросно-ответной системы с поиском.

```python
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4", temperature=0)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
    return_source_documents=True
)

# Выполнение запроса
result = qa_chain({"query": "Ваш вопрос"})
print(result["result"])
print(result["source_documents"])
```

#### Параметры RetrievalQA
- `chain_type` (str): Тип цепочки ("stuff", "map_reduce", "refine", "map_rerank")
- `return_source_documents` (bool): Возвращать ли исходные документы
- `verbose` (bool): Подробный вывод

## Продвинутые возможности

### EnsembleRetriever
Комбинирует несколько методов поиска.

```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

# BM25 поиск по ключевым словам
bm25_retriever = BM25Retriever.from_documents(documents)

# Векторный поиск
vector_retriever = vectorstore.as_retriever()

# Комбинированный поиск
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.5, 0.5]
)
```

### ConversationBufferMemory
Память для диалоговых систем.

```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Сохранение сообщения
memory.save_context(
    {"input": "Вопрос пользователя"},
    {"output": "Ответ системы"}
)
```

### ConversationSummaryBufferMemory
Память с автоматическим суммированием.

```python
from langchain.memory import ConversationSummaryBufferMemory

memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=1000,
    return_messages=True
)
```

## Обработка ошибок

### Типичные ошибки

#### RateLimitError
```python
try:
    result = qa_chain({"query": "вопрос"})
except Exception as e:
    if "rate limit" in str(e).lower():
        print("Превышен лимит запросов. Повторите позже.")
        time.sleep(60)
```

#### InvalidRequestError
```python
try:
    embeddings = OpenAIEmbeddings()
except Exception as e:
    if "invalid api key" in str(e).lower():
        print("Неверный API ключ OpenAI")
```

## Лучшие практики API

### 1. Управление API ключами
```python
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
```

### 2. Batch обработка
```python
# Обрабатывайте документы батчами
batch_size = 100
for i in range(0, len(documents), batch_size):
    batch = documents[i:i+batch_size]
    vectorstore.add_documents(batch)
```

### 3. Кеширование
```python
from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache

set_llm_cache(InMemoryCache())
```

### 4. Асинхронная обработка
```python
import asyncio
from langchain.callbacks import AsyncCallbackHandler

async def process_query(query):
    result = await qa_chain.acall({"query": query})
    return result
```

## Мониторинг и отладка

### Callbacks
```python
from langchain.callbacks import StdOutCallbackHandler

callback_handler = StdOutCallbackHandler()

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    callbacks=[callback_handler],
    verbose=True
)
```

### Логирование
```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Начинаем обработку запроса")
```
