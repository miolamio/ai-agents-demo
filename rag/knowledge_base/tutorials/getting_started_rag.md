# Начало работы с RAG: Пошаговое руководство

## Введение

Это руководство поможет вам создать вашу первую RAG (Retrieval-Augmented Generation) систему с нуля. Мы пройдем все этапы от подготовки данных до создания работающего чат-бота.

## Что вам понадобится

### Предварительные требования
- Python 3.8+
- Базовые знания Python
- OpenAI API ключ
- 15-30 минут времени

### Установка зависимостей
```bash
pip install langchain langchain-community langchain-openai
pip install faiss-cpu chromadb
pip install python-dotenv
```

## Шаг 1: Подготовка окружения

### Создание .env файла
```bash
# .env
OPENAI_API_KEY=your_openai_api_key_here
```

### Базовая настройка
```python
import os
from dotenv import load_dotenv

load_dotenv()

# Проверка API ключа
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("Установите OPENAI_API_KEY в .env файле")

print("Окружение настроено успешно!")
```

## Шаг 2: Подготовка данных

### Создание тестовых документов
```python
# Создаем папку для документов
import os
os.makedirs("./documents", exist_ok=True)

# Создаем тестовый документ
with open("./documents/python_basics.md", "w", encoding="utf-8") as f:
    f.write("""
# Основы Python

## Переменные
В Python переменные создаются простым присваиванием:
```python
name = "Алексей"
age = 25
is_student = True
```

## Списки
Списки - это упорядоченные коллекции элементов:
```python
fruits = ["яблоко", "банан", "апельсин"]
numbers = [1, 2, 3, 4, 5]
```

## Функции
Функции определяются с помощью ключевого слова def:
```python
def greet(name):
    return f"Привет, {name}!"
```
""")

print("Тестовые документы созданы!")
```

## Шаг 3: Загрузка и обработка документов

### Загрузка документов
```python
from langchain_community.document_loaders import DirectoryLoader

# Загружаем все markdown файлы
loader = DirectoryLoader("./documents", glob="**/*.md")
documents = loader.load()

print(f"Загружено {len(documents)} документов")
for doc in documents:
    print(f"- {doc.metadata['source']}: {len(doc.page_content)} символов")
```

### Разбиение на чанки
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Настраиваем сплиттер
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # Размер чанка
    chunk_overlap=50,  # Перекрытие между чанками
    separators=["\n\n", "\n", " ", ""]  # Разделители
)

# Разбиваем документы
chunks = text_splitter.split_documents(documents)

print(f"Создано {len(chunks)} чанков")
for i, chunk in enumerate(chunks[:3]):  # Показываем первые 3
    print(f"\nЧанк {i+1}:")
    print(f"Содержимое: {chunk.page_content[:100]}...")
    print(f"Метаданные: {chunk.metadata}")
```

## Шаг 4: Создание векторной базы данных

### Вариант 1: Использование FAISS
```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# Создаем модель эмбеддингов
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# Создаем FAISS индекс
print("Создаем FAISS индекс...")
vectorstore = FAISS.from_documents(chunks, embeddings)

# Сохраняем индекс
vectorstore.save_local("./faiss_index")
print("FAISS индекс создан и сохранен!")

# Тестируем поиск
results = vectorstore.similarity_search("Как создать переменную в Python?", k=2)
print(f"\nНайдено {len(results)} релевантных чанков:")
for i, result in enumerate(results):
    print(f"{i+1}. {result.page_content[:100]}...")
```

### Вариант 2: Использование ChromaDB
```python
from langchain_community.vectorstores import Chroma

# Создаем ChromaDB индекс
print("Создаем ChromaDB индекс...")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db",
    collection_name="python_docs"
)

print("ChromaDB индекс создан!")

# Тестируем поиск с фильтрацией
results = vectorstore.similarity_search(
    "функции Python",
    k=2,
    filter={"source": "./documents/python_basics.md"}
)
print(f"\nНайдено {len(results)} релевантных чанков с фильтрацией:")
for result in results:
    print(f"- {result.page_content[:100]}...")
```

## Шаг 5: Создание QA системы

### Простая QA цепочка
```python
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

# Создаем языковую модель
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0  # Для более детерминированных ответов
)

# Создаем QA цепочку
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # Простейший тип цепочки
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    verbose=True  # Для отладки
)

print("QA система создана!")
```

### Тестирование системы
```python
# Функция для удобного тестирования
def ask_question(question):
    print(f"\n🤔 Вопрос: {question}")
    print("=" * 50)
    
    result = qa_chain({"query": question})
    
    print(f"🤖 Ответ: {result['result']}")
    print(f"\n📚 Использованные источники:")
    for i, doc in enumerate(result['source_documents']):
        print(f"{i+1}. {doc.metadata['source']}")
        print(f"   Содержимое: {doc.page_content[:150]}...")

# Тестируем различные вопросы
questions = [
    "Как создать переменную в Python?",
    "Что такое списки в Python?",
    "Покажи пример функции",
    "Как работают функции в Python?"
]

for question in questions:
    ask_question(question)
```

## Шаг 6: Улучшение системы

### Добавление памяти для диалогов
```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Создаем память для диалога
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Создаем диалоговую цепочку
conversation_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=memory,
    return_source_documents=True
)

print("Диалоговая система создана!")

# Функция для диалога
def chat():
    print("💬 Чат-бот готов! (введите 'выход' для завершения)")
    
    while True:
        question = input("\nВы: ")
        
        if question.lower() in ['выход', 'exit', 'quit']:
            break
            
        result = conversation_chain({"question": question})
        print(f"\nБот: {result['answer']}")

# Запускаем чат (раскомментируйте для интерактивного режима)
# chat()
```

### Кастомизация промптов
```python
from langchain.prompts import PromptTemplate

# Создаем кастомный промпт
custom_prompt = PromptTemplate(
    template="""Ты - полезный ассистент по программированию на Python.
    Используй следующий контекст для ответа на вопрос пользователя.
    Если ответа нет в контексте, честно скажи об этом.

    Контекст:
    {context}

    Вопрос: {question}

    Ответ на русском языке:""",
    input_variables=["context", "question"]
)

# Создаем цепочку с кастомным промптом
custom_qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": custom_prompt}
)

# Тестируем кастомную цепочку
result = custom_qa_chain({"query": "Как работают переменные?"})
print(f"Ответ с кастомным промптом: {result['result']}")
```

## Шаг 7: Мониторинг и оптимизация

### Логирование запросов
```python
import logging
from datetime import datetime

# Настраиваем логирование
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_system.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def ask_with_logging(question):
    start_time = datetime.now()
    logger.info(f"Получен вопрос: {question}")
    
    try:
        result = qa_chain({"query": question})
        duration = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Ответ сгенерирован за {duration:.2f}с")
        logger.info(f"Использовано {len(result['source_documents'])} документов")
        
        return result
    except Exception as e:
        logger.error(f"Ошибка при обработке вопроса: {e}")
        raise

# Используем функцию с логированием
result = ask_with_logging("Что такое Python?")
```

### Оценка качества
```python
def evaluate_answer(question, expected_keywords):
    """Простая оценка качества ответа"""
    result = qa_chain({"query": question})
    answer = result['result'].lower()
    
    found_keywords = [kw for kw in expected_keywords if kw.lower() in answer]
    score = len(found_keywords) / len(expected_keywords)
    
    print(f"Вопрос: {question}")
    print(f"Найдено ключевых слов: {found_keywords}")
    print(f"Оценка качества: {score:.2f}")
    
    return score

# Тестируем качество
test_cases = [
    ("Как создать переменную?", ["присваивание", "=", "переменная"]),
    ("Что такое список?", ["список", "коллекция", "элемент"]),
    ("Как определить функцию?", ["def", "функция", "return"])
]

total_score = 0
for question, keywords in test_cases:
    score = evaluate_answer(question, keywords)
    total_score += score
    print("-" * 40)

print(f"\nСредняя оценка качества: {total_score/len(test_cases):.2f}")
```

## Следующие шаги

### Что можно улучшить:
1. **Добавить больше документов** - расширить базу знаний
2. **Настроить чанкинг** - экспериментировать с размерами чанков
3. **Использовать гибридный поиск** - комбинировать векторный и keyword поиск
4. **Добавить переранжирование** - улучшить качество поиска
5. **Реализовать веб-интерфейс** - создать удобный UI

### Полезные ресурсы:
- [Документация LangChain](https://python.langchain.com/)
- [FAISS документация](https://faiss.ai/)
- [ChromaDB документация](https://docs.trychroma.com/)
- [OpenAI API документация](https://platform.openai.com/docs)

## Заключение

Поздравляем! Вы создали свою первую RAG систему. Теперь у вас есть:
- ✅ Система загрузки и обработки документов
- ✅ Векторная база данных для поиска
- ✅ QA система с возможностью диалога
- ✅ Инструменты для мониторинга и оценки

Экспериментируйте с различными настройками и добавляйте новые возможности!
