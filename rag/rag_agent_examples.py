#!/usr/bin/env python3
"""
Практические примеры интеграции RAG с агентами
Демонстрация совместной работы поиска и генерации с агентным подходом
"""

import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Загрузка переменных окружения
from dotenv import load_dotenv
load_dotenv()

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS, Chroma
from langchain.chains import RetrievalQA
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory
from langchain.schema import Document
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.callbacks import StdOutCallbackHandler

# OpenRouter integration
class ChatOpenRouter(ChatOpenAI):
    """
    OpenRouter интеграция для LangChain
    Совместимый с OpenAI API провайдер с доступом к множественным моделям
    """
    def __init__(self, model_name: str = "anthropic/claude-3.5-sonnet", **kwargs):
        # Получаем API ключ из переменных окружения
        openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
        if not openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY не найден в переменных окружения")
        
        # Инициализируем родительский класс с OpenRouter настройками
        super().__init__(
            model=model_name,  # Используем model вместо model_name для новых версий
            openai_api_key=openrouter_api_key,
            openai_api_base="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "https://github.com/your-repo",  # Опционально для статистики
                "X-Title": "RAG Agent Examples",  # Опционально для статистики
            },
            **kwargs
        )

class OpenRouterEmbeddings(OpenAIEmbeddings):
    """
    OpenRouter эмбеддинги (используем OpenAI эмбеддинги через OpenRouter)
    """
    def __init__(self, model: str = "openai/text-embedding-ada-002", **kwargs):
        openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
        if not openrouter_api_key:
            # Fallback к OpenAI если OpenRouter ключ не найден
            openrouter_api_key = os.getenv('OPENAI_API_KEY')
            if not openrouter_api_key:
                raise ValueError("Ни OPENROUTER_API_KEY, ни OPENAI_API_KEY не найдены")
        
        super().__init__(
            model=model,
            openai_api_key=openrouter_api_key,
            openai_api_base="https://openrouter.ai/api/v1",
            **kwargs
        )

# Дополнительные библиотеки
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class AgentConfig:
    """Конфигурация RAG-агента"""
    # Модели для разных провайдеров
    model_name: str = "gpt-4o-mini"  # OpenAI модель по умолчанию
    openrouter_model: str = "anthropic/claude-3.5-sonnet"  # OpenRouter модель
    embedding_model: str = "text-embedding-ada-002"  # Стандартные эмбеддинги OpenAI
    openrouter_embedding_model: str = "openai/text-embedding-ada-002"  # Эмбеддинги через OpenRouter
    temperature: float = 0.1
    max_tokens: int = 2000
    chunk_size: int = 1000
    chunk_overlap: int = 200
    retrieval_k: int = 4
    similarity_threshold: float = 0.7
    use_openrouter: bool = False  # Флаг для выбора провайдера


class MultiSourceRAGAgent:
    """
    Агент с множественными источниками знаний и динамическим выбором
    """
    
    def __init__(self, config: AgentConfig = None):
        self.config = config or AgentConfig()
        self.knowledge_bases = {}
        
        # Выбираем провайдер в зависимости от конфигурации
        if self.config.use_openrouter:
            self.llm = ChatOpenRouter(
                model_name=self.config.openrouter_model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                request_timeout=30,
                max_retries=1
            )
        else:
            self.llm = ChatOpenAI(
                model=self.config.model_name,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                request_timeout=30,
                max_retries=1
            )
        # Используем простую память для избежания deprecation warnings
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.conversation_history = []
        
    def add_knowledge_base(self, name: str, documents_path: str, description: str):
        """Добавление источника знаний"""
        print(f"Создание базы знаний: {name}")
        
        # Загрузка документов
        if os.path.isfile(documents_path):
            if documents_path.endswith('.pdf'):
                loader = PyPDFLoader(documents_path)
            else:
                raise ValueError(f"Неподдерживаемый формат файла: {documents_path}")
        else:
            loader = DirectoryLoader(documents_path, glob="**/*.md")
            
        documents = loader.load()
        
        # Обработка текста
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        
        # Добавление метаданных
        for chunk in chunks:
            chunk.metadata['knowledge_base'] = name
            chunk.metadata['timestamp'] = datetime.now().isoformat()
        
        # Создание векторного хранилища
        if self.config.use_openrouter:
            embeddings = OpenRouterEmbeddings(model=self.config.openrouter_embedding_model)
        else:
            embeddings = OpenAIEmbeddings(model=self.config.embedding_model)
        
        vectorstore = FAISS.from_documents(chunks, embeddings)
        
        # Создание QA цепочки
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(
                search_kwargs={"k": self.config.retrieval_k}
            ),
            return_source_documents=True
        )
        
        self.knowledge_bases[name] = {
            "qa_chain": qa_chain,
            "vectorstore": vectorstore,
            "description": description,
            "document_count": len(chunks)
        }
        
        print(f"База знаний '{name}' создана: {len(chunks)} чанков")
    
    def classify_query(self, query: str) -> List[str]:
        """Классификация запроса для выбора релевантных источников"""
        classification_prompt = f"""
        Определи, какие источники знаний наиболее релевантны для данного запроса.
        
        Доступные источники:
        {chr(10).join([f"- {name}: {info['description']}" for name, info in self.knowledge_bases.items()])}
        
        Запрос пользователя: {query}
        
        Верни список названий релевантных источников через запятую.
        Если не уверен, верни все источники.
        """
        
        try:
            llm_response = self.llm.invoke(classification_prompt)
            # Проверяем тип ответа
            if hasattr(llm_response, 'content'):
                response = llm_response.content
            else:
                response = str(llm_response)
        except Exception as e:
            print(f"⚠️ Ошибка при классификации запроса: {e}")
            return list(self.knowledge_bases.keys())  # Возвращаем все источники при ошибке
        
        # Парсинг ответа
        suggested_sources = [s.strip() for s in response.split(",")]
        valid_sources = [s for s in suggested_sources if s in self.knowledge_bases]
        
        return valid_sources if valid_sources else list(self.knowledge_bases.keys())
    
    def search_knowledge_bases(self, query: str, sources: List[str] = None) -> Dict[str, Any]:
        """Поиск информации в указанных источниках"""
        if sources is None:
            sources = self.classify_query(query)
        
        results = {}
        all_documents = []
        
        for source_name in sources:
            if source_name in self.knowledge_bases:
                kb = self.knowledge_bases[source_name]
                result = kb["qa_chain"].invoke({"query": query})
                
                results[source_name] = {
                    "answer": result["result"],
                    "source_documents": result["source_documents"],
                    "confidence": self._calculate_confidence(result["source_documents"], query)
                }
                all_documents.extend(result["source_documents"])
        
        return {
            "source_results": results,
            "all_documents": all_documents,
            "sources_used": sources
        }
    
    def _calculate_confidence(self, documents: List[Document], query: str) -> float:
        """Вычисление уверенности на основе схожести документов с запросом"""
        if not documents:
            return 0.0
        
        try:
            # Упрощенный расчет уверенности
            # В реальной системе можно использовать более сложные метрики
            if self.config.use_openrouter:
                embeddings = OpenRouterEmbeddings(model=self.config.embedding_model)
            else:
                embeddings = OpenAIEmbeddings(model=self.config.embedding_model)
            
            query_embedding = embeddings.embed_query(query)
            
            similarities = []
            for doc in documents:
                doc_embedding = embeddings.embed_documents([doc.page_content])[0]
                similarity = cosine_similarity([query_embedding], [doc_embedding])[0][0]
                similarities.append(similarity)
            
            return float(np.mean(similarities)) if similarities else 0.0
        except Exception as e:
            print(f"⚠️ Ошибка при вычислении уверенности: {e}")
            return 0.5  # Возвращаем среднюю уверенность при ошибке
    
    def synthesize_answer(self, query: str, search_results: Dict[str, Any]) -> str:
        """Синтез финального ответа на основе результатов из разных источников"""
        source_answers = search_results["source_results"]
        
        if not source_answers:
            return "Извините, не удалось найти релевантную информацию для вашего запроса."
        
        if len(source_answers) == 1:
            # Один источник - возвращаем его ответ
            return list(source_answers.values())[0]["answer"]
        
        # Множественные источники - синтезируем ответ
        synthesis_context = []
        for source_name, result in source_answers.items():
            synthesis_context.append(f"Источник '{source_name}': {result['answer']}")
        
        synthesis_prompt = f"""
        На основе информации из различных источников дайте комплексный и точный ответ на вопрос.
        
        Вопрос: {query}
        
        Информация из источников:
        {chr(10).join(synthesis_context)}
        
        Требования к ответу:
        1. Объедините информацию из всех источников
        2. Укажите противоречия, если они есть
        3. Сделайте ответ структурированным и понятным
        4. Укажите источники для ключевых утверждений
        
        Комплексный ответ:
        """
        
        llm_response = self.llm.invoke(synthesis_prompt)
        if hasattr(llm_response, 'content'):
            return llm_response.content
        else:
            return str(llm_response)
    
    def chat(self, message: str) -> str:
        """Основной метод для общения с агентом"""
        # Добавление в историю
        self.conversation_history.append({
            "timestamp": datetime.now(),
            "user_message": message,
            "type": "user"
        })
        
        # Поиск в базах знаний
        search_results = self.search_knowledge_bases(message)
        
        # Формирование контекста для LLM
        context = self._format_context(message, search_results)
        
        # Генерация ответа с учётом памяти
        response_prompt = f"""
        Ты - полезный ИИ-ассистент с доступом к базам знаний.
        
        История разговора (если есть):
        {self._get_conversation_context()}
        
        Контекст из баз знаний:
        {context}
        
        Текущий вопрос пользователя: {message}
        
        Дай полный и точный ответ, основанный на предоставленной информации.
        Если информации недостаточно, честно об этом скажи.
        """
        
        try:
            llm_response = self.llm.invoke(response_prompt)
            if hasattr(llm_response, 'content'):
                response = llm_response.content
            else:
                response = str(llm_response)
        except Exception as e:
            print(f"⚠️ Ошибка при генерации ответа: {e}")
            return f"Извините, произошла ошибка при обработке вашего запроса: {str(e)}"
        
        # Добавление ответа в историю
        self.conversation_history.append({
            "timestamp": datetime.now(),
            "assistant_response": response,
            "sources_used": search_results["sources_used"],
            "type": "assistant"
        })
        
        return response
    
    def _format_context(self, query: str, search_results: Dict[str, Any]) -> str:
        """Форматирование контекста из результатов поиска"""
        if not search_results["source_results"]:
            return "Релевантная информация не найдена."
        
        formatted_context = []
        
        for source_name, result in search_results["source_results"].items():
            formatted_context.append(f"\n--- Источник: {source_name} ---")
            formatted_context.append(f"Ответ: {result['answer']}")
            formatted_context.append(f"Уверенность: {result['confidence']:.2f}")
            
            if result["source_documents"]:
                formatted_context.append("Документы:")
                for i, doc in enumerate(result["source_documents"][:2]):  # Показываем первые 2
                    formatted_context.append(f"  {i+1}. {doc.page_content[:200]}...")
        
        return "\n".join(formatted_context)
    
    def _get_conversation_context(self) -> str:
        """Получение контекста предыдущих сообщений"""
        if len(self.conversation_history) <= 2:
            return "Начало разговора."
        
        recent_messages = self.conversation_history[-6:]  # Последние 3 пары
        context = []
        
        for msg in recent_messages:
            if msg["type"] == "user":
                context.append(f"Пользователь: {msg['user_message']}")
            else:
                context.append(f"Ассистент: {msg['assistant_response'][:100]}...")
        
        return "\n".join(context)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Получение статистики работы агента"""
        stats = {
            "knowledge_bases": len(self.knowledge_bases),
            "total_documents": sum(kb["document_count"] for kb in self.knowledge_bases.values()),
            "conversation_turns": len([msg for msg in self.conversation_history if msg["type"] == "user"]),
            "knowledge_base_details": {}
        }
        
        for name, kb in self.knowledge_bases.items():
            stats["knowledge_base_details"][name] = {
                "description": kb["description"],
                "document_count": kb["document_count"]
            }
        
        return stats


class SmartRetrievalAgent:
    """
    Агент с умной системой поиска и адаптивным обучением
    """
    
    def __init__(self, config: AgentConfig = None):
        self.config = config or AgentConfig()
        
        # Выбираем провайдер
        if self.config.use_openrouter:
            self.llm = ChatOpenRouter(
                model_name=self.config.openrouter_model,
                temperature=0,
                request_timeout=30,
                max_retries=1
            )
        else:
            self.llm = ChatOpenAI(
                model=self.config.model_name,
                temperature=0,
                request_timeout=30,
                max_retries=1
            )
        self.vectorstore = None
        self.feedback_history = []
        self.query_success_rate = {}
        
    def setup_knowledge_base(self, documents_path: str):
        """Настройка базы знаний с адаптивными параметрами"""
        # Загрузка документов
        loader = DirectoryLoader(documents_path, glob="**/*.md")
        documents = loader.load()
        
        # Адаптивный чанкинг на основе типа документов
        chunk_size = self._determine_optimal_chunk_size(documents)
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=int(chunk_size * 0.2),
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        
        # Обогащение метаданных
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                "chunk_id": f"chunk_{i}",
                "doc_type": self._classify_document_type(chunk.page_content),
                "complexity_score": self._calculate_complexity_score(chunk.page_content),
                "created_at": datetime.now().isoformat()
            })
        
        # Создание векторного хранилища
        if self.config.use_openrouter:
            embeddings = OpenRouterEmbeddings(model=self.config.openrouter_embedding_model)
        else:
            embeddings = OpenAIEmbeddings(model=self.config.embedding_model)
        
        self.vectorstore = FAISS.from_documents(chunks, embeddings)
        
        print(f"База знаний создана: {len(chunks)} чанков с размером {chunk_size}")
    
    def _determine_optimal_chunk_size(self, documents: List[Document]) -> int:
        """Определение оптимального размера чанка на основе анализа документов"""
        # Анализ длины документов
        doc_lengths = [len(doc.page_content) for doc in documents]
        avg_length = np.mean(doc_lengths)
        
        if avg_length < 2000:
            return 500  # Короткие документы
        elif avg_length < 10000:
            return 1000  # Средние документы
        else:
            return 1500  # Длинные документы
    
    def _classify_document_type(self, content: str) -> str:
        """Классификация типа документа"""
        content_lower = content.lower()
        
        if "class " in content_lower or "def " in content_lower:
            return "code"
        elif "api" in content_lower or "endpoint" in content_lower:
            return "api_docs"
        elif "tutorial" in content_lower or "step" in content_lower:
            return "tutorial"
        else:
            return "general"
    
    def _calculate_complexity_score(self, content: str) -> float:
        """Вычисление оценки сложности текста"""
        # Упрощенная оценка на основе длины предложений и технических терминов
        sentences = content.split('.')
        avg_sentence_length = np.mean([len(s.split()) for s in sentences if s.strip()])
        
        technical_terms = ["api", "function", "class", "method", "parameter", "algorithm"]
        tech_term_count = sum(1 for term in technical_terms if term in content.lower())
        
        # Нормализованная оценка от 0 до 1
        complexity = min(1.0, (avg_sentence_length / 20) + (tech_term_count / len(technical_terms)))
        return complexity
    
    def intelligent_search(self, query: str, user_level: str = "intermediate") -> Dict[str, Any]:
        """Умный поиск с учётом уровня пользователя и контекста"""
        if not self.vectorstore:
            return {"error": "База знаний не инициализирована"}
        
        # Анализ сложности запроса
        query_complexity = self._analyze_query_complexity(query)
        
        # Определение количества документов для поиска
        k = self._determine_retrieval_k(query_complexity, user_level)
        
        # Поиск с фильтрацией по сложности
        complexity_filter = self._get_complexity_filter(user_level)
        
        # Выполнение поиска
        docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=k*2)
        
        # Фильтрация по сложности и релевантности
        filtered_docs = []
        for doc, score in docs_with_scores:
            doc_complexity = doc.metadata.get("complexity_score", 0.5)
            if self._is_suitable_complexity(doc_complexity, complexity_filter):
                filtered_docs.append((doc, score))
        
        # Ограничение количества результатов
        final_docs = filtered_docs[:k]
        
        if not final_docs:
            return {
                "answer": "Не найдено подходящих документов для вашего уровня.",
                "documents": [],
                "suggestion": "Попробуйте переформулировать вопрос или изменить уровень сложности."
            }
        
        # Формирование контекста и ответа
        context = self._build_adaptive_context(final_docs, user_level)
        answer = self._generate_adaptive_answer(query, context, user_level)
        
        return {
            "answer": answer,
            "documents": [doc for doc, _ in final_docs],
            "relevance_scores": [float(score) for _, score in final_docs],
            "query_complexity": query_complexity,
            "user_level": user_level
        }
    
    def _analyze_query_complexity(self, query: str) -> float:
        """Анализ сложности пользовательского запроса"""
        # Простые индикаторы сложности
        complex_words = ["implement", "architecture", "algorithm", "optimization", "advanced"]
        simple_words = ["what", "how", "example", "basic", "simple"]
        
        query_lower = query.lower()
        complexity_score = 0.5  # Базовая оценка
        
        for word in complex_words:
            if word in query_lower:
                complexity_score += 0.1
        
        for word in simple_words:
            if word in query_lower:
                complexity_score -= 0.1
        
        return max(0.1, min(1.0, complexity_score))
    
    def _determine_retrieval_k(self, query_complexity: float, user_level: str) -> int:
        """Определение количества документов для поиска"""
        base_k = {"beginner": 2, "intermediate": 4, "advanced": 6}
        k = base_k.get(user_level, 4)
        
        # Увеличиваем k для сложных запросов
        if query_complexity > 0.7:
            k += 2
        
        return k
    
    def _get_complexity_filter(self, user_level: str) -> tuple:
        """Получение фильтра сложности для уровня пользователя"""
        filters = {
            "beginner": (0.0, 0.4),
            "intermediate": (0.2, 0.8),
            "advanced": (0.6, 1.0)
        }
        return filters.get(user_level, (0.0, 1.0))
    
    def _is_suitable_complexity(self, doc_complexity: float, complexity_filter: tuple) -> bool:
        """Проверка подходящей сложности документа"""
        min_complexity, max_complexity = complexity_filter
        return min_complexity <= doc_complexity <= max_complexity
    
    def _build_adaptive_context(self, docs_with_scores: List[tuple], user_level: str) -> str:
        """Построение адаптивного контекста"""
        context_parts = []
        
        for i, (doc, score) in enumerate(docs_with_scores):
            doc_type = doc.metadata.get("doc_type", "general")
            complexity = doc.metadata.get("complexity_score", 0.5)
            
            context_parts.append(f"--- Документ {i+1} (тип: {doc_type}, сложность: {complexity:.1f}) ---")
            
            # Адаптация длины контекста для разных уровней
            if user_level == "beginner":
                content = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
            elif user_level == "intermediate":
                content = doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content
            else:  # advanced
                content = doc.page_content
            
            context_parts.append(content)
        
        return "\n\n".join(context_parts)
    
    def _generate_adaptive_answer(self, query: str, context: str, user_level: str) -> str:
        """Генерация адаптивного ответа"""
        level_instructions = {
            "beginner": "Объясни простым языком, используй аналогии, избегай сложной терминологии",
            "intermediate": "Дай подробное объяснение с примерами, используй техническую терминологию умеренно",
            "advanced": "Дай глубокий технический анализ, включи детали реализации и best practices"
        }
        
        instruction = level_instructions.get(user_level, level_instructions["intermediate"])
        
        prompt = f"""
        {instruction}.
        
        Контекст из документации:
        {context}
        
        Вопрос пользователя (уровень {user_level}): {query}
        
        Ответ:
        """
        
        llm_response = self.llm.invoke(prompt)
        if hasattr(llm_response, 'content'):
            return llm_response.content
        else:
            return str(llm_response)
    
    def learn_from_feedback(self, query: str, answer: str, feedback_score: int, comments: str = ""):
        """Обучение на основе обратной связи"""
        feedback_entry = {
            "query": query,
            "answer": answer,
            "score": feedback_score,  # 1-5
            "comments": comments,
            "timestamp": datetime.now().isoformat()
        }
        
        self.feedback_history.append(feedback_entry)
        
        # Обновление статистики успешности запросов
        query_type = self._classify_query_type(query)
        if query_type not in self.query_success_rate:
            self.query_success_rate[query_type] = []
        
        self.query_success_rate[query_type].append(feedback_score >= 4)  # 4-5 считаем успешными
        
        # Адаптивное обучение (упрощенный пример)
        if feedback_score < 3:
            print(f"Низкая оценка для запроса типа '{query_type}'. Требует улучшения.")
    
    def _classify_query_type(self, query: str) -> str:
        """Классификация типа запроса для аналитики"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["how to", "tutorial", "guide"]):
            return "how-to"
        elif any(word in query_lower for word in ["what is", "define", "explain"]):
            return "definition"
        elif any(word in query_lower for word in ["error", "bug", "problem", "fix"]):
            return "troubleshooting"
        elif any(word in query_lower for word in ["example", "sample", "demo"]):
            return "example"
        else:
            return "general"
    
    def get_learning_analytics(self) -> Dict[str, Any]:
        """Получение аналитики обучения"""
        if not self.feedback_history:
            return {"message": "Нет данных для анализа"}
        
        # Общая статистика
        total_feedback = len(self.feedback_history)
        avg_score = np.mean([f["score"] for f in self.feedback_history])
        
        # Статистика по типам запросов
        type_stats = {}
        for query_type, success_rates in self.query_success_rate.items():
            type_stats[query_type] = {
                "total_queries": len(success_rates),
                "success_rate": np.mean(success_rates) * 100,
                "success_count": sum(success_rates)
            }
        
        # Последние низкие оценки
        low_scores = [f for f in self.feedback_history[-10:] if f["score"] < 3]
        
        return {
            "total_feedback": total_feedback,
            "average_score": round(avg_score, 2),
            "query_type_performance": type_stats,
            "recent_low_scores": low_scores,
            "improvement_suggestions": self._generate_improvement_suggestions()
        }
    
    def _generate_improvement_suggestions(self) -> List[str]:
        """Генерация предложений по улучшению"""
        suggestions = []
        
        for query_type, success_rates in self.query_success_rate.items():
            success_rate = np.mean(success_rates) * 100
            if success_rate < 70:
                suggestions.append(f"Улучшить обработку запросов типа '{query_type}' (успешность: {success_rate:.1f}%)")
        
        return suggestions


def demo_multi_source_agent():
    """Демонстрация работы агента с множественными источниками"""
    print("=== Демонстрация MultiSourceRAGAgent с OpenRouter ===")
    
    # Проверяем наличие API ключей
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if not openrouter_key and not openai_key:
        print("⚠️  Ни OPENROUTER_API_KEY, ни OPENAI_API_KEY не установлены.")
        print("Установите хотя бы один из них в .env файле для полной функциональности.")
        return
    
    # Определяем какой провайдер использовать
    # Временно принудительно используем OpenAI для стабильности
    use_openrouter = False  # bool(openrouter_key)
    provider_name = "OpenRouter" if use_openrouter else "OpenAI"
    print(f"🚀 Используем провайдер: {provider_name}")
    
    try:
        # Создание агента с правильной конфигурацией
        config = AgentConfig(use_openrouter=use_openrouter)
        agent = MultiSourceRAGAgent(config)
        
        # Добавление реальных источников знаний
        print("Добавление баз знаний...")
        
        knowledge_base_path = os.path.join(os.path.dirname(__file__), "knowledge_base")
        
        # Проверяем существование папки
        if not os.path.exists(knowledge_base_path):
            print(f"❌ Папка с документами не найдена: {knowledge_base_path}")
            print("Создайте папку knowledge_base с подпапками technical_docs, api_reference, tutorials")
            return
        
        # Добавляем источники знаний
        sources = [
            ("technical_docs", os.path.join(knowledge_base_path, "technical_docs"), 
             "Техническая документация по векторным БД и RAG архитектуре"),
            ("api_reference", os.path.join(knowledge_base_path, "api_reference"), 
             "API документация LangChain, FAISS и ChromaDB"),
            ("tutorials", os.path.join(knowledge_base_path, "tutorials"), 
             "Пошаговые руководства и продвинутые техники")
        ]
        
        for name, path, description in sources:
            if os.path.exists(path) and os.listdir(path):
                agent.add_knowledge_base(name, path, description)
            else:
                print(f"⚠️  Папка {path} пуста или не существует")
        
        # Реальный диалог с агентом (сокращенная версия для быстрой демонстрации)
        queries = [
            "Как создать FAISS индекс?",
            "Что такое ChromaDB?",
            "Основы RAG архитектуры"
        ]
        
        for query in queries:
            print(f"\n🤔 Пользователь: {query}")
            print("=" * 60)
            
            try:
                import signal
                import time
                
                def timeout_handler(signum, frame):
                    raise TimeoutError("Запрос превысил время ожидания")
                
                # Устанавливаем таймаут на 45 секунд
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(45)
                
                start_time = time.time()
                response = agent.chat(query)
                elapsed_time = time.time() - start_time
                
                signal.alarm(0)  # Отключаем таймаут
                
                print(f"🤖 Ассистент ({elapsed_time:.1f}с): {response}")
                
            except TimeoutError:
                print(f"⏰ Запрос превысил время ожидания (45с). Пропускаем...")
                signal.alarm(0)
            except KeyboardInterrupt:
                print(f"⏹️  Пользователь прервал выполнение")
                signal.alarm(0)
                break
            except Exception as e:
                print(f"❌ Ошибка при обработке запроса: {e}")
                signal.alarm(0)
        
        # Статистика
        stats = agent.get_statistics()
        print(f"\n📊 Статистика работы агента:")
        print(f"- Источников знаний: {stats['knowledge_bases']}")
        print(f"- Общее количество документов: {stats['total_documents']}")
        print(f"- Обработано запросов: {stats['conversation_turns']}")
        
        print("\n📚 Детали баз знаний:")
        for name, details in stats['knowledge_base_details'].items():
            print(f"  • {name}: {details['document_count']} документов")
            print(f"    Описание: {details['description']}")
            
    except Exception as e:
        print(f"❌ Ошибка при создании агента: {e}")
        print("Убедитесь, что установлены все зависимости и настроен OPENAI_API_KEY")


def demo_smart_retrieval_agent():
    """Демонстрация работы умного агента поиска с реальными данными"""
    print("\n=== Демонстрация SmartRetrievalAgent с OpenRouter ===")
    
    # Проверяем наличие API ключей
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if not openrouter_key and not openai_key:
        print("⚠️  Ни OPENROUTER_API_KEY, ни OPENAI_API_KEY не установлены.")
        return
    
    # Временно принудительно используем OpenAI для стабильности
    use_openrouter = False  # bool(openrouter_key)
    provider_name = "OpenRouter" if use_openrouter else "OpenAI"
    print(f"🚀 Используем провайдер: {provider_name}")
    
    try:
        config = AgentConfig(use_openrouter=use_openrouter)
        agent = SmartRetrievalAgent(config)
        
        # Настройка базы знаний с реальными документами
        knowledge_base_path = os.path.join(os.path.dirname(__file__), "knowledge_base")
        
        if os.path.exists(knowledge_base_path):
            agent.setup_knowledge_base(knowledge_base_path)
            print("✅ База знаний настроена с реальными документами")
        else:
            print("❌ Папка knowledge_base не найдена. Создайте её с документами.")
            return
        
        # Демонстрация адаптивного поиска для разных уровней пользователей
        test_queries = [
            ("Что такое векторная база данных?", "beginner"),
            ("Как настроить FAISS индекс для оптимальной производительности?", "intermediate"),
            ("Реализация кастомного переранжирования с использованием кросс-энкодеров", "advanced"),
            ("Объясни принципы работы RAG архитектуры", "intermediate")
        ]
        
        for query, level in test_queries:
            print(f"\n🎯 Запрос ({level}): {query}")
            print("=" * 70)
            
            try:
                result = agent.intelligent_search(query, level)
                
                if 'error' in result:
                    print(f"❌ {result['error']}")
                else:
                    print(f"📝 Ответ: {result['answer'][:300]}...")
                    print(f"📊 Сложность запроса: {result['query_complexity']:.2f}")
                    print(f"👤 Уровень пользователя: {result['user_level']}")
                    print(f"📚 Найдено документов: {len(result['documents'])}")
                    
                    if result['relevance_scores']:
                        avg_relevance = sum(result['relevance_scores']) / len(result['relevance_scores'])
                        print(f"🎯 Средняя релевантность: {avg_relevance:.3f}")
                        
            except Exception as e:
                print(f"❌ Ошибка при обработке запроса: {e}")
        
        # Демонстрация обучения на основе обратной связи
        print("\n🧠 Демонстрация обучения на основе обратной связи:")
        print("=" * 50)
        
        feedback_examples = [
            ("Что такое эмбеддинги?", "Эмбеддинги - это векторные представления текста...", 5, "Отличное объяснение!"),
            ("Как работает HNSW алгоритм?", "HNSW это граф...", 4, "Хорошо, но можно больше деталей"),
            ("Настройка ChromaDB", "Неполный ответ", 2, "Нужно больше практических примеров"),
            ("API LangChain", "Подробное объяснение с примерами кода", 5, "Очень полезно!"),
            ("Оптимизация поиска", "Базовая информация", 3, "Средне, нужно больше конкретики")
        ]
        
        for query, answer, score, comment in feedback_examples:
            agent.learn_from_feedback(query, answer, score, comment)
            print(f"✅ Обратная связь записана для: '{query}' (оценка: {score}/5)")
        
        # Получение аналитики обучения
        analytics = agent.get_learning_analytics()
        
        if 'message' in analytics:
            print(f"\n📈 {analytics['message']}")
        else:
            print(f"\n📈 Аналитика обучения:")
            print(f"  • Всего отзывов: {analytics['total_feedback']}")
            print(f"  • Средняя оценка: {analytics['average_score']}/5")
            
            print(f"\n📊 Производительность по типам запросов:")
            for query_type, stats in analytics['query_type_performance'].items():
                print(f"  • {query_type}: {stats['success_rate']:.1f}% успешности ({stats['success_count']}/{stats['total_queries']})")
            
            if analytics['recent_low_scores']:
                print(f"\n⚠️  Недавние низкие оценки:")
                for feedback in analytics['recent_low_scores']:
                    print(f"  • '{feedback['query']}': {feedback['score']}/5 - {feedback['comments']}")
            
            if analytics['improvement_suggestions']:
                print(f"\n💡 Предложения по улучшению:")
                for suggestion in analytics['improvement_suggestions']:
                    print(f"  • {suggestion}")
                    
    except Exception as e:
        print(f"❌ Ошибка при создании агента: {e}")
        print("Убедитесь, что установлены все зависимости и настроен OPENAI_API_KEY")


def interactive_chat_demo():
    """Интерактивный чат с RAG агентом"""
    print("\n=== Интерактивный чат с RAG агентом (OpenRouter) ===")
    
    # Проверяем API ключи
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if not openrouter_key and not openai_key:
        print("⚠️  Ни OPENROUTER_API_KEY, ни OPENAI_API_KEY не установлены.")
        return
    
    # Временно принудительно используем OpenAI для стабильности
    use_openrouter = False  # bool(openrouter_key)
    provider_name = "OpenRouter" if use_openrouter else "OpenAI"
    print(f"🚀 Используем провайдер: {provider_name}")
    
    try:
        print("Инициализация агента...")
        config = AgentConfig(use_openrouter=use_openrouter)
        agent = MultiSourceRAGAgent(config)
        
        # Настройка баз знаний
        knowledge_base_path = os.path.join(os.path.dirname(__file__), "knowledge_base")
        
        if not os.path.exists(knowledge_base_path):
            print("❌ Папка knowledge_base не найдена")
            return
        
        sources = [
            ("technical_docs", os.path.join(knowledge_base_path, "technical_docs"), 
             "Техническая документация"),
            ("api_reference", os.path.join(knowledge_base_path, "api_reference"), 
             "API документация"),
            ("tutorials", os.path.join(knowledge_base_path, "tutorials"), 
             "Руководства и туториалы")
        ]
        
        for name, path, description in sources:
            if os.path.exists(path) and os.listdir(path):
                agent.add_knowledge_base(name, path, description)
        
        print("✅ Агент готов к работе!")
        print("\n💬 Начинайте задавать вопросы (введите 'выход' для завершения)")
        print("Примеры вопросов:")
        print("- Как создать FAISS индекс?")
        print("- Что такое RAG архитектура?")
        print("- Покажи пример использования ChromaDB")
        print("- Какие есть продвинутые техники RAG?")
        
        while True:
            try:
                user_input = input("\n🤔 Вы: ").strip()
                
                if user_input.lower() in ['выход', 'exit', 'quit', 'q']:
                    print("👋 До свидания!")
                    break
                
                if not user_input:
                    continue
                
                print("🤖 Обрабатываю запрос...")
                response = agent.chat(user_input)
                print(f"🤖 Ассистент: {response}")
                
            except KeyboardInterrupt:
                print("\n\n👋 Чат завершен пользователем")
                break
            except Exception as e:
                print(f"❌ Ошибка: {e}")
                print("Попробуйте еще раз")
        
        # Показываем статистику сессии
        stats = agent.get_statistics()
        print(f"\n📊 Статистика сессии:")
        print(f"- Обработано запросов: {stats['conversation_turns']}")
        print(f"- Использовано источников: {stats['knowledge_bases']}")
        
    except Exception as e:
        print(f"❌ Ошибка при запуске чата: {e}")


def comparison_demo():
    """Демонстрация сравнения FAISS и ChromaDB"""
    print("\n=== Сравнение FAISS и ChromaDB ===")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY не установлен")
        return
    
    try:
        from langchain_openai import OpenAIEmbeddings
        from langchain_community.vectorstores import FAISS, Chroma
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
        
        # Выбираем провайдер эмбеддингов
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        # Временно принудительно используем OpenAI для стабильности
        if False and openrouter_key:  # Отключаем OpenRouter временно
            embeddings = OpenRouterEmbeddings(model="openai/text-embedding-ada-002")
            print("🔗 Используем OpenRouter для эмбеддингов")
        else:
            embeddings = OpenAIEmbeddings()
            print("🔗 Используем OpenAI для эмбеддингов")
        
        # Тестируем FAISS
        print("\n🚀 Тестируем FAISS...")
        start_time = time.time()
        faiss_store = FAISS.from_documents(chunks, embeddings)
        faiss_build_time = time.time() - start_time
        
        # Тестируем поиск в FAISS
        start_time = time.time()
        faiss_results = faiss_store.similarity_search("векторная база данных", k=3)
        faiss_search_time = time.time() - start_time
        
        print(f"⏱️  FAISS - Построение: {faiss_build_time:.2f}с, Поиск: {faiss_search_time:.3f}с")
        print(f"📄 Найдено результатов: {len(faiss_results)}")
        
        # Тестируем ChromaDB
        print("\n🚀 Тестируем ChromaDB...")
        start_time = time.time()
        chroma_store = Chroma.from_documents(
            chunks, 
            embeddings,
            persist_directory="./demo_chroma_comparison",
            collection_name="comparison_test"
        )
        chroma_build_time = time.time() - start_time
        
        # Тестируем поиск в ChromaDB
        start_time = time.time()
        chroma_results = chroma_store.similarity_search("векторная база данных", k=3)
        chroma_search_time = time.time() - start_time
        
        print(f"⏱️  ChromaDB - Построение: {chroma_build_time:.2f}с, Поиск: {chroma_search_time:.3f}с")
        print(f"📄 Найдено результатов: {len(chroma_results)}")
        
        # Сравнение результатов
        print(f"\n📊 Сравнение:")
        print(f"{'Метрика':<20} {'FAISS':<15} {'ChromaDB':<15}")
        print("-" * 50)
        print(f"{'Время построения':<20} {faiss_build_time:.2f}с{'':<10} {chroma_build_time:.2f}с")
        print(f"{'Время поиска':<20} {faiss_search_time:.3f}с{'':<10} {chroma_search_time:.3f}с")
        
        # Показываем примеры результатов
        print(f"\n📝 Пример результата FAISS:")
        if faiss_results:
            print(f"   {faiss_results[0].page_content[:150]}...")
        
        print(f"\n📝 Пример результата ChromaDB:")
        if chroma_results:
            print(f"   {chroma_results[0].page_content[:150]}...")
        
    except Exception as e:
        print(f"❌ Ошибка при сравнении: {e}")


def demo_openrouter_models():
    """Демонстрация различных моделей через OpenRouter"""
    import time
    
    print("\n=== Демонстрация различных моделей OpenRouter ===")
    
    if not os.getenv("OPENROUTER_API_KEY"):
        print("⚠️  OPENROUTER_API_KEY не установлен.")
        print("📝 Демонстрируем доступные модели без API вызовов:")
        
        models_info = [
            ("anthropic/claude-3.5-sonnet", "Claude 3.5 Sonnet", "Высокое качество, отлично для сложных задач"),
            ("openai/gpt-4o-mini", "GPT-4o Mini", "Быстрая и дешевая модель OpenAI"),
            ("meta-llama/llama-3.1-8b-instruct", "Llama 3.1 8B", "Open source, часто бесплатная"),
            ("google/gemini-flash-1.5", "Gemini Flash", "Быстрая модель Google"),
            ("mistralai/mistral-7b-instruct", "Mistral 7B", "Эффективная европейская модель"),
            ("anthropic/claude-3-haiku", "Claude 3 Haiku", "Самая быстрая модель Anthropic")
        ]
        
        print("\n🤖 Доступные модели OpenRouter:")
        for model_id, name, description in models_info:
            print(f"   • {name}")
            print(f"     ID: {model_id}")
            print(f"     Описание: {description}")
            print()
        
        print("💡 Для тестирования установите OPENROUTER_API_KEY в .env файле")
        return
    
    # Список популярных моделей OpenRouter
    models_to_test = [
        ("anthropic/claude-3.5-sonnet", "Claude 3.5 Sonnet - Высокое качество"),
        ("openai/gpt-4o-mini", "GPT-4o Mini - Быстрая и дешевая"),
        ("meta-llama/llama-3.1-8b-instruct", "Llama 3.1 8B - Open source"),
        ("google/gemini-flash-1.5", "Gemini Flash - Быстрая модель Google")
    ]
    
    test_query = "Что такое векторная база данных?"
    
    for model_name, description in models_to_test:
        print(f"\n🤖 Тестируем: {description}")
        print(f"   Модель: {model_name}")
        
        try:
            # Создаем конфигурацию для конкретной модели
            config = AgentConfig(
                model_name=model_name,
                use_openrouter=True,
                max_tokens=500  # Короткие ответы для демо
            )
            
            # Создаем LLM напрямую для быстрого тестирования
            llm = ChatOpenRouter(
                model_name=model_name,
                temperature=0.1,
                max_tokens=500,
                request_timeout=20,
                max_retries=1
            )
            
            start_time = time.time()
            response = llm.invoke(f"Кратко объясни: {test_query}").content
            elapsed_time = time.time() - start_time
            
            print(f"   ⏱️  Время ответа: {elapsed_time:.1f}с")
            print(f"   📝 Ответ: {response[:150]}...")
            
        except Exception as e:
            print(f"   ❌ Ошибка: {e}")
        
        print("-" * 50)


if __name__ == "__main__":
    import sys
    
    print("🚀 RAG Agent Examples - Демонстрация с реальными данными")
    print("=" * 60)
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        if mode == 'chat':
            interactive_chat_demo()
        elif mode == 'compare':
            comparison_demo()
        elif mode == 'multi':
            demo_multi_source_agent()
        elif mode == 'smart':
            demo_smart_retrieval_agent()
        elif mode == 'models':
            demo_openrouter_models()
        else:
            print(f"Неизвестный режим: {mode}")
            print("Доступные режимы: chat, compare, multi, smart, models")
    else:
        # Запускаем все демо по очереди
        demo_multi_source_agent()
        demo_smart_retrieval_agent()
        
        print("\n" + "=" * 60)
        print("✨ Дополнительные режимы:")
        print("  python rag_agent_examples.py chat     - Интерактивный чат")
        print("  python rag_agent_examples.py compare  - Сравнение FAISS vs ChromaDB")
        print("  python rag_agent_examples.py multi    - Только MultiSource демо")
        print("  python rag_agent_examples.py smart    - Только Smart Retrieval демо")
        print("  python rag_agent_examples.py models   - Тестирование разных моделей OpenRouter")
    
    print("\n=== Заключение ===")
    print("Примеры демонстрируют:")
    print("1. Интеграцию RAG с агентным подходом")
    print("2. Работу с множественными источниками знаний")
    print("3. Адаптивный поиск и персонализацию")
    print("4. Обучение на основе обратной связи")
    print("5. Аналитику и мониторинг качества")
    print("6. Интерактивное взаимодействие с пользователем")