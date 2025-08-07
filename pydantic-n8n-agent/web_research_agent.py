"""
Web Research Agent using Pydantic AI
Эквивалент n8n workflow для веб-исследований с структурированным JSON выводом
"""

import os
import asyncio
from datetime import datetime
from typing import List, Dict, Optional, Literal, Union
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext, Tool
from pydantic_ai.messages import ModelMessage
from tavily import TavilyClient
import json

import logfire

# Шаг 0: Добавляем Logfire
logfire.configure(token="pylf_v1_us_DQN6NX83mbX9hcJz6yWWpRdgVFS47xxHJmKRzPhjfgsp") # Настраиваем Logfire
logfire.instrument_pydantic_ai() # Включаем автоматическую трассировку Pydantic!

# ============================================================================
# Конфигурация
# ============================================================================

# Установите переменные окружения или замените на свои ключи
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-openai-api-key")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "your-tavily-api-key")

# ============================================================================
# Модели данных для структурированного вывода (эквивалент Structured Output Parser)
# ============================================================================

class KeyFinding(BaseModel):
    """Ключевая находка из исследования"""
    title: str = Field(description="Заголовок находки")
    description: str = Field(description="Детальное описание")
    source: str = Field(description="URL источника или название")
    relevance_score: int = Field(ge=1, le=10, description="Оценка релевантности от 1 до 10")


class Categories(BaseModel):
    """Категории контента"""
    main_topic: str = Field(description="Основная тема")
    subtopics: List[str] = Field(description="Подтемы")


class Entities(BaseModel):
    """Извлеченные сущности"""
    people: List[str] = Field(default_factory=list, description="Люди")
    organizations: List[str] = Field(default_factory=list, description="Организации")
    locations: List[str] = Field(default_factory=list, description="Локации")
    dates: List[str] = Field(default_factory=list, description="Даты")


class ResearchOutput(BaseModel):
    """Структурированный вывод исследования (эквивалент JSON schema из n8n)"""
    query: str = Field(description="Оригинальный поисковый запрос")
    timestamp: str = Field(description="ISO 8601 временная метка")
    summary: str = Field(description="Краткий обзор находок в 2-3 предложения")
    key_findings: List[KeyFinding] = Field(description="Ключевые находки")
    categories: Categories = Field(description="Категории")
    entities: Entities = Field(description="Извлеченные сущности")
    sentiment: Literal["positive", "neutral", "negative"] = Field(description="Общий тон информации")
    confidence_level: Literal["high", "medium", "low"] = Field(description="Уровень уверенности")
    additional_queries: List[str] = Field(description="Предложения для дальнейшего исследования")


# ============================================================================
# Зависимости агента (эквивалент подключения к Tavily)
# ============================================================================

class AgentDependencies(BaseModel):
    """Зависимости для агента"""
    tavily_client: TavilyClient
    
    class Config:
        arbitrary_types_allowed = True


# ============================================================================
# Web Research Agent (основной агент)
# ============================================================================

# Системный промпт из n8n workflow
SYSTEM_PROMPT = """<AgentInstructions>
  <Role>
    <Name>Web Research Assistant</Name>
    <Description>You are an AI agent specialized in conducting thorough web research and returning results in a structured JSON format for easy integration with other systems.</Description>
    <currentDate>{current_date}</currentDate>
  </Role>
  
  <Goal>
    <Primary>Conduct comprehensive web research on any given topic and return findings in a strictly structured JSON format that can be easily parsed and processed by downstream applications.</Primary>
  </Goal>
  
  <Instructions>
    <!-- 1. Research Process -->
    <Instruction>
      Always begin by using the Tavily tool to search for relevant, up-to-date information. Conduct multiple searches if needed to gather comprehensive data from various perspectives and sources.
    </Instruction>
    
    <!-- 2. Output Structure -->
    <Instruction>
      You MUST return your response as a valid structured object with all required fields filled properly.
    </Instruction>
    
    <!-- 3. Research Guidelines -->
    <Instruction>
      Follow these research guidelines:
      - Verify information across multiple sources when possible
      - Focus on recent and authoritative sources
      - Extract key entities (people, places, organizations, dates)
      - Identify the main topic and related subtopics
      - Assess the overall sentiment of the information
      - Rate your confidence based on source quality and consistency
      - Suggest follow-up queries for deeper investigation
    </Instruction>
    
    <!-- 4. Tools Available -->
    <Instruction>
      Tools accessible to this Agent:
      1) tavily_search - for gathering web information from multiple search engines
    </Instruction>
    
    <!-- 5. Critical Requirements -->
    <Instruction>
      CRITICAL: 
      - Fill all required fields with appropriate values
      - Relevance scores must be integers between 1-10
      - confidence_level must be exactly "high", "medium", or "low"
      - sentiment must be exactly "positive", "neutral", or "negative"
      - ISO 8601 timestamp format: YYYY-MM-DDTHH:mm:ssZ
    </Instruction>
  </Instructions>
</AgentInstructions>"""

# Создание агента
research_agent = Agent(
    'openai:gpt-4o-mini',  # Эквивалент gpt-4.1-mini из n8n
    deps_type=AgentDependencies,
    output_type=ResearchOutput,  # Структурированный вывод
    system_prompt=SYSTEM_PROMPT.format(current_date=datetime.now().isoformat()),
)

# ============================================================================
# Инструменты агента (эквивалент Tavily Tool из n8n)
# ============================================================================

@research_agent.tool
async def tavily_search(
    ctx: RunContext[AgentDependencies], 
    query: str,
    search_depth: Literal["basic", "advanced"] = "basic",
    max_results: int = 10
) -> Dict:
    """
    Search the web using Tavily API.
    
    Args:
        query: Search query string
        search_depth: Depth of search - 'basic' or 'advanced'
        max_results: Maximum number of results to return
    
    Returns:
        Dictionary with search results
    """
    try:
        # Используем Tavily client из зависимостей
        response = ctx.deps.tavily_client.search(
            query=query,
            search_depth=search_depth,
            max_results=max_results,
            include_raw_content=True,
            include_domains=[],
            exclude_domains=[]
        )
        return response
    except Exception as e:
        return {
            "error": str(e),
            "results": []
        }


# ============================================================================
# Chat Memory Manager (эквивалент Simple Memory из n8n)
# ============================================================================

class ChatMemory:
    """Простой менеджер памяти для хранения истории сообщений"""
    
    def __init__(self, window_size: int = 10):
        self.messages: List[ModelMessage] = []
        self.window_size = window_size
    
    def add_messages(self, messages: List[ModelMessage]):
        """Добавить сообщения в память"""
        self.messages.extend(messages)
        # Ограничиваем размер окна памяти
        if len(self.messages) > self.window_size * 2:
            self.messages = self.messages[-self.window_size:]
    
    def get_messages(self) -> List[ModelMessage]:
        """Получить историю сообщений"""
        return self.messages
    
    def clear(self):
        """Очистить память"""
        self.messages = []


# ============================================================================
# Основной класс для работы с агентом (эквивалент Chat Trigger)
# ============================================================================

class WebResearchAssistant:
    """Класс для управления веб-исследовательским ассистентом"""
    
    def __init__(self, openai_api_key: str, tavily_api_key: str):
        """
        Инициализация ассистента
        
        Args:
            openai_api_key: API ключ OpenAI
            tavily_api_key: API ключ Tavily
        """
        # Устанавливаем API ключ OpenAI через переменную окружения
        os.environ["OPENAI_API_KEY"] = openai_api_key
        
        # Создаем клиент Tavily
        self.tavily_client = TavilyClient(api_key=tavily_api_key)
        
        # Создаем зависимости
        self.deps = AgentDependencies(tavily_client=self.tavily_client)
        
        # Инициализируем память
        self.memory = ChatMemory()
    
    async def process_message(self, user_message: str) -> ResearchOutput:
        """
        Обработать сообщение пользователя и получить структурированный ответ
        
        Args:
            user_message: Сообщение от пользователя
            
        Returns:
            Структурированный вывод ResearchOutput
        """
        # Получаем историю сообщений
        message_history = self.memory.get_messages()
        
        # Запускаем агента
        result = await research_agent.run(
            user_message,
            deps=self.deps,
            message_history=message_history
        )
        
        # Сохраняем новые сообщения в память
        self.memory.add_messages(result.new_messages())
        
        return result.output
    
    def process_message_sync(self, user_message: str) -> ResearchOutput:
        """
        Синхронная версия обработки сообщения
        
        Args:
            user_message: Сообщение от пользователя
            
        Returns:
            Структурированный вывод ResearchOutput
        """
        # Получаем историю сообщений
        message_history = self.memory.get_messages()
        
        # Запускаем агента синхронно
        result = research_agent.run_sync(
            user_message,
            deps=self.deps,
            message_history=message_history
        )
        
        # Сохраняем новые сообщения в память
        self.memory.add_messages(result.new_messages())
        
        return result.output
    
    def clear_memory(self):
        """Очистить память чата"""
        self.memory.clear()
    
    def get_json_output(self, output: ResearchOutput) -> str:
        """
        Преобразовать вывод в JSON строку
        
        Args:
            output: Структурированный вывод
            
        Returns:
            JSON строка
        """
        return output.model_dump_json(indent=2)


# ============================================================================
# Пример использования
# ============================================================================

async def main():
    """Пример использования Web Research Assistant"""
    
    # Создаем ассистента
    assistant = WebResearchAssistant(
        openai_api_key=OPENAI_API_KEY,
        tavily_api_key=TAVILY_API_KEY
    )
    
    # Пример запроса
    query = "Последние тренды в области искусственного интеллекта в августе 2025 года"
    
    print(f"🔍 Исследую: {query}\n")
    print("⏳ Обработка запроса...\n")
    
    # Обрабатываем запрос
    result = await assistant.process_message(query)
    
    # Выводим результат в формате JSON
    json_output = assistant.get_json_output(result)
    print("📊 Результат исследования:\n")
    print(json_output)
    
    # Можем также работать с отдельными полями
    print("\n📌 Ключевые находки:")
    for finding in result.key_findings:
        print(f"  • {finding.title} (Релевантность: {finding.relevance_score}/10)")
        print(f"    {finding.description[:100]}...")
    
    print(f"\n💡 Уровень уверенности: {result.confidence_level}")
    print(f"📈 Общий тон: {result.sentiment}")
    
    # Пример продолжения диалога с сохранением контекста
    follow_up = "Расскажи подробнее про мультимодальные модели"
    print(f"\n🔍 Дополнительный запрос: {follow_up}\n")
    
    follow_up_result = await assistant.process_message(follow_up)
    print("📊 Дополнительные результаты:")
    print(assistant.get_json_output(follow_up_result))


def main_sync():
    """Синхронная версия примера"""
    
    # Создаем ассистента
    assistant = WebResearchAssistant(
        openai_api_key=OPENAI_API_KEY,
        tavily_api_key=TAVILY_API_KEY
    )
    
    # Пример запроса
    query = "What are the latest developments in quantum computing?"
    
    print(f"🔍 Researching: {query}\n")
    print("⏳ Processing request...\n")
    
    # Обрабатываем запрос синхронно
    result = assistant.process_message_sync(query)
    
    # Выводим результат
    json_output = assistant.get_json_output(result)
    print("📊 Research Result:\n")
    print(json_output)


# ============================================================================
# Точка входа
# ============================================================================

if __name__ == "__main__":
    # Для асинхронного выполнения
    # asyncio.run(main())
    
    # Для синхронного выполнения
    main_sync()