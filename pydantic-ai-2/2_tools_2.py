from pydantic_ai import Agent
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool

# Создаем агента с инструментом интернет-поиска DuckDuckGo.
agent = Agent(
    model='openai:gpt-3.5-turbo',                   # Можно заменить на 'ollama:название-модели' для локальной модели Ollama
    tools=[duckduckgo_search_tool()],              # Подключаем инструмент веб-поиска
    system_prompt='Ты – интернет-поисковик. Используй DuckDuckGo для поиска и выдавай пользователю найденную информацию.',
)

# Запрос пользователя
query = "Какие фильмы стали самыми кассовыми в 2025 году?"
result = agent.run_sync(query)
print(result.output)
