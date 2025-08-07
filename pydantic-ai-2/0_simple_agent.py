from pydantic_ai import Agent

# Инициализируем агента с указанием модели и системного сообщения
agent = Agent(
    'openai:gpt-4',                              # (1) Используем модель GPT-4 через провайдер OpenAI
    system_prompt='Отвечай кратко и по существу.'  # (2) Статическая системная подсказка для модели
)

# Запускаем агента на запросе пользователя
result = agent.run_sync('Откуда произошло выражение "Hello, World!"?')  # (3)
print(result.output)
"""
The first known use of "hello, world" was in a 1974 textbook about the C programming language.
"""
