from pydantic_ai import Agent, RunContext

# Инициализируем агента с моделью (OpenAI GPT-3.5-турбо в данном случае) и системной инструкцией.
agent = Agent(
    model='openai:gpt-3.5-turbo',                     # Модель LLM (можно заменить на совместимую локальную через Ollama)
    system_prompt='Ты – помощник-калькулятор. Если вопрос содержит математику, вызови функцию calc для вычисления.', 
)

# Регистируем инструмент calc: он получает выражение и возвращает результат.
@agent.tool
def calc(ctx: RunContext[None], expression: str) -> float:
    """Вычисляет значение математического выражения."""
    try:
        result = eval(expression)   # В учебных целях используем eval для простоты.
    except Exception as e:
        raise ValueError(f"Ошибка вычисления: {e}")
    return result

# Примеры использования агента:
query = "Сколько будет 12 * (7 + 5)?"
result = agent.run_sync(query)
print(result.output)  # Ожидаемый ответ: 144.0 (с пояснениями модели)
