from pydantic_ai import Agent, RunContext, ModelRetry
from pydantic import BaseModel

# 1. Имитация базы данных пользователей (по имени выдаёт ID).
class UserDatabase:
    def __init__(self):
        self.users = {"John Doe": 123, "Alice Smith": 456}

# 2. Pydantic-модель результата: кому (ID) и текст сообщения.
class MessageResult(BaseModel):
    user_id: int
    message: str

# 3. Создаём агента с моделью, зависимостью и ожидаемым типом вывода.
agent = Agent(
    model='openai:gpt-3.5-turbo',
    deps_type=UserDatabase,        # агент ожидает объект UserDatabase в качестве зависимости
    output_type=MessageResult,     # агент должен вернуть MessageResult
    system_prompt="Ты – почтовый агент. Твоя задача: отправить вежливое сообщение пользователю по имени.",
)

# 4. Инструмент для получения ID пользователя по имени. Разрешаем 2 попытки (retries=2).
@agent.tool(retries=2)
def get_user_by_name(ctx: RunContext[UserDatabase], name: str) -> int:
    """Возвращает ID пользователя по полному имени."""
    user_id = ctx.deps.users.get(name)
    if user_id is None:
        # Если не найдено, просим модель уточнить запрос (через исключение ModelRetry)
        raise ModelRetry(f"Пользователь с именем {name!r} не найден. Укажи полное имя.")
    return user_id

# 5. Запускаем агента с неполным именем, чтобы инициировать ретрай.
db = UserDatabase()
query = "Отправь сообщение пользователю John с приглашением на встречу."
result = agent.run_sync(query, deps=db)
print(result.output)
# Возможный итоговый вывод (MessageResult):
# user_id=123 
# message="Hello John, would you be free for a meeting next week? Let me know what works for you!"
