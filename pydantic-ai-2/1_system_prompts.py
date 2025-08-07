from dataclasses import dataclass
from pydantic_ai import Agent, RunContext

@dataclass
class UserData:
    name: str

# Создаём агента со статической ролью ассистента
agent = Agent(
    'openai:gpt-3.5-turbo',
    deps_type=UserData,
    system_prompt='Ты – вежливый помощник, который знает некоторые данные о пользователе.'
)

# Динамическое системное сообщение добавляет имя пользователя из зависимостей
@agent.system_prompt
def add_user_name(ctx: RunContext[UserData]) -> str:
    return f"Имя пользователя: {ctx.deps.name!r}"

user = UserData(name="Иван")
result = agent.run_sync("Как меня зовут?", deps=user)
print(result.output)