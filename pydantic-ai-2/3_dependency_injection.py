from dataclasses import dataclass
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

# Определяем класс зависимостей
@dataclass
class SupportDependencies:
    customer_id: int
    db: DatabaseConn  # некий класс для работы с базой данных

# Определяем модель вывода (структуру ответа)
class SupportOutput(BaseModel):
    support_advice: str = Field(description="Совет или ответ, который вернётся клиенту")
    block_card: bool = Field(description="Нужно ли блокировать карту клиента")
    risk: int = Field(description="Уровень риска запроса клиента", ge=0, le=10)

# Создаём агента технической поддержки
support_agent = Agent(
    'openai:gpt-4',
    deps_type=SupportDependencies,
    output_type=SupportOutput,
    system_prompt=(
        "Ты — агент первой линии поддержки в нашем банке. "
        "Отвечай на вопросы клиентов и оценивай уровень риска запроса."
    )
)

# Динамическое системное сообщение: добавить имя клиента из базы по его ID
@support_agent.system_prompt
async def add_customer_name(ctx: RunContext[SupportDependencies]) -> str:
    customer_name = await ctx.deps.db.customer_name(id=ctx.deps.customer_id)
    return f"Имя клиента: {customer_name!r}"

# Инструмент: получение баланса клиента (с возможностью включить незавершённые транзакции)
@support_agent.tool
async def customer_balance(ctx: RunContext[SupportDependencies], include_pending: bool) -> float:
    """Возвращает текущий баланс счёта клиента."""
    return await ctx.deps.db.customer_balance(id=ctx.deps.customer_id, include_pending=include_pending)
