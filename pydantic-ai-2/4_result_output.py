from pydantic_ai import Agent, RunContext
from pydantic import BaseModel

# 1. Определяем Pydantic-модель для структурированного ответа.
class PersonInfo(BaseModel):
    name: str
    age: int
    city: str

# 2. Создаем агента с указанным типом вывода (PersonInfo).
agent = Agent(
    model='openai:gpt-3.5-turbo',
    output_type=PersonInfo,  # Агент должен вернуть PersonInfo; Pydantic проверит соответствие.
    system_prompt=(
        "Ты – помощник, который извлекает информацию о человеке из текста. "
        "Возвращай ответ *только* в формате: имя, возраст и город."
    )
)

# 3. Динамическая инструкция: пример, как можно использовать RunContext для дополнительной логики (опционально).
@agent.instructions
def example_instruction(ctx: RunContext):
    # Простая инструкция без проверки содержимого сообщения
    return "Убедись, что правильно определил все три параметра: имя, возраст и город."

# 4. Пример запуска агента
text = "Привет! Меня зовут Ольга, мне двадцать пять лет, живу в городе Казань."
result = agent.run_sync(text)
info: PersonInfo = result.output
print(info)            # Вывод в виде PersonInfo, например: name='Ольга' age=25 city='Казань'
print(info.name)       # Ольга
print(info.age)        # 25
print(info.city)       # Казань