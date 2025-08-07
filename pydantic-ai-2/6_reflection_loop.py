"""
reflection_loop_example.py
--------------------------
Мини скрипт демонстрирует *reflection loop* в Pydantic AI v2:
LLM-агент сам пишет функцию `factorial(n)`, сам придумывает
unit-тест, вызывает инструмент `run_tests`, — и при неудаче
исправляет код, пока тест не пройдёт.

"""
import logfire

# Шаг 0: Добавляем Logfire
logfire.configure(token="pylf_v1_us_DQN6NX83mbX9hcJz6yWWpRdgVFS47xxHJmKRzPhjfgsp") # Настраиваем Logfire
logfire.instrument_pydantic_ai() # Включаем автоматическую трассировку Pydantic!

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext


# ---------- 1. Формат ответа, который агент обязан вернуть ----------
class CodeOutput(BaseModel):
    code: str = Field(description="Исходный код функции")
    result: str = Field(description="Строка 'pass' либо 'fail: …'")


# ---------- 2. Создаём агента ----------
agent = Agent(
    "openai:gpt-3.5-turbo",      # поменяйте модель при желании
    output_type=CodeOutput,
    system_prompt=(
        "Ты пишешь код на Python. Задача: реализовать рекурсивную "
        "функцию factorial(n).\n"
        "1) Напиши функцию, 2) придумай unit-тест (assert), 3) "
        "вызови инструмент run_tests, передав ему код и тесты.\n"
        "Если ответ run_tests != 'pass', исправь код и повтори.\n"
        "Верни JSON с полями code и result."
    ),
    instrument=True
)


# ---------- 3. Инструмент, который запускает код + тесты ----------
@agent.tool
def run_tests(ctx: RunContext, code: str, tests: str) -> str:
    """Выполняет код и тесты; возвращает 'pass' или 'fail: <ошибка>'."""
    ns: dict[str, object] = {}   # изолируем пространство имён
    try:
        exec(code, ns)           # выполняем функцию
        exec(tests, ns)          # выполняем тесты (должны быть assert)
        return "pass"
    except Exception as exc:
        return f"fail: {exc}"    # сообщаем об ошибке LLM-у


# ---------- 4. Запуск ----------
if __name__ == "__main__":
    result = agent.run_sync("Напиши функцию factorial")
    # Выводим красиво отформатированный JSON-ответ
    print(result.output.model_dump_json(indent=2))
