"""
Пример 2: Демонстрация восстановления после сбоя
"""
from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    task_status: str
    attempt_count: int

def unreliable_node(state: AgentState):
    """Узел, который падает при первом запуске"""
    attempt_count = state.get("attempt_count", 0)
    
    if attempt_count == 0:
        # Первая попытка — симуляция сбоя
        raise Exception("Симуляция сбоя сервиса")
    else:
        # Вторая и последующие попытки — успешное выполнение
        return {
            "messages": [("ai", f"Задача выполнена после восстановления (попытка {attempt_count + 1})")],
            "task_status": "completed",
            "attempt_count": attempt_count + 1
        }

def check_status_node(state: AgentState):
    """Узел проверки состояния для принятия решения о продолжении"""
    if state["task_status"] == "failed":
        return {
            "messages": [("system", "Обнаружен сбой, подготовка к восстановлению")],
            "task_status": "retrying"
        }
    return state

def should_retry(state: AgentState) -> str:
    """Условная логика для повторных попыток"""
    if state["task_status"] == "retrying":
        return "unreliable_task"
    return "__end__"

if __name__ == "__main__":
    # Используем контекстный менеджер для SqliteSaver
    with SqliteSaver.from_conn_string("recovery_checkpoints.db") as memory:
        # Настройка графа с восстановлением
        workflow = StateGraph(AgentState)
        workflow.add_node("unreliable_task", unreliable_node)
        workflow.add_node("status_check", check_status_node)

        workflow.set_entry_point("unreliable_task")
        workflow.add_edge("unreliable_task", "status_check")

        workflow.add_conditional_edges("status_check", should_retry)

        app = workflow.compile(checkpointer=memory)

        # Демонстрация восстановления после сбоя
        config_recovery = {"configurable": {"thread_id": "recovery-demo-123"}}
        
        print("=== Демонстрация восстановления ===")
        
        # Первая попытка
        print("\n1. Первая попытка выполнения:")
        try:
            result1 = app.invoke({
                "messages": [("human", "Выполни сложную задачу")],
                "task_status": "started",
                "attempt_count": 0
            }, config=config_recovery)
            print("Результат первой попытки:", result1["task_status"])
        except Exception as e:
            print(f"Первая попытка неудачна: {e}")

        # Вторая попытка — восстановление с увеличенным счетчиком
        print("\n2. Восстановление работы:")
        try:
            result2 = app.invoke({
                "messages": [("human", "Повторная попытка")],
                "task_status": "retrying", 
                "attempt_count": 1
            }, config=config_recovery)
            if result2:
                print("Результат восстановления:", result2["task_status"])
                print("Количество попыток:", result2["attempt_count"])
                # Показываем только последние несколько сообщений
                recent_messages = result2["messages"][-4:]
                print("Последние сообщения:", [msg[1] if isinstance(msg, tuple) else str(msg) for msg in recent_messages])
            else:
                print("Результат восстановления: None")
        except Exception as e:
            print(f"Ошибка при восстановлении: {e}")