"""
Пример 1: Базовое состояние агента для персистентности
"""
from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph

class AgentState(TypedDict):
    # История сообщений автоматически накапливается с помощью Annotated и operator.add
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # Можно добавить другие поля, например, статус задачи
    task_status: str
    step_count: int

# Пример узла агента
def process_node(state: AgentState):
    """Простой узел для демонстрации работы с состоянием"""
    messages = state["messages"]
    current_count = state.get("step_count", 0)
    
    return {
        "messages": [("ai", f"Обработан шаг {current_count + 1}")],
        "task_status": "processing",
        "step_count": current_count + 1
    }

if __name__ == "__main__":
    # Используем контекстный менеджер для SqliteSaver
    with SqliteSaver.from_conn_string("checkpoints.db") as memory:
        # Создание и компиляция графа
        workflow = StateGraph(AgentState)
        workflow.add_node("process", process_node)
        workflow.set_entry_point("process")
        workflow.set_finish_point("process")

        # Компилируем граф, передавая checkpointer
        app = workflow.compile(checkpointer=memory)

        # Конфигурация для конкретной сессии пользователя
        config = {"configurable": {"thread_id": "user-alice-task-data-analysis"}}

        # Первый запуск — создаётся новый поток
        response1 = app.invoke(
            {"messages": [("human", "Анализируй продажи за Q1")], "task_status": "started", "step_count": 0}, 
            config=config
        )
        print("Первый ответ:", response1)

        # Последующие запуски автоматически загружают состояние
        response2 = app.invoke(
            {"messages": [("human", "Добавь сравнение с прошлым годом")]}, 
            config=config
        )
        print("Второй ответ:", response2)
        print(f"Общее количество шагов: {response2['step_count']}")