"""
Пример 3: Управление переполнением контекста
"""
from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    task_status: str
    total_messages: int

def trim_messages(messages: list, max_tokens: int = 6000) -> list:
    """Обрезка сообщений с сохранением последних и системных"""
    if not messages:
        return messages
    
    # Всегда сохраняем системные сообщения
    system_msgs = [msg for msg in messages if msg.type == "system"]
    other_msgs = [msg for msg in messages if msg.type != "system"]
    
    # Подсчёт приблизительных токенов (1 токен ≈ 4 символа для русского)
    total_chars = sum(len(str(msg.content)) for msg in other_msgs)
    
    if total_chars > max_tokens * 4:
        # Берём последние 70% от лимита
        keep_chars = int(max_tokens * 4 * 0.7)
        kept_msgs = []
        current_chars = 0
        
        for msg in reversed(other_msgs):
            msg_chars = len(str(msg.content))
            if current_chars + msg_chars <= keep_chars:
                kept_msgs.insert(0, msg)
                current_chars += msg_chars
            else:
                break
        
        return system_msgs + kept_msgs
    
    return messages

def context_manager_node(state: AgentState):
    """Узел управления контекстом с автоматической обрезкой"""
    messages = state["messages"]
    total_count = len(messages)
    
    # Проверяем, нужно ли обрезать сообщения
    if total_count > 10:  # Порог для демонстрации
        trimmed_messages = trim_messages(messages, max_tokens=2000)
        trimmed_count = len(trimmed_messages)
        
        if trimmed_count < total_count:
            # Добавляем системное сообщение о обрезке
            system_notice = SystemMessage(content=f"Контекст обрезан: оставлено {trimmed_count} из {total_count} сообщений")
            trimmed_messages.append(system_notice)
            
            return {
                "messages": trimmed_messages,
                "task_status": "context_trimmed",
                "total_messages": total_count
            }
    
    # Обычная обработка
    response_msg = AIMessage(content=f"Обработано сообщение #{total_count}. Контекст в норме.")
    
    return {
        "messages": [response_msg],
        "task_status": "processing",
        "total_messages": total_count
    }

def simulate_long_conversation():
    """Симуляция долгого разговора для тестирования обрезки контекста"""
    with SqliteSaver.from_conn_string("context_checkpoints.db") as memory:
        # Настройка графа с управлением контекстом
        workflow = StateGraph(AgentState)
        workflow.add_node("context_manager", context_manager_node)
        workflow.set_entry_point("context_manager")
        workflow.set_finish_point("context_manager")

        app = workflow.compile(checkpointer=memory)
        
        config = {"configurable": {"thread_id": "long-conversation-test"}}
        
        # Добавляем системное сообщение
        app.invoke({
            "messages": [SystemMessage(content="Система управления контекстом активна")],
            "task_status": "started",
            "total_messages": 0
        }, config=config)
        
        # Симулируем много сообщений
        for i in range(15):
            human_msg = f"Сообщение пользователя #{i+1}: " + "А" * 100  # Длинное сообщение
            
            result = app.invoke({
                "messages": [HumanMessage(content=human_msg)]
            }, config=config)
            
            print(f"Шаг {i+1}:")
            print(f"  Статус: {result['task_status']}")
            print(f"  Всего сообщений в состоянии: {len(result['messages'])}")
            print(f"  Последнее сообщение: {result['messages'][-1].content[:50]}...")
            
            if result['task_status'] == "context_trimmed":
                print("  ⚠️  Контекст был обрезан!")
            print()

if __name__ == "__main__":
    print("=== Демонстрация управления контекстом ===\n")
    simulate_long_conversation()