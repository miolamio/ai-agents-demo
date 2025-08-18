"""
Пример 4: Human-in-the-Loop и Guardrails
"""
from typing import TypedDict, Annotated, Sequence
import operator
import re
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.messages.tool import ToolCall
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    task_status: str
    pending_action: dict
    safety_approved: bool

# Паттерны опасных команд для Guardrails
DANGEROUS_PATTERNS = [
    r'rm\s+-rf', r'DELETE\s+FROM', r'DROP\s+TABLE', 
    r'sudo', r'chmod\s+777', r'> /dev/null'
]

def safety_check(tool_calls: list) -> tuple[bool, str]:
    """Проверка планируемых действий на безопасность"""
    if not tool_calls:
        return True, "Нет инструментов для проверки"
        
    for tool_call in tool_calls:
        if isinstance(tool_call, dict) and tool_call.get('name') in ['shell', 'database_query']:
            command = str(tool_call.get('args', {}).get('command', ''))
            for pattern in DANGEROUS_PATTERNS:
                if re.search(pattern, command, re.IGNORECASE):
                    return False, f"Обнаружена опасная команда: {pattern}"
    return True, "Проверка пройдена"

def planning_node(state: AgentState):
    """Узел планирования действий"""
    messages = state["messages"]
    last_human_msg = None
    
    for msg in reversed(messages):
        if msg.type == "human":
            last_human_msg = msg.content
            break
    
    if not last_human_msg:
        return {"messages": [AIMessage(content="Не получено задание для выполнения")]}
    
    # Симулируем планирование инструмента (безопасная команда для демонстрации)
    planned_tool = {
        "name": "shell", 
        "args": {"command": "ls -la /tmp && echo 'Directory listed'"}
    }
    
    # Создаем AI сообщение с tool_calls
    ai_msg = AIMessage(
        content="Планирую выполнить очистку старых файлов",
        tool_calls=[ToolCall(
            name=planned_tool["name"],
            args=planned_tool["args"],
            id="tool_1"
        )]
    )
    
    return {
        "messages": [ai_msg],
        "pending_action": planned_tool,
        "task_status": "planned"
    }

def safety_gate_node(state: AgentState):
    """Узел проверки безопасности"""
    messages = state["messages"]
    last_ai_msg = None
    
    for msg in reversed(messages):
        if msg.type == "ai":
            last_ai_msg = msg
            break
    
    if not last_ai_msg or not hasattr(last_ai_msg, 'tool_calls') or not last_ai_msg.tool_calls:
        return {"task_status": "no_tools_to_check"}
    
    # Проверяем безопасность
    tool_calls_dict = []
    for tc in last_ai_msg.tool_calls:
        if hasattr(tc, 'name') and hasattr(tc, 'args'):
            tool_calls_dict.append({"name": tc.name, "args": tc.args})
        elif isinstance(tc, dict):
            tool_calls_dict.append(tc)
    
    is_safe, reason = safety_check(tool_calls_dict)
    
    if not is_safe:
        return {
            "messages": [AIMessage(content=f"Действие заблокировано системой безопасности: {reason}")],
            "task_status": "blocked",
            "safety_approved": False
        }
    
    return {
        "task_status": "needs_approval",
        "safety_approved": True
    }

def execution_node(state: AgentState):
    """Узел выполнения одобренных действий"""
    if not state.get("safety_approved", False):
        return {
            "messages": [AIMessage(content="Выполнение отклонено: действие не одобрено")],
            "task_status": "rejected"
        }
    
    # Симулируем успешное выполнение
    return {
        "messages": [
            ToolMessage(content="Команда выполнена успешно", tool_call_id="tool_1"),
            AIMessage(content="Задача завершена успешно")
        ],
        "task_status": "completed"
    }

def should_execute(state: AgentState) -> str:
    """Условная логика для выполнения"""
    status = state.get("task_status", "")
    if status == "blocked":
        return "__end__"
    elif status == "needs_approval":
        return "execution"  # В реальности здесь было бы прерывание
    else:
        return "__end__"

def demonstrate_hitl():
    """Демонстрация Human-in-the-Loop workflow"""
    with SqliteSaver.from_conn_string("hitl_checkpoints.db") as memory:
        # Настройка графа с HITL
        workflow = StateGraph(AgentState)
        workflow.add_node("planning", planning_node)
        workflow.add_node("safety_gate", safety_gate_node)
        workflow.add_node("execution", execution_node)

        workflow.set_entry_point("planning")
        workflow.add_edge("planning", "safety_gate")

        workflow.add_conditional_edges("safety_gate", should_execute)
        workflow.add_edge("execution", "__end__")

        # Компилируем с прерыванием перед выполнением
        app_with_hitl = workflow.compile(
            checkpointer=memory,
            interrupt_before=["execution"]
        )
        
        config_hitl = {"configurable": {"thread_id": "hitl-demo-456"}}
        
        print("=== Демонстрация Human-in-the-Loop ===\n")
        
        # 1. Запуск до прерывания
        print("1. Запуск задачи с автоматической остановкой перед выполнением:")
        response = app_with_hitl.invoke({
            "messages": [HumanMessage(content="Удали старые временные файлы из системы")],
            "task_status": "started",
            "safety_approved": False
        }, config=config_hitl)
        
        print(f"   Статус: {response.get('task_status', 'unknown')}")
        print(f"   Последнее сообщение: {response['messages'][-1].content}")
        
        # 2. Инспекция состояния
        print("\n2. Инспекция планируемых действий:")
        snapshot = app_with_hitl.get_state(config_hitl)
        print(f"   Следующий узел для выполнения: {snapshot.next}")
        
        if snapshot.values.get('pending_action'):
            action = snapshot.values['pending_action']
            print(f"   Планируемое действие: {action['name']}")
            print(f"   Команда: {action['args']['command']}")
        
        # 3. Симуляция одобрения пользователем
        print("\n3. Процесс одобрения:")
        user_decision = "y"  # Симулируем одобрение
        print(f"   Пользователь решил: {'Одобрить' if user_decision.lower() == 'y' else 'Отклонить'}")
        
        if user_decision.lower() == 'y':
            # Проверяем, не заблокировано ли действие
            if response.get('task_status') == 'blocked':
                print("   Действие заблокировано системой безопасности, одобрение невозможно")
            else:
                # Обновляем состояние для одобрения
                app_with_hitl.update_state(config_hitl, {"safety_approved": True})
                
                # Возобновляем выполнение
                final_result = app_with_hitl.invoke(None, config=config_hitl)
                if final_result:
                    print(f"   Результат: {final_result['task_status']}")
                    print(f"   Финальное сообщение: {final_result['messages'][-1].content}")
                else:
                    print("   Выполнение завершено без дополнительных результатов")
        else:
            # Отклонение действия
            print("   Действие отклонено пользователем")

if __name__ == "__main__":
    demonstrate_hitl()