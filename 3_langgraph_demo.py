"""
Демонстрация LangGraph: Многошаговый планировщик задач
Показывает создание графа для последовательной обработки задач
"""

# 1. Импорты
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from dotenv import load_dotenv
import os
import time

# 2. Определение состояния
class PlannerState(TypedDict):
    """Состояние планировщика"""
    task: str
    steps: List[str]
    current_step: int
    results: List[str]

# 3. Определение узлов графа
def analyze_task(state: PlannerState) -> PlannerState:
    """Анализирует задачу и создает план"""
    print("Анализирую задачу...")
    
    # Создаем LLM для анализа
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
    
    # Промпт для разбиения задачи на шаги
    prompt = f"""
    Разбей следующую задачу на 3 конкретных, выполнимых шага:
    
    Задача: {state['task']}
    
    Верни ответ в виде списка из 3 пунктов, каждый пункт должен начинаться с цифры.
    Например:
    1. Первый шаг
    2. Второй шаг
    3. Третий шаг
    """
    
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        steps_text = response.content
        
        # Парсим шаги из ответа
        steps = []
        for line in steps_text.split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-')):
                # Убираем номер и точку/тире
                step = line.split('.', 1)[-1].strip()
                if step.startswith(' '):
                    step = step[1:]
                steps.append(step)
        
        # Если не удалось распарсить, создаем общие шаги
        if len(steps) != 3:
            steps = [
                f"Подготовить материалы для: {state['task']}",
                f"Выполнить основную часть: {state['task']}",
                f"Завершить и проверить результат: {state['task']}"
            ]
        
        print("План создан:")
        for i, step in enumerate(steps, 1):
            print(f"{i}. {step}")
        print()
        
        # Обновляем состояние
        return {
            **state,
            "steps": steps,
            "current_step": 0,
            "results": []
        }
        
    except Exception as e:
        print(f"⚠️ Ошибка при анализе задачи: {e}")
        # Создаем простой план в случае ошибки
        steps = [
            f"Подготовить материалы для: {state['task']}",
            f"Выполнить основную часть: {state['task']}",
            f"Завершить и проверить результат: {state['task']}"
        ]
        
        return {
            **state,
            "steps": steps,
            "current_step": 0,
            "results": []
        }

def execute_step(state: PlannerState) -> PlannerState:
    """Выполняет текущий шаг"""
    current_step = state["current_step"]
    step_description = state["steps"][current_step]
    
    print(f"Выполняю шаг {current_step + 1}: {step_description}")
    
    # Имитация выполнения шага
    time.sleep(1)
    
    # Создаем результат выполнения
    result = f"Шаг {current_step + 1} выполнен успешно"
    
    print("✓ Шаг выполнен")
    print()
    
    # Обновляем состояние
    new_results = state["results"] + [result]
    new_current_step = current_step + 1
    
    return {
        **state,
        "results": new_results,
        "current_step": new_current_step
    }

def should_continue(state: PlannerState) -> str:
    """Определяет, продолжать ли выполнение"""
    if state["current_step"] < len(state["steps"]):
        return "continue"
    else:
        return "end"

def main():
    # Загрузка переменных окружения
    load_dotenv()
    
    # Проверка наличия ключа
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Ошибка: Не найден OPENAI_API_KEY в файле .env")
        print("Создайте файл .env и добавьте ваш ключ OpenAI")
        exit(1)
    
    print("=== LangGraph Demo: Планировщик задач ===\n")
    
    try:
        # 4. Создание графа
        # Создаем StateGraph с типом состояния PlannerState
        workflow = StateGraph(PlannerState)
        
        # Добавляем узлы
        workflow.add_node("analyze", analyze_task)
        workflow.add_node("execute", execute_step)
        
        # Настраиваем начальную точку
        workflow.set_entry_point("analyze")
        
        # Добавляем ребра
        workflow.add_edge("analyze", "execute")
        workflow.add_conditional_edges(
            "execute",
            should_continue,
            {
                "continue": "execute",
                "end": END
            }
        )
        
        # Компилируем граф
        app = workflow.compile()
        
        # 5. Тестовый запуск
        # Задача для планирования
        task = "Организовать день рождения для друга"
        print(f"Задача: {task}\\n")
        
        # Начальное состояние
        initial_state = PlannerState(
            task=task,
            steps=[],
            current_step=0,
            results=[]
        )
        
        # Запускаем граф
        final_state = app.invoke(initial_state)
        
        # Выводим финальный результат
        print("Задача завершена успешно!")
        print("\\nСводка выполнения:")
        for i, result in enumerate(final_state["results"], 1):
            print(f"- {result}")
            
    except Exception as e:
        print(f"❌ Ошибка при работе с LangGraph: {e}")
        print("Убедитесь, что:")
        print("1. Установлены все зависимости: pip install -r requirements.txt")
        print("2. Правильно указан OPENAI_API_KEY в файле .env")
        print("3. Есть доступ к интернету для обращения к OpenAI API")

if __name__ == "__main__":
    main()