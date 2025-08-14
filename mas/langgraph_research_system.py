"""
Система генерации исследовательских отчётов на LangGraph
Демонстрация многоагентной архитектуры с супервизором и специализированными агентами
"""

import os
import json
import functools
from typing import Annotated, TypedDict, Literal

try:
    from langchain_core.messages import HumanMessage, BaseMessage
    from langchain_openai import ChatOpenAI
    from langchain_core.tools import tool
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langgraph.graph import StateGraph, START, END
    from langgraph.graph.message import add_messages
    from langgraph.checkpoint.sqlite import SqliteSaver
    from dotenv import load_dotenv
except ImportError as e:
    print(f"❌ Ошибка импорта: {e}")
    print("Установите необходимые пакеты:")
    print("pip install langchain langchain-openai langgraph tavily-python python-dotenv")
    exit(1)

# Загрузка переменных окружения из .env файла
load_dotenv()

# Проверка наличия необходимых API ключей
def check_api_keys():
    """Проверка наличия необходимых API ключей"""
    required_keys = {
        "OPENAI_API_KEY": "OpenAI API ключ для работы с GPT моделями",
        "TAVILY_API_KEY": "Tavily API ключ для веб-поиска"
    }
    
    missing_keys = []
    for key, description in required_keys.items():
        if not os.getenv(key):
            missing_keys.append(f"  - {key}: {description}")
    
    if missing_keys:
        print("❌ Отсутствуют необходимые API ключи:")
        print("\n".join(missing_keys))
        print("\nСоздайте файл .env в корне проекта со следующими переменными:")
        print("OPENAI_API_KEY=your-openai-api-key-here")
        print("TAVILY_API_KEY=your-tavily-api-key-here")
        return False
    
    return True

# Проверяем API ключи при импорте
if not check_api_keys():
    print("\n💡 Получите API ключи:")
    print("- OpenAI: https://platform.openai.com/api-keys")
    print("- Tavily: https://app.tavily.com/")
    exit(1)

# Определение состояния системы
class ResearchState(TypedDict):
    """Состояние многоагентной исследовательской системы"""
    messages: Annotated[list[BaseMessage], add_messages]
    topic: str
    research_data: str
    analysis: str
    report: str
    next_agent: str

# Инициализация модели
model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# Инструменты для агентов
@tool
def web_search(query: str) -> str:
    """Поиск актуальной информации в интернете"""
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
        results = client.search(query, max_results=5)
        
        formatted_results = []
        for result in results.get("results", []):
            formatted_results.append({
                "title": result.get("title", ""),
                "content": result.get("content", "")[:300] + "...",
                "url": result.get("url", "")
            })
        
        return json.dumps(formatted_results, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"Ошибка поиска: {str(e)}"

# Создание специализированных агентов
def create_agent_node(agent_name: str, system_prompt: str, tools: list = None):
    """Фабрика для создания узлов-агентов"""
    if tools is None:
        tools = []
    
    def agent_function(state: ResearchState) -> dict:
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])
        
        # Определяем входные данные для агента
        if agent_name == "researcher":
            input_text = f"Тема для исследования: {state['topic']}"
        elif agent_name == "analyst":
            input_text = f"Данные для анализа: {state.get('research_data', '')}"
        elif agent_name == "writer":
            analysis_data = state.get('analysis', '')
            research_data = state.get('research_data', '')
            input_text = f"Исследование: {research_data}\n\nАнализ: {analysis_data}"
        else:
            input_text = state.get("topic", "")

        if tools:
            # Используем модель с инструментами
            llm_with_tools = model.bind_tools(tools)
            response = llm_with_tools.invoke(prompt.format_messages(input=input_text))
            
            # Если есть вызовы инструментов, выполняем их
            if response.tool_calls:
                tool_results = []
                for tool_call in response.tool_calls:
                    if tool_call["name"] == "web_search":
                        result = web_search.invoke(tool_call["args"])
                        tool_results.append(result)
                
                # Объединяем результаты
                combined_result = "\n".join(tool_results)
                final_response = f"{response.content}\n\nДанные поиска:\n{combined_result}"
            else:
                final_response = response.content
        else:
            response = model.invoke(prompt.format_messages(input=input_text))
            final_response = response.content
        
        # Обновляем соответствующее поле состояния
        update_dict = {"messages": [HumanMessage(content=final_response, name=agent_name)]}
        
        if agent_name == "researcher":
            update_dict["research_data"] = final_response
        elif agent_name == "analyst":
            update_dict["analysis"] = final_response
        elif agent_name == "writer":
            update_dict["report"] = final_response
            
        return update_dict
    
    return agent_function

# Создание узлов агентов
research_node = create_agent_node(
    "researcher",
    """Вы — эксперт-исследователь. Ваша задача — найти актуальную и достоверную информацию по заданной теме.
    Используйте web_search для поиска последних данных. Представьте результат в виде списка из 10 ключевых фактов.
    Текущий год: 2025. Фокусируйтесь на самых свежих данных.""",
    tools=[web_search]
)

analysis_node = create_agent_node(
    "analyst",
    """Вы — аналитик данных. Проанализируйте предоставленную информацию и выделите:
    1. Основные тенденции и тренды
    2. Ключевые цифры и статистику
    3. Важные выводы и инсайты
    4. Потенциальные направления развития"""
)

writer_node = create_agent_node(
    "writer",
    """Вы — профессиональный автор технических отчётов. На основе исследования и анализа создайте 
    структурированный отчёт в формате Markdown со следующими разделами:
    
    # Исполнительное резюме
    # Ключевые находки
    # Детальный анализ
    # Выводы и рекомендации
    
    Используйте заголовки, списки и форматирование для читаемости."""
)

# Супервизор для маршрутизации
def supervisor_node(state: ResearchState) -> dict:
    """Супервизор определяет следующий шаг в процессе"""
    if not state.get("research_data"):
        return {"next_agent": "researcher"}
    elif not state.get("analysis"):
        return {"next_agent": "analyst"}
    elif not state.get("report"):
        return {"next_agent": "writer"}
    else:
        return {"next_agent": "FINISH"}

def route_to_next_agent(state: ResearchState) -> Literal["researcher", "analyst", "writer", "end"]:
    """Маршрутизация к следующему агенту"""
    next_agent = state.get("next_agent", "researcher")
    if next_agent == "FINISH":
        return "end"
    return next_agent

# Построение графа
def create_research_graph():
    """Создание и компиляция графа исследовательской системы"""
    workflow = StateGraph(ResearchState)
    
    # Добавление узлов
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("researcher", research_node)
    workflow.add_node("analyst", analysis_node)
    workflow.add_node("writer", writer_node)
    
    # Определение рёбер
    workflow.add_edge(START, "supervisor")
    
    # Условные рёбра от супервизора
    workflow.add_conditional_edges(
        "supervisor",
        route_to_next_agent,
        {
            "researcher": "researcher",
            "analyst": "analyst", 
            "writer": "writer",
            "end": END
        }
    )
    
    # Возврат к супервизору после каждого агента
    workflow.add_edge("researcher", "supervisor")
    workflow.add_edge("analyst", "supervisor")
    workflow.add_edge("writer", "supervisor")
    
    # Компиляция с персистентностью (используем InMemorySaver для простоты)
    from langgraph.checkpoint.memory import InMemorySaver
    memory = InMemorySaver()
    return workflow.compile(checkpointer=memory)

# Главная функция для запуска системы
def run_research_system(topic: str, verbose: bool = True):
    """
    Запуск многоагентной исследовательской системы
    
    Args:
        topic: Тема для исследования
        verbose: Показывать промежуточные шаги
    
    Returns:
        str: Итоговый отчёт
    """
    app = create_research_graph()
    
    # Конфигурация с уникальным thread_id
    config = {"configurable": {"thread_id": f"research_{hash(topic) % 10000}"}}
    
    # Начальное состояние
    initial_state = {
        "messages": [HumanMessage(content=f"Исследовать тему: {topic}")],
        "topic": topic,
        "next_agent": "researcher"
    }
    
    if verbose:
        print(f"🔍 Начинаю исследование темы: {topic}\n")
    
    # Выполнение с потоковым выводом
    final_state = None
    for step, state in enumerate(app.stream(initial_state, config), 1):
        if "__end__" not in state:
            current_step = list(state.keys())[0]
            if verbose:
                print(f"Шаг {step}: Агент '{current_step}' выполняет задачу...")
            final_state = state
    
    # Получение финального состояния через get_state
    try:
        state_snapshot = app.get_state(config)
        final_result = state_snapshot.values
    except Exception as e:
        if verbose:
            print(f"⚠️  Не удалось получить состояние через get_state: {e}")
        # Fallback: используем обычный invoke
        final_result = app.invoke(initial_state, config)
    
    return final_result.get("report", "Отчёт не сгенерирован")

# Пример использования
if __name__ == "__main__":
    # Пример запуска системы
    try:
        topic = "Искусственный интеллект в медицине: последние достижения 2025"
        report = run_research_system(topic, verbose=True)
        
        print("\n" + "="*80)
        print("ИТОГОВЫЙ ОТЧЁТ")
        print("="*80)
        print(report)
        
        # Сохранение отчёта
        with open("research_report_langgraph.md", "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\n✅ Отчёт сохранён в 'research_report_langgraph.md'")
        
    except Exception as e:
        print(f"❌ Ошибка выполнения: {e}")
        print("Проверьте настройку API ключей в переменных окружения")

# Дополнительные утилиты для тестирования
def test_individual_agents():
    """Тестирование отдельных агентов"""
    print("🧪 Тестирование отдельных компонентов...")
    
    # Тест поискового инструмента
    search_result = web_search.invoke({"query": "последние новости ИИ 2025"})
    print(f"Поиск работает: {len(search_result) > 100}")
    
    # Тест состояния
    test_state = ResearchState(
        messages=[],
        topic="тест",
        research_data="",
        analysis="",
        report="",
        next_agent="researcher"
    )
    
    supervisor_result = supervisor_node(test_state)
    print(f"Супервизор работает: {supervisor_result.get('next_agent') == 'researcher'}")
    
    print("✅ Компоненты протестированы")

def visualize_graph():
    """Комплексная визуализация графа"""
    print("📊 Создание визуализации графа...")
    
    try:
        app = create_research_graph()
        graph = app.get_graph()
        
        # ASCII
        print("\n🔤 ASCII структура:")
        try:
            graph.print_ascii()
        except:
            print("ASCII недоступно")
        
        # Mermaid
        print("\n🌊 Mermaid код:")
        try:
            mermaid = graph.draw_mermaid()
            print(mermaid)
            with open("research_graph.mmd", "w") as f:
                f.write(mermaid)
            print("✅ Сохранено в research_graph.mmd")
        except Exception as e:
            print(f"Mermaid ошибка: {e}")
        
        # PNG
        print("\n🖼️ PNG изображение:")
        try:
            png_data = graph.draw_mermaid_png()
            with open("research_graph.png", "wb") as f:
                f.write(png_data)
            print("✅ Сохранено в research_graph.png")
        except Exception as e:
            print(f"PNG ошибка: {e}")
            print("💡 Установите: npm install -g @mermaid-js/mermaid-cli")
            
    except Exception as e:
        print(f"❌ Общая ошибка: {e}")

def create_interactive_graph():
    """Создать интерактивную HTML визуализацию"""
    try:
        from pyvis.network import Network
        
        net = Network(height="600px", width="100%", directed=True)
        
        # Добавляем узлы
        net.add_node("START", label="🚀 START", color="#e1f5fe")
        net.add_node("supervisor", label="🎯 Supervisor", color="#fff3e0")
        net.add_node("researcher", label="🔍 Researcher", color="#f3e5f5")
        net.add_node("analyst", label="📊 Analyst", color="#e8f5e8")
        net.add_node("writer", label="✍️ Writer", color="#fff8e1")
        net.add_node("END", label="🏁 END", color="#ffebee")
        
        # Добавляем связи
        edges = [
            ("START", "supervisor"),
            ("supervisor", "researcher"),
            ("supervisor", "analyst"),
            ("supervisor", "writer"),
            ("supervisor", "END"),
            ("researcher", "supervisor"),
            ("analyst", "supervisor"),
            ("writer", "supervisor")
        ]
        
        for src, dst in edges:
            net.add_edge(src, dst)
        
        # Сохраняем
        net.save_graph("research_graph_interactive.html")
        print("✅ Интерактивный граф сохранён в research_graph_interactive.html")
        
    except ImportError:
        print("💡 Установите PyVis: pip install pyvis")

def show_graph_ascii():
    """Показать ASCII структуру графа"""
    app = create_research_graph()
    print("🔤 ASCII структура графа:")
    print("=" * 40)
    app.get_graph().print_ascii()

# Вызовите функцию:
show_graph_ascii()

def show_graph_mermaid():
    """Показать Mermaid код графа"""
    app = create_research_graph()
    print("🌊 Mermaid код:")
    print("=" * 40)
    mermaid_code = app.get_graph().draw_mermaid()
    print(mermaid_code)
    
    # Сохраняем в файл
    with open("research_graph.mmd", "w", encoding="utf-8") as f:
        f.write(mermaid_code)
    print("\n✅ Код сохранён в research_graph.mmd")
    print("💡 Вставьте код в https://mermaid.live для интерактивного просмотра")

# Вызовите функцию:
show_graph_mermaid()

def save_graph_png():
    """Сохранить граф как PNG"""
    try:
        app = create_research_graph()
        graph_png = app.get_graph().draw_mermaid_png()
        with open("research_graph.png", "wb") as f:
            f.write(graph_png)
        print("✅ PNG граф сохранён в research_graph.png")
    except Exception as e:
        print(f"❌ Ошибка PNG: {e}")
        print("💡 Установите mermaid-cli: npm install -g @mermaid-js/mermaid-cli")