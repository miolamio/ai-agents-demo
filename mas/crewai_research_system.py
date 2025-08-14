"""
Система генерации исследовательских отчётов на CrewAI
Демонстрация ролевой многоагентной команды для автоматизации исследований
"""

import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool, FileWriterTool
from langchain_openai import ChatOpenAI

# Загрузка переменных окружения из .env файла
load_dotenv()

# Инициализация инструментов
search_tool = SerperDevTool()
file_tool = FileWriterTool()

# Настройка модели (опционально)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# Определение агентов команды
class ResearchCrew:
    """Класс для организации исследовательской команды агентов"""
    
    def __init__(self):
        self.llm = llm
    
    def create_researcher_agent(self) -> Agent:
        """Создание агента-исследователя"""
        return Agent(
            role='Старший научный исследователь',
            goal='Найти самую актуальную и достоверную информацию по заданной теме {topic}',
            backstory="""Вы — опытный исследователь с 15-летним стажем в области технологий и инноваций. 
            Вы известны своей способностью находить самые релевантные и свежие данные из различных источников.
            Вы всегда проверяете информацию на актуальность и достоверность, особенно учитывая, что текущий год — 2025.""",
            tools=[search_tool],
            verbose=True,
            memory=True,
            allow_delegation=False,
            llm=self.llm
        )
    
    def create_analyst_agent(self) -> Agent:
        """Создание агента-аналитика"""
        return Agent(
            role='Ведущий аналитик данных',
            goal='Проанализировать собранную информацию по теме {topic} и выявить ключевые инсайты',
            backstory="""Вы — эксперт в области анализа данных и трендов с глубоким пониманием технологических процессов.
            Вы умеете выявлять скрытые паттерны в больших объёмах информации и делать обоснованные выводы.
            Ваша специализация — превращение сырых данных в структурированные инсайты.""",
            verbose=True,
            memory=True,
            allow_delegation=True,
            llm=self.llm
        )
    
    def create_writer_agent(self) -> Agent:
        """Создание агента-писателя"""
        return Agent(
            role='Ведущий технический писатель',
            goal='Создать профессиональный и структурированный отчёт по теме {topic}',
            backstory="""Вы — профессиональный писатель, специализирующийся на создании технических отчётов 
            и аналитических документов. У вас есть талант представлять сложную информацию в ясной и доступной форме.
            Ваши отчёты всегда хорошо структурированы и содержат практические выводы.""",
            tools=[file_tool],
            verbose=True,
            memory=True,
            allow_delegation=False,
            llm=self.llm
        )
    
    def create_research_task(self, agent: Agent) -> Task:
        """Создание задачи исследования"""
        return Task(
            description="""Провести глубокое исследование по теме: {topic}
            
            Ваши задачи:
            1. Найти последние новости и разработки (обязательно 2025 год)
            2. Собрать статистические данные и цифры
            3. Выявить ключевых игроков и компании
            4. Определить основные тренды и направления
            5. Найти практические применения и кейсы
            
            Используйте веб-поиск для получения самой актуальной информации.""",
            expected_output="""Подробный список из 10-12 ключевых фактов с указанием источников, 
            включающий конкретные цифры, даты, имена компаний и статистику. 
            Каждый факт должен быть актуальным и содержать практическую информацию.""",
            agent=agent
        )
    
    def create_analysis_task(self, agent: Agent, context_tasks: list) -> Task:
        """Создание задачи анализа"""
        return Task(
            description="""Проанализировать собранную исследователем информацию по теме {topic}
            
            Выполните следующий анализ:
            1. Выявите 3-4 основных тренда
            2. Определите ключевые возможности и риски
            3. Проанализируйте статистические данные
            4. Сделайте прогнозы на ближайшие 2-3 года
            5. Выделите наиболее значимые инсайты
            
            Сконцентрируйтесь на практической значимости находок.""",
            expected_output="""Структурированный аналитический отчёт с четкими разделами:
            - Основные тренды (3-4 пункта)
            - Ключевые возможности и риски
            - Статистический анализ
            - Прогнозы и рекомендации""",
            agent=agent,
            context=context_tasks
        )
    
    def create_writing_task(self, agent: Agent, context_tasks: list) -> Task:
        """Создание задачи написания отчёта"""
        return Task(
            description="""Создать профессиональный исследовательский отчёт по теме {topic}
            
            Структура отчёта:
            1. Исполнительное резюме (2-3 абзаца)
            2. Ключевые находки (список с пояснениями)
            3. Подробный анализ по разделам
            4. Статистика и цифры
            5. Выводы и рекомендации
            
            Используйте форматирование Markdown для лучшей читаемости.
            Включите все важные данные из исследования и анализа.""",
            expected_output="""Полный исследовательский отчёт в формате Markdown (2000-3000 слов)
            с заголовками, списками, и профессиональным форматированием. 
            Отчёт должен быть самодостаточным и информативным.""",
            agent=agent,
            context=context_tasks,
            output_file='research_report_crewai.md'
        )

def create_research_crew(topic: str) -> Crew:
    """Создание и настройка исследовательской команды"""
    
    # Инициализация класса команды
    research_crew = ResearchCrew()
    
    # Создание агентов
    researcher = research_crew.create_researcher_agent()
    analyst = research_crew.create_analyst_agent()
    writer = research_crew.create_writer_agent()
    
    # Создание задач
    research_task = research_crew.create_research_task(researcher)
    analysis_task = research_crew.create_analysis_task(analyst, [research_task])
    writing_task = research_crew.create_writing_task(writer, [research_task, analysis_task])
    
    # Создание команды
    crew = Crew(
        agents=[researcher, analyst, writer],
        tasks=[research_task, analysis_task, writing_task],
        process=Process.sequential,  # Последовательное выполнение
        memory=True,                 # Включение памяти команды
        verbose=True,               # Подробное логирование
        output_log_file='crew_execution_log.txt'
    )
    
    return crew

def create_hierarchical_crew(topic: str) -> Crew:
    """Создание команды с иерархическим процессом"""
    
    research_crew = ResearchCrew()
    
    # Создание специального менеджера
    manager = Agent(
        role='Менеджер исследовательского проекта',
        goal='Эффективно координировать команду исследователей для создания качественного отчёта по теме {topic}',
        backstory="""Вы — опытный менеджер исследовательских проектов с 20-летним стажем.
        Вы умеете эффективно распределять задачи между специалистами, контролировать качество 
        и обеспечивать своевременную доставку результатов. Ваша задача — получить лучший возможный результат.""",
        allow_delegation=True,
        verbose=True,
        llm=llm
    )
    
    # Создание агентов (без менеджера в списке)
    researcher = research_crew.create_researcher_agent()
    analyst = research_crew.create_analyst_agent()
    writer = research_crew.create_writer_agent()
    
    # Создание задач (менеджер сам их распределит)
    research_task = research_crew.create_research_task(researcher)
    analysis_task = research_crew.create_analysis_task(analyst, [research_task])
    writing_task = research_crew.create_writing_task(writer, [research_task, analysis_task])
    
    # Создание иерархической команды
    crew = Crew(
        agents=[researcher, analyst, writer],
        tasks=[research_task, analysis_task, writing_task],
        process=Process.hierarchical,    # Иерархический процесс
        manager_agent=manager,           # Назначение менеджера
        memory=True,
        verbose=True,
        output_log_file='hierarchical_crew_log.txt'
    )
    
    return crew

def run_research_project(topic: str, use_hierarchical: bool = False):
    """
    Запуск исследовательского проекта
    
    Args:
        topic: Тема для исследования
        use_hierarchical: Использовать иерархический процесс
    """
    
    print(f"🚀 Запуск исследовательского проекта")
    print(f"📋 Тема: {topic}")
    print(f"👥 Режим: {'Иерархический' if use_hierarchical else 'Последовательный'}")
    print("="*80)
    
    try:
        # Создание команды
        if use_hierarchical:
            crew = create_hierarchical_crew(topic)
        else:
            crew = create_research_crew(topic)
        
        # Запуск проекта
        result = crew.kickoff(inputs={'topic': topic})
        
        print("\n" + "="*80)
        print("✅ ПРОЕКТ ЗАВЕРШЁН УСПЕШНО")
        print("="*80)
        print("Результаты:")
        print(f"- Отчёт сохранён в: research_report_crewai.md")
        print(f"- Лог выполнения: {'hierarchical_crew_log.txt' if use_hierarchical else 'crew_execution_log.txt'}")
        
        return result
        
    except Exception as e:
        print(f"❌ Ошибка выполнения проекта: {e}")
        print("Проверьте настройку API ключей и подключение к интернету")
        return None

def analyze_crew_performance(crew: Crew):
    """Анализ производительности команды"""
    try:
        print("\n📊 АНАЛИЗ ПРОИЗВОДИТЕЛЬНОСТИ КОМАНДЫ")
        print("="*50)
        
        # Информация об агентах
        print(f"Количество агентов: {len(crew.agents)}")
        for i, agent in enumerate(crew.agents, 1):
            print(f"{i}. {agent.role}")
        
        # Информация о задачах
        print(f"\nКоличество задач: {len(crew.tasks)}")
        for i, task in enumerate(crew.tasks, 1):
            print(f"{i}. {task.description[:50]}...")
        
        print(f"\nРежим выполнения: {crew.process}")
        print(f"Память включена: {crew.memory}")
        
    except Exception as e:
        print(f"Ошибка анализа: {e}")

# Демонстрационные функции
def demo_sequential_crew():
    """Демонстрация последовательной команды"""
    topic = "Квантовые вычисления в криптографии: прорывы 2025 года"
    print("🔄 Демонстрация последовательной команды")
    result = run_research_project(topic, use_hierarchical=False)
    return result

def demo_hierarchical_crew():
    """Демонстрация иерархической команды"""
    topic = "Генеративный ИИ в образовании: трансформация обучения"
    print("🏢 Демонстрация иерархической команды")
    result = run_research_project(topic, use_hierarchical=True)
    return result

def compare_approaches(topic: str):
    """Сравнение двух подходов на одной теме"""
    print(f"⚡ СРАВНЕНИЕ ПОДХОДОВ: {topic}")
    print("="*80)
    
    print("\n1️⃣ Запуск последовательной команды:")
    sequential_result = run_research_project(f"{topic} (Последовательный)", False)
    
    print("\n2️⃣ Запуск иерархической команды:")
    hierarchical_result = run_research_project(f"{topic} (Иерархический)", True)
    
    return sequential_result, hierarchical_result

# Основная программа
if __name__ == "__main__":
    print("🎯 CrewAI Research System - Демонстрация")
    print("="*80)
    
    # Проверка настроек
    if not os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY") == "your-openai-api-key-here":
        print("⚠️  Установите OPENAI_API_KEY в переменных окружения")
        print("⚠️  Установите SERPER_API_KEY для веб-поиска")
        print("\nПример настройки:")
        print("export OPENAI_API_KEY='your-actual-key'")
        print("export SERPER_API_KEY='your-actual-key'")
    else:
        # Выбор режима демонстрации
        print("\nВыберите режим демонстрации:")
        print("1. Последовательная команда")
        print("2. Иерархическая команда")
        print("3. Сравнение подходов")
        print("4. Пользовательская тема")
        
        try:
            choice = input("\nВведите номер (1-4): ").strip()
            
            if choice == "1":
                demo_sequential_crew()
            elif choice == "2":
                demo_hierarchical_crew()
            elif choice == "3":
                compare_approaches("Искусственный интеллект в медицине")
            elif choice == "4":
                custom_topic = input("Введите тему для исследования: ").strip()
                if custom_topic:
                    run_research_project(custom_topic)
                else:
                    print("Тема не может быть пустой")
            else:
                print("Неверный выбор, запускаю демонстрацию по умолчанию...")
                demo_sequential_crew()
                
        except KeyboardInterrupt:
            print("\n👋 Программа прервана пользователем")
        except Exception as e:
            print(f"❌ Неожиданная ошибка: {e}")

# Дополнительные утилиты
def export_crew_config(crew: Crew, filename: str):
    """Экспорт конфигурации команды в файл"""
    try:
        config = {
            "agents": [{"role": agent.role, "goal": agent.goal} for agent in crew.agents],
            "tasks": [{"description": task.description[:100]} for task in crew.tasks],
            "process": str(crew.process),
            "memory": crew.memory
        }
        
        import json
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Конфигурация команды сохранена в {filename}")
        
    except Exception as e:
        print(f"❌ Ошибка сохранения конфигурации: {e}")

def load_crew_from_config(filename: str):
    """Загрузка команды из файла конфигурации (заготовка)"""
    # Эта функция может быть расширена для загрузки конфигурации
    # из внешних файлов (YAML, JSON)
    print(f"🔄 Загрузка конфигурации из {filename} (функция в разработке)")
    pass