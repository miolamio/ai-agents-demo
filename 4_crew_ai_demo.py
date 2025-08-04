"""
Демонстрация Crew AI: Команда для исследования темы
Показывает работу нескольких агентов с разными ролями
"""

# 1. Импорты
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# 2. Создание агентов
def create_researcher():
    """Создает агента-исследователя"""
    return Agent(
        role='Исследователь',
        goal='Найти ключевую информацию по теме',
        backstory='''Ты опытный исследователь, специализирующийся на поиске 
        и структурировании информации. Ты умеешь находить самые важные факты 
        и представлять их в понятном виде.''',
        verbose=True,
        allow_delegation=False,
        llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
    )

def create_analyst():
    """Создает агента-аналитика"""
    return Agent(
        role='Аналитик',
        goal='Проанализировать собранную информацию и сделать выводы',
        backstory='''Ты опытный аналитик с большим опытом структурирования данных 
        и создания кратких, но информативных резюме. Ты можешь выделить главное 
        из большого объема информации.''',
        verbose=True,
        allow_delegation=False,
        llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
    )

def main():
    # Загрузка переменных окружения
    load_dotenv()
    
    # Проверка наличия ключа
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Ошибка: Не найден OPENAI_API_KEY в файле .env")
        print("Создайте файл .env и добавьте ваш ключ OpenAI")
        exit(1)
    
    print("=== Crew AI Demo: Исследовательская команда ===\\n")
    
    try:
        # Тема исследования
        topic = "Применение ИИ в медицине"
        print(f"Тема исследования: {topic}\\n")
        
        # Создание агентов
        researcher = create_researcher()
        analyst = create_analyst()
        
        # 3. Создание задач
        # Задача для исследователя
        research_task = Task(
            description=f'''Найди 3 ключевых факта о теме "{topic}".
            
            Каждый факт должен:
            1. Быть конкретным и точным
            2. Содержать практическую информацию
            3. Быть актуальным на сегодняшний день
            
            Представь результат в виде пронумерованного списка.''',
            agent=researcher,
            expected_output="Список из 3 пронумерованных фактов о применении ИИ в медицине"
        )
        
        # Задача для аналитика  
        analysis_task = Task(
            description=f'''На основе фактов, собранных исследователем, создай краткое резюме 
            о теме "{topic}".
            
            Резюме должно:
            1. Быть не длиннее 2-3 предложений
            2. Выделять главные тенденции
            3. Показывать значимость темы
            
            Начни резюме словами "Резюме: "''',
            agent=analyst,
            expected_output="Краткое резюме в 2-3 предложениях о применении ИИ в медицине"
        )
        
        # 4. Создание команды (Crew)
        # Объединяем агентов и задачи
        crew = Crew(
            agents=[researcher, analyst],
            tasks=[research_task, analysis_task],
            verbose=True  # Детальное логирование
        )
        
        print("🔍 Исследователь начинает работу...")
        
        # 5. Запуск
        # Запускаем crew.kickoff()
        result = crew.kickoff()
        
        print(f"\\n📊 Аналитик обрабатывает информацию...")
        print(f"\\n✅ Исследование завершено!")
        print(f"\\n{'-'*50}")
        print("ИТОГОВЫЙ РЕЗУЛЬТАТ:")
        print(f"{'-'*50}")
        print(result)
        
    except Exception as e:
        print(f"❌ Ошибка при работе с Crew AI: {e}")
        print("Убедитесь, что:")
        print("1. Установлены все зависимости: pip install -r requirements.txt")
        print("2. Правильно указан OPENAI_API_KEY в файле .env")
        print("3. Есть доступ к интернету для обращения к OpenAI API")
        print("\\nВозможные решения:")
        print("- Обновите версию crewai: pip install --upgrade crewai")
        print("- Проверьте совместимость версий в requirements.txt")

if __name__ == "__main__":
    main()