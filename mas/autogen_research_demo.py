"""
Демонстрация AutoGen: диалоговая многоагентная система
Система генерации отчёта через беседу между агентами
"""

import os
from typing import Dict, List
import autogen
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

from dotenv import load_dotenv

# Загрузка переменных окружения
load_dotenv()

# Настройка окружения
#os.environ["OPENAI_API_KEY"] = "your-openai-api-key-here"

# Конфигурация модели для всех агентов
llm_config = {
    "model": "gpt-4o-mini",
    "temperature": 0.7,
    "api_key": os.environ.get("OPENAI_API_KEY"),
}

class AutoGenResearchSystem:
    """Система исследования на базе AutoGen с диалоговым взаимодействием"""
    
    def __init__(self):
        self.agents = {}
        self.chat_history = []
        self.setup_agents()
    
    def setup_agents(self):
        """Создание и настройка агентов для исследовательской команды"""
        
        # Агент-исследователь
        self.agents['researcher'] = AssistantAgent(
            name="Researcher",
            system_message="""Вы — ведущий исследователь технологий. Ваши задачи:
            1. Найти актуальную информацию по заданной теме (2025 год)
            2. Проверить достоверность источников
            3. Собрать ключевые факты и статистику
            4. Представить находки в структурированном виде
            
            Если нужна дополнительная информация, попросите Coordinator'а организовать поиск.
            Всегда указывайте, что информация актуальна на 2025 год.""",
            llm_config=llm_config,
        )
        
        # Агент-аналитик  
        self.agents['analyst'] = AssistantAgent(
            name="Analyst", 
            system_message="""Вы — эксперт по анализу данных и трендов. Ваши задачи:
            1. Анализировать информацию, предоставленную Researcher
            2. Выявлять закономерности и тренды
            3. Делать обоснованные выводы и прогнозы
            4. Предоставлять структурированный анализ для Writer
            
            Фокусируйтесь на практической значимости и бизнес-применении.""",
            llm_config=llm_config,
        )
        
        # Агент-писатель
        self.agents['writer'] = AssistantAgent(
            name="Writer",
            system_message="""Вы — профессиональный технический писатель. Ваши задачи:
            1. Создавать структурированные отчёты на основе исследования и анализа
            2. Использовать чёткий и профессиональный стиль
            3. Форматировать текст в Markdown
            4. Включать все ключевые данные и выводы
            
            Структура отчёта:
            # Исполнительное резюме
            # Ключевые находки  
            # Детальный анализ
            # Выводы и рекомендации""",
            llm_config=llm_config,
        )
        
        # Агент-координатор (UserProxy для управления процессом)
        self.agents['coordinator'] = UserProxyAgent(
            name="Coordinator",
            system_message="""Вы координируете работу исследовательской команды. 
            Направляйте диалог между агентами и следите за выполнением всех этапов:
            1. Исследование (Researcher)
            2. Анализ (Analyst)
            3. Написание отчёта (Writer)
            
            Завершите работу, когда отчёт будет готов.""",
            human_input_mode="NEVER",  # Автономная работа
            max_consecutive_auto_reply=10,
            code_execution_config={"use_docker": False},
        )
        
        # Агент-критик для качества
        self.agents['critic'] = AssistantAgent(
            name="Critic",
            system_message="""Вы — критик и эксперт по качеству отчётов. Ваши задачи:
            1. Оценивать качество исследования и анализа
            2. Проверять полноту и актуальность информации
            3. Предлагать улучшения и дополнения
            4. Валидировать финальный отчёт
            
            Будьте конструктивными и фокусируйтесь на практической ценности.""",
            llm_config=llm_config,
        )
    
    def create_group_chat(self, topic: str) -> GroupChat:
        """Создание группового чата для исследовательской команды"""
        
        # Список участников чата (порядок важен для flow)
        participants = [
            self.agents['coordinator'],
            self.agents['researcher'], 
            self.agents['analyst'],
            self.agents['writer'],
            self.agents['critic']
        ]
        
        # Создание группового чата
        group_chat = GroupChat(
            agents=participants,
            messages=[],
            max_round=20,  # Максимальное количество раундов диалога
            speaker_selection_method="round_robin",  # Метод выбора следующего спикера
        )
        
        return group_chat
    
    def run_research_project(self, topic: str, save_to_file: bool = True) -> str:
        """
        Запуск исследовательского проекта через групповой диалог
        
        Args:
            topic: Тема для исследования
            save_to_file: Сохранить результат в файл
            
        Returns:
            str: Результат исследования
        """
        
        print(f"🤖 Запуск AutoGen исследовательской системы")
        print(f"📋 Тема: {topic}")
        print("👥 Участники: Coordinator, Researcher, Analyst, Writer, Critic")
        print("="*80)
        
        try:
            # Создание группового чата
            group_chat = self.create_group_chat(topic)
            
            # Создание менеджера группового чата
            manager = GroupChatManager(
                groupchat=group_chat,
                llm_config=llm_config,
            )
            
            # Начальное сообщение для запуска процесса
            initial_message = f"""Начинаем исследовательский проект по теме: "{topic}"

Этапы работы:
1. Researcher: найти актуальную информацию (факты, статистику, тренды)
2. Analyst: проанализировать данные и выявить закономерности  
3. Writer: создать структурированный отчёт в формате Markdown
4. Critic: оценить качество и предложить улучшения

Начинаем с исследования. Researcher, пожалуйста, найдите ключевую информацию по теме."""
            
            # Запуск группового диалога
            result = self.agents['coordinator'].initiate_chat(
                manager,
                message=initial_message,
            )
            
            # Извлечение финального отчёта из истории чата
            final_report = self.extract_final_report(group_chat.messages)
            
            if save_to_file:
                filename = "research_report_autogen.md"
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(final_report)
                print(f"\n✅ Отчёт сохранён в {filename}")
            
            return final_report
            
        except Exception as e:
            print(f"❌ Ошибка выполнения: {e}")
            return f"Ошибка: {str(e)}"
    
    def extract_final_report(self, messages: List[Dict]) -> str:
        """Извлечение финального отчёта из истории сообщений"""
        
        # Ищем последние сообщения от Writer'а, содержащие отчёт
        report_content = ""
        
        for message in reversed(messages):
            if message.get('name') == 'Writer':
                content = message.get('content', '')
                if '# ' in content and len(content) > 500:  # Признаки отчёта
                    report_content = content
                    break
        
        if not report_content:
            # Если не найден готовый отчёт, собираем ключевую информацию
            research_data = []
            analysis_data = []
            
            for message in messages:
                name = message.get('name', '')
                content = message.get('content', '')
                
                if name == 'Researcher' and len(content) > 100:
                    research_data.append(content)
                elif name == 'Analyst' and len(content) > 100:
                    analysis_data.append(content)
            
            # Создаём базовый отчёт из собранных данных
            report_content = f"""# Исследовательский отчёт (AutoGen)

## Результаты исследования
{chr(10).join(research_data[-2:]) if research_data else 'Данные исследования недоступны'}

## Аналитические выводы  
{chr(10).join(analysis_data[-1:]) if analysis_data else 'Анализ недоступен'}

## Заключение
Отчёт сгенерирован системой AutoGen на основе группового диалога между агентами.
"""
        
        return report_content
    
    def demo_two_agent_conversation(self, topic: str):
        """Демонстрация простого диалога между двумя агентами"""
        
        print(f"💬 Демонстрация диалога двух агентов по теме: {topic}")
        print("="*60)
        
        # Создание простого диалога Researcher -> Writer
        researcher = self.agents['researcher']
        writer = self.agents['writer'] 
        
        # Researcher инициирует диалог
        result = researcher.initiate_chat(
            writer,
            message=f"""Привет! Я провёл исследование по теме "{topic}". 
            Вот ключевые находки:
            
            1. Актуальные тренды в области
            2. Статистические данные за 2025 год
            3. Ключевые игроки рынка
            4. Перспективы развития
            
            Можешь создать краткий отчёт на основе этой информации?""",
            max_turns=4,
        )
        
        return result

def demo_group_research():
    """Демонстрация группового исследования"""
    system = AutoGenResearchSystem()
    topic = "Генеративный ИИ в образовании: революция обучения 2025"
    
    result = system.run_research_project(topic)
    
    print("\n" + "="*80)
    print("РЕЗУЛЬТАТ ГРУППОВОГО ИССЛЕДОВАНИЯ")
    print("="*80)
    print(result)
    
    return result

def demo_simple_conversation():
    """Демонстрация простого диалога"""
    system = AutoGenResearchSystem()
    topic = "Квантовые вычисления в криптографии"
    
    result = system.demo_two_agent_conversation(topic)
    return result

def compare_autogen_approaches():
    """Сравнение разных подходов AutoGen"""
    system = AutoGenResearchSystem()
    
    print("🔄 Сравнение подходов AutoGen")
    print("="*50)
    
    # Простой диалог
    print("\n1️⃣ Простой диалог двух агентов:")
    simple_result = demo_simple_conversation()
    
    # Групповой чат
    print("\n2️⃣ Групповой чат команды:")
    group_result = demo_group_research()
    
    return simple_result, group_result

# Главная программа
if __name__ == "__main__":
    print("🎯 AutoGen Research Demo - Диалоговые агенты")
    print("="*80)
    
    # Проверка настроек
    if not os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY") == "your-openai-api-key-here":
        print("⚠️  Установите OPENAI_API_KEY в переменных окружения")
        print("\nПример: export OPENAI_API_KEY='your-actual-key'")
    else:
        print("\nВыберите демонстрацию:")
        print("1. Простой диалог двух агентов")
        print("2. Групповое исследование (5 агентов)")
        print("3. Сравнение подходов")
        print("4. Пользовательская тема")
        
        try:
            choice = input("\nВведите номер (1-4): ").strip()
            
            if choice == "1":
                demo_simple_conversation()
            elif choice == "2":
                demo_group_research()
            elif choice == "3":
                compare_autogen_approaches()
            elif choice == "4":
                custom_topic = input("Введите тему: ").strip()
                if custom_topic:
                    system = AutoGenResearchSystem()
                    system.run_research_project(custom_topic)
                else:
                    print("Тема не может быть пустой")
            else:
                print("Неверный выбор, запускаю демонстрацию по умолчанию...")
                demo_group_research()
                
        except KeyboardInterrupt:
            print("\n👋 Программа прервана пользователем")
        except Exception as e:
            print(f"❌ Ошибка: {e}")

# Дополнительные утилиты для экспериментов
def create_custom_agent(name: str, role: str, instructions: str) -> AssistantAgent:
    """Создание пользовательского агента"""
    return AssistantAgent(
        name=name,
        system_message=f"""Ваша роль: {role}
        
        Инструкции: {instructions}
        
        Всегда отвечайте конструктивно и помогайте достичь цели исследования.""",
        llm_config=llm_config,
    )

def save_chat_history(messages: List[Dict], filename: str):
    """Сохранение истории чата в файл"""
    try:
        import json
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(messages, f, ensure_ascii=False, indent=2)
        print(f"✅ История чата сохранена в {filename}")
    except Exception as e:
        print(f"❌ Ошибка сохранения истории: {e}")

def analyze_conversation_flow(messages: List[Dict]):
    """Анализ потока диалога между агентами"""
    print("\n📊 АНАЛИЗ ДИАЛОГА")
    print("="*40)
    
    speakers = {}
    for msg in messages:
        speaker = msg.get('name', 'Unknown')
        speakers[speaker] = speakers.get(speaker, 0) + 1
    
    print("Участники диалога:")
    for speaker, count in speakers.items():
        print(f"- {speaker}: {count} сообщений")
    
    print(f"\nОбщее количество сообщений: {len(messages)}")
    print(f"Активных участников: {len(speakers)}")

# Экспериментальные функции
def experiment_with_roles():
    """Эксперимент с различными ролями агентов"""
    print("🧪 Эксперимент с ролями агентов")
    
    # Создание агентов с необычными ролями
    devil_advocate = create_custom_agent(
        "DevilAdvocate",
        "Адвокат дьявола",
        "Критикуйте все предложения и находите слабые места в исследовании"
    )
    
    optimist = create_custom_agent(
        "Optimist", 
        "Оптимист",
        "Находите положительные стороны и возможности во всех находках"
    )
    
    # Можно создать диалог между ними
    print("Создание экспериментальных агентов завершено")
    return devil_advocate, optimist