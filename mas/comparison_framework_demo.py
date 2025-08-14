"""
Демонстрационный скрипт сравнения фреймворков
Прямое сравнение LangGraph, CrewAI и AutoGen на одной задаче
"""

import os
import time
import json
from typing import Dict, Any, Optional

# Настройка окружения (все ключи должны быть установлены)
os.environ.setdefault("OPENAI_API_KEY", "your-openai-api-key-here")
os.environ.setdefault("TAVILY_API_KEY", "your-tavily-api-key-here") 
os.environ.setdefault("SERPER_API_KEY", "your-serper-api-key-here")

class FrameworkComparison:
    """Класс для сравнения производительности разных фреймворков"""
    
    def __init__(self):
        self.results = {}
        self.test_topic = "Влияние искусственного интеллекта на рынок труда в 2025 году"
        
    def test_langgraph(self) -> Dict[str, Any]:
        """Тест LangGraph системы"""
        print("🔗 Тестирование LangGraph...")
        start_time = time.time()
        
        try:
            # Импорт и запуск LangGraph системы
            from langgraph_research_system import run_research_system
            
            result = run_research_system(self.test_topic, verbose=False)
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            return {
                "framework": "LangGraph",
                "success": True,
                "execution_time": round(execution_time, 2),
                "result_length": len(result) if result else 0,
                "result_preview": result[:200] + "..." if result and len(result) > 200 else result,
                "error": None
            }
            
        except Exception as e:
            return {
                "framework": "LangGraph", 
                "success": False,
                "execution_time": time.time() - start_time,
                "result_length": 0,
                "result_preview": "",
                "error": str(e)
            }
    
    def test_crewai(self) -> Dict[str, Any]:
        """Тест CrewAI системы"""
        print("👥 Тестирование CrewAI...")
        start_time = time.time()
        
        try:
            # Импорт и запуск CrewAI системы
            from crewai_research_system import run_research_project
            
            result = run_research_project(self.test_topic, use_hierarchical=False)
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Результат CrewAI может быть объектом, конвертируем в строку
            result_str = str(result) if result else ""
            
            return {
                "framework": "CrewAI",
                "success": True,
                "execution_time": round(execution_time, 2),
                "result_length": len(result_str),
                "result_preview": result_str[:200] + "..." if len(result_str) > 200 else result_str,
                "error": None
            }
            
        except Exception as e:
            return {
                "framework": "CrewAI",
                "success": False, 
                "execution_time": time.time() - start_time,
                "result_length": 0,
                "result_preview": "",
                "error": str(e)
            }
    
    def test_autogen(self) -> Dict[str, Any]:
        """Тест AutoGen системы"""
        print("🤖 Тестирование AutoGen...")
        start_time = time.time()
        
        try:
            # Импорт и запуск AutoGen системы
            from autogen_research_demo import AutoGenResearchSystem
            
            system = AutoGenResearchSystem()
            result = system.run_research_project(self.test_topic, save_to_file=False)
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            return {
                "framework": "AutoGen",
                "success": True,
                "execution_time": round(execution_time, 2),
                "result_length": len(result) if result else 0,
                "result_preview": result[:200] + "..." if result and len(result) > 200 else result,
                "error": None
            }
            
        except Exception as e:
            return {
                "framework": "AutoGen",
                "success": False,
                "execution_time": time.time() - start_time,
                "result_length": 0,
                "result_preview": "",
                "error": str(e)
            }
    
    def run_comparison(self, save_results: bool = True) -> Dict[str, Dict[str, Any]]:
        """Запуск полного сравнения всех фреймворков"""
        
        print("🚀 ЗАПУСК СРАВНИТЕЛЬНОГО ТЕСТИРОВАНИЯ ФРЕЙМВОРКОВ")
        print("="*80)
        print(f"📋 Тестовая задача: {self.test_topic}")
        print("🔄 Тестируем: LangGraph, CrewAI, AutoGen")
        print("="*80)
        
        # Проверка API ключей
        missing_keys = []
        required_keys = ["OPENAI_API_KEY", "TAVILY_API_KEY", "SERPER_API_KEY"]
        
        for key in required_keys:
            if not os.environ.get(key) or os.environ.get(key).startswith("your-"):
                missing_keys.append(key)
        
        if missing_keys:
            print(f"⚠️  Отсутствуют API ключи: {', '.join(missing_keys)}")
            print("Некоторые тесты могут не работать корректно")
            print()
        
        # Запуск тестов
        results = {}
        
        # Тест 1: LangGraph
        results['langgraph'] = self.test_langgraph()
        print(f"✅ LangGraph завершён за {results['langgraph']['execution_time']}с")
        
        # Небольшая пауза между тестами
        time.sleep(2)
        
        # Тест 2: CrewAI  
        results['crewai'] = self.test_crewai()
        print(f"✅ CrewAI завершён за {results['crewai']['execution_time']}с")
        
        # Пауза
        time.sleep(2)
        
        # Тест 3: AutoGen
        results['autogen'] = self.test_autogen()
        print(f"✅ AutoGen завершён за {results['autogen']['execution_time']}с")
        
        # Сохранение результатов
        if save_results:
            self.save_comparison_results(results)
        
        # Вывод сравнительной таблицы
        self.print_comparison_table(results)
        
        return results
    
    def print_comparison_table(self, results: Dict[str, Dict[str, Any]]):
        """Вывод сравнительной таблицы результатов"""
        
        print("\n" + "="*80)
        print("📊 РЕЗУЛЬТАТЫ СРАВНЕНИЯ ФРЕЙМВОРКОВ")
        print("="*80)
        
        # Заголовок таблицы
        print(f"{'Фреймворк':<15} {'Статус':<10} {'Время (с)':<10} {'Размер':<12} {'Ошибки':<20}")
        print("-" * 80)
        
        for framework_key, result in results.items():
            framework = result['framework']
            status = "✅ Успех" if result['success'] else "❌ Ошибка"
            exec_time = f"{result['execution_time']:.2f}"
            result_size = f"{result['result_length']} симв."
            error = result['error'][:15] + "..." if result['error'] and len(result['error']) > 15 else result['error'] or "-"
            
            print(f"{framework:<15} {status:<10} {exec_time:<10} {result_size:<12} {error:<20}")
        
        # Анализ результатов
        print("\n" + "="*40)
        print("📈 АНАЛИЗ ПРОИЗВОДИТЕЛЬНОСТИ")
        print("="*40)
        
        successful_results = {k: v for k, v in results.items() if v['success']}
        
        if successful_results:
            # Самый быстрый
            fastest = min(successful_results.items(), key=lambda x: x[1]['execution_time'])
            print(f"🏆 Самый быстрый: {fastest[1]['framework']} ({fastest[1]['execution_time']:.2f}с)")
            
            # Самый подробный результат
            most_detailed = max(successful_results.items(), key=lambda x: x[1]['result_length'])
            print(f"📝 Самый подробный результат: {most_detailed[1]['framework']} ({most_detailed[1]['result_length']} символов)")
            
            # Среднее время выполнения
            avg_time = sum(r['execution_time'] for r in successful_results.values()) / len(successful_results)
            print(f"⏱️  Среднее время: {avg_time:.2f}с")
        
        # Рекомендации
        print("\n" + "="*40)
        print("💡 РЕКОМЕНДАЦИИ")
        print("="*40)
        
        print("🔗 LangGraph: Лучший контроль и отладка, сложная настройка")
        print("👥 CrewAI: Простота использования, быстрый старт")
        print("🤖 AutoGen: Диалоговый подход, естественные взаимодействия")
    
    def save_comparison_results(self, results: Dict[str, Dict[str, Any]]):
        """Сохранение результатов сравнения в файл"""
        try:
            # Добавляем метаданные
            output_data = {
                "test_metadata": {
                    "topic": self.test_topic,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "frameworks_tested": len(results)
                },
                "results": results
            }
            
            filename = "framework_comparison_results.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            
            print(f"\n💾 Результаты сохранены в {filename}")
            
        except Exception as e:
            print(f"❌ Ошибка сохранения результатов: {e}")
    
    def detailed_feature_comparison(self):
        """Подробное сравнение возможностей фреймворков"""
        
        print("\n🔍 ДЕТАЛЬНОЕ СРАВНЕНИЕ ВОЗМОЖНОСТЕЙ")
        print("="*80)
        
        features = {
            "LangGraph": {
                "Сложность настройки": "Высокая",
                "Контроль над процессом": "Максимальный", 
                "Отладка": "Отличная (LangSmith)",
                "Условная логика": "Встроенная",
                "Персистентность": "Да (Checkpoints)",
                "Streaming": "Да",
                "Кривая обучения": "Крутая",
                "Экосистема": "LangChain"
            },
            "CrewAI": {
                "Сложность настройки": "Низкая",
                "Контроль над процессом": "Ограниченный",
                "Отладка": "Базовая (verbose)",
                "Условная логика": "Сложная реализация", 
                "Персистентность": "Да (память агентов)",
                "Streaming": "Нет",
                "Кривая обучения": "Пологая",
                "Экосистема": "Независимая"
            },
            "AutoGen": {
                "Сложность настройки": "Средняя",
                "Контроль над процессом": "Гибкий",
                "Отладка": "История диалогов",
                "Условная логика": "Через диалог",
                "Персистентность": "Частично",
                "Streaming": "Да",
                "Кривая обучения": "Средняя", 
                "Экосистема": "Microsoft"
            }
        }
        
        # Вывод таблицы сравнения
        feature_names = list(next(iter(features.values())).keys())
        
        # Заголовок
        print(f"{'Характеристика':<25} {'LangGraph':<20} {'CrewAI':<20} {'AutoGen':<20}")
        print("-" * 85)
        
        # Строки с данными
        for feature in feature_names:
            langgraph_val = features["LangGraph"][feature]
            crewai_val = features["CrewAI"][feature]
            autogen_val = features["AutoGen"][feature]
            
            print(f"{feature:<25} {langgraph_val:<20} {crewai_val:<20} {autogen_val:<20}")

def quick_comparison():
    """Быстрое сравнение без полного запуска"""
    print("⚡ БЫСТРОЕ СРАВНЕНИЕ ФРЕЙМВОРКОВ")
    print("="*60)
    
    comparison = FrameworkComparison()
    comparison.detailed_feature_comparison()
    
    print("\n💭 Выводы:")
    print("• LangGraph — для сложных production систем с максимальным контролем")
    print("• CrewAI — для быстрого прототипирования и простых workflow")
    print("• AutoGen — для исследований и диалоговых систем")

def full_performance_test():
    """Полное тестирование производительности"""
    comparison = FrameworkComparison()
    results = comparison.run_comparison()
    return results

def interactive_comparison():
    """Интерактивное сравнение с выбором пользователя"""
    print("🎛️  ИНТЕРАКТИВНОЕ СРАВНЕНИЕ ФРЕЙМВОРКОВ")
    print("="*60)
    
    comparison = FrameworkComparison()
    
    # Позволяем пользователю выбрать тему
    custom_topic = input("Введите тему для сравнения (или Enter для использования по умолчанию): ").strip()
    if custom_topic:
        comparison.test_topic = custom_topic
        print(f"✅ Установлена тема: {custom_topic}")
    
    # Выбор фреймворков для тестирования
    print("\nКакие фреймворки протестировать?")
    print("1. Все (LangGraph + CrewAI + AutoGen)")
    print("2. Только LangGraph и CrewAI")
    print("3. Выборочно")
    
    choice = input("Выбор (1-3): ").strip()
    
    if choice == "1":
        results = comparison.run_comparison()
    elif choice == "2":
        results = {}
        results['langgraph'] = comparison.test_langgraph()
        results['crewai'] = comparison.test_crewai()
        comparison.print_comparison_table(results)
    elif choice == "3":
        results = {}
        if input("Тестировать LangGraph? (y/n): ").lower() == 'y':
            results['langgraph'] = comparison.test_langgraph()
        if input("Тестировать CrewAI? (y/n): ").lower() == 'y':
            results['crewai'] = comparison.test_crewai()
        if input("Тестировать AutoGen? (y/n): ").lower() == 'y':
            results['autogen'] = comparison.test_autogen()
        
        if results:
            comparison.print_comparison_table(results)
        else:
            print("Фреймворки не выбраны для тестирования")
    else:
        print("Неверный выбор")
        return
    
    return results

# Главная программа
if __name__ == "__main__":
    print("🎯 Framework Comparison Tool")
    print("Инструмент для сравнения LangGraph, CrewAI и AutoGen")
    print("="*80)
    
    # Проверка зависимостей
    missing_deps = []
    try:
        import autogen
    except ImportError:
        missing_deps.append("autogen-agentchat")
    
    try:
        import crewai
    except ImportError:
        missing_deps.append("crewai")
    
    try:
        import langgraph
    except ImportError:
        missing_deps.append("langgraph")
    
    if missing_deps:
        print(f"⚠️  Отсутствуют зависимости: {', '.join(missing_deps)}")
        print("Установите: pip install " + " ".join(missing_deps))
        print()
    
    # Меню выбора
    print("\nВыберите режим сравнения:")
    print("1. Быстрое сравнение (только таблица возможностей)")
    print("2. Полное тестирование производительности")
    print("3. Интерактивное сравнение")
    
    try:
        choice = input("\nВведите номер (1-3): ").strip()
        
        if choice == "1":
            quick_comparison()
        elif choice == "2":
            full_performance_test()
        elif choice == "3":
            interactive_comparison()
        else:
            print("Неверный выбор, запускаю быстрое сравнение...")
            quick_comparison()
            
    except KeyboardInterrupt:
        print("\n👋 Сравнение прервано пользователем")
    except Exception as e:
        print(f"❌ Неожиданная ошибка: {e}")

# Утилиты для анализа
def analyze_execution_logs():
    """Анализ логов выполнения фреймворков"""
    print("📋 Анализ логов выполнения...")
    
    log_files = [
        "crew_execution_log.txt",
        "hierarchical_crew_log.txt", 
        "research_report_langgraph.md",
        "research_report_crewai.md",
        "research_report_autogen.md"
    ]
    
    existing_logs = []
    for log_file in log_files:
        if os.path.exists(log_file):
            size = os.path.getsize(log_file)
            existing_logs.append(f"• {log_file} ({size} байт)")
    
    if existing_logs:
        print("Найденные файлы результатов:")
        for log in existing_logs:
            print(log)
    else:
        print("Файлы результатов не найдены. Запустите тесты сначала.")

def cleanup_test_files():
    """Очистка тестовых файлов"""
    test_files = [
        "research_report_langgraph.md",
        "research_report_crewai.md", 
        "research_report_autogen.md",
        "crew_execution_log.txt",
        "hierarchical_crew_log.txt",
        "framework_comparison_results.json",
        "research_graph.png"
    ]
    
    cleaned = 0
    for file in test_files:
        if os.path.exists(file):
            os.remove(file)
            cleaned += 1
    
    print(f"🧹 Очищено {cleaned} тестовых файлов")