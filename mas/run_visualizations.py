"""
Простой скрипт для запуска всех визуализаций LangGraph
Запустите после основной работы системы: python run_visualizations.py
"""

# Импортируем функции из основного файла
from langgraph_research_system import (
    show_graph_ascii,
    show_graph_mermaid, 
    save_graph_png,
    create_interactive_graph,
    visualize_graph
)

def run_all_visualizations():
    """Запуск всех доступных визуализаций"""
    print("🎨 Создание всех визуализаций графа...")
    print("=" * 60)
    
    try:
        # 1. ASCII структура (быстро)
        print("\n📝 1. ASCII структура графа:")
        print("-" * 40)
        show_graph_ascii()
        
        # 2. Mermaid код (сохраняется в файл)
        print("\n🌊 2. Mermaid диаграмма:")
        print("-" * 40)
        show_graph_mermaid()
        
        # 3. Интерактивный HTML (требует pyvis)
        print("\n🌐 3. Интерактивная HTML визуализация:")
        print("-" * 40)
        create_interactive_graph()
        
        # 4. PNG изображение (требует mermaid-cli)
        print("\n🖼️ 4. PNG изображение:")
        print("-" * 40)
        save_graph_png()
        
        # 5. Комплексная визуализация (все методы сразу)
        print("\n🎯 5. Комплексная визуализация:")
        print("-" * 40)
        visualize_graph()
        
        print("\n" + "=" * 60)
        print("✅ Все визуализации завершены!")
        print("\n📁 Созданные файлы:")
        print("   • research_graph.mmd - Mermaid код")
        print("   • research_graph_interactive.html - Интерактивная версия")
        print("   • research_graph.png - PNG изображение (если установлен mermaid-cli)")
        print("\n💡 Советы:")
        print("   • Откройте .html файл в браузере для интерактивной визуализации")
        print("   • Вставьте код из .mmd файла в https://mermaid.live")
        print("   • Для PNG установите: npm install -g @mermaid-js/mermaid-cli")
        
    except Exception as e:
        print(f"❌ Общая ошибка: {e}")
        print("💡 Проверьте:")
        print("   • Установлены ли API ключи (OPENAI_API_KEY, TAVILY_API_KEY)")
        print("   • Установлены ли зависимости: pip install pyvis")

def check_dependencies():
    """Проверка установленных зависимостей"""
    print("🔍 Проверка зависимостей...")
    
    # Проверка pyvis
    try:
        import pyvis
        print("✅ PyVis установлен - интерактивная HTML визуализация доступна")
    except ImportError:
        print("⚠️  PyVis не установлен - запустите: pip install pyvis")
    
    # Проверка основных пакетов
    try:
        import langgraph
        import langchain_openai
        print("✅ LangGraph и LangChain установлены")
    except ImportError as e:
        print(f"❌ Отсутствуют основные пакеты: {e}")
    
    # Проверка mermaid-cli (через subprocess)
    import subprocess
    try:
        result = subprocess.run(['mmdc', '--version'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✅ Mermaid CLI установлен - PNG визуализация доступна")
        else:
            print("⚠️  Mermaid CLI не найден - PNG может не работать")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("⚠️  Mermaid CLI не установлен - запустите: npm install -g @mermaid-js/mermaid-cli")
    
    print()

if __name__ == "__main__":
    print("🚀 Запуск визуализатора LangGraph")
    print("=" * 60)
    
    # Проверяем зависимости
    check_dependencies()
    
    # Запускаем все визуализации
    run_all_visualizations()