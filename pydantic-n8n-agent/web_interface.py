"""
web_interface.py - Веб-интерфейс для Web Research Agent

Streamlit приложение для удобного использования агента через браузер.
Запуск: streamlit run web_interface.py
"""

import streamlit as st
import asyncio
import json
from datetime import datetime
import os
from dotenv import load_dotenv
from web_research_agent import WebResearchAssistant, ResearchOutput
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Загружаем переменные окружения
load_dotenv()

# ============================================================================
# Конфигурация Streamlit
# ============================================================================

st.set_page_config(
    page_title="🔍 Web Research Agent",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Кастомные стили
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .stAlert {
        border-radius: 10px;
    }
    .finding-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# Инициализация состояния сессии
# ============================================================================

if 'assistant' not in st.session_state:
    st.session_state.assistant = None

if 'history' not in st.session_state:
    st.session_state.history = []

if 'api_keys_set' not in st.session_state:
    st.session_state.api_keys_set = False

# ============================================================================
# Функции помощники
# ============================================================================

def initialize_assistant(openai_key: str, tavily_key: str):
    """Инициализировать ассистента с API ключами"""
    try:
        st.session_state.assistant = WebResearchAssistant(
            openai_api_key=openai_key,
            tavily_api_key=tavily_key
        )
        st.session_state.api_keys_set = True
        return True
    except Exception as e:
        st.error(f"Ошибка инициализации: {str(e)}")
        return False


def get_sentiment_color(sentiment: str) -> str:
    """Получить цвет для тональности"""
    colors = {
        "positive": "#10b981",
        "neutral": "#6b7280",
        "negative": "#ef4444"
    }
    return colors.get(sentiment, "#6b7280")


def get_confidence_emoji(level: str) -> str:
    """Получить эмодзи для уровня уверенности"""
    emojis = {
        "high": "🟢",
        "medium": "🟡",
        "low": "🔴"
    }
    return emojis.get(level, "⚪")


# ============================================================================
# Боковая панель
# ============================================================================

with st.sidebar:
    st.markdown("## ⚙️ Настройки")
    
    # API ключи
    st.markdown("### 🔑 API Ключи")
    
    # Проверяем переменные окружения
    env_openai = os.getenv("OPENAI_API_KEY", "")
    env_tavily = os.getenv("TAVILY_API_KEY", "")
    
    openai_key = st.text_input(
        "OpenAI API Key",
        value=env_openai,
        type="password",
        help="Получить на https://platform.openai.com/api-keys"
    )
    
    tavily_key = st.text_input(
        "Tavily API Key",
        value=env_tavily,
        type="password",
        help="Получить на https://tavily.com/"
    )
    
    if st.button("💾 Сохранить ключи", type="primary"):
        if openai_key and tavily_key:
            if initialize_assistant(openai_key, tavily_key):
                st.success("✅ Ключи сохранены!")
        else:
            st.error("Пожалуйста, введите оба ключа")
    
    st.divider()
    
    # Управление памятью
    st.markdown("### 🧠 Управление памятью")
    
    if st.button("🗑️ Очистить память"):
        if st.session_state.assistant:
            st.session_state.assistant.clear_memory()
            st.success("Память очищена")
    
    if st.button("📜 Очистить историю"):
        st.session_state.history = []
        st.success("История очищена")
    
    st.divider()
    
    # Статистика
    st.markdown("### 📊 Статистика")
    st.metric("Запросов в истории", len(st.session_state.history))
    
    if st.session_state.history:
        sentiments = [h['output'].sentiment for h in st.session_state.history]
        st.markdown("**Тональность запросов:**")
        for s in ["positive", "neutral", "negative"]:
            count = sentiments.count(s)
            if count > 0:
                st.write(f"{s.capitalize()}: {count}")
    
    st.divider()
    
    # Экспорт
    st.markdown("### 💾 Экспорт данных")
    
    if st.session_state.history:
        # Создаем JSON для экспорта
        export_data = {
            "timestamp": datetime.now().isoformat(),
            "total_queries": len(st.session_state.history),
            "queries": [
                {
                    "query": h['query'],
                    "timestamp": h['timestamp'],
                    "summary": h['output'].summary,
                    "sentiment": h['output'].sentiment,
                    "confidence": h['output'].confidence_level,
                    "findings_count": len(h['output'].key_findings)
                }
                for h in st.session_state.history
            ]
        }
        
        st.download_button(
            label="📥 Скачать историю (JSON)",
            data=json.dumps(export_data, ensure_ascii=False, indent=2),
            file_name=f"research_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

# ============================================================================
# Основной контент
# ============================================================================

# Заголовок
st.markdown('<h1 class="main-header">🔍 Web Research Agent</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #6b7280;">Powered by Pydantic AI & Tavily</p>', unsafe_allow_html=True)

# Проверка инициализации
if not st.session_state.api_keys_set:
    st.warning("⚠️ Пожалуйста, введите API ключи в боковой панели для начала работы")
    st.stop()

# Табы
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Исследование", "📊 Аналитика", "📜 История", "ℹ️ О системе"])

# ============================================================================
# Таб 1: Исследование
# ============================================================================

with tab1:
    # Форма ввода
    with st.form("research_form"):
        col1, col2 = st.columns([4, 1])
        
        with col1:
            query = st.text_area(
                "Введите ваш запрос для исследования:",
                placeholder="Например: Последние достижения в области искусственного интеллекта",
                height=100
            )
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button("🚀 Исследовать", type="primary", use_container_width=True)
    
    # Обработка запроса
    if submitted and query:
        with st.spinner("🔄 Выполняю исследование... Это может занять несколько секунд"):
            try:
                # Выполняем исследование
                result = st.session_state.assistant.process_message_sync(query)
                
                # Сохраняем в историю
                st.session_state.history.append({
                    'query': query,
                    'timestamp': datetime.now().isoformat(),
                    'output': result
                })
                
                # Отображаем результаты
                st.success("✅ Исследование завершено!")
                
                # Метрики
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Тональность",
                        result.sentiment.capitalize(),
                        delta=None,
                        delta_color="normal"
                    )
                
                with col2:
                    confidence_emoji = get_confidence_emoji(result.confidence_level)
                    st.metric(
                        "Уверенность",
                        f"{confidence_emoji} {result.confidence_level.capitalize()}"
                    )
                
                with col3:
                    st.metric(
                        "Ключевых находок",
                        len(result.key_findings)
                    )
                
                with col4:
                    entities_count = (
                        len(result.entities.people) +
                        len(result.entities.organizations) +
                        len(result.entities.locations)
                    )
                    st.metric(
                        "Извлечено сущностей",
                        entities_count
                    )
                
                # Резюме
                st.markdown("### 📝 Резюме")
                st.info(result.summary)
                
                # Ключевые находки
                st.markdown("### 🎯 Ключевые находки")
                
                for i, finding in enumerate(result.key_findings, 1):
                    with st.expander(f"{i}. {finding.title} (Релевантность: {finding.relevance_score}/10)"):
                        st.write(finding.description)
                        st.caption(f"📌 Источник: {finding.source}")
                
                # Категории и сущности
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 📂 Категории")
                    st.write(f"**Основная тема:** {result.categories.main_topic}")
                    if result.categories.subtopics:
                        st.write("**Подтемы:**")
                        for subtopic in result.categories.subtopics:
                            st.write(f"• {subtopic}")
                
                with col2:
                    st.markdown("### 🏷️ Извлеченные сущности")
                    
                    if result.entities.people:
                        st.write("**Люди:**")
                        st.write(", ".join(result.entities.people))
                    
                    if result.entities.organizations:
                        st.write("**Организации:**")
                        st.write(", ".join(result.entities.organizations))
                    
                    if result.entities.locations:
                        st.write("**Места:**")
                        st.write(", ".join(result.entities.locations))
                
                # Рекомендации для дальнейшего исследования
                if result.additional_queries:
                    st.markdown("### 💡 Рекомендации для углубленного изучения")
                    for i, suggestion in enumerate(result.additional_queries, 1):
                        st.write(f"{i}. {suggestion}")
                
                # JSON вывод (опционально)
                with st.expander("🔧 Показать JSON вывод"):
                    json_output = st.session_state.assistant.get_json_output(result)
                    st.code(json_output, language="json")
                    
                    # Кнопка копирования
                    st.download_button(
                        label="📥 Скачать JSON",
                        data=json_output,
                        file_name=f"research_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                
            except Exception as e:
                st.error(f"❌ Ошибка при выполнении исследования: {str(e)}")

# ============================================================================
# Таб 2: Аналитика
# ============================================================================

with tab2:
    st.markdown("### 📊 Аналитика исследований")
    
    if not st.session_state.history:
        st.info("Пока нет данных для анализа. Выполните несколько исследований.")
    else:
        # Подготовка данных
        df_data = []
        for item in st.session_state.history:
            df_data.append({
                'Запрос': item['query'][:50] + '...' if len(item['query']) > 50 else item['query'],
                'Время': item['timestamp'],
                'Тональность': item['output'].sentiment,
                'Уверенность': item['output'].confidence_level,
                'Находок': len(item['output'].key_findings),
                'Сущностей': (
                    len(item['output'].entities.people) +
                    len(item['output'].entities.organizations) +
                    len(item['output'].entities.locations)
                )
            })
        
        df = pd.DataFrame(df_data)
        
        # График тональности
        col1, col2 = st.columns(2)
        
        with col1:
            sentiment_counts = df['Тональность'].value_counts()
            fig_sentiment = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                title="Распределение тональности",
                color_discrete_map={
                    'positive': '#10b981',
                    'neutral': '#6b7280',
                    'negative': '#ef4444'
                }
            )
            st.plotly_chart(fig_sentiment, use_container_width=True)
        
        with col2:
            confidence_counts = df['Уверенность'].value_counts()
            fig_confidence = px.bar(
                x=confidence_counts.index,
                y=confidence_counts.values,
                title="Уровни уверенности",
                labels={'x': 'Уровень', 'y': 'Количество'},
                color=confidence_counts.index,
                color_discrete_map={
                    'high': '#10b981',
                    'medium': '#fbbf24',
                    'low': '#ef4444'
                }
            )
            st.plotly_chart(fig_confidence, use_container_width=True)
        
        # График находок во времени
        if len(df) > 1:
            df['Время_parsed'] = pd.to_datetime(df['Время'])
            fig_timeline = go.Figure()
            
            fig_timeline.add_trace(go.Scatter(
                x=df['Время_parsed'],
                y=df['Находок'],
                mode='lines+markers',
                name='Ключевые находки',
                line=dict(color='#667eea', width=2),
                marker=dict(size=8)
            ))
            
            fig_timeline.add_trace(go.Scatter(
                x=df['Время_parsed'],
                y=df['Сущностей'],
                mode='lines+markers',
                name='Извлеченные сущности',
                line=dict(color='#764ba2', width=2),
                marker=dict(size=8)
            ))
            
            fig_timeline.update_layout(
                title="Динамика исследований",
                xaxis_title="Время",
                yaxis_title="Количество",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_timeline, use_container_width=True)
        
        # Таблица данных
        st.markdown("### 📋 Сводная таблица")
        st.dataframe(df, use_container_width=True)

# ============================================================================
# Таб 3: История
# ============================================================================

with tab3:
    st.markdown("### 📜 История исследований")
    
    if not st.session_state.history:
        st.info("История пуста. Выполните исследование для начала.")
    else:
        # Отображаем историю в обратном порядке (новые сверху)
        for i, item in enumerate(reversed(st.session_state.history), 1):
            with st.expander(f"#{len(st.session_state.history) - i + 1}: {item['query'][:100]}..."):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write(f"**Время:** {item['timestamp']}")
                
                with col2:
                    st.write(f"**Тональность:** {item['output'].sentiment}")
                
                with col3:
                    st.write(f"**Уверенность:** {item['output'].confidence_level}")
                
                st.write(f"**Резюме:** {item['output'].summary}")
                
                if st.button(f"🔄 Повторить запрос", key=f"repeat_{i}"):
                    st.session_state.repeat_query = item['query']
                    st.rerun()

# ============================================================================
# Таб 4: О системе
# ============================================================================

with tab4:
    st.markdown("""
    ### ℹ️ О системе Web Research Agent
    
    **Web Research Agent** - это интеллектуальная система для проведения веб-исследований,
    построенная на основе передовых технологий искусственного интеллекта.
    
    #### 🛠️ Технологии:
    - **Pydantic AI** - фреймворк для построения AI агентов с типизированным выводом
    - **OpenAI GPT** - языковая модель для анализа и генерации текста
    - **Tavily API** - продвинутый поисковый движок для веб-исследований
    - **Streamlit** - фреймворк для создания веб-интерфейса
    
    #### ✨ Возможности:
    - 🔍 Глубокий поиск информации в интернете
    - 📊 Структурированный анализ данных
    - 🧠 Сохранение контекста между запросами
    - 📈 Оценка тональности и уверенности
    - 🏷️ Извлечение ключевых сущностей
    - 💡 Рекомендации для дальнейших исследований
    
    #### 📋 Структура вывода:
    Система возвращает структурированные данные, включающие:
    - Краткое резюме находок
    - Ключевые находки с оценкой релевантности
    - Категоризацию информации
    - Извлеченные сущности (люди, организации, места)
    - Оценку тональности и уровня уверенности
    - Предложения для углубленного изучения
    
    #### 🔗 Полезные ссылки:
    - [Pydantic AI Documentation](https://ai.pydantic.dev/)
    - [Tavily API](https://tavily.com/)
    - [OpenAI Platform](https://platform.openai.com/)
    
    #### 📝 Версия:
    v1.0.0 - Август 2025
    """)

# ============================================================================
# Обработка повторного запроса
# ============================================================================

if 'repeat_query' in st.session_state:
    st.rerun()

# ============================================================================
# Footer
# ============================================================================

st.markdown("---")
st.markdown(
    '<p style="text-align: center; color: #9ca3af;">Built with ❤️ using Pydantic AI</p>',
    unsafe_allow_html=True
)