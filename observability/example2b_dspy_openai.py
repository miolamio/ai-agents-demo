#!/usr/bin/env python3
"""
Пример 2b: DSPy программа с реальной интеграцией OpenAI

Демонстрирует:
- Настройку DSPy с реальной OpenAI моделью
- Создание простых сигнатур для Q&A
- Базовую оптимизацию промптов
- Сравнение результатов до и после оптимизации
"""

import os
import dspy
from typing import List, Dict, Any
import time

# Загрузка переменных окружения
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Переменные окружения загружены")
except ImportError:
    print("⚠️  python-dotenv не установлен")

# Проверяем доступность библиотек
try:
    import dspy
    print("✅ DSPy доступен")
    DSPY_AVAILABLE = True
except ImportError:
    print("❌ DSPy не установлен")
    print("Установите: pip install dspy-ai")
    DSPY_AVAILABLE = False


# 1. Определяем сигнатуры задач
class SimpleQA(dspy.Signature):
    """Отвечает на вопросы коротко и по существу"""
    
    question = dspy.InputField(desc="Вопрос пользователя")
    answer = dspy.OutputField(desc="Краткий и точный ответ")


class ReasoningQA(dspy.Signature):
    """Отвечает на вопросы с пошаговыми рассуждениями"""
    
    question = dspy.InputField(desc="Вопрос пользователя") 
    reasoning = dspy.OutputField(desc="Пошаговые рассуждения")
    answer = dspy.OutputField(desc="Финальный ответ на основе рассуждений")


class FactCheckQA(dspy.Signature):
    """Проверяет факты и дает обоснованные ответы"""
    
    question = dspy.InputField(desc="Вопрос пользователя")
    confidence = dspy.OutputField(desc="Уровень уверенности (высокий/средний/низкий)")
    sources = dspy.OutputField(desc="Возможные источники информации")
    answer = dspy.OutputField(desc="Ответ с указанием степени достоверности")


# 2. Создаем DSPy модуль
class SmartQAAgent(dspy.Module):
    """Умный агент вопросов и ответов с разными режимами"""
    
    def __init__(self):
        super().__init__()
        
        # Различные режимы работы
        self.simple_qa = dspy.Predict(SimpleQA)
        self.reasoning_qa = dspy.ChainOfThought(ReasoningQA)
        self.fact_check_qa = dspy.ChainOfThought(FactCheckQA)
        
        # Для more complex reasoning
        self.complex_reasoning = dspy.ChainOfThought("question -> analysis, evidence, conclusion")
        
    def forward(self, question: str, mode: str = "simple") -> dspy.Prediction:
        """
        Основная логика агента с разными режимами
        
        Args:
            question: Вопрос пользователя
            mode: Режим работы ("simple", "reasoning", "fact_check")
            
        Returns:
            dspy.Prediction: Результат обработки
        """
        
        if mode == "simple":
            return self.simple_qa(question=question)
            
        elif mode == "reasoning":
            return self.reasoning_qa(question=question)
            
        elif mode == "fact_check":
            return self.fact_check_qa(question=question)
            
        else:
            # Default to simple mode
            return self.simple_qa(question=question)


# 3. Функции оценки качества
def evaluate_answer_quality(example, prediction, trace=None) -> float:
    """
    Оценивает качество ответа по различным критериям
    
    Args:
        example: Ожидаемый пример
        prediction: Предсказание модели  
        trace: Трассировка (опционально)
        
    Returns:
        float: Оценка от 0.0 до 1.0
    """
    
    # Извлекаем ответ из предсказания
    if hasattr(prediction, 'answer'):
        answer = prediction.answer
    elif hasattr(prediction, 'completion'):
        answer = prediction.completion
    else:
        answer = str(prediction)
    
    if not answer or len(answer.strip()) < 5:
        return 0.0
    
    score = 0.0
    
    # 1. Длина ответа (не слишком короткий, не слишком длинный)
    answer_length = len(answer.strip())
    if 20 <= answer_length <= 300:
        score += 0.25
    elif 10 <= answer_length <= 500:
        score += 0.15
    
    # 2. Релевантность к вопросу
    question = example.question if hasattr(example, 'question') else ""
    if question:
        question_words = set(question.lower().split())
        answer_words = set(answer.lower().split())
        overlap = len(question_words.intersection(answer_words))
        if overlap > 0:
            score += 0.3 * min(overlap / len(question_words), 1.0)
    
    # 3. Структурированность ответа
    if any(marker in answer for marker in [':', '.', '?', '!', ',']):
        score += 0.15
    
    # 4. Отсутствие "пустых" ответов
    empty_phrases = ["не знаю", "не могу ответить", "недостаточно информации", "ошибка"]
    if not any(phrase in answer.lower() for phrase in empty_phrases):
        score += 0.3
    
    return min(score, 1.0)


def evaluate_reasoning_quality(example, prediction, trace=None) -> float:
    """Оценивает качество рассуждений"""
    
    if not hasattr(prediction, 'reasoning'):
        return 0.0
    
    reasoning = prediction.reasoning
    if not reasoning or len(reasoning.strip()) < 10:
        return 0.0
    
    score = 0.0
    
    # Проверяем наличие логической структуры
    reasoning_indicators = [
        "потому что", "поскольку", "следовательно", "во-первых", "во-вторых", 
        "таким образом", "в результате", "из этого следует"
    ]
    
    for indicator in reasoning_indicators:
        if indicator in reasoning.lower():
            score += 0.2
    
    # Длина рассуждений
    if len(reasoning.split()) >= 10:
        score += 0.4
    
    return min(score, 1.0)


# 4. Создание обучающих данных
def create_qa_dataset() -> List[dspy.Example]:
    """Создает набор данных для обучения и тестирования"""
    
    examples = [
        dspy.Example(
            question="Что такое машинное обучение?",
            answer="Машинное обучение - это область искусственного интеллекта, которая позволяет компьютерам обучаться и принимать решения на основе данных без явного программирования."
        ),
        dspy.Example(
            question="Как работают нейронные сети?",
            answer="Нейронные сети состоят из связанных узлов (нейронов), которые обрабатывают информацию через веса и функции активации, имитируя работу человеческого мозга."
        ),
        dspy.Example(
            question="Что такое Python?",
            answer="Python - это высокоуровневый язык программирования общего назначения, известный своей простотой и читаемостью кода."
        ),
        dspy.Example(
            question="Для чего используется Git?",
            answer="Git - это система контроля версий, которая позволяет отслеживать изменения в коде, работать в команде и управлять различными версиями проекта."
        ),
        dspy.Example(
            question="Что такое API?",
            answer="API (Application Programming Interface) - это набор правил и протоколов, который позволяет разным программам взаимодействовать друг с другом."
        ),
        dspy.Example(
            question="Как работает интернет?",
            answer="Интернет работает через сеть взаимосвязанных компьютеров, которые обмениваются данными по протоколам TCP/IP, используя маршрутизаторы и серверы."
        )
    ]
    
    return examples


def check_openai_configuration() -> bool:
    """Проверяет конфигурацию OpenAI"""
    
    openai_key = os.getenv("OPENAI_API_KEY")
    
    print(f"\n🔧 Конфигурация OpenAI:")
    print(f"  API Key: {'✅ установлен' if openai_key else '❌ не установлен'}")
    
    if not openai_key:
        print("\n⚠️  ВНИМАНИЕ: OpenAI API ключ не настроен!")
        print("   Добавьте в .env: OPENAI_API_KEY=ваш-ключ")
        return False
    
    return True


def demonstrate_simple_qa():
    """Демонстрация простого Q&A"""
    
    print("\n🎯 Демонстрация простого Q&A")
    print("=" * 50)
    
    # Создаем агента
    qa_agent = SmartQAAgent()
    
    test_questions = [
        "Что такое искусственный интеллект?",
        "Как изучать программирование?", 
        "Что такое облачные вычисления?"
    ]
    
    results = []
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n📋 Вопрос {i}: {question}")
        
        try:
            start_time = time.time()
            
            # Простой режим
            simple_result = qa_agent(question=question, mode="simple")
            simple_time = time.time() - start_time
            
            simple_answer = simple_result.answer if hasattr(simple_result, 'answer') else "Нет ответа"
            print(f"🤖 Простой ответ ({simple_time:.1f}с): {simple_answer}")
            
            # Режим с рассуждениями
            start_time = time.time()
            reasoning_result = qa_agent(question=question, mode="reasoning")
            reasoning_time = time.time() - start_time
            
            if hasattr(reasoning_result, 'reasoning') and hasattr(reasoning_result, 'answer'):
                print(f"🧠 Рассуждения ({reasoning_time:.1f}с): {reasoning_result.reasoning}")
                print(f"📝 Итоговый ответ: {reasoning_result.answer}")
            else:
                reasoning_answer = reasoning_result.answer if hasattr(reasoning_result, 'answer') else "Нет ответа"
                print(f"🧠 Ответ с рассуждениями ({reasoning_time:.1f}с): {reasoning_answer}")
            
            # Оценка качества
            example = dspy.Example(question=question, answer="демо-ответ")
            simple_score = evaluate_answer_quality(example, simple_result)
            reasoning_score = evaluate_reasoning_quality(example, reasoning_result)
            
            print(f"📊 Оценки: Простой={simple_score:.2f} | Рассуждения={reasoning_score:.2f}")
            
            results.append({
                'question': question,
                'simple_score': simple_score,
                'reasoning_score': reasoning_score,
                'simple_time': simple_time,
                'reasoning_time': reasoning_time
            })
            
        except Exception as e:
            print(f"❌ Ошибка при обработке вопроса: {e}")
            continue
    
    # Общая статистика
    if results:
        avg_simple_score = sum(r['simple_score'] for r in results) / len(results)
        avg_reasoning_score = sum(r['reasoning_score'] for r in results) / len(results)
        avg_simple_time = sum(r['simple_time'] for r in results) / len(results)
        avg_reasoning_time = sum(r['reasoning_time'] for r in results) / len(results)
        
        print(f"\n📈 Общая статистика:")
        print(f"   Простые ответы: {avg_simple_score:.2f} (среднее время: {avg_simple_time:.1f}с)")
        print(f"   С рассуждениями: {avg_reasoning_score:.2f} (среднее время: {avg_reasoning_time:.1f}с)")
        
        improvement = ((avg_reasoning_score - avg_simple_score) / avg_simple_score) * 100 if avg_simple_score > 0 else 0
        print(f"   Улучшение качества: {improvement:.1f}%")


def demonstrate_optimization():
    """Демонстрация базовой оптимизации"""
    
    print(f"\n🚀 Демонстрация оптимизации DSPy")
    print("=" * 50)
    
    # Создаем данные для обучения
    training_data = create_qa_dataset()
    
    print(f"📚 Создан обучающий набор: {len(training_data)} примеров")
    
    # Разделяем на тренировочную и тестовую выборки
    train_size = int(0.7 * len(training_data))
    trainset = training_data[:train_size]
    testset = training_data[train_size:]
    
    print(f"   Обучающих примеров: {len(trainset)}")
    print(f"   Тестовых примеров: {len(testset)}")
    
    # Создаем и тестируем неоптимизированную модель
    qa_agent = SmartQAAgent()
    
    print(f"\n🧪 Тестирование неоптимизированной модели:")
    unoptimized_scores = []
    
    for example in testset:
        try:
            result = qa_agent(question=example.question, mode="simple")
            score = evaluate_answer_quality(example, result)
            unoptimized_scores.append(score)
            print(f"   Q: {example.question[:50]}...")
            print(f"   A: {result.answer[:100] if hasattr(result, 'answer') else 'Нет ответа'}...")
            print(f"   Оценка: {score:.2f}\n")
        except Exception as e:
            print(f"   Ошибка: {e}")
            continue
    
    avg_unoptimized = sum(unoptimized_scores) / len(unoptimized_scores) if unoptimized_scores else 0
    print(f"📊 Средняя оценка до оптимизации: {avg_unoptimized:.2f}")
    
    # Базовая оптимизация с Few-Shot примерами  
    try:
        print(f"\n⚙️ Запуск оптимизации...")
        
        # Простая оптимизация - добавляем примеры в промпт
        optimizer = dspy.BootstrapFewShot(
            metric=evaluate_answer_quality,
            max_bootstrapped_demos=2,
            max_labeled_demos=1
        )
        
        optimized_qa = optimizer.compile(qa_agent, trainset=trainset)
        
        print(f"✅ Оптимизация завершена!")
        
        # Тестируем оптимизированную модель
        print(f"\n🧪 Тестирование оптимизированной модели:")
        optimized_scores = []
        
        for example in testset:
            try:
                result = optimized_qa(question=example.question, mode="simple")
                score = evaluate_answer_quality(example, result)
                optimized_scores.append(score)
                print(f"   Q: {example.question[:50]}...")
                print(f"   A: {result.answer[:100] if hasattr(result, 'answer') else 'Нет ответа'}...")
                print(f"   Оценка: {score:.2f}\n")
            except Exception as e:
                print(f"   Ошибка: {e}")
                continue
        
        avg_optimized = sum(optimized_scores) / len(optimized_scores) if optimized_scores else 0
        print(f"📊 Средняя оценка после оптимизации: {avg_optimized:.2f}")
        
        if avg_unoptimized > 0:
            improvement = ((avg_optimized - avg_unoptimized) / avg_unoptimized) * 100
            print(f"🎯 Улучшение: {improvement:.1f}%")
        
        # Сохранение оптимизированной модели
        try:
            optimized_qa.save("optimized_qa_model.json")
            print(f"💾 Модель сохранена в optimized_qa_model.json")
        except Exception as e:
            print(f"⚠️  Не удалось сохранить модель: {e}")
        
    except Exception as e:
        print(f"❌ Ошибка при оптимизации: {e}")
        print(f"   Это может быть связано с ограничениями API или конфигурацией")


def main():
    """Главная функция демонстрации DSPy с OpenAI"""
    
    print("=== DSPy с реальной интеграцией OpenAI ===")
    
    if not DSPY_AVAILABLE:
        print("❌ DSPy недоступен")
        return
    
    # Проверяем конфигурацию OpenAI
    if not check_openai_configuration():
        return
    
    try:
        # Настройка DSPy с OpenAI
        print(f"\n🔧 Настройка DSPy с OpenAI...")
        
        # Инициализируем OpenAI модель (новый API DSPy)
        try:
            # Попробуем новый способ подключения
            lm = dspy.LM(
                model='openai/gpt-4o-mini',
                api_key=os.getenv("OPENAI_API_KEY"),
                max_tokens=1000,
                temperature=0.7
            )
        except Exception as e:
            # Fallback на старый API
            try:
                import openai
                from dspy.clients.openai_client import OpenAIClient
                
                client = OpenAIClient(
                    api_key=os.getenv("OPENAI_API_KEY"),
                    model='gpt-4o-mini'
                )
                lm = client
            except Exception as e2:
                # Последний вариант - прямое использование openai
                print(f"⚠️  Используем прямую интеграцию с OpenAI")
                import openai
                
                class OpenAIWrapper:
                    def __init__(self, model='gpt-4o-mini'):
                        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                        self.model = model
                    
                    def __call__(self, prompt, **kwargs):
                        response = self.client.chat.completions.create(
                            model=self.model,
                            messages=[{"role": "user", "content": str(prompt)}],
                            max_tokens=kwargs.get('max_tokens', 1000),
                            temperature=kwargs.get('temperature', 0.7)
                        )
                        return [response.choices[0].message.content]
                
                lm = OpenAIWrapper()
        
        dspy.configure(lm=lm)
        print(f"✅ DSPy настроен с моделью gpt-4o-mini")
        
        # Выбор режима демонстрации
        print(f"\n🎯 Выберите режим демонстрации:")
        print("   1. Простое Q&A с разными режимами")
        print("   2. Демонстрация оптимизации")
        print("   3. Оба режима")
        
        choice = input("\nВведите номер (1-3) или Enter для обоих режимов: ").strip()
        
        if choice in ['1', '']:
            demonstrate_simple_qa()
            
        elif choice == '2':
            demonstrate_optimization()
            
        elif choice == '3':
            demonstrate_simple_qa()
            demonstrate_optimization()
            
        else:
            print("⚠️  Неверный выбор, запускаю оба режима")
            demonstrate_simple_qa()
            demonstrate_optimization()
        
        print(f"\n✅ Демонстрация DSPy завершена!")
        print(f"🔍 Для просмотра истории промптов используйте:")
        print(f"   dspy.inspect_history(n=3)")
        
    except Exception as e:
        print(f"❌ Ошибка при настройке DSPy: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
