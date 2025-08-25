#!/usr/bin/env python3
"""
Пример 2: DSPy программа с оптимизацией

Демонстрирует:
- Создание сигнатуры для задачи Q&A
- Модуль DSPy с логикой обработки
- Компиляцию программы с оптимизатором
- Сравнение до и после оптимизации
"""

import dspy
from typing import List
import random


# Настройка модели (для демо используем заглушку)
class DemoLLM(dspy.LM):
    """Демо-модель для примера"""
    
    def __init__(self):
        super().__init__("demo-model")
        
    def __call__(self, prompt, **kwargs):
        # Эмулируем ответ LLM
        if "вопрос" in prompt.lower() or "question" in prompt.lower():
            return [f"Это ответ на ваш вопрос. Prompt: {prompt[:50]}..."]
        return [f"Обработанный ответ для: {prompt[:50]}..."]


# 1. Определяем сигнатуру задачи
class BasicQA(dspy.Signature):
    """Отвечает на вопросы короткими фактологическими ответами"""
    
    question = dspy.InputField(desc="Вопрос пользователя")
    answer = dspy.OutputField(desc="Краткий фактологический ответ")


class DetailedQA(dspy.Signature):
    """Предоставляет развернутые ответы с объяснениями"""
    
    question = dspy.InputField(desc="Вопрос пользователя")
    context = dspy.InputField(desc="Дополнительный контекст", default="")
    answer = dspy.OutputField(desc="Развернутый ответ с объяснениями")
    confidence = dspy.OutputField(desc="Уровень уверенности от 0 до 1")


# 2. Создаем модуль DSPy
class QAAgent(dspy.Module):
    """Агент вопросов и ответов с поддержкой оптимизации"""
    
    def __init__(self):
        super().__init__()
        
        # Основные компоненты
        self.basic_qa = dspy.Predict(BasicQA)
        self.detailed_qa = dspy.ChainOfThought(DetailedQA)
        self.reasoning = dspy.ChainOfThought("question -> reasoning, answer")
        
    def forward(self, question, use_detailed=False):
        """
        Основная логика агента
        
        Args:
            question (str): Вопрос пользователя
            use_detailed (bool): Использовать развернутый ответ
            
        Returns:
            dspy.Prediction: Результат обработки
        """
        
        if use_detailed:
            # Используем развернутый режим с рассуждениями
            result = self.detailed_qa(
                question=question,
                context="Это демонстрационный контекст для обучения"
            )
            return result
        else:
            # Простой режим
            result = self.basic_qa(question=question)
            return result


# 3. Функция оценки качества
def validate_answer(example, prediction, trace=None):
    """
    Метрика для оценки качества ответов
    
    Args:
        example: Пример из датасета
        prediction: Предсказание модели
        trace: Трассировка выполнения (опционально)
        
    Returns:
        float: Оценка от 0 до 1
    """
    
    # Для демо используем простые эвристики
    answer = prediction.answer if hasattr(prediction, 'answer') else str(prediction)
    expected = example.answer if hasattr(example, 'answer') else example.get('answer', '')
    
    # Проверки качества
    score = 0.0
    
    # Длина ответа (не слишком короткий и не слишком длинный)
    if 10 <= len(answer) <= 200:
        score += 0.3
    
    # Содержит ключевые слова из вопроса
    question_words = set(example.question.lower().split())
    answer_words = set(answer.lower().split())
    overlap = len(question_words.intersection(answer_words))
    if overlap > 0:
        score += 0.4
        
    # Не является очевидно неправильным
    if not ("не знаю" in answer.lower() or "ошибка" in answer.lower()):
        score += 0.3
    
    return min(score, 1.0)


# 4. Создание тренировочного набора
def create_training_data():
    """Создает демонстрационный набор данных для обучения"""
    
    examples = [
        dspy.Example(
            question="Что такое машинное обучение?",
            answer="Машинное обучение - это область ИИ, где алгоритмы учатся на данных"
        ),
        dspy.Example(
            question="Как работают нейронные сети?",
            answer="Нейронные сети имитируют работу мозга через связанные узлы"
        ),
        dspy.Example(
            question="Что такое Python?",
            answer="Python - высокоуровневый язык программирования"
        ),
        dspy.Example(
            question="Для чего нужен DSPy?",
            answer="DSPy помогает программировать языковые модели декларативно"
        ),
        dspy.Example(
            question="Что такое prompt engineering?",
            answer="Prompt engineering - искусство создания эффективных запросов к LLM"
        )
    ]
    
    return examples


def main():
    """Демонстрация DSPy оптимизации"""
    
    print("=== Пример 2: DSPy программа с оптимизацией ===\n")
    
    # Настройка DSPy
    demo_lm = DemoLLM()
    dspy.configure(lm=demo_lm)
    
    # Создаем программу
    qa_agent = QAAgent()
    
    # Подготавливаем данные
    trainset = create_training_data()
    
    print("1. Тестирование неоптимизированной программы:")
    print("-" * 50)
    
    test_questions = [
        "Что такое искусственный интеллект?",
        "Как изучать программирование?",
        "Зачем нужна оптимизация промптов?"
    ]
    
    # Тест до оптимизации
    unoptimized_results = []
    for question in test_questions:
        result = qa_agent(question=question)
        score = validate_answer(
            dspy.Example(question=question, answer="демо-ответ"), 
            result
        )
        unoptimized_results.append(score)
        print(f"Q: {question}")
        print(f"A: {result.answer if hasattr(result, 'answer') else 'Нет ответа'}")
        print(f"Оценка: {score:.2f}\n")
    
    avg_unoptimized = sum(unoptimized_results) / len(unoptimized_results)
    print(f"Средняя оценка до оптимизации: {avg_unoptimized:.2f}\n")
    
    print("2. Компиляция программы с оптимизатором:")
    print("-" * 50)
    
    # Настройка оптимизатора
    try:
        optimizer = dspy.BootstrapFewShot(
            metric=validate_answer,
            max_bootstrapped_demos=3,
            max_labeled_demos=2
        )
        
        # Компиляция (в реальной ситуации это займет время)
        print("Запуск оптимизации...")
        compiled_qa = optimizer.compile(qa_agent, trainset=trainset)
        
        print("Оптимизация завершена!\n")
        
        print("3. Тестирование оптимизированной программы:")
        print("-" * 50)
        
        # Тест после оптимизации
        optimized_results = []
        for question in test_questions:
            result = compiled_qa(question=question)
            score = validate_answer(
                dspy.Example(question=question, answer="демо-ответ"), 
                result
            )
            optimized_results.append(score)
            print(f"Q: {question}")
            print(f"A: {result.answer if hasattr(result, 'answer') else 'Нет ответа'}")
            print(f"Оценка: {score:.2f}\n")
        
        avg_optimized = sum(optimized_results) / len(optimized_results)
        print(f"Средняя оценка после оптимизации: {avg_optimized:.2f}")
        
        improvement = ((avg_optimized - avg_unoptimized) / avg_unoptimized) * 100
        print(f"Улучшение: {improvement:.1f}%\n")
        
        # Сохранение оптимизированной программы
        compiled_qa.save("examples/optimized_qa_agent.json")
        print("Оптимизированная программа сохранена в optimized_qa_agent.json")
        
    except Exception as e:
        print(f"Ошибка при оптимизации: {e}")
        print("В демо-режиме показываем концепцию оптимизации")
        
        # Эмулируем улучшение для демо
        print("Эмулируем улучшение после оптимизации:")
        for i, (question, old_score) in enumerate(zip(test_questions, unoptimized_results)):
            new_score = min(old_score + random.uniform(0.1, 0.3), 1.0)
            print(f"Q: {question}")
            print(f"Старая оценка: {old_score:.2f} → Новая: {new_score:.2f}")
    
    print("\n4. Инспекция промптов:")
    print("-" * 50)
    
    # Показываем, как можно инспектировать промпты
    print("Для инспекции промптов используйте:")
    print("dspy.inspect_history(n=1)  # Показать последний промпт")
    print("\nПример структуры промпта после оптимизации:")
    print("System: Отвечайте на вопросы короткими фактологическими ответами")
    print("Few-shot examples: [автоматически сгенерированные примеры]")
    print("User: [текущий вопрос]")


if __name__ == "__main__":
    # Проверяем установку DSPy
    try:
        import dspy
        main()
    except ImportError:
        print("Для запуска примера установите DSPy:")
        print("pip install dspy-ai")