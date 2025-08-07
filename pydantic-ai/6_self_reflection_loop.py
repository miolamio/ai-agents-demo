# self_correcting_coder.py
import os
import subprocess
import tempfile
from pydantic_ai import Agent

if "OPENAI_API_KEY" not in os.environ:
    raise ValueError("Необходимо установить переменную окружения OPENAI_API_KEY")

# Используем более мощную модель, так как задача генерации и отладки кода сложнее
agent = Agent("openai:gpt-4o")

def run_python_code(code: str) -> tuple[bool, str]:
    """Запускает код в виде строки и возвращает (успех, вывод)."""
    try:
        # Используем subprocess для выполнения кода в отдельном процессе
        result = subprocess.run(
            ["python", "-c", code],
            capture_output=True,
            text=True,
            check=True,
            timeout=10
        )
        return True, result.stdout
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        # Возвращаем stderr, если была ошибка
        return False, e.stderr if hasattr(e, 'stderr') and e.stderr else str(e)
    except Exception as e:
        return False, str(e)

def run_simple_test(code: str, test_code: str) -> tuple[bool, str]:
    """Альтернативный механизм тестирования без pytest."""
    # Очищаем код от markdown разметки
    clean_main_code = clean_code(code)
    
    # Объединяем код и тесты в один скрипт
    combined_code = f"""
{clean_main_code}

# Простые тесты
def run_tests():
    results = []
    try:
        # Тест 1: fibonacci(0)
        result_0 = fibonacci(0)
        if result_0 == 0:
            results.append("✓ fibonacci(0) = 0 - PASSED")
        else:
            results.append(f"✗ fibonacci(0) = {{result_0}}, ожидалось 0 - FAILED")
    except Exception as e:
        results.append(f"✗ fibonacci(0) - ERROR: {{e}}")
    
    try:
        # Тест 2: fibonacci(1)
        result_1 = fibonacci(1)
        if result_1 == 1:
            results.append("✓ fibonacci(1) = 1 - PASSED")
        else:
            results.append(f"✗ fibonacci(1) = {{result_1}}, ожидалось 1 - FAILED")
    except Exception as e:
        results.append(f"✗ fibonacci(1) - ERROR: {{e}}")
    
    try:
        # Тест 3: fibonacci(10) (ожидаем 55)
        result_10 = fibonacci(10)
        if result_10 == 55:
            results.append("✓ fibonacci(10) = 55 - PASSED")
        else:
            results.append(f"✗ fibonacci(10) = {{result_10}}, ожидалось 55 - FAILED")
    except Exception as e:
        results.append(f"✗ fibonacci(10) - ERROR: {{e}}")
    
    return results

if __name__ == "__main__":
    test_results = run_tests()
    for result in test_results:
        print(result)
    
    # Проверяем, все ли тесты прошли
    failed_count = sum(1 for r in test_results if "FAILED" in r or "ERROR" in r)
    if failed_count == 0:
        print("\\nВсе тесты прошли успешно!")
    else:
        print(f"\\n{{failed_count}} тест(ов) не прошли.")
        exit(1)
"""
    
    return run_python_code(combined_code)

def clean_code(code: str) -> str:
    """Очищает код от markdown разметки и лишних символов."""
    # Удаляем markdown код блоки
    code = code.replace('```python', '').replace('```', '')
    # Удаляем лишние пробелы в начале и конце
    code = code.strip()
    return code

def run_pytest(code: str, test_code: str) -> tuple[bool, str]:
    """Запускает pytest для переданного кода и тестов."""
    # Очищаем код от markdown разметки
    clean_main_code = clean_code(code)
    clean_test_code = clean_code(test_code)
    
    # Создаем временные файлы для кода и теста
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.py', delete=False) as code_file, \
         tempfile.NamedTemporaryFile(mode='w+', suffix='_test.py', delete=False) as test_file:
        
        code_file.write(clean_main_code)
        code_file_path = code_file.name
        
        # Импортируем тестируемую функцию в тестовый файл
        module_name = os.path.basename(code_file_path).replace('.py', '')
        test_file.write(f"import sys\nimport os\nsys.path.insert(0, os.path.dirname('{code_file_path}'))\n")
        test_file.write(f"from {module_name} import fibonacci\n\n")
        test_file.write(clean_test_code)
        test_file_path = test_file.name

    try:
        # Проверяем, установлен ли pytest
        try:
            subprocess.run(["pytest", "--version"], capture_output=True, check=True, timeout=5)
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False, "ERROR: pytest не установлен. Установите его командой: pip install pytest"
        
        # Запускаем pytest
        result = subprocess.run(
            ["pytest", test_file_path, "-v"],
            capture_output=True,
            text=True,
            timeout=15
        )
        if result.returncode == 0:
            return True, result.stdout
        else:
            return False, result.stdout + "\n" + result.stderr
    except subprocess.TimeoutExpired:
        return False, "ERROR: Тайм-аут при выполнении тестов"
    except Exception as e:
        return False, f"ERROR: Неожиданная ошибка при запуске pytest: {str(e)}"
    finally:
        # Удаляем временные файлы
        try:
            os.unlink(code_file_path)
            os.unlink(test_file_path)
        except OSError:
            pass  # Игнорируем ошибки удаления файлов


async def reflection_loop():
    print("--- Шаг 1: Генерация кода (роль: Разработчик) ---")
    # Промпт, который может привести к неэффективному рекурсивному решению
    code_generation_prompt = (
        "Напиши Python-функцию `fibonacci(n)` для вычисления n-го числа Фибоначчи. "
        "Используй простой рекурсивный подход. Верни только код функции, без пояснений и без markdown разметки. "
        "Не используй ```python или ``` - только чистый Python код."
    )
    initial_code_result = await agent.run(code_generation_prompt)
    initial_code = initial_code_result.output
    print("Сгенерированный код:\n", initial_code)

    print("\n--- Шаг 2: Генерация теста (роль: QA-инженер) ---")
    test_generation_prompt = (
        f"Ты — QA-инженер. Напиши юнит-тест `pytest` для следующей функции:\n\n"
        f"{initial_code}\n\n"
        "Тест должен называться `test_fibonacci`. Проверь базовые случаи (n=0, n=1) "
        "и один обычный случай (например, n=10). Верни только код теста без markdown разметки. "
        "Не используй ```python или ``` - только чистый Python код."
    )
    test_code_result = await agent.run(test_generation_prompt)
    test_code = test_code_result.output
    print("Сгенерированный тест:\n", test_code)

    print("\n--- Шаг 3: Выполнение теста и получение обратной связи ---")
    # Мы ожидаем, что тест может упасть из-за неэффективности или ошибки
    is_success, test_output = run_pytest(initial_code, test_code)
    
    # Если pytest не работает, используем простой тестирующий механизм
    if "pytest не установлен" in test_output:
        print("pytest недоступен, используем простой тестирующий механизм...")
        is_success, test_output = run_simple_test(initial_code, test_code)
    
    print("Результат выполнения теста:")
    print(test_output)

    if is_success:
        print("\nУдивительно, но код прошел тесты с первого раза! Цикл завершен.")
        return initial_code

    print("\n--- Шаг 4: Рефлексия и исправление (роль: Отладчик) ---")
    correction_prompt = (
        f"Твоя функция `fibonacci` не прошла тесты. Вот отчет об ошибке:\n\n"
        f"{test_output}\n\n"
        f"Исходный код функции был:\n\n"
        f"{initial_code}\n\n"
        "Проанализируй ошибку и предоставь исправленную, более эффективную версию функции `fibonacci`. "
        "Вероятно, стоит использовать итеративный подход вместо рекурсии. "
        "Верни только исправленный код функции без markdown разметки. "
        "Не используй ```python или ``` - только чистый Python код."
    )
    corrected_code_result = await agent.run(correction_prompt)
    corrected_code = corrected_code_result.output
    print("Исправленный код:\n", corrected_code)

    print("\n--- Шаг 5: Верификация ---")
    is_success_final, final_test_output = run_pytest(corrected_code, test_code)
    
    # Если pytest не работает, используем простой тестирующий механизм
    if "pytest не установлен" in final_test_output:
        print("pytest недоступен, используем простой тестирующий механизм...")
        is_success_final, final_test_output = run_simple_test(corrected_code, test_code)
    
    print("Результат выполнения теста для исправленного кода:")
    print(final_test_output)

    if is_success_final:
        print("\nОтлично! Исправленный код успешно прошел тесты.")
        return corrected_code
    else:
        print("\nК сожалению, даже исправленный код не прошел тесты. Требуется дальнейшая отладка.")
        return None

if __name__ == "__main__":
    import asyncio
    final_code = asyncio.run(reflection_loop())
    if final_code:
        print("\n--- Итоговый надежный код ---")
        print(final_code)