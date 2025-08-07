# user_model.py
from datetime import datetime
from pydantic import BaseModel, EmailStr, ValidationError, Field

# Шаг 2: Определяем структуру с помощью Pydantic
class User(BaseModel):
    """Модель данных для пользователя с валидацией."""
    id: int
    name: str = Field(min_length=2, description="Полное имя пользователя")
    age: int = Field(gt=0, le=120, description="Возраст пользователя")
    email: EmailStr  # Специальный тип для валидации email
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)

# --- Демонстрация работы ---

# 1. Успешная валидация и приведение типов (type coercion)
clean_data = {
    "id": 123,
    "name": "Alice",
    "age": "35",  # Pydantic автоматически преобразует строку "35" в число 35
    "email": "alice@pydantic.dev"
}

try:
    user = User(**clean_data)
    print("Пользователь успешно создан:")
    print(user)
    # Мы можем обращаться к полям как к атрибутам объекта
    print(f"Имя: {user.name}, Активен: {user.is_active}")
    # Модель можно легко сериализовать в словарь или JSON
    print("В виде словаря:", user.model_dump())
except ValidationError as e:
    print("Ошибка валидации:", e)

print("-" * 20)

# 2. Неуспешная валидация
invalid_data = {
    "id": 456,
    "name": "Bob",
    "age": "ninety", # Не может быть преобразовано в число
    "email": "bob@invalid-email" # Некорректный формат email
}

try:
    user = User(**invalid_data)
except ValidationError as e:
    print("Перехвачена ожидаемая ошибка валидации:")
    print(e)