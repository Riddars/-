# Dockerfile для фронтенда

# Используем официальный Python образ
FROM python:3.10-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем requirements.txt
COPY ../requirements.txt .

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь проект
COPY . .

# Открываем порт для Streamlit
EXPOSE 8501

# Запускаем Streamlit приложение
CMD ["streamlit", "run", "frontend/main.py"]
