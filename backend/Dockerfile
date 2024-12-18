# Используем образ с Python 3.10
FROM python:3.10-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем requirements.txt
COPY ../requirements.txt .

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем остальные файлы приложения
COPY . .

# Открываем порт для бэкенда
EXPOSE 8000

# Запускаем приложение
CMD ["python", "backend/app/main.py"]
