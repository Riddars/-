FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY server/ /app/server/
RUN pip install --no-cache-dir -r /app/server/requirements.txt

COPY client/ /app/client/
RUN pip install --no-cache-dir -r /app/client/requirements.txt

EXPOSE 5000 8501

CMD ["sh", "-c", "uvicorn server.server:app --host 0.0.0.0 --port 5000 & streamlit run client/client.py --server.port 8501 --server.address 0.0.0.0"]
