import uvicorn
import yape
from transformers import AutoTokenizer, AutoModel
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from weaviate.client_base import ConnectionParams
from weaviate import WeaviateClient
from weaviate.classes.config import Property, DataType
import weaviate

class Document(BaseModel):
    content: str
    dataframe: str = None
    keywords: list[str] = []

# FastAPI приложение
app = FastAPI()

# Создание параметров подключения
connection_params = ConnectionParams(
    http={
        "host": "localhost",  # Хост Weaviate-сервера
        "port": 8080,         # Порт Weaviate-сервера
        "secure": False,      # Использовать ли HTTPS (False - HTTP)
    },
    grpc={
        "host": "localhost",  # Хост для gRPC (если не используете gRPC, все равно указывайте)
        "port": 50051,        # Порт gRPC
        "secure": False       # Без шифрования для gRPC
    }
)

# Инициализация клиента
client = WeaviateClient(connection_params=connection_params)
client.connect()

if client.is_ready():
    print("Weaviate server is ready!")
else:
    print("Failed to connect to Weaviate server.")

# Модель E5 для генерации эмбеддингов
model_name = "intfloat/e5-large-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Функция для векторизации текста
def embed_text(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings.squeeze().numpy()

# Функция для извлечения ключевых слов с использованием Yape
def extract_keywords(text):
    extractor = yape.KeywordExtractor()
    keywords = extractor.extract_keywords(text)
    return [keyword for keyword, score in keywords]


def create_test_article_collection():
    # Создание новой коллекции (класса)
    collection_name = "TestArticle"  # имя класса (коллекции)
    properties = [
        Property(name="content", data_type=DataType.TEXT),  # Corrected key name
        Property(name="dataframe", data_type=DataType.TEXT),
        Property(name="keywords", data_type=DataType.TEXT_ARRAY),  # Use TEXT_ARRAY for array type
    ]

    # Создание коллекции (класса) через Weaviate client
    client.collections.create(
        name=collection_name,
        properties=properties
    )
    print("Коллекция 'TestArticle' создана.")

#create_test_article_collection()

# Индексация документов с использованием векторизации
@app.post("/indexing")
async def index_docs_with_embeddings(docs: list[Document]):
    try:
        for doc in docs:
            embedding = embed_text(doc.content)  # Vectorization
            response = client.data_object.create(
                data_object={
                    "content": embedding,
                    "dataframe": doc.dataframe,
                    "keywords": doc.keywords,
                },
                class_name="TestArticle",
                vector=embedding.tolist()  # Ensure the vector is a list
            )
        return {"message": "Documents indexed with embeddings"}
    except Exception as e:
        return {"error": str(e)}

@app.post("/searching")
async def search_with_llm(text: str, top_k: int = 3):
    # Извлекаем ключевые слова из запроса
    query_keywords = extract_keywords(text)

    query_embedding = embed_text(text)  # Векторизуем запрос
    near_vector = {"vector": query_embedding.tolist()}

    # Выполняем запрос к Weaviate с фильтрацией по ключевым словам
    result = ((((client.query.get("Paragraph", ["content", "dataframe", "keywords"])
              .with_near_vector(near_vector))
              .with_limit(top_k))
              .with_where({
                    "operator": "Like",
                    "path": ["keywords"],
                    "valueString": query_keywords[0]  # Пример фильтрации по первому ключевому слову
              }))
              .do())
    return {"results": result["data"]["Get"]["Paragraph"]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
