import json
import uvicorn
import yape
from transformers import AutoTokenizer, AutoModel
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from weaviate.client_base import ConnectionParams
from weaviate import WeaviateClient
from weaviate.classes.config import Property, DataType
from weaviate.classes.config import Configure


class Document(BaseModel):
    content: str
    dataframe: str = None
    keywords: list[str] = []


app = FastAPI()

# Создание параметров подключения
connection_params = ConnectionParams(
    http={
        "host": "localhost",  # Хост Weaviate-сервера
        "port": 8080,  # Порт Weaviate-сервера
        "secure": False,  # Использовать ли HTTPS (False - HTTP)
    },
    grpc={
        "host": "localhost",  # Хост для gRPC (если не используете gRPC, все равно указывайте)
        "port": 50051,  # Порт gRPC
        "secure": False  # Без шифрования для gRPC
    }
)

client = WeaviateClient(connection_params=connection_params)


def ensure_client_connected():
    if not client.is_connected():
        print("Client is not connected. Connecting to Weaviate...")
        client.connect()
        print("Weaviate is connected")


def create_test_article_collection():
    try:
        client.collections.create(
            "Paragraphs",
            vectorizer_config=Configure.Vectorizer.text2vec_huggingface(),
            vector_index_config=Configure.VectorIndex.hnsw(),
            properties=[
                Property(name="content", data_type=DataType.TEXT),
                Property(name="dataframe", data_type=DataType.TEXT),
                Property(name="keywords", data_type=DataType.TEXT_ARRAY),
            ]
        )
        print("Коллекция 'Paragraphs' создана.")
    except Exception as e:
        print(f"Error creating collection: {e}")


# create_test_article_collection()

# Модель E5 для генерации эмбеддингов
model_name = "intfloat/e5-large-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


def embed_text(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings.squeeze().numpy()


# Функция для извлечения ключевых слов с использованием Yape
# def extract_keywords(text):
#     extractor = yape.KeywordExtractor()
#     keywords = extractor.extract_keywords(text)
#     return [keyword for keyword, score in keywords]


@app.post("/indexing")
async def index_docs_with_embeddings(docs: list[Document]):
    try:
        ensure_client_connected()
        for doc in docs:
            embedding = embed_text(doc.content)
            with client.batch.dynamic() as batch:
                batch.add_object(
                    properties={
                        "content": doc.content,
                        "dataframe": doc.dataframe,
                        "keywords": doc.keywords,
                    },
                    vector=embedding.tolist(),
                    collection="Paragraphs"
                )
        return {"message": "Documents indexed with embeddings"}
    except Exception as e:
        print(f"Failed to add document: {e}")
        if client.batch.failed_objects:
            print("Failed objects:", client.batch.failed_objects)
        return {"error": f"Document failed to index: {e}"}


class SearchQuery(BaseModel):
    text: str
    top_k: int


@app.post("/searching")
async def search_with_llm(query: SearchQuery):
    text = query.text
    top_k = query.top_k

    query_embedding = embed_text(text)
    near_vector = {"vector": query_embedding.tolist()}

    paragraphs = client.collections.get("Paragraphs")
    result = paragraphs.query.near_vector(
        query_embedding,
        limit=top_k
    )
    res = ""
    for obj in result.objects:
        res += json.dumps(obj.properties, indent=2)
    return res


@app.on_event("shutdown")
async def shutdown_event():
    client.close()
    print("Weaviate client closed.")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
