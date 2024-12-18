from weaviate import WeaviateClient
from weaviate.classes.config import Configure
from weaviate.classes.config import Property, DataType
from weaviate.client_base import ConnectionParams

connection_params = ConnectionParams(
    http={
        "host": "localhost",  # Хост Weaviate-сервера
        "port": 8080
        ,
        "secure": False,
    },
    grpc={
        "host": "localhost",
        "port": 50051,
        "secure": False
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
        pass
