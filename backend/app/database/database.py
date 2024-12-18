import weaviate
from weaviate import WeaviateClient
from weaviate.auth import Auth
from weaviate.classes.config import Configure
from weaviate.classes.config import Property, DataType
from weaviate.client_base import ConnectionParams
import weaviate.classes as wvc

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
weaviate_url = "https://yk800nalqwyizpyqiipfw.c0.europe-west3.gcp.weaviate.cloud"
weaviate_api_key = "r7ulnPvkmroOfUB1LnvcD5xu5HViFUYtrDqM"
# client = WeaviateClient(connection_params=connection_params)

client = weaviate.connect_to_weaviate_cloud(
    cluster_url=weaviate_url,
    auth_credentials=Auth.api_key(weaviate_api_key),
)

def ensure_client_connected():
    if not client.is_connected():
        print("Client is not connected. Connecting to Weaviate...")
        client.connect()
        print("Weaviate is connected")


# def create_test_article_collection():
#     try:
#         client.collections.create(
#             "Paragraphs",
#             vectorizer_config=Configure.Vectorizer.text2vec_huggingface(),
#             vector_index_config=Configure.VectorIndex.hnsw(),
#             properties=[
#                 Property(name="content", data_type=DataType.TEXT),
#                 Property(name="dataframe", data_type=DataType.TEXT),
#                 Property(name="keywords", data_type=DataType.TEXT_ARRAY),
#             ]
#         )
#         print("Коллекция 'Paragraphs' создана.")
#     except Exception as e:
#         pass

def client_article():
    try:
        paragraphs = client.collections.create(
            name="Data_base_paragraphs",
            vectorizer_config=wvc.config.Configure.Vectorizer.none())
    except Exception as e:
        print(e)

