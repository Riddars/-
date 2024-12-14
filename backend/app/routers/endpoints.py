import json

from fastapi import APIRouter

from ..database.database import ensure_client_connected, client
from ..embending.embending import embed_text
from ..models.models import Document, SearchQuery

router = APIRouter()


@router.post("/indexing")
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


@router.post("/searching")
async def search_with_llm(query: SearchQuery):
    ensure_client_connected()
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
