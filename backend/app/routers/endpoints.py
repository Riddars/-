import json
from weaviate.classes.query import Filter

from fastapi import APIRouter

from ..database.database import ensure_client_connected, client
from ..embending.embending import embed_text, extract_keywords
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
    dataframe = query.dataframe
    top_k = query.top_k

    query_embedding = embed_text(text)
    keywords = extract_keywords(text)

    paragraphs = client.collections.get("Paragraphs")
    result = paragraphs.query.hybrid(
        query=text,
        filters=(
            Filter.all_of([
                Filter.by_property("dataframe").equal(dataframe),
                Filter.by_property("keywords").contains_any(keywords)
            ])
        ),
        vector=query_embedding,
        limit=top_k
    )
    seen = set()
    unique_objects = []
    for obj in result.objects:
        obj_properties_json = json.dumps(obj.properties, sort_keys=True)
        if obj_properties_json not in seen:
            seen.add(obj_properties_json)
            unique_objects.append(obj.properties)

    response = json.dumps(unique_objects, indent=2, ensure_ascii=False)
    return response
