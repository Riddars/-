from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.sql import or_
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pymorphy3
import uvicorn
from typing import List, Optional

app = FastAPI()
DATABASE_URL = "sqlite:///paragraphs.db"
engine = create_engine(DATABASE_URL, echo=False)
Base = declarative_base()
Session = sessionmaker(bind=engine)

morph = pymorphy3.MorphAnalyzer()
search_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

class Paragraph(Base):
    __tablename__ = "paragraphs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    content = Column(Text, nullable=False)
    dataframe = Column(String(255), nullable=True)
    keywords = Column(Text, nullable=True)
    embedding = Column(Text, nullable=False)

Base.metadata.create_all(engine)

def normalize_keywords(keywords):
    normalized = []
    for phrase in keywords:
        words = phrase.split()
        normalized_words = [morph.parse(word)[0].normal_form for word in words]
        normalized.append(" ".join(normalized_words))
    return normalized

class IndexRequest(BaseModel):
    dataset_name_or_docs: list

class SearchRequest(BaseModel):
    text: str
    keywords: List[str]
    filter_by: Optional[List[str]] = [] 
    top_k: int

@app.post("/indexing")
async def indexing(request: IndexRequest):
    try:
        session = Session()
        for item in request.dataset_name_or_docs:
            content = item.get("content", "")
            dataframe = item.get("dataframe", None)
            keywords = item.get("keywords", [])

            normalized_keywords = normalize_keywords(keywords) if keywords else []

            embedding = search_model.encode(content, convert_to_tensor=False)
            paragraph = Paragraph(
                content=content,
                dataframe=dataframe,
                keywords=", ".join(normalized_keywords),
                embedding=np.array2string(embedding, separator=',')
            )
            session.add(paragraph)

        session.commit()
        session.close()
        return {"message": "Данные успешно сохранены."}

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/searching")
async def search_paragraphs(request: SearchRequest):
    try:
        session = Session()
        query = session.query(Paragraph)

        if request.filter_by:
            query = query.filter(Paragraph.dataframe.in_(request.filter_by))

        paragraphs = query.all()
        # print(paragraphs)
        session.close()

        if not paragraphs:
            return {"message": "Нет подходящих параграфов."}

        contents = [p.content for p in paragraphs]
        embeddings = [np.fromstring(p.embedding[1:-1], sep=',') for p in paragraphs]

        normalized_keywords = normalize_keywords(request.keywords)
        query_text = " ".join(normalized_keywords)

        query_embedding = search_model.encode(query_text, convert_to_tensor=False)

        similarities = cosine_similarity([query_embedding], embeddings)[0]

        top_k_indices = np.argsort(-similarities)[:request.top_k]

        top_paragraphs = [
            {
                "id": paragraphs[i].id,
                "content": paragraphs[i].content,
                "similarity": round(float(similarities[i]), 4),
                "dataframe": paragraphs[i].dataframe,
                "keywords": paragraphs[i].keywords.split(", ")
            }
            for i in top_k_indices
        ]

        return {"response": top_paragraphs}

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/paragraphs")
async def get_paragraphs():
    try:
        session = Session()
        paragraphs = session.query(Paragraph).all()
        session.close()

        result = [
            {
                "id": p.id,
                "content": p.content,
                "dataframe": p.dataframe,
                "keywords": p.keywords.split(", ")
            }
            for p in paragraphs
        ]

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/restart")
async def restart():
    try:
        session = Session()
        session.query(Paragraph).delete()
        session.commit()
        session.close()
        return {"message": "Таблица успешно очищена."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
