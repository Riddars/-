from pydantic import BaseModel


class SearchQuery(BaseModel):
    text: str
    top_k: int


class Document(BaseModel):
    content: str
    dataframe: str = None
    keywords: list[str] = []
