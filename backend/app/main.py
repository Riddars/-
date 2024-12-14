import uvicorn
from fastapi import FastAPI

from backend.app.database.database import create_test_article_collection, client
from backend.app.routers import endpoints

app = FastAPI()
app.include_router(endpoints.router)

if __name__ == "__main__":

    client.connect()
    create_test_article_collection()

    uvicorn.run(app, host="0.0.0.0", port=8000)
