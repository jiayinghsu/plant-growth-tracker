# app/main.py

from fastapi import FastAPI
from app.api.endpoints import router

app = FastAPI(title="Plant Growth Tracker API")

app.include_router(router)

# Root endpoint for testing
@app.get("/")
async def root():
    return {"message": "Welcome to the Plant Growth Tracker API!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
