from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import os

# Import your RAG logic (optional)
import rag  

app = FastAPI()

# Mount static directories
app.mount("/css", StaticFiles(directory="css"), name="css")
app.mount("/js", StaticFiles(directory="js"), name="js")
app.mount("/images", StaticFiles(directory="images"), name="images")

# Serve index.html
@app.get("/", response_class=HTMLResponse)
def serve_home():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "FastAPI running on Vercel"}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
