from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
import uvicorn

# import RAG instance
from rag import rag_instance

app = FastAPI(title="Pulsevo AI Assistant")

# CORS (optional but useful)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------- STATIC FILES (Frontend) ---------
app.mount("/css", StaticFiles(directory="css"), name="css")
app.mount("/js", StaticFiles(directory="js"), name="js")
app.mount("/images", StaticFiles(directory="images"), name="images")

# Serve the main webpage
@app.get("/", response_class=HTMLResponse)
def home():
    with open("index.html", "r", encoding="utf-8") as file:
        return file.read()

# --------- BACKEND API (RAG) ---------
@app.post("/process_user_input", response_class=PlainTextResponse)
async def process_user_input(prompt_form_input: str = Form(...)):
    try:
        answer = rag_instance.ask(prompt_form_input)
        return f"💬 {answer}"
    except Exception as e:
        return f"❌ Error: {str(e)}"

# --------- HEALTH CHECK ---------
@app.get("/health")
def health():
    return {"status": "ok", "message": "Pulsevo AI FastAPI server running"}

# Run locally
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
