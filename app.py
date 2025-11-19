from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GROQ_KEY = os.getenv("GROQ_KEY")

app = FastAPI(title="Pulsevo AI Assistant")

# ---------------------------------------------------
# CORS
# ---------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------
# STATIC FILES
# ---------------------------------------------------
app.mount("/css", StaticFiles(directory="css"), name="css")
app.mount("/js", StaticFiles(directory="js"), name="js")
app.mount("/images", StaticFiles(directory="images"), name="images")

# ---------------------------------------------------
# RAG INITIALIZATION (LAZY)
# ---------------------------------------------------
rag_instance = None

def get_rag():
    global rag_instance
    if rag_instance is None:
        print("🔄 Initializing RAG engine...")
        from rag import RAG
        rag_instance = RAG(api_key=GROQ_KEY)

        if rag_instance.db.col.count() == 0:
            rag_instance.build()
        else:
            print("📦 Using existing vector DB.")

    return rag_instance

# ---------------------------------------------------
# ROUTES
# ---------------------------------------------------

# Render needs GET or HEAD / to return 200
@app.get("/")
def root():
    return {"status": "running", "message": "Pulsevo AI backend is live!"}

@app.get("/ui", response_class=HTMLResponse)
def ui():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/process_user_input", response_class=PlainTextResponse)
async def process_user_input(prompt_form_input: str = Form(...)):
    try:
        rag = get_rag()
        answer = rag.ask(prompt_form_input)
        return f"💬 {answer}"
    except Exception as e:
        return f"❌ Error: {str(e)}"

@app.get("/health")
def health():
    return {"status": "ok", "message": "Pulsevo AI running"}

# ---------------------------------------------------
# MAIN (Render auto-handles uvicorn)
# ---------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=port)
