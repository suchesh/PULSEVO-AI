# app.py
from fastapi import FastAPI, Form
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from rag import rag_instance
from datetime import datetime

app = FastAPI(title="Company PDF RAG Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/process_user_input", response_class=PlainTextResponse)
async def process_user_input(prompt_form_input: str = Form(...)):
    """
    Accepts user text query -> RAG -> returns LLM answer.
    """
    try:
        answer = rag_instance.ask(prompt_form_input)
    except Exception as e:
        return f"❌ Internal error: {str(e)}"

    response_lines = [
        f"💬 {answer}"
    ]
    return "\n".join(response_lines)


@app.get("/", response_class=PlainTextResponse)
async def root():
    return "✅ Company RAG Assistant is live! Use POST /process_user_input"
