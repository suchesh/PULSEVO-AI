# ============================================
# rag.py

# ============================================
# Smart Real-Time RAG Assistant for Company PDFs and Dashboard Queries
# - Place PDFs in the SAME folder as this file.
# - Embeds on first run, uses Groq LLM for context-driven, natural chat.
# ============================================

import os
import uuid
import chromadb
from pathlib import Path
import numpy as np
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from dotenv import load_dotenv


load_dotenv()
GROQ_KEY = os.getenv("GROQ_KEY")

# =============================
# EMBEDDING MANAGER
# =============================
class EmbeddingManager:
    """Handles text embedding using SentenceTransformer."""

    def __init__(self):
        print("🔹 Loading Embedding Model: all-MiniLM-L6-v2")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed(self, texts):
        print(f"🧠 Generating embeddings for {len(texts)} chunks...")
        return np.array(self.model.encode(texts, show_progress_bar=False))


# =============================
# VECTOR DATABASE MANAGER
# =============================
class VectorStore:
    """Handles persistent vector storage and search using ChromaDB."""

    def __init__(self):
        os.makedirs("./vector_store", exist_ok=True)
        self.client = chromadb.PersistentClient(path="./vector_store")
        self.col = self.client.get_or_create_collection(name="company_docs")
        print(f"📦 DB Loaded: {self.col.count()} records")

    def add(self, docs, embeds):
        ids = [f"doc_{uuid.uuid4()}" for _ in docs]
        self.col.add(
            ids=ids,
            documents=[d.page_content for d in docs],
            embeddings=embeds.tolist(),
            metadatas=[d.metadata for d in docs]
        )
        print(f"✅ Added {len(docs)} chunks to Vector DB. Now: {self.col.count()}")

    def search(self, q_embed, k=3):
        """Query the vector store for top-k similar chunks."""
        return self.col.query(query_embeddings=[q_embed.tolist()], n_results=k)


# =============================
# RAG (Retrieve & Generate) PIPELINE
# =============================
class RAG:
    """Main RAG logic: load PDFs, embed, search, and generate conversational answers."""

    def __init__(self, api_key):
        self.embedder = EmbeddingManager()
        self.db = VectorStore()
        self.llm = ChatGroq(api_key=api_key, model="groq/compound-mini")

    # ---------------------------
    # Build Knowledge Base
    # ---------------------------
    def build(self):
        """Load PDFs, split into chunks, embed, and store in ChromaDB."""
        pdfs = list(Path(".").glob("*.pdf"))
        if not pdfs:
            print("⚠️ No PDFs found in folder.")
            return

        docs = []
        for pdf in pdfs:
            loader = PyMuPDFLoader(str(pdf))
            loaded = loader.load()
            for d in loaded:
                d.metadata["source_file"] = pdf.name
            docs.extend(loaded)
            print(f"✅ Loaded {len(loaded)} pages from {pdf.name}")

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)

        embeds = self.embedder.embed([c.page_content for c in chunks])
        self.db.add(chunks, embeds)
        print("🚀 RAG setup complete.")

    # ---------------------------
    # ASK A QUESTION (INTERACTIVE CHAT)
    # ---------------------------
    def ask(self, query, k=3):
        """Generate a conversational, real-time answer aligned with context."""
        print(f"\n🔎 Query: {query}")

        # Create query embedding
        q_embed = self.embedder.embed([query])[0]
        res = self.db.search(q_embed, k)

        # Gather context (if found)
        context = ""
        if res.get("documents") and res["documents"][0]:
            context = "\n\n".join(res["documents"][0])

        # ==============================
        # HIGHLY QUALIFIED CONVERSATIONAL PROMPT
        # ==============================
        prompt = f"""
You are **SAI**, a world-class AI corporate assistant that speaks naturally like a human.
Your tone is friendly, confident, and professional — never robotic or verbose.

🎯 **Your Mission**
- Act like a real-time chat assistant.
- Use short sentences, clean structure, and natural flow.
- Make replies easy to scan — use line breaks and bullet points where helpful.
- Always connect your answers to the retrieved dashboard or company context if relevant.
- If question is irrelevant, answer politely and redirect.

---

🧭 **Behavior Framework**

1️⃣ **Greetings / Casual Talk**  
→ Respond warmly in 1–2 lines.  
Example: “👋 Hey there! Great to see you. What can I help you with today?”

2️⃣ **Relevant or Dashboard Queries (Data / Stats / Company Info)**  
→ Use the provided context below.  
→ Provide a short **summary + key points (bullets or short lines)**.  
→ Include numbers naturally (no raw dumps).  
→ Conclude politely (e.g., “Would you like me to expand on that?”)

3️⃣ **Irrelevant / Personal Queries**  
→ Decline gently.  
Example: “😊 I’m designed to focus on company and dashboard insights only. Want me to show related analytics instead?”

4️⃣ **Formatting Rules**  
- Use a maximum of **5 short lines**.  
- Add **line breaks** between ideas.  
- Use **bullets or dashes** for clarity.  
- Never echo the user’s question verbatim.

---

📊 **Relevant Context:**
{context if context else "No matching company or dashboard data found."}

💬 **User Message:**
"{query}"

---

Now, respond as **SAI**, your intelligent corporate assistant.
Keep tone human, helpful, and concise.
Avoid essays — reply like a professional who explains clearly in a chat.
"""

        reply = self.llm.invoke(prompt)
        return reply.content.strip()


# =============================
# GLOBAL INSTANCE (For FastAPI)
# =============================
rag_instance = RAG(GROQ_KEY)

if rag_instance.db.col.count() == 0:
    rag_instance.build()
else:
    print("✅ Using existing vector DB.")



