import os
import shutil
import gc
import time
from dotenv import load_dotenv
import streamlit as st
from git import Repo
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import google.generativeai as genai
from typing import List, Dict, Tuple

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("API key not found. Please create a .env file with GEMINI_API_KEY")
    st.stop()

class CodebaseQA:
    def __init__(self):
        self.model = None
        self.index = None
        self.chunks = []
        self.metas = []

    def reset(self):
        self.model = None
        self.index = None
        self.chunks = []
        self.metas = []

    def clone_repository(self, repo_url: str, repo_dir: str = 'repo') -> str:
        if os.path.exists(repo_dir):
            shutil.rmtree(repo_dir, ignore_errors=True)
        st.info(f"Cloning repository from {repo_url}...")
        Repo.clone_from(repo_url, repo_dir)
        st.success("Repository cloned successfully!")
        return repo_dir

    def load_files(self, path: str, exts: List[str] = ['.py', '.md', '.js', '.html', '.css']) -> List[Dict]:
        docs = []
        for root, _, files in os.walk(path):
            for file in files:
                if any(file.endswith(ext) for ext in exts):
                    try:
                        with open(os.path.join(root, file), 'r', errors='ignore') as f:
                            content = f.read()
                            docs.append({
                                "filename": os.path.join(root, file),
                                "content": content
                            })
                    except Exception as e:
                        st.warning(f"Could not read file {file}: {str(e)}")
        return docs

    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
        chunks = []
        for i in range(0, len(text), chunk_size - overlap):
            chunks.append(text[i:i+chunk_size])
        return chunks

    def initialize_embedding_model(self):
        if self.model is None:
            with st.spinner("Loading embedding model..."):
                self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    def embed_documents(self, docs: List[Dict]) -> Tuple[np.ndarray, List[str], List[Dict]]:
        self.initialize_embedding_model()
        chunks, metas = [], []
        with st.spinner("Processing documents..."):
            for doc in docs:
                for chunk in self.chunk_text(doc["content"]):
                    chunks.append(chunk)
                    metas.append({"filename": doc["filename"]})
            embeddings = self.model.encode(chunks, convert_to_tensor=False)
        return embeddings, chunks, metas

    def create_index(self, embeddings: np.ndarray) -> faiss.Index:
        dim = len(embeddings[0])
        index = faiss.IndexFlatL2(dim)
        index.add(np.array(embeddings).astype("float32"))
        return index

    def search_index(self, query: str, top_k: int = 5) -> List[Dict]:
        if self.index is None or len(self.chunks) == 0:
            return []
        query_vec = self.model.encode([query])
        D, I = self.index.search(np.array(query_vec).astype("float32"), top_k)
        return [{
            "text": self.chunks[i],
            "meta": self.metas[i],
            "score": float(D[0][idx])
        } for idx, i in enumerate(I[0])]

    def ask_gemini(self, query: str, context_chunks: List[Dict]) -> str:
        if not context_chunks:
            return "No relevant context found to answer this question."
        context = "\n\n".join(
            f"From {chunk['meta']['filename']}:\n{chunk['text']}"
            for chunk in context_chunks
        )
        prompt = f"""You are a code assistant. Use the following code snippets to answer the question:

--- CONTEXT START ---
{context}
--- CONTEXT END ---

Question: {query}
Answer:"""
        try:
            model = genai.GenerativeModel("gemini-2.0-flash")
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error querying Gemini: {str(e)}"

def setup_gemini():
    try:
        genai.configure(api_key=GEMINI_API_KEY)
    except Exception as e:
        st.error(f"Failed to configure Gemini API: {str(e)}")
        st.stop()

def clear_repository_data():
    keys_to_clear = [
        'current_repo',
        'last_query',
        'qa_system',
        'repo_url_input',
        'query_input'
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

    repo_path = 'repo'
    if os.path.exists(repo_path):
        try:
            shutil.rmtree(repo_path)
        except PermissionError:
            # Handle file lock (especially for .git on Windows)
            for root, dirs, files in os.walk(repo_path, topdown=False):
                for name in files:
                    try:
                        file_path = os.path.join(root, name)
                        os.chmod(file_path, 0o777)
                        os.remove(file_path)
                    except Exception:
                        pass
                for name in dirs:
                    try:
                        os.rmdir(os.path.join(root, name))
                    except Exception:
                        pass
            try:
                shutil.rmtree(repo_path)
            except Exception as e:
                st.error(f"Could not delete repo: {str(e)}")

    gc.collect()
    time.sleep(0.1)


