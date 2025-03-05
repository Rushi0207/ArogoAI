import faiss
import numpy as np
import fitz 
from sentence_transformers import SentenceTransformer
import pandas as pd
from src.rag.chunking import chunk_text

model = SentenceTransformer("all-MiniLM-L6-v2")

class VectorStore:
    def __init__(self):
        """Initialize FAISS vector index and store document sections"""
        self.index = faiss.IndexFlatL2(384)
        self.docs = []

    def _extract_text_from_pdf(self, pdf_path):
        """Extract text from a PDF file"""
        text = ""
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text("text") + "\n\n"
        return text.split("\n\n")

    def add_document(self, text):
        """Convert document text to vectors and store"""
        chunks = chunk_text(text)
        for chunk in chunks:
            vector = model.encode([chunk])[0]
            self.index.add(np.array([vector]))
            self.docs.append(chunk)

    def add_pdf(self, pdf_path):
        """Extract text from PDF and add it to the vector store"""
        with fitz.open(pdf_path) as doc:
            full_text = ""
            for page in doc:
                full_text += page.get_text("text") + "\n"
        for chunk in chunk_text(full_text):
            self.add_document(chunk)

    def add_csv(self, csv_path):
        """Reads CSV, extracts text from all columns, and adds it to the vector store."""
        df = pd.read_csv(csv_path)
        text_data = "\n\n".join(df.astype(str).agg(" ".join, axis=1))
        self.add_document(text_data)

    def add_json(self, json_path):
        """Reads JSON, extracts text, and adds it to the vector store."""
        with open(json_path, "r", encoding="utf-8") as f:
            data = pd.json_normalize(pd.read_json(f))
        text_data = "\n\n".join(data.astype(str).agg(" ".join, axis=1))
        self.add_document(text_data)

    def search(self, query, top_k=3):
        """Retrieve top-K most relevant document sections"""
        query_vector = model.encode([query])[0]
        _, indices = self.index.search(np.array([query_vector]), top_k)
        return [self.docs[i] for i in indices[0]]
