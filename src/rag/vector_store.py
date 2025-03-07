import faiss
import numpy as np
import fitz
from sentence_transformers import SentenceTransformer
import pandas as pd
import json
from src.rag.chunking import chunk_text
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

nltk.download('punkt')
nltk.download('stopwords')

model = SentenceTransformer("all-MiniLM-L6-v2")

class TempFileCleaner(FileSystemEventHandler):
    def on_closed(self, event):
        if event.is_directory: return
        os.unlink(event.src_path)

class VectorStore:
    def __init__(self):
        self.dimension = 384
        self.num_clusters = 100
        quantizer = faiss.IndexFlatL2(self.dimension)
        self.index = faiss.IndexIVFFlat(quantizer, self.dimension, self.num_clusters, faiss.METRIC_L2)
        self.doc_vectors = []
        self.docs = []
        self.index.train(np.random.rand(1000, self.dimension).astype('float32'))

    def _preprocess_text(self, text):
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        return " ".join(tokens)

    def _extract_text_from_pdf(self, pdf_path):
        text = ""
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text("text") + "\n\n"
        return text.split("\n\n")

    def add_document(self, text):
        processed_text = self._preprocess_text(text)
        chunks = chunk_text(processed_text)
        for chunk in chunks:
            vector = model.encode([chunk])[0]
            self.index.add(np.array([vector]).astype('float32'))
            self.doc_vectors.append(vector)
            self.docs.append(chunk)

    def add_pdf(self, pdf_path):
        with fitz.open(pdf_path) as doc:
            full_text = ""
            for page in doc:
                full_text += page.get_text("text") + "\n"
        self.add_document(full_text)

    def add_csv(self, csv_path):
        try:
            df = pd.read_csv(csv_path)
            if df.empty:
                raise ValueError("CSV file is empty or invalid.")
            structured_text = self._format_csv(df)
            self.add_document(structured_text)
        except (pd.errors.ParserError, ValueError) as e:
            print(f"CSV Error ({csv_path}): {str(e)}")

    def add_json(self, json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            structured_text = self._format_json(data)
            self.add_document(structured_text)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"JSON Error ({json_path}): {str(e)}")

    def _format_csv(self, df):
        return "\n\n".join([f"Row {i+1}: " + ", ".join([f"{col}: {df.at[i, col]}" for col in df.columns]) for i in range(len(df))])

    def _format_json(self, data):
        if isinstance(data, dict):
            return json.dumps(data, indent=2)
        elif isinstance(data, list):
            return "\n\n".join([json.dumps(item, indent=2) for item in data])
        return ""
    
    def search(self, query, top_k=5):
        query_vector = model.encode([query])[0]
        distances, indices = self.index.search(np.array([query_vector]).astype('float32'), top_k)
        
        ranked_results = [
            (self.docs[i], distances[0][idx])
            for idx, i in enumerate(indices[0]) if i < len(self.docs)
        ]
        ranked_results.sort(key=lambda x: x[1])
        return [doc for doc, _ in ranked_results]
