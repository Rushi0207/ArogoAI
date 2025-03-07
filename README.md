# Arogo AI Assistant

## Overview
This project is a multi-functional AI assistant designed for the **Arogo AI LLM Engineer Intern Assignment**. It integrates **Large Language Models (LLMs)** to perform various **Natural Language Processing (NLP) tasks**, including:

- **Text Summarization**
- **Sentiment Analysis**
- **Named Entity Recognition (NER)**
- **Question Answering**
- **Code Generation & Review**
- **Retrieval-Augmented Generation (RAG) for Document Search**

The assistant is implemented with a **modular architecture**, supports **multiple LLM providers** (OpenAI, Gemini, Hugging Face), and offers a **Streamlit-based UI** for user interaction.

---
## Features
### 1️⃣ LLM Integration & Abstraction
- Supports **OpenAI, Gemini, and Hugging Face** models.
- Easy switching between LLM providers using `llm_wrapper.py`.
- API keys managed via environment variables.

### 2️⃣ Data Ingestion & Preprocessing
- Supports **TXT, PDF, CSV, and JSON** file uploads.
- Uses **vector embeddings (FAISS)** to store and retrieve relevant document sections.
- Implements **text chunking** for better retrieval performance.

### 3️⃣ Core NLP Functionalities
✅ **Summarization** – Generates concise summaries of input text.
✅ **Sentiment Analysis** – Classifies text as **Positive, Negative, or Neutral**.
✅ **NER** – Extracts named entities such as **People, Locations, and Organizations**.
✅ **Question Answering** – Answers questions based on a given context.
✅ **Code Generation** – Generates Python, JavaScript, or other programming code.
✅ **Code Review** – Reviews code for **bugs, security issues, and improvements**.

### 4️⃣ Retrieval-Augmented Generation (RAG)
- Stores and retrieves documents using **FAISS vector search**.
- Enhances LLM responses with **dynamically retrieved knowledge**.
- Supports **semantic search** for improved accuracy.

### 5️⃣ Conversational Interface
- Maintains **multi-turn chat history**.
- Allows **persona switching** (Casual, Professional, Technical).
- Implements **content moderation** to filter harmful responses.

### 6️⃣ Performance Optimizations
✅ **Caching** – Reduces redundant API calls using `cache_manager.py`.
✅ **Logging** – Tracks model usage, response times, and errors.
✅ **Prompt Engineering** – Optimized for high-quality outputs.

### 7️⃣ Testing & Evaluation
- **Unit tests** for NLP tasks (`test_nlp_tasks.py`).
- **ROUGE & BLEU evaluation** for summarization and QA.
- **Code execution validation** for generated code snippets.

---
## Installation & Setup
### **1️⃣ Prerequisites**
Ensure you have **Python 3.8+** installed. Install dependencies using:
```bash
pip install -r requirements.txt
```

### **2️⃣ Set API Keys**
Create a `.env` file and add the required API keys:
```
OPENAI_API_KEY=your_openai_api_key
GEMINI_API_KEY=your_gemini_api_key
```

### **3️⃣ Run the Application**
Start the Streamlit UI:
```bash
streamlit run src/interface/app.py
```

---
## Folder Structure
```
├── src/
│   ├── interface/
│   │   ├── app.py             # Streamlit Web App
│   ├── models/
│   │   ├── llm_wrapper.py    # LLM Abstraction Layer
│   │   ├── context_manager.py # Multi-turn chat history
│   ├── nlp_tasks/
│   │   ├── summarization.py
│   │   ├── sentiment_analysis.py
│   │   ├── ner.py
│   │   ├── question_answering.py
│   │   ├── code_generation.py
│   │   ├── code_review.py
│   ├── rag/
│   │   ├── vector_store.py    # FAISS-based Document Retrieval
│   │   ├── chunking.py        # Splitting Text into Chunks
│   ├── utils/
│   │   ├── cache_manager.py   # Caching API Calls
│   │   ├── logger.py          # Logging System Performance
│   │   ├── moderation.py      # Content Moderation
├── tests/
│   ├── test_app.py            # UI Tests
│   ├── test_evaluation.py     # Evaluation Tests
│   ├── test_nlp_tasks.py      # NLP Functional Tests
├── requirements.txt           # Dependencies
├── README.md                  # Project Documentation
```

---
## Future Improvements
- **Real-time token usage tracking** to manage API costs.
- **Enhanced UI** with chat history search and response explanations.
- **Custom fine-tuned LLM models** for domain-specific accuracy.

---
## Contributors
- **Rushikesh Phadtare** – Developer
- **Arogo AI Team** – Project Assignment

---
## License
This project is for the **Arogo AI LLM Engineer Intern Assignment** and follows standard open-source licensing policies.