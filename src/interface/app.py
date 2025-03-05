import asyncio
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)
import streamlit as st
from src.models.llm import LLMWrapper
from src.nlp_tasks.summarization import summarize_text
from src.nlp_tasks.sentiment_analysis import analyze_sentiment
from src.nlp_tasks.ner import extract_entities
from src.nlp_tasks.question_answering import answer_question
from src.nlp_tasks.code_generation import generate_code
from src.rag.vector_store import VectorStore
from src.nlp_tasks.code_review import review_code

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

vector_store = VectorStore()

st.title("Arogo AI Assistant")

st.sidebar.title("RAG - Upload Knowledge Base 📂")
uploaded_files = st.sidebar.file_uploader(
    "Upload TXT, PDF, CSV, or JSON files:",
    type=["txt", "pdf", "csv", "json"],
    accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        file_type = uploaded_file.name.split(".")[-1]
        file_path = os.path.join("temp_docs", uploaded_file.name)
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

        if file_type == "txt":
            with open(file_path, "r", encoding="utf-8") as f:
                doc_text = f.read()
            vector_store.add_document(doc_text)

        elif file_type == "pdf":
            vector_store.add_pdf(file_path)

        elif file_type == "csv":
            vector_store.add_csv(file_path)

        elif file_type == "json":
            vector_store.add_json(file_path)

    st.sidebar.success("Documents processed successfully!")




if "messages" not in st.session_state:
    st.session_state.messages = []

persona = st.selectbox(
    "Select Persona:", 
    ["Default", "Casual", "Professional", "Technical"]
)


provider = st.radio("Choose AI Model:", ["OpenAI", "Gemini","MetaAI"])
task = st.radio(
    "Choose Task:", 
    ["General Chat", "Summarization", "Sentiment Analysis", "NER", "Question Answering", "Code Generation", "Code Review"]
)

llm = LLMWrapper(provider=provider.lower())

selected_model = provider.lower()

query, context, question = None, None, None

if task not in ["Question Answering", "Code Generation"]:
    query = st.text_area("Enter your text:")
if task == "Question Answering":
    context = st.text_area("Enter context for the question:")
    question = st.text_area("Enter your question:")
if task == "Code Generation":
    query = st.text_area("Enter a description for the code you want:")

response = ""
if st.button("Generate Response"):
    if uploaded_files and query.strip():
        relevant_texts = vector_store.search(query, top_k=5)
        retrieved_context = "\n\n".join(relevant_texts)
        response = llm.generate_response(f"Context:\n{retrieved_context}\n\nQuestion: {query}")  

    elif task == "Code Review" and query.strip():
        response = review_code(query, provider=provider.lower())     

    elif task == "General Chat" and query.strip():
        response = llm.generate_response(query)

    elif task == "Summarization" and query.strip():
        response = summarize_text(query, provider=provider.lower())

    elif task == "Sentiment Analysis" and query.strip():
        response = analyze_sentiment(query, provider=provider.lower())

    elif task == "NER" and query.strip():
        response = extract_entities(query, provider=provider.lower())

    elif task == "Question Answering":
        if context and question and context.strip() and question.strip():
            response = answer_question(question, context, provider=provider.lower())
        else:
            response = "Please provide both a **context** and a **question**."
    elif task == "Code Generation" and query.strip():
        response = generate_code(query, provider=provider.lower())
    else:
        response = llm.generate_response(q)

    st.write(response)
    if persona == "Casual":
        response = f"Here's what I found:\n\n{response}"
    elif persona == "Professional":
        response = f"Here's a well-structured response:\n\n{response}"
    elif persona == "Technical":
        response = f"Technical details:\n\n{response}"

    st.session_state.messages.append(("You", query or question))
    st.session_state.messages.append(("AI", response))
    
st.subheader("Chat History")
for role, message in st.session_state.messages:
    st.markdown(f"**{role}:** {message}")
