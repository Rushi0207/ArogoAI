import asyncio
import sys
import os
import json
import streamlit as st

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

from src.nlp_tasks.summarization import summarize_text
from src.nlp_tasks.sentiment_analysis import analyze_sentiment
from src.nlp_tasks.ner import extract_entities
from src.nlp_tasks.question_answering import answer_question
from src.nlp_tasks.code_generation import generate_code
from src.rag.vector_store import VectorStore
from src.nlp_tasks.code_review import review_code
from src.evaluation.evaluator import ResponseEvaluator
from src.models.llm_wrapper import LLMWrapper

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

vector_store = VectorStore()

st.title("Arogo AI Assistant")

eval_tab, chat_tab = st.tabs(["Evaluation", "Chat"])

with eval_tab:
    uploaded_eval_data = st.file_uploader("Upload test cases (JSON)", key="eval_uploader")
    if uploaded_eval_data:
        test_cases = json.load(uploaded_eval_data)
        evaluator = ResponseEvaluator()
        results = evaluator.run_full_evaluation(test_cases)
        st.write("Evaluation Results:", results)

with chat_tab:
    if st.checkbox("Enable multi-turn context", key="context_checkbox"):
        st.session_state.context_enabled = True
    else:
        st.session_state.context_enabled = False

    st.subheader("RAG - Upload Knowledge Base")
    uploaded_files = st.file_uploader(
        "Upload TXT, PDF, CSV, or JSON files:",
        type=["txt", "pdf", "csv", "json"],
        accept_multiple_files=True,
        key="knowledge_uploader"
    )
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_type = uploaded_file.name.split(".")[-1]
            file_path = os.path.join("temp_docs", uploaded_file.name)

            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())

            if file_type == "txt":
                with open(file_path, "r", encoding="utf-8") as f:
                    vector_store.add_document(f.read())
            elif file_type == "pdf":
                vector_store.add_pdf(file_path)
            elif file_type == "csv":
                vector_store.add_csv(file_path)
            elif file_type == "json":
                vector_store.add_json(file_path)
            os.remove(file_path)
        
        st.sidebar.success("Documents processed and temporary files deleted successfully!")


    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    persona = st.selectbox("Select Persona:", ["Default", "Casual", "Professional", "Technical"], key="persona")
    provider = st.radio("Choose AI Model:", ["OpenAI", "Gemini", "HuggingFace"], key="provider")
    if provider == "HuggingFace":
        st.warning("Hugging Face models may run slower on low-RAM devices.")
    
    task = st.radio(
        "Choose Task:",
        ["General Chat", "Summarization", "Sentiment Analysis", "NER", "Question Answering", "Code Generation", "Code Review"],
        key="task"
    )
    
    llm = LLMWrapper(provider=provider.lower())
    
    query, context_text, question = None, None, None
    if task not in ["Question Answering", "Code Generation", "Code Review"]:
        query = st.text_area("Enter your text:", key="chat_text")
    elif task == "Question Answering":
        context_text = st.text_area("Enter context for the question:", key="qa_context")
        question = st.text_area("Enter your question:", key="qa_question")
    elif task == "Code Generation":
        query = st.text_area("Enter a description for the code you want:", key="codegen_text")
    elif task == "Code Review":
        query = st.text_area("Enter the code snippet for review:", key="code_review_text")
    
    response = ""
    if st.button("Generate Response", key="generate_response"):
        if task == "Question Answering" and (not context_text or not question):
            st.warning("⚠ Please provide both a **context** and a **question**.")
        elif task in ["General Chat", "Summarization", "Sentiment Analysis", "NER", "Code Generation", "Code Review"] and not (query and query.strip()):
            st.warning("⚠ Please enter your text before submitting.")
        else:
            if uploaded_files and query and query.strip():
                relevant_texts = vector_store.search(query, top_k=5)
                retrieved_context = "\n\n".join(relevant_texts)
                response = llm.generate_response(f"Context:\n{retrieved_context}\n\nQuestion: {query}")
            elif task == "Code Review":
                response = review_code(query, provider=provider.lower())
            elif task == "General Chat":
                response = llm.generate_response(query)
            elif task == "Summarization":
                response = summarize_text(query, provider=provider.lower())
            elif task == "Sentiment Analysis":
                response = analyze_sentiment(query, provider=provider.lower())
            elif task == "NER":
                response = extract_entities(query, provider=provider.lower())
            elif task == "Question Answering":
                response = answer_question(question, context_text, provider=provider.lower())
            elif task == "Code Generation":
                response = generate_code(query, provider=provider.lower())
    
            if persona == "Casual":
                response = f"Here's what I found:\n\n{response}"
            elif persona == "Professional":
                response = f"Here's a well-structured response:\n\n{response}"
            elif persona == "Technical":
                response = f"Technical Breakdown:\n\n{response}"
    
            st.write(response)
    
            st.session_state.messages.append(("You", query or question))
            st.session_state.messages.append(("AI", response))
    
    st.subheader("Chat History")
    for role, message in st.session_state.messages:
        st.markdown(f"**{role}:** {message}")
