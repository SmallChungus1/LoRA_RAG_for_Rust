import streamlit as st
from streamlit import session_state as se_state
from main import course_rag
import os

if 'course_rag_instance' not in se_state:
    se_state.course_rag_instance = course_rag()
    rag_doc_folder = "./rag_docs"
    rag_docs = [os.path.join(rag_doc_folder, file_path) for file_path in os.listdir(rag_doc_folder)]
    se_state.course_rag_instance.add_pdf(rag_docs)
    print("init course_rag instance and populated vector db")

st.title("RAG Powered LLM for your coursework")

user_query = st.text_input("Enter your question here:")

if user_query:
    response = se_state.course_rag_instance.ask(user_query)
    st.write("### Response:")
    st.write(response)
