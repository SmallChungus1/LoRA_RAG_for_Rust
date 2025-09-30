import streamlit as st
from streamlit import session_state as se_state
from main import course_rag
import os
import sys

rag_doc_folder=None

#first arg is app.py
if len(sys.argv) > 1:
    rag_doc_folder = sys.argv[1]
if 'course_rag_instance' not in se_state:
    se_state.course_rag_instance = course_rag()

    if rag_doc_folder:
        print(f"adding {os.path.basename(rag_doc_folder)} to vecDB")
        rag_docs = [os.path.join(rag_doc_folder, file_path) for file_path in os.listdir(rag_doc_folder)]
        se_state.course_rag_instance.add_pdf(rag_docs)
    else:
        print("not adding to vectorDB")

    print("finished init course_rag instance")

st.title("RAG Powered LLM for your questions about Rust")

user_query = st.text_input("Enter your question here:")

if user_query:
    response = se_state.course_rag_instance.ask(user_query)
    st.write("### Response:")
    st.write(response)
