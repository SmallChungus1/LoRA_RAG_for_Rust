import transformers
import torch 
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
import argparse
import os

class course_rag():
    def __init__(self, llm_model_name = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
, embedding_model_name = "BAAI/bge-small-en", persist_directory = "./chroma_langchain_db"):
        
        self.persist_directory = persist_directory
        #device setup source: https://python.langchain.com/docs/integrations/llms/huggingface_pipelines/
        device_name = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {device_name}")

        #embedding source: https://python.langchain.com/docs/integrations/text_embedding/bge_huggingface/
        model_kwargs = {"device": device_name}
        encode_kwargs = {"normalize_embeddings": True}
        hf_bge_embed = HuggingFaceBgeEmbeddings(
            model_name=embedding_model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
        )
        embedding_model = hf_bge_embed


        #chroma and retriever set up
        self.vector_store = Chroma(
            collection_name="rag_collection",
            embedding_function=embedding_model,
            persist_directory=self.persist_directory,  # Where to save data locally, remove if not necessary
        )
        self.vector_store.reset_collection()

        # Retriever (MMR reduces redundancy)
        self.retriever = self.vector_store.as_retriever(
            search_type="mmr", search_kwargs={"k": 8, "fetch_k": 40, "lambda_mult": 0.5}
        )

        #llm setup source: https://python.langchain.com/docs/integrations/llms/huggingface_pipelines/
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        tokenizer = AutoTokenizer.from_pretrained(llm_model_name, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            llm_model_name,
            device_map="auto",
            torch_dtype=dtype,
        ).to(device_name)
        hf_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512, use_fast=True, return_full_text=False)
        llm = HuggingFacePipeline(pipeline=hf_pipe)
        
        #chain setup
        self.RAG_PROMPT = PromptTemplate.from_template(
            "You are a college computer science teacher.\n"
            "you can use the context to help with answering. If unknown, say you don't know. DO NOT HALLUCINATE. DO NOT MAKE UP SOURCES.\n"
            "Question:\n{question}\n\n"
            "Context:\n{context}\n\n"
            "Answer:"
        )


        # Chain: retrieve -> format -> prompt -> llm -> string
        self.chain = (
            {"context": self.retriever | self.format_docs, "question": RunnablePassthrough()}
            | self.RAG_PROMPT
            | llm
            | StrOutputParser()
        )

        self.text_splitter =  RecursiveCharacterTextSplitter(
            chunk_size=1000,  # chunk size (characters)
            chunk_overlap=200,  # chunk overlap (characters)
            add_start_index=True,  # track index in original document
        )


    def add_pdf(self, pdf_paths):
        #for loading pdf docs: https://python.langchain.com/docs/integrations/document_loaders/pypdfloader/
        for pdf_path in pdf_paths:
            loader = PyPDFLoader(pdf_path)
            docs = loader.load_and_split(text_splitter=self.text_splitter)
            print(f"Number of chunks created from {os.path.basename(pdf_path)}: {len(docs)}")
            self.vector_store.add_documents(docs)
    
    
    def format_docs(self, docs):
        '''turns all docs into single string for llm ingestion'''
        return "\n\n".join(f"[{i+1}] {d.page_content}" for i,d in enumerate(docs, start=1))

    def populate_vector_store(self, docs):
        self.vector_store.add_documents(docs)

    def ask(self, question):
        return self.chain.invoke(question) #question and not {'question': question} because of RunnablePassthrough

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Course RAG")
    parser.add_argument("--q", type=str, help="Question to ask")
    args = parser.parse_args() 
    rag_doc_folder = "./rag_docs"

    course_rag_instance = course_rag()
    rag_docs = [os.path.join(rag_doc_folder, file_path) for file_path in os.listdir(rag_doc_folder)]
    course_rag_instance.add_pdf(rag_docs)
    response = course_rag_instance.ask(args.q)
    print(response)