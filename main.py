import transformers
import torch
from langchain_chroma import Chroma
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
import argparse
import os

class course_rag():
    def __init__(self, llm_model_name = "HuggingFaceTB/SmolLM2-135M", embedding_model_name = "BAAI/bge-small-en", persist_directory = "./chroma_langchain_db"):
        

        #device setup source: https://python.langchain.com/docs/integrations/llms/huggingface_pipelines/
        device_name = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using device: {device_name}")

        #embedding source: https://python.langchain.com/docs/integrations/text_embedding/bge_huggingface/
        model_kwargs = {"device": device_name}
        encode_kwargs = {"normalize_embeddings": True}
        hf_bge_embed = HuggingFaceBgeEmbeddings(
            model_name=embedding_model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
        )
        embedding_model = hf_bge_embed


        vector_store = Chroma(
            collection_name="rag_collection",
            embedding_function=embedding_model,
            persist_directory=persist_directory,  # Where to save data locally, remove if not necessary
        )

        #llm setup source: https://python.langchain.com/docs/integrations/llms/huggingface_pipelines/
        tokenizer = AutoTokenizer.from_pretrained(llm_model_name, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            llm_model_name,
            device_map="auto",
            torch_dtype=torch.float16,
        ).to(device_name)
        hf_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=20, use_fast=True)
        llm = HuggingFacePipeline(pipeline=hf_pipe)

        #source for chaining: https://python.langchain.com/docs/integrations/llms/huggingface_pipelines/
        template = """Question: {question}

        Answer: """
        prompt = PromptTemplate.from_template(template)

        chain = prompt | llm

        self.vector_store = vector_store
        self.chain = chain

    def add_pdf(self, pdf_paths):
        #for loading pdf docs: https://python.langchain.com/docs/integrations/document_loaders/pypdfloader/
        for pdf_path in pdf_paths:
            loader = PyPDFLoader(pdf_path)
            docs = loader.load_and_split()
            print(docs[0].metadata)
        #     self.vector_store.add_documents(documents)
        # self.vector_store.persist()
        
    def test_response(self, question):

        response = self.chain.invoke({"question": question})
        return response


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Course RAG")
    parser.add_argument("--q", type=str, help="Question to ask")
    args = parser.parse_args() 
    rag_doc_folder = "./rag_docs"

    course_rag_instance = course_rag()
    rag_docs = [os.path.join(rag_doc_folder, file_path) for file_path in os.listdir(rag_doc_folder)]
    course_rag_instance.add_pdf(rag_docs)
    response = course_rag_instance.test_response(args.q)
    print(response)