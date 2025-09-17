import transformers
import torch
from langchain_chroma import Chroma
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate





device_name = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device_name}")

#llm setup source: https://python.langchain.com/docs/integrations/llms/huggingface_pipelines/
model_name = "HuggingFaceTB/SmolLM2-135M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
).to(device_name)
hf_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256)
llm = HuggingFacePipeline(pipeline=hf_pipe)

#source for chaining: https://python.langchain.com/docs/integrations/llms/huggingface_pipelines/
template = """Question: {question}

Answer: Let's think step by step."""
prompt = PromptTemplate.from_template(template)

chain = prompt | llm

question = "What is electroencephalography?"

print(chain.invoke({"question": question}))

#embedding source: https://python.langchain.com/docs/integrations/text_embedding/bge_huggingface/
model_name = "BAAI/bge-small-en"
model_kwargs = {"device": device_name}
encode_kwargs = {"normalize_embeddings": True}
hf_bge_embed = HuggingFaceBgeEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)
embedding_model = hf_bge_embed


vector_store = Chroma(
    collection_name="rag_collection",
    embedding_function=embedding_model,
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
)


