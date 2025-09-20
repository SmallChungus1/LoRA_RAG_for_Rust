In this project I finetuned HuggingFace SLMs (Qwen 0.6B, Smollm2 135M) with Lora on 5k synthetically generated Rust Programming question-answer pairs. The model is paired with RAG (chromaDB, LangChain) to answer Rust-related questions using a Rust Programming textbook for grounding. <br>
# Repo overview: <br>
**cs_rag_llm_lora_finetune.ipynb**: LORA finetuning code using HuggingFace's PEFT and TRL libs and their models. Training was done through Google Colab, using an L4 GPU.<br><br>
**data_gen**: folder with code for calling larger Qwen model from Openrouter to generate the syntehtic Rust q-a dataset (**rust_qa_dataset_5k.jsonl**) <br><br>
**main.py**: RAG implementation with ChromaDB and Langchain <br><br>
**app.py**: Simple frontend for User q-a built with Streamlit <br><br>
