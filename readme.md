In this project I finetuned HuggingFace SLMs (Qwen 0.6B, Smollm2 135M) with Lora on 5k synthetically generated Rust Programming question-answer pairs. The model is paired with RAG (chromaDB, LangChain) to answer Rust-related questions using a Rust Programming textbook for grounding. It also contains a evaluation script that uses a Judge LLM (Qwen 4b model) to evaluate the finetuned model against baseline on rust_qa_500_bench.jsonl.<br>

# Usage: <br>
* create venv and install dependencies ```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt ``` <br>
* to start up the StreamLit App: ```
streamlit run app.py ``` <br>
* to run the evaluation script: ```
python eval/eval_script.py ``` <br>   

# Repo overview: <br>
**cs_rag_llm_lora_finetune.ipynb**: LORA finetuning code using HuggingFace's PEFT and TRL libs and their models. Training was done through Google Colab, using an L4 GPU.<br><br>
**data_gen**: folder with code for calling larger Qwen model from Openrouter to generate the syntehtic Rust q-a dataset (**rust_qa_dataset_5k.jsonl**) <br><br>
**eval**: contains eval_script.py for evaluating the finetuned model on rust_qa_500_bench.jsonl using a Judge LLM (Qwen 4b model) <br><br>
**main.py**: RAG implementation with ChromaDB and Langchain <br><br>
**app.py**: Simple frontend for User q-a built with Streamlit <br><br>
