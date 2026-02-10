import os
import json
import torch
import gc
from tqdm import tqdm
from llama_cpp import Llama
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
# Using Unsloth GGUF repos
BASE_MODEL_REPO = "unsloth/Qwen3-0.6B-GGUF"
BASE_MODEL_FILE = "*Q4_K_M.gguf" # Pattern to match q4_k_m

GGUF_FT_MODEL_PATH = os.path.abspath("qwen3_06b_Q4_K_M_Rust_FT.gguf")

DATASET_PATH = os.path.abspath("data_gen/rust_qa_28_bench.jsonl")
OUTPUT_PATH = os.path.abspath("eval/eval_results.json")

JUDGE_MODEL_REPO = "unsloth/Qwen3-4B-Instruct-2507-GGUF"
JUDGE_MODEL_FILE = "*Q4_K_M.gguf"

# Judge System Prompt
JUDGE_SYSTEM_PROMPT = """You are an impartial judge evaluating the quality of answers to Rust programming questions.
Your task is to compare two responses against a Ground Truth answer.
"""

# Judge User Prompt Template
JUDGE_USER_TEMPLATE = """Instruction to Judge: Compare Response 1 and Response 2 against the Ground Truth.

Question: {question}

Ground Truth: {ground_truth}

Response 1 (Base Model):
{response_1}

Response 2 (Fine-Tuned Model):
{response_2}

Evaluation Criteria:
1. Accuracy: Does the code/explanation adhere to Rust specific rules (ownership, borrowing, lifetimes)?
2. Conciseness: Is the answer direct and to the point?
3. Correctness: Is the syntax correct?

Which response is better?

[[A]]: Response 1 is clearly superior.
[[B]]: Response 2 is clearly superior.
[[C]]: Both responses are effectively equivalent in quality.

Please respond with ONLY [[A]], [[B]], or [[C]].
"""

def generate_response_gguf(llm, question):
    messages = [
        {"role": "system", "content": "You are a helpful Rust assistant. Answer the following question concisely."},
        {"role": "user", "content": question}
    ]
    # Llama-cpp-python chat completion
    response = llm.create_chat_completion(
        messages=messages,
        max_tokens=256,
        temperature=0.7
    )
    return response["choices"][0]["message"]["content"]

def call_judge_gguf(llm, question, gt, resp1, resp2):
    prompt_content = JUDGE_USER_TEMPLATE.format(
        question=question,
        ground_truth=gt,
        response_1=resp1,
        response_2=resp2
    )

    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": prompt_content},
    ]

    try:
        # Judge with low temperature
        response = llm.create_chat_completion(
             messages=messages,
             max_tokens=100,
             temperature=0.1
        )
        content = response["choices"][0]["message"]["content"]
        print(f"DEBUG JUDGE RESPONSE: {content}")
        
        if "[[A]]" in content:
            return "A", content
        elif "[[B]]" in content:
            return "B", content
        elif "[[C]]" in content:
            return "C", content
        else:
            return "Unknown", content
    except Exception as e:
        print(f"Judge Error: {e}")
        return "Error", str(e)

def main():
    # Load Dataset
    print(f"Loading dataset from {DATASET_PATH}...")
    dataset = []
    if os.path.exists(DATASET_PATH):
        with open(DATASET_PATH, "r") as f:
            for line in f:
                if line.strip():
                    dataset.append(json.loads(line))
        print(f"Loaded {len(dataset)} examples (full dataset).")
    else:
        # Fallback to older path if user moved it in edits or just in case
        fallback_path = os.path.abspath("rust_qa_dataset.jsonl")
        print(f"Warning: {DATASET_PATH} not found. Trying {fallback_path}...")
        with open(fallback_path, "r") as f:
            for line in f:
                if line.strip():
                    dataset.append(json.loads(line))
        print(f"Loaded {len(dataset)} examples from fallback path.")
        
    # Testing Limit (optional, currently full)
    dataset_eval = dataset

    # 1. Generate with Base Model (GGUF via Llama-cpp)
    print(f"--- Generating Responses with Base Model: {BASE_MODEL_REPO} ---")
    
    # Download/Load Base model
    base_llm = Llama.from_pretrained(
        repo_id=BASE_MODEL_REPO,
        filename=BASE_MODEL_FILE,
        n_gpu_layers=-1, # Metal support check
        n_ctx=2048,
        verbose=False
    )
    
    base_responses = []
    for item in tqdm(dataset_eval, desc="Base Model Inference"):
        resp = generate_response_gguf(base_llm, item["question"])
        base_responses.append(resp)
    
    # Cleanup Base model
    del base_llm
    gc.collect()

    # 2. Generate with Fine-Tuned Model (Local GGUF)
    print(f"--- Generating Responses with Fine-Tuned Model: {GGUF_FT_MODEL_PATH} ---")
    
    ft_llm = Llama(
        model_path=GGUF_FT_MODEL_PATH,
        n_gpu_layers=-1, 
        n_ctx=2048,
        verbose=False
    )

    ft_responses = []
    for item in tqdm(dataset_eval, desc="Fine-Tuned Model Inference"):
        resp = generate_response_gguf(ft_llm, item["question"])
        ft_responses.append(resp)
    
    # Cleanup FT model
    del ft_llm
    gc.collect()

    # 3. Judge (GGUF via Llama-cpp)
    print(f"--- Loading Judge Model: {JUDGE_MODEL_REPO} ---")
    
    judge_llm = Llama.from_pretrained(
        repo_id=JUDGE_MODEL_REPO,
        filename=JUDGE_MODEL_FILE,
        n_gpu_layers=-1,
        n_ctx=4096, # Give judge more context
        verbose=False
    )

    print("--- Running AI Judge ---")
    results = []
    score_base = 0
    score_ft = 0
    total = 0

    for i, item in enumerate(tqdm(dataset_eval, desc="Judging")):
        question = item["question"]
        gt = item["answer"]
        r_base = base_responses[i]
        r_ft = ft_responses[i]
        
        winner, rationale = call_judge_gguf(judge_llm, question, gt, r_base, r_ft)
        
        if winner == "A":
            score_base += 1
        elif winner == "B":
            score_ft += 1
        elif winner == "C":
            score_base += 0.5
            score_ft += 0.5
        
        if winner in ["A", "B", "C"]:
            total += 1

        results.append({
            "question": question,
            "ground_truth": gt,
            "base_response": r_base,
            "ft_response": r_ft,
            "judge_decision": winner,
            "judge_rationale": rationale
        })
    
    # Cleanup Judge
    del judge_llm
    gc.collect()

    # 4. Summary and Save
    print("\n=== Evaluation Results ===")
    print(f"Total Evaluated: {total}")
    if total > 0:
        print(f"Base Model Score: {score_base} (Win Rate: {score_base/total:.2%})")
        print(f"Fine-Tuned Model Score: {score_ft} (Win Rate: {score_ft/total:.2%})")
    else:
        print("No valid judgments collected.")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump({
            "summary": {
                "total": total,
                "base_score": score_base,
                "ft_score": score_ft,
                "base_win_rate": score_base/total if total > 0 else 0,
                "ft_win_rate": score_ft/total if total > 0 else 0
            },
            "details": results
        }, f, indent=2)
    print(f"Detailed results saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
