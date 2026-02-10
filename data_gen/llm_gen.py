#!/usr/bin/env python3
# Minimal OpenRouter → NDJSON Rust QA generator.
# Output: each line is {"question":"...","answer":"..."} (no wrapper).

import os, json, time, requests, random
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL   = os.getenv("OPENROUTER_MODEL")  # e.g., "openai/gpt-4o-mini" or your pick
if not API_KEY or not MODEL:
    raise SystemExit("Set OPENROUTER_API_KEY and OPENROUTER_MODEL in your env")

OUT     = "./data_gen/rust_qa_500_bench.jsonl"
MODEL_MAX_TOKENS = 30000  # adjust to your model’s limit
TOTAL   = 500           # total pairs to generate
BATCH   = 50            # pairs per request
TEMP    = 0.4
TIMEOUT = 60
RETRIES = 4              # simple retry

SYSTEM = (
    "You are a concise Rust tutor. Produce short, correct Q&A pairs across fundamentals, intermediate, and concurrency.\n"
    "OUTPUT FORMAT: **NDJSON** — exactly N lines; each line is a compact JSON object with keys 'question' and 'answer'.\n"
    "Rules:\n"
    "- Output ONLY NDJSON (no preface/trailer text).\n"
    "- No Markdown fences around the NDJSON.\n"
    "- Prefer std; include a tiny ```rust code block``` inside 'answer' only when helpful.\n"
    "- Escape newlines inside strings as \\n."
)

USER_TPL = (
    "Generate {n} diverse Rust Q&A pairs (mixed difficulty: fundamentals, intermediate, concurrency).\n"
    "Return EXACTLY {n} lines of NDJSON. Each line must be a single JSON object of question answer pairs like: (question:answer)\n"
    "Do not wrap in an array. No extra lines or commentary."
)

def call_openrouter(n: int) -> str:
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "temperature": TEMP,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user",   "content": USER_TPL.format(n=n)},
        ],
        # Keep it simple; many providers handle arrays better than NDJSON,
        # but NDJSON makes our parsing trivial. No response_format used.
        "max_tokens": MODEL_MAX_TOKENS,   # adjust to your model’s limit
    }
    r = requests.post(url, headers=headers, json=payload, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

def parse_ndjson(text: str):
    """Yield dicts from NDJSON content; skip bad lines quietly."""
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            q, a = obj.get("question"), obj.get("answer")
            if isinstance(q, str) and isinstance(a, str) and q.strip() and a.strip():
                yield {"question": q.strip(), "answer": a.strip()}
        except json.JSONDecodeError:
            # Skip malformed lines; keep it simple.
            continue

def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    written = 0
    random.seed(42)

    with open(OUT, "w", encoding="utf-8") as f:
        while written < TOTAL:
            need = min(BATCH, TOTAL - written)
            for attempt in range(1, RETRIES + 1):
                try:
                    raw = call_openrouter(need)
                    count_before = written
                    for item in parse_ndjson(raw):
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
                        written += 1
                        if written >= TOTAL:
                            break
                    print(f"[ok] wrote {written - count_before}, total {written}/{TOTAL}")
                    break
                except Exception as e:
                    print(f"[retry {attempt}] {e}")
                    time.sleep(2.0 * attempt)
            else:
                print("[skip] batch failed; continuing...")
    print(f"Done: {written} lines → {OUT}")

if __name__ == "__main__":
    main()
