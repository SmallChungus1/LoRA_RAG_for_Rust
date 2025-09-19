"""
Generate a mixed, tierless Rust QA dataset via OpenRouter based on 3 tiers of topics. 
Tier1: foundational rust concepts. Tier2: intermediate rust concepts. Tier3: concurrency concepts.

Output format (JSONL):
  {"question": "...", "answer": "..."}
"""

import os, json, random, time, requests, re
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")
if not API_KEY:
    raise SystemExit("Please set OPENROUTER_API_KEY")

MODEL = os.getenv("OPENROUTER_MODEL")
if not MODEL:
    raise SystemExit("Please set OPENROUTER_MODEL")

OUT_PATH = "rust_qa_dataset.jsonl"

# How many QAs (edit as you like)
N_FOUNDATIONS = 10
N_INTERMEDIATE = 8
N_CONCURRENCY = 10

TEMPERATURE = 0.7
TIMEOUT = 60
RETRIES = 3

# --- topic pools (we sample from these; tiers are NOT emitted) ---
FOUNDATIONS = [
    "let vs let mut (tiny example)",
    "ownership basics, moving a String",
    "borrowing: &T vs &mut T",
    "slices for strings and arrays",
    "struct + method example",
    "enums and match with Option",
    "Result and ? operator",
    "iterator map/filter/collect",
    "modules and pub",
    "string vs &str",
]
INTERMEDIATE = [
    "lifetimes: function returning &str",
    "impl with explicit lifetime when holding refs",
    "trait bounds: a generic max_of",
    "IntoIterator vs Iterator in for loops",
    "custom iterator adaptor example",
    "RefCell and interior mutability",
    "Box vs Rc vs Arc tradeoffs",
    "manual custom error type with Display",
]
CONCURRENCY = [
    "threads + mpsc: send work and collect results",
    "Arc<Mutex<_>> shared counter",
    "explain Send vs Sync",
    "thread::scope splitting borrows",
    "async vs threads: when to choose",
    "Tokio: concurrent HTTP GET skeleton",
    "Arc<RwLock<T>> for read-heavy access",
    "atomics: lock-free counter and Ordering",
    "why Rc is not Send; fix with Arc",
    "bounded channel and backpressure idea",
]

JSON_RE = re.compile(r"\{.*\}", re.S)

SYSTEM = (
    "You are a concise Rust tutor.\n"
    "Return ONLY a single-line JSON object with keys 'question' and 'answer'.\n"
    "Rules:\n"
    " - No text before or after the JSON.\n"
    " - No Markdown code fences anywhere.\n"
    " - Escape ALL newlines in strings as \\n.\n"
    " - Escape double quotes inside strings as \\\".\n"
    " - Keep it short; one minimal Rust snippet is OK inside 'answer'.\n"
)

USER_TEMPLATE = (
    "Create ONE Rust Q&A pair about: {topic}\n"
    "Output strictly as ONE LINE of JSON in (question, answer) format\n"
)

def call_openrouter(system: str, user: str) -> str:
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "temperature": TEMPERATURE,
        "messages": [
            {"role":"system","content": system},
            {"role":"user","content": user},
        ],
        # If the model supports it, this *greatly* reduces parse issues:
        "response_format": {"type": "json_object"},
        # Optional: cap tokens to avoid verbose answers
        "max_tokens": 400,
    }
    r = requests.post(url, headers=headers, json=payload, timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]

def extract_json_object(text: str):
    m = JSON_RE.search(text)
    if not m: return None
    blob = m.group(0).strip()
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        return None

def make_samples(topics, n, out_list):
    # sample with wrap-around if n > len(topics)
    pool = topics[:]
    random.shuffle(pool)
    if n > len(pool):
        pool = (pool * ((n + len(pool) - 1) // len(pool)))[:n]
    else:
        pool = pool[:n]

    for i, topic in enumerate(pool, 1):
        for attempt in range(1, RETRIES + 1):
            try:
                content = call_openrouter(SYSTEM, USER_TEMPLATE.format(topic=topic))
                obj = extract_json_object(content)
                if not obj or "question" not in obj or "answer" not in obj:
                    raise ValueError("model did not return valid {question, answer} JSON")
                # minimal cleanup: ensure strings
                q = str(obj["question"]).strip()
                a = str(obj["answer"]).strip()
                out_list.append({"question": q, "answer": a})
                print(f"[ok] {topic}")
                break
            except Exception as e:
                print(f"[retry {attempt}] {topic}: {e}")
                time.sleep(2.0 * attempt)
        else:
            print(f"[skip] {topic}")

def main():
    random.seed(42)
    out = []
    make_samples(FOUNDATIONS, N_FOUNDATIONS, out)
    make_samples(INTERMEDIATE, N_INTERMEDIATE, out)
    make_samples(CONCURRENCY, N_CONCURRENCY, out)
    random.shuffle(out)  # mixed, tierless
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for row in out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(out)} QAs to {OUT_PATH}")

if __name__ == "__main__":
    main()
