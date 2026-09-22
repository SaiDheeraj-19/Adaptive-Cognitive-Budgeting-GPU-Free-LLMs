import os
from huggingface_hub import hf_hub_download

MODELS = [
    ("bartowski/Llama-3.2-1B-Instruct-GGUF", "Llama-3.2-1B-Instruct-Q4_K_M.gguf"),
    ("bartowski/Llama-3.2-1B-Instruct-GGUF", "Llama-3.2-1B-Instruct-Q8_0.gguf"),
    ("Qwen/Qwen2.5-3B-Instruct-GGUF", "qwen2.5-3b-instruct-q4_k_m.gguf"),
    ("Qwen/Qwen2.5-3B-Instruct-GGUF", "qwen2.5-3b-instruct-q8_0.gguf"),
    ("bartowski/Meta-Llama-3.1-8B-Instruct-GGUF", "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"),
    ("bartowski/Meta-Llama-3.1-8B-Instruct-GGUF", "Meta-Llama-3.1-8B-Instruct-Q8_0.gguf"),
]

local_dir = "/Users/saidheeraj/.gemini/antigravity-ide/scratch/acb-evaluation/models"
os.makedirs(local_dir, exist_ok=True)

print("Downloading models...")
for repo_id, filename in MODELS:
    print(f"Downloading {filename} from {repo_id}...")
    try:
        path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir)
        print(f"Downloaded to {path}")
    except Exception as e:
        print(f"Failed to download {filename}: {e}")

print("All downloads complete.")
