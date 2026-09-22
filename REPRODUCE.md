# Reproducibility Instructions

## Hardware Requirements
- Apple M2 (or any CPU-only machine), 8 GB RAM minimum
- ~25 GB free disk space for models

## Setup

```bash
# 1. Clone this repo
git clone https://github.com/SaiDheeraj-19/Adaptive-Cognitive-Budgeting-GPU-Free-LLMs.git
cd Adaptive-Cognitive-Budgeting-GPU-Free-LLMs

# 2. Create virtualenv
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 3. Build llama.cpp (CPU only)
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp && cmake -B build -DGGML_METAL=OFF && cmake --build build --config Release -j4
cd ..

# 4. Download models (requires Hugging Face token)
export HF_TOKEN="your_token_here"
python download_models.py

# 5. Run smoke test (1 query per config, ~5 min)
python eval_harness.py --smoke-test

# 6. Run full controlled experiment (20 queries per config, ~8 hrs)
python eval_harness.py --n-queries 20
```

## Output
Results are written to `results_v2_corrected.csv` with the following schema:

| Column | Description |
|--------|-------------|
| run_id | Sequential row ID |
| configuration | Static-Safe / Static-Capable / ACB |
| model | Model family name |
| quantization | Q4_K_M or Q8_0 |
| query_type | short_factual / multi_turn / long_context |
| context_limit | Tokens admitted to context window |
| generated_tokens | Actual tokens generated |
| latency_seconds | Wall-clock time from request to last token |
| tokens_per_second | generated_tokens / latency_seconds |
| free_memory_mb | psutil free RAM at completion |
| available_memory_mb | psutil available RAM at completion |
| memory_pressure | macOS memory_pressure -Q output |
| swap_mb | Point-in-time swap usage at completion |
| thread_count | CPU threads used |
| truncated | True if re-negotiation hook triggered early stop |
| process_failure | True if server crash or timeout |
| data_status | VALID / CONTAMINATED |
