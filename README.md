<div align="center">

# Adaptive Cognitive Budgeting for GPU-Free LLMs

**Dynamically allocating inference resources on memory-constrained personal computers**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SaiDheeraj-19/Adaptive-Cognitive-Budgeting-GPU-Free-LLMs/blob/main/demo.ipynb)
[![Platform: macOS](https://img.shields.io/badge/platform-macOS%20%7C%20Linux-lightgrey.svg)]()
[![Hardware: CPU-Only](https://img.shields.io/badge/hardware-CPU--only-orange.svg)]()

*R. Sai Dheeraj — Independent Researcher, Kurnool, Andhra Pradesh, India*

📄 [Read the Paper](paper/Adaptive_Cognitive_Budgeting_GPU_Free_LLMs.pdf) · 📊 [View Raw Results](results/results_v2_corrected.csv) · 🔁 [Reproduce](REPRODUCE.md)

</div>

---

## The Problem

When you run a local LLM on an 8 GB laptop with no GPU, you face a hard constraint: the model, the OS, your browser, and every other app fight over the same pool of RAM. The standard approach is **static configuration** — you pick one quantization level and one context window at install time and use it forever.

This is wrong in both directions:

- **Too conservative** (small context, low precision): Fast on simple queries, but stalls pathologically when fed a long document — the model can't fit the prompt and disk-swaps for minutes.
- **Too aggressive** (large context, high precision): Handles complex queries, but pushes 8 GB machines over the "memory cliff" — a single query that triggers swapping can take **67 minutes** to complete.

---

## The Solution: Adaptive Cognitive Budgeting (ACB)

ACB is a lightweight control layer that sits **above** an unmodified llama.cpp server. Before each query, it answers three questions:

1. **How complex is this query?** (cheap heuristic, no model call)
2. **How much RAM is free right now?** (background OS telemetry)
3. **What's the trend — is pressure increasing?** (EWMA)

Then it selects a **budget vector** `c = (p, w, g, k, n)`:

| Knob | Meaning | Example values |
|------|---------|----------------|
| **p** | Quantization precision | Q4_K_M → Q8_0 |
| **w** | Context window (tokens) | 512 – 8192 |
| **g** | Max generation length | 64 – 2048 |
| **k** | KV-cache retention policy | full → sliding → evict |
| **n** | CPU threads | 4 – 7 |

The rule: **`F(t)` alone gates memory-safety decisions. `s(q)` only spends headroom `F(t)` has already made available.** A simple query on a loaded machine gets a tight budget. A complex query on an idle machine gets a generous one.

If pressure spikes mid-generation, a re-negotiation hook fires and issues an early stop rather than letting the OS kill the process.

---

## Architecture

```
┌─────────────────────────────────┐
│     Application / Chat UI       │
└────────────────┬────────────────┘
                 │ query q
                 ▼
┌────────────────────────────────────────────────────┐
│                   ACB Control Layer                │
│                                                    │
│  ┌──────────────────┐   ┌──────────────────────┐  │
│  │ Complexity        │   │ Resource Telemetry   │  │
│  │ Estimator         │   │ Monitor              │  │
│  │ s(q) ∈ [0,1]     │   │ F(t), ΔF @ 250ms    │  │
│  └────────┬─────────┘   └──────────┬───────────┘  │
│           │                        │               │
│           └──────────┬─────────────┘               │
│                      ▼                             │
│           ┌──────────────────────┐                 │
│           │  Budget Controller   │                 │
│           │  c = (p, w, g, k, n) │                 │
│           └──────────┬───────────┘                 │
│                      │                             │
│           ┌──────────▼───────────┐                 │
│           │  Re-negotiation Hook │                 │
│           │  (every τ=32 tokens) │                 │
│           └──────────┬───────────┘                 │
└──────────────────────┼─────────────────────────────┘
                       │ HTTP (no engine changes)
                       ▼
        ┌──────────────────────────────┐
        │  llama.cpp / GGUF engine     │
        │  (unmodified)                │
        └──────────────────────────────┘
```

---

## Measured Results

Tested on **Apple M2, 8 GB unified memory, CPU-only** (`--n-gpu-layers 0`).  
Workload: 20 identical queries (short factual, multi-turn, long-context) per configuration.  
Generation cap: 128 tokens, applied uniformly to all three configurations.

### Throughput (tokens/second)

| Configuration | Mean TPS | Median TPS | Min TPS | Max TPS | Std Dev |
|---|---|---|---|---|---|
| Static-Safe (Q4_K_M, c=512) | 41.75 | 49.69 | 0.32 | 67.93 | 25.38 |
| Static-Capable (Q8_0, c=8192) | 21.61 | 28.05 | 0.03 | 45.75 | 18.64 |
| **ACB (Dynamic)** | **34.51** | **36.83** | **0.16** | **56.48** | **15.19** |

### Latency (seconds per query)

| Configuration | Mean | Median | Min | **Max** |
|---|---|---|---|---|
| Static-Safe (Q4_K_M, c=512) | 45.31 | 2.05 | 0.10 | **402.42s (6.7 min)** |
| Static-Capable (Q8_0, c=8192) | 330.63 | 4.43 | 2.04 | **4057.91s (67 min)** |
| **ACB (Dynamic)** | **15.49** | **8.50** | **0.20** | **53.14s** |

### ACB Context Decisions (measured, N=20)

| Query Type | Context Window Selected | Threads |
|---|---|---|
| `short_factual` | 2,084 – 2,772 tokens (dynamic) | 7 |
| `multi_turn` | 4,173 tokens (consistent) | 7 |
| `long_context` | 4,505 tokens (max dynamic budget) | 7 |

### Key Findings

> **1. ACB prevents catastrophic latency spikes.** Static-Capable's Q8_0 model with 8192 context pushed the 8 GB M2 over the memory cliff. A single 108-token generation took 4,057 seconds (67 minutes) due to SSD-backed swap thrashing. ACB bounded the context budget so no query ever exceeded 53.14 seconds.

> **2. Static-Safe fails on long inputs.** A rigid 512-token context window causes prompt-processing stalls on 16,000+ character documents, with one query stalling for 402 seconds. ACB expanded its budget dynamically to absorb the same query.

> **3. ACB's standard deviation is 40% lower than Static-Safe's.** A median TPS of 36.83 and a std dev of 15.19 vs Static-Safe's 25.38 shows ACB delivers consistently predictable performance, not just a better average.

---

## Repository Structure

```
├── acb_server.py           # Core ACB implementation
│                           #   - Complexity estimator (heuristic scorer)
│                           #   - Budget controller (decision table)
│                           #   - Re-negotiation hook (mid-generation pressure check)
│                           #   - Server readiness polling (GET /health)
│                           #   - Streaming error handling + timeout protection
│
├── eval_harness.py         # Experiment orchestrator
│                           #   - Configures Static-Safe, Static-Capable, ACB runs
│                           #   - Writes results to CSV with full telemetry columns
│                           #   - --smoke-test flag (1 query/config, fast validation)
│                           #   - --n-queries N flag (controlled batch size)
│
├── telemetry.py            # Background memory monitor
│                           #   - psutil.virtual_memory() + psutil.swap_memory()
│                           #   - macOS memory_pressure -Q @ 250ms intervals
│                           #   - EWMA pressure trend (alpha=0.3)
│
├── generate_workload.py    # Workload generator
│                           #   - Produces mixed short_factual/multi_turn/long_context
│                           #   - Fixed seed for reproducibility
│
├── download_models.py      # Hugging Face model downloader
│
├── workload.json           # Fixed 200-query evaluation workload
├── requirements.txt        # Python dependencies (requests, psutil, huggingface_hub)
├── REPRODUCE.md            # Step-by-step setup and run guide
├── LICENSE                 # MIT
│
├── results/
│   ├── results_v2_corrected.csv      # ✅ CLEAN — 60 rows, 20 per config, all VALID
│   └── results_v1_contaminated.csv   # ⚠️  ARCHIVED — early run, 13 rows contaminated
│                                     #    by HTTP 503 pre-ready bug (harness fired
│                                     #    requests before llama-server finished loading)
│
├── figures/
│   ├── fig1_architecture.png         # ACB system architecture
│   ├── fig2_budget_map.png           # Budget controller decision mapping
│   ├── alg1_algorithm.png            # Algorithm 1 pseudocode
│   └── fig3_control_flow.png         # Per-query control flow diagram
│
└── paper/
    ├── Adaptive_Cognitive_Budgeting_GPU_Free_LLMs.pdf          # ✅ Final paper with results
    └── Adaptive_Cognitive_Budgeting_GPU_Free_LLMs_original_draft.pdf  # Pre-experiment draft
```

---

## Quick Start

```bash
# Clone
git clone https://github.com/SaiDheeraj-19/Adaptive-Cognitive-Budgeting-GPU-Free-LLMs.git
cd Adaptive-Cognitive-Budgeting-GPU-Free-LLMs

# Install Python deps
pip install -r requirements.txt

# Build llama.cpp (CPU-only, no GPU)
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
cmake -B build -DGGML_METAL=OFF -DGGML_CUDA=OFF
cmake --build build --config Release -j4
cd ..

# Download models (needs HF token for gated models)
export HF_TOKEN="hf_your_token_here"
python download_models.py

# Smoke test — validates the full pipeline in ~5 minutes
python eval_harness.py --smoke-test

# Full controlled run — 20 queries × 3 configurations
python eval_harness.py --n-queries 20
```

Results are written to `results_v2_corrected.csv`.

---

## Dataset Schema

| Column | Type | Description |
|--------|------|-------------|
| `run_id` | int | Sequential row ID |
| `timestamp` | ISO-8601 | Wall-clock time of query completion |
| `configuration` | str | `Static-Safe` / `Static-Capable` / `ACB` |
| `model` | str | Model family name |
| `quantization` | str | `Q4_K_M` or `Q8_0` |
| `query_type` | str | `short_factual` / `multi_turn` / `long_context` |
| `query_id` | int | Index into `workload.json` (0–19) |
| `context_limit` | int | Context window allocated (`-c` flag) |
| `generated_tokens` | int | Actual tokens produced |
| `latency_seconds` | float | Wall-clock time from POST to last token |
| `tokens_per_second` | float | `generated_tokens / latency_seconds` |
| `free_memory_mb` | float | `psutil.virtual_memory().free` at completion |
| `available_memory_mb` | float | `psutil.virtual_memory().available` at completion |
| `memory_pressure` | str | `Normal` / `Warn` / `Critical` from `memory_pressure -Q` |
| `swap_mb` | float | Point-in-time swap usage at completion |
| `thread_count` | int | CPU threads assigned (`-t` flag) |
| `truncated` | bool | `True` if re-negotiation hook triggered early stop |
| `process_failure` | bool | `True` if server crash, OOM, or timeout |
| `data_status` | str | `VALID` / `CONTAMINATED` |

---

## Analytical Memory Model

KV-cache memory grows as:

```
m_kv ≈ 2 × L × H × d_h × w × b
```

Where `L` = transformer layers, `H` = KV heads, `d_h` = head dimension, `w` = context length (tokens), `b` = bytes per element (2 for fp16).

For the 3B Qwen model, each additional 1,000 tokens of context costs ~150–250 MB of KV-cache — a cost invisible in the static weight size but directly controlled by ACB's `w` and `k` knobs.

---

## Testbed

| Field | Value |
|---|---|
| SoC | Apple M2 |
| RAM | 8 GB Unified Memory |
| Storage | External HDD (Crucial X9 Pro) |
| GPU offload | **DISABLED** (`--n-gpu-layers 0`) |
| Inference engine | llama.cpp b3799 |
| Model | Llama-3.2-1B-Instruct |
| Quantizations tested | Q4_K_M, Q8_0 |
| OS | macOS |
| Memory telemetry | `psutil` + `memory_pressure -Q` @ 250 ms |

---

## Limitations

- Evaluated on a single 8 GB Apple M2 machine only
- Full controlled run used a 128-token generation cap (uniformly applied) to make Static-Capable tractable
- Workload is semi-synthetic (repeated prompt templates)
- Generalisation to diverse hardware and real-world query distributions requires further study

---

## Citation

```bibtex
@article{dheeraj2026acb,
  title   = {Adaptive Cognitive Budgeting for GPU-Free Language Models
             on Memory-Constrained Personal Computers},
  author  = {R. Sai Dheeraj},
  year    = {2026},
  url     = {https://github.com/SaiDheeraj-19/Adaptive-Cognitive-Budgeting-GPU-Free-LLMs}
}
```

---

## License

MIT — see [LICENSE](LICENSE).
