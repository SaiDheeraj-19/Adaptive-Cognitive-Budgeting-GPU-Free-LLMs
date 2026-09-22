# Adaptive Cognitive Budgeting for GPU-Free LLMs

**"Adaptive Cognitive Budgeting for GPU-Free Language Models on Memory-Constrained Personal Computers"**  
R. Sai Dheeraj — Independent Researcher, Kurnool, Andhra Pradesh, India

---

## Overview

This repository contains the full implementation, experiment harness, raw results, and reproducibility instructions for the **Adaptive Cognitive Budgeting (ACB)** framework.

ACB is a control layer that sits above a CPU-based GGUF inference engine (llama.cpp) and dynamically allocates five inference knobs **per query** based on query complexity and live host memory pressure:

| Knob | Description |
|------|-------------|
| **p** – Quantization precision | Q4_0 → Q5_K → Q8_0 |
| **w** – Context window | Scaled to query complexity |
| **g** – Generation length | Capped per complexity estimate |
| **k** – KV-cache retention | Full → sliding → evict |
| **n** – CPU thread count | Reduced under memory pressure |

---

## Key Results (Measured, N=20 per configuration)

| Configuration | Mean TPS | Mean Latency | Max Latency | Pressure |
|---|---|---|---|---|
| Static-Safe (Q4_K_M, c=512) | 41.75 | 45.31s | **402.42s** | Normal |
| Static-Capable (Q8_0, c=8192) | 21.61 | 330.63s | **4057.91s (67 min)** | Normal |
| **ACB (Dynamic)** | **34.51** | **15.49s** | **53.14s** | **Normal** |

> ACB prevented catastrophic latency spikes that caused Static-Capable to stall for **67 minutes** on a single query, while maintaining consistent throughput across all query types.

---

## Repository Structure

```
├── acb_server.py          # ACB session wrapper — complexity estimator, budget controller,
│                          # re-negotiation hook, server readiness polling
├── eval_harness.py        # Experiment orchestrator — runs workload across all 3 configs
├── telemetry.py           # Background memory/pressure telemetry monitor (250ms polling)
├── generate_workload.py   # Workload generator — produces mixed query set
├── download_models.py     # Hugging Face model downloader
├── workload.json          # Fixed 200-query workload (short_factual, multi_turn, long_context)
├── requirements.txt       # Python dependencies
├── REPRODUCE.md           # Full step-by-step reproducibility guide
│
├── results/
│   ├── results_v2_corrected.csv    # CLEAN — 60 rows, 20 per config, all valid
│   └── results_v1_contaminated.csv # DO NOT USE — early run with HTTP 503 pre-ready bug
│
├── figures/
│   ├── fig1_architecture.png   # ACB architecture block diagram
│   ├── fig2_budget_map.png     # Budget controller mapping
│   ├── alg1_algorithm.png      # Algorithm 1 pseudocode
│   └── fig3_control_flow.png   # Per-query control flow diagram
│
└── paper/
    └── ACB_Paper_Final.pdf     # Full IEEE-format paper with measured results
```

---

## Hardware Tested

| Field | Value |
|---|---|
| SoC | Apple M2 |
| RAM | 8 GB Unified Memory |
| GPU offload | **DISABLED** (`--n-gpu-layers 0`) |
| Model | Llama-3.2-1B-Instruct |
| Quantizations | Q4_K_M, Q8_0 |
| Inference engine | llama.cpp b3799 |

---

## Quick Start

```bash
git clone https://github.com/SaiDheeraj-19/Adaptive-Cognitive-Budgeting-GPU-Free-LLMs.git
cd Adaptive-Cognitive-Budgeting-GPU-Free-LLMs
pip install -r requirements.txt
python eval_harness.py --smoke-test   # Validates harness (1 query/config)
python eval_harness.py --n-queries 20  # Full controlled run
```

See [REPRODUCE.md](REPRODUCE.md) for full setup instructions including llama.cpp build and model download.

---

## Citation

```
@article{dheeraj2026acb,
  title     = {Adaptive Cognitive Budgeting for GPU-Free Language Models on
               Memory-Constrained Personal Computers},
  author    = {Dheeraj, R. Sai},
  year      = {2026},
  note      = {Independent research}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
