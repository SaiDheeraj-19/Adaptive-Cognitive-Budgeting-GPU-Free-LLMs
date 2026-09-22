import json
import csv
import time
from acb_server import ACBSession

def run_evaluation(model_family, models_dir, llama_dir, workload_file="workload.json"):
    with open(workload_file, "r") as f:
        queries = json.load(f)
        
    session = ACBSession(model_family, models_dir, llama_dir)
    
    # Define static configs
    static_safe = {
        "p": "Q4_K_M",
        "w": 512,
        "g": 256,
        "k": 0,
        "n": 4
    }
    
    static_capable = {
        "p": "Q8_0",
        "w": 8192,
        "g": 2048,
        "k": 0,
        "n": 7
    }
    
    results = []
    
    configs = [
        ("Static-Safe", False, static_safe),
        ("Static-Capable", False, static_capable),
        ("ACB", True, None)
    ]
    
    try:
        for config_name, is_acb, static_cfg in configs:
            print(f"\n--- Running Configuration: {config_name} ---")
            
            for i, q in enumerate(queries):
                prompt_text = q.get("prompt", "")
                if "messages" in q:
                    prompt_text = "\n".join(q["messages"])
                    
                print(f"[{i+1}/5] Query type: {q['type']} (Len: {len(prompt_text)})")
                
                result = session.generate(prompt_text, is_acb=is_acb, static_config=static_cfg)
                
                # Capture max RSS for this query
                telemetry_state = session.telemetry.get_state()
                
                row = {
                    "config": config_name,
                    "query_index": i,
                    "query_type": q["type"],
                    "latency": result["latency"],
                    "tokens": result["tokens"],
                    "truncated": result["truncated"],
                    "precision": result["config"]["p"],
                    "context": result["config"]["w"],
                    "free_mb": telemetry_state["free_memory_mb"],
                    "swap_used_mb": telemetry_state["swap_used_mb"],
                    "pressure_level": telemetry_state["pressure_level"]
                }
                results.append(row)
                print(f"  -> Latency: {result['latency']:.2f}s | Tokens: {result['tokens']} | Truncated: {result['truncated']}")
                print(f"  -> RAM Free: {telemetry_state['free_memory_mb']:.0f} MB | Pressure: {telemetry_state['pressure_level']}")
                
    finally:
        session.stop()
        
    # Write results
    with open("results.csv", "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
        
    print("\nEvaluation complete. Results saved to results.csv")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        model_family = sys.argv[1]
    else:
        model_family = "Llama-3.2-1B-Instruct"
        
    base = "/Users/saidheeraj/.gemini/antigravity-ide/scratch/acb-evaluation"
    run_evaluation(
        model_family=model_family,
        models_dir=f"{base}/models",
        llama_dir=f"{base}/llama.cpp/build/bin"
    )
