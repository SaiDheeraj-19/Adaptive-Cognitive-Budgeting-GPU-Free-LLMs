import time
import re
import requests
import subprocess
import os
import json
from telemetry import TelemetryMonitor

class ComplexityEstimator:
    def estimate(self, query: str) -> float:
        # A simple rule-based heuristic
        score = 0.0
        
        # Length feature
        words = len(query.split())
        score += min(words / 500.0, 0.4) # up to 0.4 for length
        
        # Code blocks or structured data
        if "```" in query or "def " in query or "{" in query:
            score += 0.3
            
        # Distinct sub-questions
        questions = query.count("?")
        score += min(questions * 0.1, 0.3)
        
        return min(score, 1.0)

class BudgetController:
    def __init__(self, model_family, base_dir):
        self.model_family = model_family # e.g. "Llama-3.2-1B-Instruct"
        self.base_dir = base_dir
        self.max_cores = 8 # M2 has 8 cores
        
    def select_budget(self, s_q, F_t, delta_F):
        # F_t is available memory in bytes
        free_mb = F_t / (1024 * 1024)
        
        # Precision p
        if free_mb < 2000:
            p = "Q4_K_M"
        else:
            p = "Q8_0"
            
        # Context w and generation g
        # Max context 8192 for our test
        w = int(2048 + (s_q * 6144))
        g = int(256 + (s_q * 1792))
        
        # KV cache policy k (represented as sliding window size, 0 = full)
        # If free_mb is low, we aggressively slide the window
        if free_mb < 1500:
            k = 1024 # sliding window 1024
        else:
            k = 0 # full retention
            
        # Threads n
        if delta_F > 0.1: # worsening pressure
            n = max(2, self.max_cores - 2)
        else:
            n = self.max_cores - 1
            
        return {
            "p": p,
            "w": w,
            "g": g,
            "k": k,
            "n": n
        }

class ACBSession:
    def __init__(self, model_family, models_dir, llama_dir):
        self.model_family = model_family
        self.models_dir = models_dir
        self.llama_dir = llama_dir
        self.estimator = ComplexityEstimator()
        self.controller = BudgetController(model_family, models_dir)
        self.telemetry = TelemetryMonitor(sample_interval=0.25)
        self.telemetry.start()
        
        self.server_process = None
        self.current_config = None
        
    def _start_server(self, config):
        if self.server_process:
            self.server_process.terminate()
            self.server_process.wait()
            
        model_path = os.path.join(self.models_dir, f"{self.model_family}-{config['p']}.gguf")
        
        # If Qwen, the filename is slightly different (lowercase), we need to handle this
        if "Qwen" in model_path:
            model_path = model_path.replace("Qwen2.5-3B-Instruct", "qwen2.5-3b-instruct").replace("Q4_K_M", "q4_k_m")
            
        cmd = [
            os.path.join(self.llama_dir, "llama-server"),
            "-m", model_path,
            "-c", str(config['w']),
            "-t", str(config['n']),
            "--n-gpu-layers", "0",
            "--port", "8080"
        ]
        
        # Add sliding window if needed
        # llama-server doesn't natively expose an easy KV retention flag that changes dynamically,
        # but we can simulate it with context size or passing the system prompt.
        
        print(f"Starting server: {' '.join(cmd)}")
        self.server_process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Wait for server to start
        for _ in range(30):
            try:
                resp = requests.get("http://localhost:8080/health")
                if resp.status_code == 200:
                    break
            except:
                pass
            time.sleep(1)
                
    def stop(self):
        self.telemetry.stop()
        if self.server_process:
            self.server_process.terminate()
            self.server_process.wait()

    def generate(self, prompt, is_acb=True, static_config=None):
        if is_acb:
            s_q = self.estimator.estimate(prompt)
            F_t, delta_F = self.telemetry.sample()
            c = self.controller.select_budget(s_q, F_t, delta_F)
        else:
            c = static_config
            
        # Hot-swap if precision, context, or threads changed
        if self.current_config is None or \
           self.current_config['p'] != c['p'] or \
           self.current_config['w'] != c['w'] or \
           self.current_config['n'] != c['n']:
            self._start_server(c)
            self.current_config = c
            
        # Stream the request
        payload = {
            "prompt": prompt,
            "n_predict": c['g'],
            "stream": True
        }
        
        start_time = time.time()
        response = requests.post("http://localhost:8080/completion", json=payload, stream=True)
        
        generated_tokens = 0
        truncated = False
        output_text = ""
        
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                if decoded_line.startswith("data: "):
                    data_str = decoded_line[6:]
                    try:
                        data = json.loads(data_str)
                        output_text += data.get("content", "")
                        generated_tokens += 1
                        
                        # Re-negotiation hook
                        if is_acb and generated_tokens % 32 == 0:
                            F_t, delta_F = self.telemetry.sample()
                            # Check hard threshold (e.g. less than 500MB available)
                            if F_t < 500 * 1024 * 1024:
                                truncated = True
                                response.close() # Abort the stream
                                break
                    except json.JSONDecodeError:
                        pass
                        
        latency = time.time() - start_time
        return {
            "output": output_text,
            "latency": latency,
            "tokens": generated_tokens,
            "truncated": truncated,
            "config": c
        }
