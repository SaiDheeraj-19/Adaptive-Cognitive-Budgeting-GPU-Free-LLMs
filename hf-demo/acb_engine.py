import psutil
import time
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

class ACBEngine:
    def __init__(self):
        self.model_id = "bartowski/Llama-3.2-1B-Instruct-GGUF"
        self.filename = "Llama-3.2-1B-Instruct-Q4_K_M.gguf"
        self.model_path = None
        self.llm = None
        
        self.SYSTEM_RAM_MB = psutil.virtual_memory().total / (1024 * 1024)
        
    def download_model(self):
        print(f"Downloading {self.filename}...")
        self.model_path = hf_hub_download(repo_id=self.model_id, filename=self.filename)
        return self.model_path

    def estimate_complexity(self, prompt: str) -> float:
        """
        Calculates s(q) ∈ [0,1]
        Combines length, structural markers, and reasoning indicators.
        """
        l_norm = min(len(prompt) / 4000.0, 1.0)
        
        structural_markers = ["step by step", "analyze", "compare", "summarize", "code", "function"]
        s_count = sum(1 for m in structural_markers if m in prompt.lower())
        s_norm = min(s_count / 3.0, 1.0)
        
        # Weighted heuristic
        sq = (0.6 * l_norm) + (0.4 * s_norm)
        return min(sq, 1.0)
        
    def get_free_memory_mb(self) -> float:
        """Returns F(t)"""
        return psutil.virtual_memory().available / (1024 * 1024)

    def allocate_budget(self, sq: float, free_mb: float):
        """
        Calculates budget c = (w, n)
        Since we only have one model loaded (Q4_K_M), we fix precision `p`.
        """
        # Context window w
        base_w = 512
        max_w = 8192
        
        # Threads n
        max_threads = psutil.cpu_count(logical=False) or 4
        
        if free_mb < (self.SYSTEM_RAM_MB * 0.1): # < 10% RAM free
            w = base_w + int((max_w - base_w) * sq * 0.2)
            n = max(1, max_threads - 2)
        elif free_mb < (self.SYSTEM_RAM_MB * 0.3): # < 30% RAM free
            w = base_w + int((max_w - base_w) * sq * 0.5)
            n = max(1, max_threads - 1)
        else: # Lots of RAM
            w = base_w + int((max_w - base_w) * sq)
            n = max_threads

        w = min(max_w, max(base_w, w))
        
        return {"context": w, "threads": n, "precision": "Q4_K_M"}

    def reload_model_if_needed(self, context_size, threads):
        # In a real environment we'd reload with new `-c` and `-t`
        # For HF spaces with 16GB RAM, we can just instantiate once with a large context
        # to avoid slow reloads on every query in the UI, but we'll simulate the allocation.
        if self.llm is None:
            if not self.model_path:
                self.download_model()
            
            # Initialize with max context for the UI speed, but logically we allocate `w`
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=8192,  # max possible
                n_threads=threads,
                verbose=False
            )

    def generate(self, prompt: str):
        sq = self.estimate_complexity(prompt)
        free_mb = self.get_free_memory_mb()
        budget = self.allocate_budget(sq, free_mb)
        
        self.reload_model_if_needed(budget['context'], budget['threads'])
        
        formatted_prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        
        start_time = time.time()
        
        # Stream response
        stream = self.llm(
            formatted_prompt,
            max_tokens=128,
            stream=True,
            temperature=0.1
        )
        
        output_text = ""
        token_count = 0
        for chunk in stream:
            token = chunk['choices'][0]['text']
            output_text += token
            token_count += 1
            yield output_text, budget, sq, free_mb, token_count, time.time() - start_time
