import subprocess
import time
import psutil
import threading
import json
from collections import deque

class TelemetryMonitor:
    def __init__(self, sample_interval=0.25):
        self.sample_interval = sample_interval
        self.running = False
        self.thread = None
        self.history = deque(maxlen=20)  # Short history for trend calculation
        self.current_state = {
            "free_memory_mb": 0,
            "available_memory_mb": 0,
            "swap_used_mb": 0,
            "pressure_level": "Normal",
            "pressure_ewma": 0.0,
            "trend": 0.0
        }

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()

    def _monitor_loop(self):
        ewma_alpha = 0.3
        while self.running:
            mem = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            # macOS specific parsing could be added here if psutil isn't enough, 
            # but psutil handles available/free on macOS.
            # For memory_pressure -Q:
            pressure_level = "Normal"
            try:
                # 'memory_pressure -Q' gives output like:
                # System-wide memory free percentage: 41%
                # Pages freed: 0
                # Pages purged: 0
                # Pages uncompressed: 0
                out = subprocess.check_output(['memory_pressure', '-Q'], stderr=subprocess.STDOUT, text=True)
                if "Warn" in out or "Warning" in out:
                    pressure_level = "Warn"
                elif "Critical" in out:
                    pressure_level = "Critical"
            except Exception:
                pass
            
            free_mb = mem.free / (1024 * 1024)
            available_mb = mem.available / (1024 * 1024)
            swap_used_mb = swap.used / (1024 * 1024)
            
            # Map pressure to a numerical value for EWMA
            p_val = 0.0
            if pressure_level == "Warn":
                p_val = 0.5
            elif pressure_level == "Critical":
                p_val = 1.0
                
            prev_ewma = self.current_state["pressure_ewma"]
            new_ewma = (ewma_alpha * p_val) + ((1 - ewma_alpha) * prev_ewma)
            
            trend = new_ewma - prev_ewma

            self.current_state = {
                "free_memory_mb": free_mb,
                "available_memory_mb": available_mb,
                "swap_used_mb": swap_used_mb,
                "pressure_level": pressure_level,
                "pressure_ewma": new_ewma,
                "trend": trend
            }
            self.history.append(self.current_state)
            
            time.sleep(self.sample_interval)

    def get_state(self):
        return self.current_state

    def sample(self):
        # Returns F(t) (available memory in bytes) and delta F (trend)
        # We use available_memory instead of strictly free memory as it accounts for caches
        F_t = self.current_state["available_memory_mb"] * 1024 * 1024
        delta_F = self.current_state["trend"]
        return F_t, delta_F

if __name__ == "__main__":
    monitor = TelemetryMonitor()
    monitor.start()
    try:
        for _ in range(5):
            print(json.dumps(monitor.get_state(), indent=2))
            time.sleep(1)
    finally:
        monitor.stop()
