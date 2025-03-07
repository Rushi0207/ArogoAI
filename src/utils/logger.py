import time
import logging
import pandas as pd
import os
from datetime import datetime

class AILogger:
    def __init__(self):
        log_dir = os.path.join(os.getcwd(), "logs")
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, "ai_assistant.log")
        
        self.metrics = pd.DataFrame(columns=["timestamp", "provider", "prompt_length", "response_time", "error"])
        
        logging.basicConfig(
            filename=self.log_file,
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )
        
        print(f"Logging initialized. Log file: {self.log_file}", flush=True)

    def log_request(self, provider, prompt):
        return time.time()

    def log_response(self, start_time, provider, prompt, response, error=None):
        response_time = time.time() - start_time
        log_entry = {
            "timestamp": datetime.now(),
            "provider": provider,
            "prompt_length": len(prompt),
            "response_time": response_time,
            "error": error
        }
        if not self.metrics.empty:
            self.metrics = pd.concat([self.metrics, pd.DataFrame([log_entry])], ignore_index=True)
        else:
            self.metrics = pd.DataFrame([log_entry])
            
        if error:
            logging.error(f"{provider} Error: {error} | Prompt: '{prompt[:50]}...'")
        else:
            logging.info(f"{provider} Response: {response_time:.2f}s | Chars: {len(response)}")

    def get_performance_stats(self):
        return {
            "avg_response_time": self.metrics["response_time"].mean(),
            "total_errors": self.metrics["error"].notna().sum(),
            "most_used_provider": self.metrics["provider"].mode()[0] if not self.metrics.empty else None
        }
