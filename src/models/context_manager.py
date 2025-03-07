from collections import deque

class ContextManager:
    def __init__(self, max_context_length=4):
        self.context_window = deque(maxlen=max_context_length)
    
    def add_exchange(self, user_input, ai_response):
        self.context_window.extend([
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": ai_response}
        ])
    
    def get_context(self):
        return list(self.context_window)