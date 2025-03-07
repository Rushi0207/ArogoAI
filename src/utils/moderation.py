from transformers import pipeline, AutoTokenizer
import torch

class ContentModerator:
    def __init__(self, max_tokens=512):
        self.tokenizer = AutoTokenizer.from_pretrained("martin-ha/toxic-comment-model")
        self.max_tokens = max_tokens
        self.moderation_model = pipeline(
            "text-classification",
            model="martin-ha/toxic-comment-model",
            device=0 if torch.cuda.is_available() else -1
        )
    
    def chunk_text(self, text):
        """Splits text into chunks with at most self.max_tokens tokens each."""
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        chunks = []
        for i in range(0, len(tokens), self.max_tokens):
            chunk_tokens = tokens[i:i+self.max_tokens]
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            chunks.append(chunk_text)
        return chunks

    def is_toxic(self, text, threshold=0.7):
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) > self.max_tokens:
            try:
                import google.generativeai as genai
                gemini_model = genai.GenerativeModel("gemini-2.0-flash")
                prompt = f"Please analyze the following text for toxicity: \"{text}\". Respond with either 'Safe' or 'Toxic'."
                gemini_response = gemini_model.generate_content(prompt)
                response_text = gemini_response.text.strip().lower()
                if "toxic" in response_text:
                    return True
                else:
                    return False
            except Exception as e:
                print(f"Gemini moderation error: {e}")
                chunks = self.chunk_text(text)
                for chunk in chunks:
                    results = self.moderation_model(chunk)
                    toxic_labels = ["toxic", "severe_toxic", "threat", "insult"]
                    if any(res["label"].lower() in toxic_labels for res in results):
                        return True
                return False
        else:
            try:
                import google.generativeai as genai
                gemini_model = genai.GenerativeModel("gemini-2.0-flash")
                prompt = f"Please analyze the following text for toxicity: \"{text}\". Respond with either 'Safe' or 'Toxic'."
                gemini_response = gemini_model.generate_content(prompt)
                response_text = gemini_response.text.strip().lower()
                if "toxic" in response_text:
                    return True
                else:
                    return False
            except Exception as e:
                print(f"Gemini moderation error: {e}")
                results = self.moderation_model(text)
                toxic_labels = ["toxic", "severe_toxic", "threat", "insult"]
                return any(res["label"].lower() in toxic_labels for res in results)