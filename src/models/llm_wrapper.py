from openai import OpenAI
import google.generativeai as genai
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
import os
import functools
from src.utils.cache_manager import CacheManager
from src.utils.logger import AILogger
from src.utils.moderation import ContentModerator
from src.models.context_manager import ContextManager

class LLMWrapper:
    def __init__(self, provider="openai"):
        self.context_manager = ContextManager()
        self.moderator = ContentModerator()
        self.logger = AILogger()
        self.cache = CacheManager(max_size=50)
        self.provider = provider.lower()
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        genai.configure(api_key=self.gemini_api_key)
        if self.provider == "huggingface":
            model_name = "facebook/opt-125m"
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    device_map="auto"
                )
            except Exception as e:
                raise RuntimeError(f"Hugging Face model loading failed: {str(e)}")
        
    def generate_response(self, prompt):
        if self.moderator.is_toxic(prompt):
            return "Request blocked: Violates content policy."
            
        context = self.context_manager.get_context()
        full_prompt = "\n".join(
            [f"{msg['role']}: {msg['content']}" for msg in context] + 
            [f"user: {prompt}"]
        )

        start_time = self.logger.log_request(self.provider, prompt)
        try:
            prompt_hash = self.cache._hash_prompt(prompt, self.provider)
            cached_response = self.cache.get_cached_response(prompt_hash)
            if cached_response:
                return f"(Cached) {cached_response}"
            print("provider: ",self.provider)
            if self.provider == "openai":
                response = self._openai_response(prompt)
            elif self.provider == "gemini":
                response = self._gemini_response(prompt)
            elif self.provider == "huggingface":
                response = self._huggingface_response(prompt)
            else:
                response = "Invalid provider."
            
            self.cache.set_cached_response(prompt_hash, response)
            self.logger.log_response(start_time, self.provider, prompt, response)
            if self.moderator.is_toxic(response):
                return "Response blocked: Potentially harmful content detected."
                
            self.context_manager.add_exchange(prompt, response)
            print("start_time",start_time)
            return response
        except Exception as e:
            self.logger.log_response(start_time, self.provider, prompt, "", error=str(e))
            return f"Error: {str(e)}"

    def _openai_response(self, prompt):
        try:
            client = OpenAI(
                base_url="https://models.inference.ai.azure.com",
                api_key=self.openai_api_key,
            )
            #client = OpenAI(api_key=self.openai_api_key)
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=1,
                max_tokens=4096,
                top_p=1
            )
            print(response)
            return response.choices[0].message.content.strip()
        except openai.OpenAIError as e:
            return f"OpenAI Error: {str(e)}"

    def _gemini_response(self, prompt):
        try:
            model = genai.GenerativeModel("gemini-2.0-flash")
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"Gemini API Error: {str(e)}")
            return f"Gemini API Error: {str(e)}"

    def _huggingface_response(self, prompt):
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True
            )
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            return f"Hugging Face Error: {str(e)}"
