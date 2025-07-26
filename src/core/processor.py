# src/core/processor.py
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class Processor:
    def __init__(self, model_path: str):
        print(f"🔄 Loading multilingual model from: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        # Load with the same settings used for training
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        print("✅ Model loaded successfully.")

    def process(self, instruction: str) -> str:
        """Processes a given instruction (e.g., 'Translate Hindi to English: ...')."""
        inputs = self.tokenizer(instruction, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_length=256)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)