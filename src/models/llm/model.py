"""
LLM-based GEC via prompting and LoRA fine-tuning.
"""

from typing import Optional, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer


GEC_PROMPT_TEMPLATE = """You are a Finnish grammar correction system. Your task is to correct grammatical errors in Finnish sentences while making minimal changes.

Instructions:
- Only fix grammatical errors
- Do not paraphrase or change the meaning
- Keep corrections minimal
- Preserve the original style and vocabulary

Input: {input_sentence}
Corrected:"""


class LLMGECModel:
    """LLM-based GEC model with prompting or LoRA fine-tuning."""

    def __init__(self, model_name: str, use_lora: bool = False, lora_config: Optional[Dict] = None):
        """
        Initialize LLM GEC model.

        Args:
            model_name: HuggingFace model name (e.g., 'meta-llama/Llama-3-8B-Instruct')
            use_lora: Whether to use LoRA fine-tuning
            lora_config: LoRA configuration dict
        """
        self.model_name = model_name
        self.use_lora = use_lora

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype="auto"
        )

        if use_lora:
            self._setup_lora(lora_config or {})

    def _setup_lora(self, lora_config: Dict):
        """Setup LoRA for efficient fine-tuning."""
        from peft import LoraConfig, get_peft_model

        config = LoraConfig(
            r=lora_config.get('r', 8),
            lora_alpha=lora_config.get('alpha', 32),
            target_modules=lora_config.get('target_modules', ["q_proj", "v_proj"]),
            lora_dropout=lora_config.get('dropout', 0.05),
            bias="none",
            task_type="CAUSAL_LM"
        )

        self.model = get_peft_model(self.model, config)
        print(f"LoRA enabled: {self.model.print_trainable_parameters()}")

    def generate(self, input_sentence: str, **generation_kwargs) -> str:
        """
        Generate correction for input sentence.

        Args:
            input_sentence: Corrupted input
            **generation_kwargs: Additional generation parameters

        Returns:
            Corrected sentence
        """
        prompt = GEC_PROMPT_TEMPLATE.format(input_sentence=input_sentence)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=generation_kwargs.get('max_new_tokens', 128),
            temperature=generation_kwargs.get('temperature', 0.7),
            do_sample=generation_kwargs.get('do_sample', False)
        )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract just the correction (after the prompt)
        if "Corrected:" in response:
            correction = response.split("Corrected:")[-1].strip()
            return correction

        return response
