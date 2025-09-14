import os
import torch
import logging
from typing import Optional


os.environ["TRITON_LOG_LEVEL"] = "ERROR"

class Model:
    """
    A class to handle model setup and initialization for different types of models.
    Supports OpenAI models, Hugging Face models, and custom fine-tuned models.
    """

    def __init__(self, model_name: str, device: Optional[str] = None, **kwargs):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.client = None
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = self._determine_model_type()
        self._setup_model(**kwargs)
        self._set_generate_function()

    def _determine_model_type(self) -> str:
        name = self.model_name.lower()
        print(f"Name model in model.py: {name}")
        if name in ["gpt-4o-mini", "gpt-4o", "o3", "o3-mini", "o4-mini", "gpt-4.1-2025-04-14"]:
            return "openai"

        try:
            from unsloth import FastLanguageModel
        except ImportError:
            print("Unsloth not installed")

        if name.startswith("deepseek"):
            return "deepseek"
        elif "aya" in name:
            return "aya-expanse"
        else:
            return "hf_model"

    def _setup_model(self, **kwargs):
        setup_methods = {
            "openai": self._setup_openai,
            "deepseek": self._setup_deepseek,
        }
        # All other models use HF setup
        if self.model_type in setup_methods:
            setup_methods[self.model_type](**kwargs)
        else:
            self._setup_hf_model(**kwargs)

    def _set_generate_function(self):
        if self.model_type == "openai":
            self.generate = self._generate_openai
        elif self.model_type == "deepseek":
            self.generate = self._generate_deepseek
        else:
            # All HF models use unified generation, with model-specific options
            self.generate = self._generate_hf

    def _setup_openai(self):
        from openai import OpenAI
        import dotenv
        dotenv.load_dotenv()
        self.client = OpenAI()
        logging.getLogger("openai").setLevel(logging.WARNING)
        logging.getLogger("openai._client").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        print("OpenAI client initialized")

    def _setup_deepseek(self):
        from openai import OpenAI
        import dotenv
        dotenv.load_dotenv()
        self.client = OpenAI()
        print("DeepSeek client (OpenAI API) initialized")

    def _setup_hf_model(self, **kwargs):
        if self.model_type == "aya-expanse":
            # Use traditional HuggingFace for Aya Expanse
            from transformers import AutoTokenizer, AutoModelForCausalLM
            model_id = self.model_name  # Use as provided
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForCausalLM.from_pretrained(model_id).to(self.device)
            logging.info(f"Loaded HuggingFace model: {model_id}")
        else:
            # Use Unsloth for all other supported models

            print(f"Model name: {self.model_name}")
            model_id = self.model_name  # Use as provided
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_id,
                max_seq_length=6000,
                dtype=torch.bfloat16,
                load_in_4bit=True,
                load_in_8bit=False,
            )
            
            FastLanguageModel.for_inference(self.model)
            logging.info(f"Loaded Unsloth model: {model_id}")
            print(f"Loaded Unsloth model: {model_id}")


    def _generate_openai(self, prompt, temperature=0.3, max_new_tokens=8):
        if self.model_name in ["o3-mini", "o4-mini", "03"]:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_completion_tokens=max_new_tokens,
            )
            return response.choices[0].message.content

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_new_tokens,
        )
        return response.choices[0].message.content

    def _generate_deepseek(self, prompt, temperature=0.7, max_new_tokens=2048):
        messages = [{"role": "user", "content": prompt}]
        completion = self.client.chat.completions.create(
            model="deepseek-ai/DeepSeek-R1-0528",
            messages=messages,
            temperature=temperature,
            max_tokens=5000
        )
        response = completion.choices[0].message.content.strip()
        # Remove thinking content
        index = response.find("</think>")
        if index != -1:
            response = response[index + len("</think>"):].strip()
        return response

    def _generate_hf(self, prompt, max_new_tokens=8, temperature=0.3, top_p=0.95):
        """
        Unified HF model generation for: Gemma, Aya-Expanse, Qwen, fine-tuned.
        Uses options according to self.model_type.
        """

        if 'aya' in self.model_name.lower():
            # Use chat template for Aya
            messages = [{"role": "user", "content": prompt}]
            input_ids = self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
            ).to(self.device)
            input_length = input_ids.shape[1]
            outputs = self.model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=True)
            new_tokens = outputs[0][input_length:]
            return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        elif 'qwen' in self.model_name.lower():
            # Use chat template for Qwen (Unsloth)
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

            # conduct text completion
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens
            )
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

            # parsing thinking content
            try:
                # rindex finding 151668 (</think>)
                index = len(output_ids) - output_ids[::-1].index(151668)
            except ValueError:
                index = 0

            # thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
            content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
            return content

        else:
            # Simple, no chat template
            input_ids = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            input_length = input_ids.input_ids.shape[1]
            outputs = self.model.generate(**input_ids, max_new_tokens=max_new_tokens)
            new_tokens = outputs[0][input_length:]
            answer = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            return answer.strip()

    def get_model_components(self):
        return self.model, self.tokenizer, self.client, self.model_type

    def is_ready(self) -> bool:
        if self.model_type == "openai" or self.model_type == "deepseek":
            return self.client is not None
        else:
            return self.model is not None and self.tokenizer is not None
