import os
from typing import Tuple
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments


ALLOWED_MODELS = {
    "unsloth/gemma-2-9b-bnb-4bit",
    "unsloth/gemma-3-12b-pt-unsloth-bnb-4bit",
    "unsloth/Qwen3-8B",
    "unsloth/Qwen3-14B",
}


class UnslothFinetuner:
    """Utility class for LoRA fine-tuning using Unsloth."""

    def __init__(
        self,
        csv_path: str,
        model_name: str,
        output_dir: str = "finetuned_model",
        test_size: float = 0.1,
        epochs: int = 1,
        max_seq_length: int = 2048,
        load_in_4bit: bool = True,
    ) -> None:
        if model_name not in ALLOWED_MODELS:
            raise ValueError(
                f"Model {model_name} not supported. Choose from: {', '.join(sorted(ALLOWED_MODELS))}"
            )

        self.csv_path = csv_path
        self.model_name = model_name
        self.output_dir = output_dir
        self.test_size = test_size
        self.epochs = epochs
        self.max_seq_length = max_seq_length
        self.load_in_4bit = load_in_4bit
        self.model = None
        self.tokenizer = None

    def _load_model(self) -> None:
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=self.max_seq_length,
            load_in_4bit=self.load_in_4bit,
            load_in_8bit=False,
            dtype=None,
            full_finetuning=False,
        )
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r = 32,           # Choose any number > 0! Suggested 8, 16, 32, 64, 128
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj",],
            lora_alpha = 32,  # Best to choose alpha = rank or rank*2
            lora_dropout = 0, # Supports any, but = 0 is optimized
            bias = "none",    # Supports any, but = "none" is optimized
            # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
            use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
            random_state = 3407,
            use_rslora = False,   # We support rank stabilized LoRA
            loftq_config = None,  # And LoftQ
        )

    def _prepare_datasets(self) -> Tuple[Dataset, Dataset]:
        df = pd.read_csv(self.csv_path)
        train_df, val_df = train_test_split(
            df, test_size=self.test_size, random_state=42
        )
        train_ds = Dataset.from_pandas(train_df)
        val_ds = Dataset.from_pandas(val_df)
        return train_ds, val_ds

    def _format_records(self, examples):
        template = "### INSTRUCTION\n{}\n\n### INPUT\n{}\n\n### RESPONSE\n{}"
        eos = self.tokenizer.eos_token
        texts = [
            template.format(inst, inp, out) + eos
            for inst, inp, out in zip(
                examples["instruction"], examples["input"], examples["output"]
            )
        ]
        return {"text": texts}

    def train(self) -> None:
        train_ds, val_ds = self._prepare_datasets()
        self._load_model()

        train_ds = train_ds.map(self._format_records, batched=True)
        val_ds = val_ds.map(self._format_records, batched=True)

        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            args=SFTConfig(
                dataset_text_field="text",
                per_device_train_batch_size=2,
                gradient_accumulation_steps=4,  # Use GA to mimic batch size!
                warmup_steps=5,
                num_train_epochs = self.epochs, # Set this for 1 full training run.
                # max_steps=30,
                learning_rate=2e-4,  # Reduce to 2e-5 for long training runs
                logging_steps=1,
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="linear",
                seed=3407,
                report_to="none",  # Use this for WandB etc
            ),
        )

        trainer.train()

        os.makedirs(self.output_dir, exist_ok=True)
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)


def train_from_csv(
    csv_path: str,
    model_name: str,
    output_dir: str = "finetuned_model",
    **kwargs,
) -> None:
    trainer = UnslothFinetuner(
        csv_path=csv_path,
        model_name=model_name,
        output_dir=output_dir,
        **kwargs,
    )
    trainer.train()
