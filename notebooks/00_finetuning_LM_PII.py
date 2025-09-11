import argparse
import os
import sys
# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from utilities.train_unsloth import UnslothFinetuner

parser = argparse.ArgumentParser(description="Fine-tune a model using Unsloth.")
parser.add_argument(
    "--csv_path",
    type=str,
    default="data/train_data_personal.csv",
    help="Path to the training CSV file.",
)
parser.add_argument(
    "--model_name",
    type=str,
    default="unsloth/gemma-2-9b-it",
    help="Model name. Supported: unsloth/gemma-2-9b-it, unsloth/Qwen3-8B, unsloth/Qwen3-14B",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="/models",
    help="Directory to save the fine-tuned model.",
)
parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs.")
args = parser.parse_args()

trainer = UnslothFinetuner(
    csv_path=args.csv_path,
    model_name=args.model_name,
    output_dir=args.output_dir,
    epochs=args.epochs,
)
trainer.train()
