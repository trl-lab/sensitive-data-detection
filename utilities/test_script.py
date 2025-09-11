import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import json
from pathlib import Path
from typing import Dict, List
import argparse
from tqdm import tqdm
import time
import psutil

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
from modules.detect_reflect.file_loader import load_data
from modules.detect_reflect.pii_module import detect_and_reflect_pii
from modules.detect_reflect.non_pii_module import detect_non_pii
from modules.detect_reflect.sensitivityClassifier import SensitivityClassifier

from modules.eval_utils import evaluate_pii_detection, evaluate_non_pii_table


def copy_clean_dataset(src: Path, dest: Path) -> None:
    """Copy non-personal dataset and remove existing prediction fields."""
    with open(src, "r") as f:
        data = json.load(f)
    for table in data.values():
        metadata = table.get("metadata", {})
        for key in list(metadata.keys()):
            if key.startswith("non_pii_") and key != "non_pii":
                del metadata[key]
            if key.startswith("pii_"):
                del metadata[key]
        for col in table.get("columns", {}).values():
            for k in list(col.keys()):
                if k.startswith("pii_") or k.startswith("non_pii_"):
                    del col[k]
    with open(dest, "w") as f:
        json.dump(data, f, indent=2)


def get_memory_usage_gb():
    # CPU RAM
    process = psutil.Process()
    cpu_gb = process.memory_info().rss / 1024 / 1024 / 1024
    # GPU RAM
    gpu_gb = 0.0
    if TORCH_AVAILABLE and torch.cuda.is_available():
        gpu_gb = torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024
    return cpu_gb, gpu_gb


def print_analysis_stats(name, n_tables, total_time, cpu_gb, gpu_gb):
    avg_time = total_time / n_tables if n_tables else 0
    print("\n" + "=" * 60)
    print(f"{name} Analysis Summary")
    print("=" * 60)
    print(f"| {'Metric':<25} | {'Value':>20} |")
    print(f"|{'-'*27}|{'-'*22}|")
    print(f"| {'Tables analyzed':<25} | {n_tables:>20} |")
    print(f"| {'Total time (s)':<25} | {total_time:>20.2f} |")
    print(f"| {'Avg time per table (s)':<25} | {avg_time:>20.2f} |")
    print(f"| {'Peak CPU RAM (GB)':<25} | {cpu_gb:>20.2f} |")
    print(f"| {'Peak GPU RAM (GB)':<25} | {gpu_gb:>20.2f} |")
    print("=" * 60 + "\n")


def run_pii_analysis(data_path: Path, model_name: str) -> Dict:
    data = load_data(str(data_path))
    classifier = SensitivityClassifier(model_name)
    start_time = time.time()
    cpu_gb_start, gpu_gb_start = get_memory_usage_gb()
    for fname, table in tqdm(data.items()):
        data[fname] = detect_and_reflect_pii(table, classifier)
    total_time = time.time() - start_time
    cpu_gb_end, gpu_gb_end = get_memory_usage_gb()
    cpu_gb = max(cpu_gb_start, cpu_gb_end)
    gpu_gb = max(gpu_gb_start, gpu_gb_end)
    print_analysis_stats("PII", len(data), total_time, cpu_gb, gpu_gb)
    return data


def run_non_pii_analysis(data_path: Path, model_name: str) -> Dict:
    data = load_data(str(data_path))
    classifier = SensitivityClassifier(model_name)
    start_time = time.time()
    cpu_gb_start, gpu_gb_start = get_memory_usage_gb()
    for fname, table in tqdm(data.items()):
        data[fname] = detect_non_pii(table, classifier, fname, method="table")
    total_time = time.time() - start_time
    cpu_gb_end, gpu_gb_end = get_memory_usage_gb()
    cpu_gb = max(cpu_gb_start, cpu_gb_end)
    gpu_gb = max(gpu_gb_start, gpu_gb_end)
    print_analysis_stats("Non-PII", len(data), total_time, cpu_gb, gpu_gb)
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run PII/non-PII analysis and evaluation."
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Path or name of the model to use."
    )
    parser.add_argument(
        "--run_pii",
        action="store_true",
        help="Run PII analysis.",
    )
    parser.add_argument(
        "--run_non_pii", action="store_true", help="Run non-PII analysis."
    )
    parser.add_argument(
        "--eval_pii", action="store_true", help="Evaluate PII detection."
    )
    parser.add_argument(
        "--eval_non_pii", action="store_true", help="Evaluate non-PII detection."
    )
    parser.add_argument(
        "--show_fps", action="store_true", help="Show false positives in evaluation."
    )
    parser.add_argument(
        "--show_fns", action="store_true", help="Show false negatives in evaluation."
    )
    parser.add_argument("--debug", action="store_true", help="Debug mode.")
    args = parser.parse_args()

    model = args.model
    run_pii = args.run_pii
    run_non_pii = args.run_non_pii
    eval_pii = args.eval_pii
    eval_non_pii = args.eval_non_pii
    show_fps = args.show_fps
    show_fns = args.show_fns

    # Run PII analysis
    pii_input = Path("modules/test/personal.json")
    if args.debug:
        pii_input = Path("modules/test/personal_dummy.json")
    if run_pii:
        pii_results = run_pii_analysis(pii_input, model)
        with open(pii_input, "w") as f:
            json.dump(pii_results, f, indent=4)
    else:
        with open(pii_input, "r") as f:
            pii_results = json.load(f)
    if eval_pii:
        pii_metrics = evaluate_pii_detection(
            pii_results, model, show_fps=show_fps, show_fns=show_fns
        )
        print("PII Detection Metrics", pii_metrics)

    # Run non-PII analysis
    non_pii_input = Path("modules/test/non_personal.json")
    if args.debug:
        non_pii_input = Path("modules/test/non_personal_dummy.json")
    if run_non_pii:
        non_pii_results = run_non_pii_analysis(non_pii_input, model)
        with open(non_pii_input, "w") as f:
            json.dump(non_pii_results, f, indent=4)
    else:
        with open(non_pii_input, "r") as f:
            non_pii_results = json.load(f)
    if eval_non_pii:
        non_pii_metrics = evaluate_non_pii_table(
            non_pii_results, model, show_fps=show_fps, show_fns=show_fns
        )
        print("Non-PII Table Metrics", non_pii_metrics)
