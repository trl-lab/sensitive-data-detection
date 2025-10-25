import argparse
import sys
import os
import time
import psutil
import torch
import logging
import json
from datetime import datetime
from typing import Dict, Any
from tqdm import tqdm
from CONFIG import *

from utilities.data_processor import DataLoader
from utilities.detect_reflect import detect_non_pii
from llm_model.model import Model
from utilities.sensitivityClassifier import SensitivityClassifier
from utilities.utils import save_json_data


def get_memory_usage() -> Dict[str, Any]:
    """Get current memory usage statistics."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    stats = {
        "ram_used_gb": round(memory_info.rss / (1024 * 1024 * 1024), 3),
        "ram_percent": process.memory_percent(),
    }
    
    if torch.cuda.is_available():
        stats.update({
            "gpu_memory_allocated_gb": round(torch.cuda.memory_allocated() / (1024 * 1024 * 1024), 3),
            "gpu_memory_reserved_gb": round(torch.cuda.memory_reserved() / (1024 * 1024 * 1024), 3),
            "gpu_max_memory_allocated_gb": round(torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024), 3)
        })
    
    return stats


def update_peak_memory(current_stats: Dict[str, Any], peak_stats: Dict[str, Any]) -> None:
    """Update peak memory usage statistics."""
    peak_stats["peak_ram_used_gb"] = max(peak_stats.get("peak_ram_used_gb", 0), current_stats["ram_used_gb"])
    peak_stats["peak_ram_percent"] = max(peak_stats.get("peak_ram_percent", 0), current_stats["ram_percent"])
    
    if torch.cuda.is_available():
        peak_stats["peak_gpu_memory_allocated_gb"] = max(
            peak_stats.get("peak_gpu_memory_allocated_gb", 0), 
            current_stats["gpu_memory_allocated_gb"]
        )
        peak_stats["peak_gpu_memory_reserved_gb"] = max(
            peak_stats.get("peak_gpu_memory_reserved_gb", 0), 
            current_stats["gpu_memory_reserved_gb"]
        )


def setup_logging(output_dir: str) -> str:
    """Setup logging configuration."""
    # Create logs directory
    logs_dir = os.path.join(output_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    # Setup logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = os.path.join(logs_dir, f"non_pii_inference_log_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stdout)  # Also log to console
        ]
    )
    
    return log_filename


def main():
    start_time = time.time()
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Detect and reflect non-PII in CSV files')
    parser.add_argument(
        '--input', '-i',
        type=str,
        default=f"{PROJECT_ROOT}/data/non_personal.json",
        help='Path to the folder containing CSV files (default: afghanistan_access_constraints.json)'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        default=NON_PII_MODEL,
        help='Model name to use for sensitivity classification (default: gpt-4o-mini)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=f"{PROJECT_ROOT}/results/non_personal.json",
        help='Path to save the results JSON file (default: results/afghanistan_access_constraints.json)'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Setup logging
    log_filename = setup_logging(output_dir)
    
    # Initialize peak memory tracking
    peak_memory_stats = {}
    
    # Validate input path
    if not os.path.exists(args.input):
        logging.error(f"Input folder '{args.input}' does not exist.")
        sys.exit(1)
    
    logging.info("Non-PII Detection Inference Configuration:")
    logging.info(f"Input: {args.input}")
    logging.info(f"Model: {args.model}")
    logging.info(f"Output: {args.output}")
    logging.info(f"Log file: {log_filename}")
    
    # Log initial memory state
    initial_memory = get_memory_usage()
    logging.info("Initial system state:")
    logging.info(f"Memory usage: {json.dumps(initial_memory, indent=2)}")
    update_peak_memory(initial_memory, peak_memory_stats)
    
    # Load CSV data
    data_load_start = time.time()
    logging.info(f"Loading CSV files from: {args.input}")
    csv_data = DataLoader().load_single_file(args.input)
    data_load_time = time.time() - data_load_start
    logging.info(f"Loaded {len(csv_data)} CSV files in {data_load_time:.2f} seconds")
    
    # Update memory after data loading
    post_load_memory = get_memory_usage()
    update_peak_memory(post_load_memory, peak_memory_stats)
    
    # Initialize the sensitivity classifier
    model_init_start = time.time()
    logging.info(f"Initializing model: {args.model}")
    generator = SensitivityClassifier(args.model)
    model_init_time = time.time() - model_init_start
    logging.info(f"Model initialized in {model_init_time:.2f} seconds")
    
    # Update memory after model loading
    post_model_memory = get_memory_usage()
    update_peak_memory(post_model_memory, peak_memory_stats)
    logging.info(f"Memory usage after model loading: {json.dumps(post_model_memory, indent=2)}")
    
    # Process each CSV file
    processing_start = time.time()
    table_processing_times = []  # Track individual table processing times
    
    for key, value in tqdm(csv_data.items(), desc="Processing tables"):
        table_start_time = time.time()
        logging.info(f"Processing: {key}")
        
        value = detect_non_pii(value, generator, key)
        csv_data[key] = value
        
        # Update peak memory during processing
        current_memory = get_memory_usage()
        update_peak_memory(current_memory, peak_memory_stats)
        
        table_processing_time = time.time() - table_start_time
        table_processing_times.append(table_processing_time)
        logging.info(f"Table '{key}' processed in {table_processing_time:.2f} seconds")
        
        # Save results after each table
        save_start = time.time()
        save_json_data(csv_data, args.output)
        save_time = time.time() - save_start
        logging.info(f"Results saved in {save_time:.2f} seconds")
        
    
    total_processing_time = time.time() - processing_start
    
    # Calculate average time per table
    average_time_per_table = sum(table_processing_times) / len(table_processing_times) if table_processing_times else 0
    
    # Final memory check
    final_memory = get_memory_usage()
    update_peak_memory(final_memory, peak_memory_stats)
    
    # Calculate total runtime
    total_runtime = time.time() - start_time
    
    # Log final statistics
    logging.info("\n" + "="*50)
    logging.info("FINAL STATISTICS")
    logging.info("="*50)
    logging.info(f"Total tables processed: {len(table_processing_times)}")
    logging.info(f"Total processing time: {total_processing_time:.2f} seconds")
    logging.info(f"Average time per table: {average_time_per_table:.2f} seconds")
    logging.info(f"Data loading time: {data_load_time:.2f} seconds")
    logging.info(f"Model initialization time: {model_init_time:.2f} seconds")
    logging.info(f"Total runtime: {total_runtime:.2f} seconds")
    
    logging.info("\nFinal memory usage:")
    logging.info(f"{json.dumps(final_memory, indent=2)}")
    
    logging.info("\nPeak memory usage:")
    logging.info(f"{json.dumps(peak_memory_stats, indent=2)}")
    
    # Log performance summary
    logging.info("\n" + "="*50)
    logging.info("PERFORMANCE SUMMARY")
    logging.info("="*50)
    logging.info(f"Model: {args.model}")
    logging.info(f"Tables processed: {len(table_processing_times)}")
    logging.info(f"Average time per table: {average_time_per_table:.2f} seconds")
    logging.info(f"Peak RAM usage: {peak_memory_stats.get('peak_ram_used_gb', 0):.3f} GB ({peak_memory_stats.get('peak_ram_percent', 0):.1f}%)")
    
    if torch.cuda.is_available():
        logging.info(f"Peak GPU memory allocated: {peak_memory_stats.get('peak_gpu_memory_allocated_gb', 0):.3f} GB")
        logging.info(f"Peak GPU memory reserved: {peak_memory_stats.get('peak_gpu_memory_reserved_gb', 0):.3f} GB")
        logging.info(f"GPU device: {torch.cuda.get_device_name(0)}")
    else:
        logging.info("GPU not available")
    
    logging.info(f"Processing complete. Results saved to: {args.output}")
    print(f"Processing complete. Results saved to: {args.output}")
    print(f"Log file: {log_filename}")
    
    return csv_data


if __name__ == "__main__":
    csv_data = main()
