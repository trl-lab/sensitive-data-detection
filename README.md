# Sensitive Data Detection

A machine learning-powered system for detecting and classifying sensitive information in humanitarian datasets, specifically designed to protect personal identifiable information (PII) and sensitive operational data according to Information Sharing Protocols (ISPs).

## Overview

This repository provides a comprehensive framework for:
- **PII Detection**: Identifying personally identifiable information in datasets
- **Sensitivity Classification**: Categorizing data sensitivity levels (LOW/NON_SENSITIVE, MODERATE_SENSITIVE, HIGH_SENSITIVE)
- **ISP Compliance**: Ensuring data sharing practices align with humanitarian information sharing protocols
- **Multi-model Support**: Working with various LLMs including OpenAI GPT models, Gemma, Qwen, and custom fine-tuned models

## Key Features

- **Detect-then-Reflect Pipeline**: Two-stage detection process with reflection for improved accuracy
- **Multiple Model Support**: Compatible with OpenAI GPT-4o-mini, Gemma 2/3, Qwen3, and Aya Expanse models
- **ISP Integration**: Automatic selection of appropriate Information Sharing Protocols based on data origin
- **Memory Monitoring**: Built-in GPU/RAM usage tracking during inference
- **Batch Processing**: Efficient processing of large datasets
- **Comprehensive Evaluation**: Built-in metrics and evaluation notebooks

## Project Structure

```
sensitive-data-detection/
├── CONFIG.py                 # Configuration settings (model selection)
├── scripts/                  # Main execution scripts
│   ├── 00_finetuning_LM_PII.py      # Fine-tune models for PII detection
│   ├── 01_inference_pii.py          # Run PII detection inference
│   ├── 02_inference_non_pii.py      # Run sensitivity classification
│   └── evaluation_*.ipynb           # Evaluation notebooks
├── utilities/                # Core utility modules
│   ├── data_processor.py            # Data loading and processing
│   ├── detect_reflect.py            # Detection and reflection logic
│   ├── sensitivityClassifier.py     # Sensitivity classification
│   ├── prompt_register.py           # Prompt templates and PII entities
│   ├── utils.py                     # Helper functions and evaluation metrics
│   ├── train_unsloth.py             # Fine-tuning utilities
│   └── isp_example.json            # Information Sharing Protocol rules
├── data/                     # Training and test datasets
├── llm_model/               # Model interface and supported models
├── results/                 # Output files and logs
└── models/                  # Fine-tuned model storage
```

This README provides a comprehensive overview of the sensitive data detection repository, covering its purpose, structure, usage instructions, and key features. It's written to be accessible to both technical users who want to implement the system and stakeholders who need to understand its capabilities and compliance features.

## Getting Started

### 1. Configuration

Edit `CONFIG.py` to set your preferred models:
```python
NON_PII_MODEL = 'gpt-4o-mini'    # For sensitivity classification
PII_MODEL = 'gpt-4o-mini'        # For PII detection
```

### 2. Data Preparation

Place your datasets in the `data/` directory. Supported formats:
- CSV files
- Excel files (.xlsx)
- JSON files (structured format)

The system can process both individual files and batch datasets.

### 3. Running Detection

**For PII Detection:**
```bash
python scripts/01_inference_pii.py --input_path data/your_dataset.csv --output_path results/pii_results.json
```

**For Sensitivity Classification:**
```bash
python scripts/02_inference_non_pii.py --input_path data/your_dataset.csv --output_path results/sensitivity_results.json
```

### 4. Model Fine-tuning (Optional)

Fine-tune models on your specific data:
```bash
python scripts/00_finetuning_LM_PII.py --csv_path data/train_data_personal.csv --model_name unsloth/gemma-2-9b-it --epochs 2
```

## Sensitivity Levels

The system classifies data into three sensitivity levels:

- **LOW/NON_SENSITIVE**: Publicly shareable data (HNO/HRP data, CODs, administrative statistics)
- **MODERATE_SENSITIVE**: Limited risk data requiring contextual approval (aggregated assessments, disaggregated data)
- **HIGH_SENSITIVE**: Data requiring strict protection (individual records, detailed locations, security incidents)

## PII Entities Detected

The system identifies 30+ types of PII including:
- Personal identifiers (names, emails, phone numbers)
- Geographic information (addresses, coordinates)
- Demographic data (age, gender, ethnicity)
- Financial information (credit cards, IBAN codes)
- Medical and sensitive attributes

## Information Sharing Protocols (ISPs)

The system automatically applies appropriate ISPs based on:
- Data origin country/region
- Humanitarian context
- Local data protection regulations
- Organizational policies

## Evaluation and Metrics

Evaluate model performance using the provided notebooks:
- `evaluation_personal.ipynb` - PII detection metrics
- `evaluation_non_personal.ipynb` - Sensitivity classification metrics

Metrics include precision, recall, F1-score, and confusion matrices with detailed false positive/negative analysis.

## Supported Models

### OpenAI Models
- gpt-4o-mini
- gpt-4o

### Open Source Models
- unsloth/gemma-2-9b-it
- unsloth/gemma-3-12b-it
- unsloth/qwen3-8b
- unsloth/qwen3-14b
- CohereLabs/aya-expanse-8b

### Fine-tuned Models
- Custom fine-tuned versions of the above models

## Output Format

Results are saved in JSON format containing:
- Table metadata and country information
- Column-wise analysis with detected PII entities
- Sensitivity classifications with explanations
- Processing statistics and memory usage

## Requirements

- Python 3.8+
- PyTorch (for local model inference)
- Transformers library
- OpenAI API access (for GPT models)
- Additional dependencies in requirements files

## Contributing

This project is developed by the TRL Lab. For questions or contributions, please refer to the project documentation or contact the development team.

## Use Cases

- **Humanitarian Organizations**: Protect beneficiary data while enabling necessary data sharing
- **Data Scientists**: Pre-process datasets to identify and handle sensitive information
- **Compliance Teams**: Ensure data sharing practices meet regulatory and organizational standards
- **Researchers**: Analyze sensitivity patterns in humanitarian datasets