# Evaluation utilities
from typing import Dict, List
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from tabulate import tabulate
import pandas as pd
import re
from typing import Dict, Any
import pycountry
from langdetect import detect, DetectorFactory, LangDetectException


def load_json_data(file_path):
    import json

    with open(file_path, "r") as file:
        data = json.load(file)
    return data


def save_json_data(data, file_path):
    import json

    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)


def table_markdown(
    table_data, pii_key=None, pii_reflection_key=None, rows=5
):
    columns_data = table_data["columns"]
    column_samples = {}

    for column_name, column_info in columns_data.items():
        if not all(x == "" for x in column_info["records"]):
            column_key = column_name

            if pii_key and column_info.get(pii_key):
                if column_info[pii_key] != "None":
                    column_key += f" - {column_info[pii_key]}"
            if pii_reflection_key and column_info.get(pii_reflection_key):
                if column_info[pii_reflection_key] != "NON_SENSITIVE":
                    column_key += f" - {column_info[pii_reflection_key]}"

            if len(column_info["records"]) > rows:
                column_samples[column_key] = column_info["records"][:rows]
            elif len(column_info["records"]) < rows:
                # Add empty strings to make it 5
                column_samples[column_key] = column_info["records"] + [""] * (
                    rows - len(column_info["records"])
                )

    df = pd.DataFrame(column_samples)
    # Delete empty rows
    df = df[df.apply(lambda row: row.astype(str).str.strip().any(), axis=1)]
    markdown_table = df.to_markdown()
    return markdown_table


def evaluate_pii_detection(
    data: Dict,
    model_name: str,
    *,
    show_fps: bool = False,
    show_fns: bool = False,
) -> Dict:
    """Evaluate PII detection results.

    Parameters
    ----------
    data: Dict
        Dataset with predictions and ground truth.
    model_name: str
        Name of the model used for predictions.
    show_fps: bool, optional
        When True, print false positive examples.
    show_fns: bool, optional
        When True, print false negative examples.
    Returns
    -------
    Dict with precision, recall, f1 and accuracy.
    """
    y_true: List[int] = []
    y_pred: List[int] = []
    fps = []
    fns = []

    for table_name, table in data.items():
        for column_name, col in table.get("columns", {}).items():
            gt = col.get("sensitivity_gt")
            pred = col.get(f"pii_reflection_{model_name}")
            if gt is None or pred is None:
                continue
            gt_bin = int(gt)
            pred_bin = 0 if pred == "NON_SENSITIVE" else 1
            y_true.append(gt_bin)
            y_pred.append(pred_bin)
            if pred_bin == 1 and gt_bin == 0:
                fps.append((table_name, column_name, gt, pred, table))
            elif pred_bin == 0 and gt_bin == 1:
                fns.append((table_name, column_name, gt, pred, table))

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)

    if show_fps and fps:
        print("\n### False Positives")
        fp_data = [(tname, cname, gt, pred) for tname, cname, gt, pred, table in fps]
        print(
            tabulate(
                fp_data,
                headers=["Table", "Column", "Ground Truth", "Prediction"],
                tablefmt="grid",
            )
        )
        # for tname, cname, gt, pred, table in fps:
        #     # print(f"\n**Table: {tname} | Column: {cname}**")
        #     print(table_markdown(table))

    if show_fns and fns:
        print("\n### False Negatives")
        fn_data = [(tname, cname, gt, pred) for tname, cname, gt, pred, table in fns]
        print(
            tabulate(
                fn_data,
                headers=["Table", "Column", "Ground Truth", "Prediction"],
                tablefmt="grid",
            )
        )
        # for tname, cname, gt, pred, table in fns:
        #     # print(f"\n**Table: {tname} | Column: {cname}**")
        #     print(table_markdown(table))

    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": acc}


def evaluate_non_pii_table(
    data: Dict,
    model_name: str,
    *,
    show_fps: bool = False,
    show_fns: bool = False,
) -> Dict:
    """Evaluate non-PII table classification with and without ISP guidance."""
    y_true: List[int] = []
    y_pred_isp: List[int] = []
    y_pred_no_isp: List[int] = []

    fps_isp = []
    fns_isp = []
    fps_no_isp = []
    fns_no_isp = []

    for table_name, table in data.items():
        meta = table.get("metadata", {})
        gt = meta.get("non_pii")
        pred_isp = meta.get(f"non_pii_{model_name}")
        pred_isp_expl = meta.get(f"non_pii_{model_name}_explanation")
        pred_no_isp = meta.get(f"non_pii_no_isp_{model_name}")
        pred_no_isp_expl = meta.get(f"non_pii_no_isp_{model_name}_explanation")
        if gt is None or pred_isp is None or pred_no_isp is None:
            continue

        gt_bin = 0 if gt == "NON_SENSITIVE" else 1
        pred_isp_bin = 0 if pred_isp == "NON_SENSITIVE" else 1
        pred_no_isp_bin = 0 if pred_no_isp == "NON_SENSITIVE" else 1

        y_true.append(gt_bin)
        y_pred_isp.append(pred_isp_bin)
        y_pred_no_isp.append(pred_no_isp_bin)

        if pred_isp_bin == 1 and gt_bin == 0:
            fps_isp.append((table_name, gt, pred_isp, pred_isp_expl, table))
        elif pred_isp_bin == 0 and gt_bin == 1:
            fns_isp.append((table_name, gt, pred_isp, pred_isp_expl, table))

        if pred_no_isp_bin == 1 and gt_bin == 0:
            fps_no_isp.append((table_name, gt, pred_no_isp, pred_no_isp_expl, table))
        elif pred_no_isp_bin == 0 and gt_bin == 1:
            fns_no_isp.append((table_name, gt, pred_no_isp, pred_no_isp_expl, table))

    precision_isp, recall_isp, f1_isp, _ = precision_recall_fscore_support(
        y_true, y_pred_isp, average="binary", zero_division=0
    )
    acc_isp = accuracy_score(y_true, y_pred_isp)
    precision_no, recall_no, f1_no, _ = precision_recall_fscore_support(
        y_true, y_pred_no_isp, average="binary", zero_division=0
    )
    acc_no = accuracy_score(y_true, y_pred_no_isp)

    if show_fps and (fps_isp or fps_no_isp):
        if fps_isp:
            print("\n### False Positives (with ISP)")
            fp_isp_data = [
                (tname, gt, pred, expl) for tname, gt, pred, expl, table in fps_isp
            ]
            print(
                tabulate(
                    fp_isp_data,
                    headers=["Table", "Ground Truth", "Prediction", "Explanation"],
                    tablefmt="grid",
                )
            )
            for tname, gt, pred, expl, table in fps_isp:
                print(f"\n**Table: {tname}**")
                print(table_markdown(table))

        if fps_no_isp:
            print("\n### False Positives (without ISP)")
            fp_no_isp_data = [
                (tname, gt, pred, expl) for tname, gt, pred, expl, table in fps_no_isp
            ]
            print(
                tabulate(
                    fp_no_isp_data,
                    headers=["Table", "Ground Truth", "Prediction", "Explanation"],
                    tablefmt="grid",
                )
            )
            for tname, gt, pred, expl, table in fps_no_isp:
                print(f"\n**Table: {tname}**")
                print(table_markdown(table))

    if show_fns and (fns_isp or fns_no_isp):
        if fns_isp:
            print("\n### False Negatives (with ISP)")
            fn_isp_data = [
                (tname, gt, pred, expl) for tname, gt, pred, expl, table in fns_isp
            ]
            print(
                tabulate(
                    fn_isp_data,
                    headers=["Table", "Ground Truth", "Prediction", "Explanation"],
                    tablefmt="grid",
                )
            )
            for tname, gt, pred, expl, table in fns_isp:
                print(f"\n**Table: {tname}**")
                print(table_markdown(table))

        if fns_no_isp:
            print("\n### False Negatives (without ISP)")
            fn_no_isp_data = [
                (tname, gt, pred, expl) for tname, gt, pred, expl, table in fns_no_isp
            ]
            print(
                tabulate(
                    fn_no_isp_data,
                    headers=["Table", "Ground Truth", "Prediction", "Explanation"],
                    tablefmt="grid",
                )
            )
            for tname, gt, pred, expl, table in fns_no_isp:
                print(f"\n**Table: {tname}**")
                print(table_markdown(table))

    return {
        "with_isp": {
            "precision": precision_isp,
            "recall": recall_isp,
            "f1": f1_isp,
            "accuracy": acc_isp,
        },
        "without_isp": {
            "precision": precision_no,
            "recall": recall_no,
            "f1": f1_no,
            "accuracy": acc_no,
        },
    }


def fetch_country(input_string: str) -> str:
    """Extract country name from a string with gpt-4o-mini"""
    from llm_model.model import Model

    prompt = f"""
    Extract country name from the following string, only return the country name:
    {input_string}
    """
    llm = Model("gpt-4o-mini")
    prediction = llm.generate(prompt)
    return prediction


# To make results deterministic (optional)
DetectorFactory.seed = 0


def detect_language(text: str) -> str:
    """
    Detects the language code of the given text using langdetect.
    Returns the ISO 639-1 language code, e.g., 'en' for English.
    If detection fails, returns 'unknown'.
    """
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"


def import_csv_xlsx(file_path: str) -> pd.DataFrame:
    """
    Imports a CSV or XLSX file into a pandas DataFrame.
    """
    if file_path.endswith(".csv"):
        return pd.read_csv(file_path, on_bad_lines="skip")
    elif file_path.endswith(".xlsx"):
        return pd.read_excel(file_path, on_bad_lines="skip")
    else:
        raise ValueError(f"Unsupported file type: {file_path}")


def evaluate_pii_detection_multiclass(
    data: Dict,
    ground_truth_key: str,
    prediction_key: str,
    *,
    show_misclassifications: bool = False,
    show_confusion_matrix: bool = False,
    show_report_table: bool = True,
) -> Dict:
    """Evaluate PII detection results with multiclass classification.

    Parameters
    ----------
    data: Dict
        Dataset with predictions and ground truth.
    ground_truth_key: str
        Key name for ground truth values in the data.
    prediction_key: str
        Key name for prediction values in the data.
    show_misclassifications: bool, optional
        When True, print misclassification examples.
    show_confusion_matrix: bool, optional
        When True, print confusion matrix.
    show_report_table: bool, optional
        When True, print classification report as a neat table.
    
    Returns
    -------
    Dict with precision, recall, f1 and accuracy for each class and overall.
    """
    from sklearn.metrics import classification_report, confusion_matrix
    import numpy as np
    
    y_true: List[str] = []
    y_pred: List[str] = []
    misclassifications = []
    
    for table_name, table in data.items():
        for column_name, col in table.get("columns", {}).items():
            gt = col.get(ground_truth_key)
            pred = col.get(prediction_key)
            if gt is None or pred is None:
                continue
            
            y_true.append(str(gt))
            y_pred.append(str(pred))
            
            if pred != gt:
                # Get sample data from the column
                sample_data = ""
                if "columns" in table and column_name in table["columns"]:
                    records = table["columns"][column_name].get("records", [])
                    # Show first few non-empty records
                    sample_records = [str(r) for r in records if r is not None and str(r).strip()][:3]
                    sample_data = ", ".join(sample_records)
                    if len(sample_records) == 3 and len(records) > 3:
                        sample_data += "..."
                
                misclassifications.append((table_name, column_name, gt, pred, sample_data))
    
    if not y_true:
        return {"error": "No valid ground truth and prediction pairs found"}
    
    # Get all unique labels
    all_labels = sorted(list(set(y_true + y_pred)))
    
    # Get classification report for all classes
    report_all = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    
    # Get classification report excluding "None" for macro/weighted averages
    pii_labels = [label for label in all_labels if label.lower() not in ['none', 'non_sensitive']]
    
    if pii_labels:
        report_pii = classification_report(
            y_true, y_pred, 
            labels=pii_labels, 
            output_dict=True, 
            zero_division=0
        )
        macro_avg_pii = report_pii["macro avg"]
        weighted_avg_pii = report_pii["weighted avg"]
    else:
        macro_avg_pii = {"precision": 0, "recall": 0, "f1-score": 0, "support": 0}
        weighted_avg_pii = {"precision": 0, "recall": 0, "f1-score": 0, "support": 0}
    
    # Display classification report as table
    if show_report_table:
        print("\n### Classification Report")
        table_data = []
        headers = ["Class", "Precision", "Recall", "F1-Score", "Support"]
        
        for class_name, metrics in report_all.items():
            if class_name in ["accuracy", "macro avg", "weighted avg"]:
                continue
            if isinstance(metrics, dict):
                table_data.append([
                    class_name,
                    f"{metrics['precision']:.3f}",
                    f"{metrics['recall']:.3f}",
                    f"{metrics['f1-score']:.3f}",
                    f"{metrics['support']:.0f}"
                ])
        
        # Add summary rows (excluding "None" from macro/weighted averages)
        table_data.append(["", "", "", "", ""])  # Empty row
        table_data.append([
            "macro avg (PII only)",
            f"{macro_avg_pii['precision']:.3f}",
            f"{macro_avg_pii['recall']:.3f}",
            f"{macro_avg_pii['f1-score']:.3f}",
            f"{macro_avg_pii['support']:.0f}"
        ])
        table_data.append([
            "weighted avg (PII only)",
            f"{weighted_avg_pii['precision']:.3f}",
            f"{weighted_avg_pii['recall']:.3f}",
            f"{weighted_avg_pii['f1-score']:.3f}",
            f"{weighted_avg_pii['support']:.0f}"
        ])
        
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        print(f"\nOverall Accuracy: {report_all['accuracy']:.3f}")
        print(f"Total Samples: {len(y_true)}")
        print(f"Total Misclassifications: {len(misclassifications)}")
        
        if pii_labels:
            print(f"PII Classes (excluded from macro avg): {', '.join([label for label in all_labels if label.lower() in ['none', 'non_sensitive']])}")
            print(f"PII Classes (included in macro avg): {', '.join(pii_labels)}")
    
    # Get confusion matrix
    if show_confusion_matrix:
        cm = confusion_matrix(y_true, y_pred)
        labels = sorted(list(set(y_true + y_pred)))
        print("\n### Confusion Matrix")
        
        # Create confusion matrix table
        cm_table = []
        headers = ["True \\ Predicted"] + labels
        
        for i, true_label in enumerate(labels):
            row = [true_label] + [str(cm[i][j]) for j in range(len(labels))]
            cm_table.append(row)
        
        print(tabulate(cm_table, headers=headers, tablefmt="grid"))
    
    # Show misclassifications
    if show_misclassifications and misclassifications:
        print(f"\n### Misclassifications ({len(misclassifications)} cases)")
        
        fp_data = []
        headers = ["Table", "Column", "True Class", "Predicted Class", "Sample Values"]
        
        for table_name, column_name, gt, pred, sample_data in misclassifications:
            fp_data.append([table_name, column_name, gt, pred, sample_data])
        
        print(tabulate(fp_data, headers=headers, tablefmt="grid"))
    
    return {
        "classification_report": report_all,
        "overall_accuracy": report_all["accuracy"],
        "macro_avg_all": report_all["macro avg"],
        "weighted_avg_all": report_all["weighted avg"],
        "macro_avg_pii_only": macro_avg_pii,
        "weighted_avg_pii_only": weighted_avg_pii,
        "per_class_metrics": {k: v for k, v in report_all.items() 
                           if k not in ["accuracy", "macro avg", "weighted avg"]},
        "total_samples": len(y_true),
        "misclassifications": len(misclassifications),
        "pii_classes": pii_labels,
        "excluded_classes": [label for label in all_labels if label.lower() in ['none', 'non_sensitive']]
    }


def evaluate_pii_reflection_binary(
    data: Dict,
    ground_truth_key: str,
    prediction_key: str,
    *,
    show_fps: bool = False,
    show_fns: bool = False,
    show_report_table: bool = True,
) -> Dict:
    """Evaluate PII reflection results with binary classification.
    
    Groups NON_SENSITIVE vs (MEDIUM_SENSITIVE + HIGH_SENSITIVE).

    Parameters
    ----------
    data: Dict
        Dataset with predictions and ground truth.
    ground_truth_key: str
        Key name for ground truth values in the data.
    prediction_key: str
        Key name for prediction values in the data.
    show_fps: bool, optional
        When True, print false positive examples.
    show_fns: bool, optional
        When True, print false negative examples.
    show_report_table: bool, optional
        When True, print classification report as a neat table.
    
    Returns
    -------
    Dict with precision, recall, f1 and accuracy for binary classification.
    """
    y_true: List[int] = []
    y_pred: List[int] = []
    fps = []
    fns = []
    
    def map_to_binary(value: str) -> int:
        """Map sensitivity values to binary: 0 for NON_SENSITIVE, 1 for others."""
        if value == "NON_SENSITIVE":
            return 0
        elif value in ["MEDIUM_SENSITIVE", "HIGH_SENSITIVE"]:
            return 1
        else:
            # Handle other possible values by treating them as sensitive
            return 1
    
    for table_name, table in data.items():
        for column_name, col in table.get("columns", {}).items():
            gt = col.get(ground_truth_key)
            pred = col.get(prediction_key)
            if gt is None or pred is None:
                continue
            
            gt_bin = map_to_binary(str(gt))
            pred_bin = map_to_binary(str(pred))
            
            y_true.append(gt_bin)
            y_pred.append(pred_bin)
            
            if pred_bin == 1 and gt_bin == 0:
                # Get sample data from the column
                sample_data = ""
                if "columns" in table and column_name in table["columns"]:
                    records = table["columns"][column_name].get("records", [])
                    # Show first few non-empty records
                    sample_records = [str(r) for r in records if r is not None and str(r).strip()][:3]
                    sample_data = ", ".join(sample_records)
                    if len(sample_records) == 3 and len(records) > 3:
                        sample_data += "..."
                
                fps.append((table_name, column_name, gt, pred, sample_data))
            elif pred_bin == 0 and gt_bin == 1:
                # Get sample data from the column
                sample_data = ""
                if "columns" in table and column_name in table["columns"]:
                    records = table["columns"][column_name].get("records", [])
                    # Show first few non-empty records
                    sample_records = [str(r) for r in records if r is not None and str(r).strip()][:3]
                    sample_data = ", ".join(sample_records)
                    if len(sample_records) == 3 and len(records) > 3:
                        sample_data += "..."
                
                fns.append((table_name, column_name, gt, pred, sample_data))
    
    if not y_true:
        return {"error": "No valid ground truth and prediction pairs found"}
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)
    
    # Calculate per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    # Display results as table
    if show_report_table:
        print("\n### Binary Classification Report")
        table_data = []
        headers = ["Class", "Precision", "Recall", "F1-Score", "Support"]
        
        # Non-sensitive class
        table_data.append([
            "NON_SENSITIVE",
            f"{precision_per_class[0]:.3f}" if len(precision_per_class) > 0 else "0.000",
            f"{recall_per_class[0]:.3f}" if len(recall_per_class) > 0 else "0.000",
            f"{f1_per_class[0]:.3f}" if len(f1_per_class) > 0 else "0.000",
            f"{support_per_class[0]:.0f}" if len(support_per_class) > 0 else "0"
        ])
        
        # Sensitive class
        table_data.append([
            "SENSITIVE",
            f"{precision_per_class[1]:.3f}" if len(precision_per_class) > 1 else "0.000",
            f"{recall_per_class[1]:.3f}" if len(recall_per_class) > 1 else "0.000",
            f"{f1_per_class[1]:.3f}" if len(f1_per_class) > 1 else "0.000",
            f"{support_per_class[1]:.0f}" if len(support_per_class) > 1 else "0"
        ])
        
        # Add summary row
        table_data.append(["", "", "", "", ""])  # Empty row
        table_data.append([
            "Overall",
            f"{precision:.3f}",
            f"{recall:.3f}",
            f"{f1:.3f}",
            f"{len(y_true)}"
        ])
        
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        # print(f"\nOverall Accuracy: {acc:.3f}")
        
        # Class distribution table
        non_sensitive_count = sum(1 for x in y_true if x == 0)
        sensitive_count = sum(1 for x in y_true if x == 1)
        
        print("\n### Class Distribution")
        dist_table = [
            ["NON_SENSITIVE", non_sensitive_count, f"{non_sensitive_count/len(y_true)*100:.1f}%"],
            ["SENSITIVE", sensitive_count, f"{sensitive_count/len(y_true)*100:.1f}%"],
            ["Total", len(y_true), "100.0%"]
        ]
        print(tabulate(dist_table, headers=["Class", "Count", "Percentage"], tablefmt="grid"))
    
    if show_fps and fps:
        print(f"\n### False Positives ({len(fps)} cases)")
        print("NON_SENSITIVE data incorrectly predicted as SENSITIVE")
        fp_data = [(tname, cname, gt, pred, sample_data) for tname, cname, gt, pred, sample_data in fps]
        print(
            tabulate(
                fp_data,
                headers=["Table", "Column", "Ground Truth", "Prediction", "Sample Values"],
                tablefmt="grid",
            )
        )
    
    if show_fns and fns:
        print(f"\n### False Negatives ({len(fns)} cases)")
        print("SENSITIVE data incorrectly predicted as NON_SENSITIVE")
        fn_data = [(tname, cname, gt, pred, sample_data) for tname, cname, gt, pred, sample_data in fns]
        print(
            tabulate(
                fn_data,
                headers=["Table", "Column", "Ground Truth", "Prediction", "Sample Values"],
                tablefmt="grid",
            )
        )
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": acc,
        "per_class": {
            "non_sensitive": {
                "precision": precision_per_class[0] if len(precision_per_class) > 0 else 0,
                "recall": recall_per_class[0] if len(recall_per_class) > 0 else 0,
                "f1": f1_per_class[0] if len(f1_per_class) > 0 else 0,
                "support": support_per_class[0] if len(support_per_class) > 0 else 0,
            },
            "sensitive": {
                "precision": precision_per_class[1] if len(precision_per_class) > 1 else 0,
                "recall": recall_per_class[1] if len(recall_per_class) > 1 else 0,
                "f1": f1_per_class[1] if len(f1_per_class) > 1 else 0,
                "support": support_per_class[1] if len(support_per_class) > 1 else 0,
            }
        },
        "total_samples": len(y_true),
        "class_distribution": {
            "non_sensitive": sum(1 for x in y_true if x == 0),
            "sensitive": sum(1 for x in y_true if x == 1)
        },
        "false_positives": len(fps),
        "false_negatives": len(fns)
    }


def evaluate_pii_as_reflection(
    data: Dict,
    ground_truth_key: str,
    prediction_key: str,
) -> Dict:
    """Evaluate PII detection results where we consider each PII as sensitive.

    """
    y_true: List[str] = []
    y_pred: List[str] = []
    
    for table_name, table in data.items():
        for column_name, col in table.get("columns", {}).items():
            gt = col.get(ground_truth_key)
            pred = col.get(prediction_key)
            if gt is None or pred is None:
                continue

            if gt == "NON_SENSITIVE":
                y_true.append(0)
            elif gt == "MEDIUM_SENSITIVE" or gt == "HIGH_SENSITIVE":
                y_true.append(1)
            
            if pred == "None":
                y_pred.append(0)
            else:
                y_pred.append(1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", labels=[0, 1], zero_division=0
    )
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "total_samples": len(y_true),
        "class_distribution": {
            "non_sensitive": sum(1 for x in y_true if x == 0),
            "sensitive": sum(1 for x in y_true if x == 1)
        }
    }


# Example usage
if __name__ == "__main__":
    print(standardize_country("Somalia_survey_results_2023.xlsx"))
    print(standardize_country("us"))
    print(standardize_country("The Netherlands"))
    print(standardize_country("SOM"))
    print(standardize_country("unrelated_filename"))
