"""
Free Text Detection and PII Analysis using Microsoft Presidio

This module provides functionality to:
1. Detect which columns in a pandas DataFrame contain free-form text
2. Analyze those columns for PII (Personally Identifiable Information) using Microsoft Presidio
3. Provide detailed reports on detected PII types and confidence levels
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
import argparse
import sys
import os

# Microsoft Presidio imports
try:
    from presidio_analyzer import AnalyzerEngine

    PRESIDIO_AVAILABLE = True
except ImportError:
    PRESIDIO_AVAILABLE = False
    print(
        "Warning: Microsoft Presidio not available. Install with: pip install presidio-analyzer presidio-structured spacy"
    )

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import utility functions
try:
    from .utils import import_csv_xlsx
except ImportError:
    try:
        from utils import import_csv_xlsx
    except ImportError:
        # Fallback if utils module is not available
        def import_csv_xlsx(file_path: str) -> pd.DataFrame:
            """Fallback import function for CSV/XLSX files."""
            if file_path.endswith(".csv"):
                return pd.read_csv(file_path, on_bad_lines="skip")
            elif file_path.endswith(".xlsx"):
                return pd.read_excel(file_path, on_bad_lines="skip")
            else:
                raise ValueError(f"Unsupported file type: {file_path}")


class FreeTextDetector:
    """
    Detects free-form text columns in pandas DataFrames and analyzes them for PII.
    """

    def __init__(
        self,
        min_text_length: int = 10,
        min_word_count: int = 3,
        max_numeric_ratio: float = 0.3,
        max_categorical_ratio: float = 0.7,
        sample_size: int = 1000,
        language: str = "en",
    ):
        """
        Initialize the FreeTextDetector.

        Args:
            min_text_length (int): Minimum length for text to be considered free text
            min_word_count (int): Minimum number of words for free text
            max_numeric_ratio (float): Maximum ratio of numeric values to still be considered text
            max_categorical_ratio (float): Maximum ratio of unique values to be categorical
            sample_size (int): Number of rows to sample for analysis
            language (str): Language code for Presidio analysis (default: "en")
        """
        self.min_text_length = min_text_length
        self.min_word_count = min_word_count
        self.max_numeric_ratio = max_numeric_ratio
        self.max_categorical_ratio = max_categorical_ratio
        self.sample_size = sample_size
        self.language = language

        # Initialize Presidio analyzer if available
        self.analyzer = None
        if PRESIDIO_AVAILABLE:
            try:
                self.analyzer = AnalyzerEngine()
                logger.info("Microsoft Presidio analyzer initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Presidio analyzer: {e}")
                self.analyzer = None

    def detect_free_text_columns(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """
        Detect which columns contain free-form text.

        Args:
            df (pd.DataFrame): Input DataFrame

        Returns:
            Dict containing analysis results for each column
        """
        results = {}

        # Sample the DataFrame if it's too large
        if len(df) > self.sample_size:
            df_sample = df.sample(n=self.sample_size, random_state=42)
            logger.info(
                f"Sampled {self.sample_size} rows from DataFrame with {len(df)} rows"
            )
        else:
            df_sample = df

        for column in df_sample.columns:
            logger.info(f"Analyzing column: {column}")

            # Get column data
            series = df_sample[column].dropna()

            if len(series) == 0:
                results[column] = {
                    "is_free_text": False,
                    "reason": "No data (all null values)",
                    "stats": {},
                }
                continue

            # Analyze the column
            analysis = self._analyze_column(series, column)
            results[column] = analysis

        return results

    def _analyze_column(self, series: pd.Series, column_name: str) -> Dict:
        """
        Analyze a single column to determine if it contains free-form text.

        Args:
            series (pd.Series): Column data
            column_name (str): Column name

        Returns:
            Dict containing analysis results
        """
        # Convert to string for analysis
        series_str = series.astype(str)

        # Calculate basic statistics
        total_values = len(series_str)
        unique_values = series_str.nunique()
        unique_ratio = unique_values / total_values if total_values > 0 else 0

        # Calculate average length and word count
        lengths = series_str.str.len()
        word_counts = series_str.str.split().str.len()

        avg_length = lengths.mean()
        avg_word_count = word_counts.mean()
        max_length = lengths.max()

        # Check for numeric patterns
        numeric_count = series_str.str.match(r"^[\d\s\-\+\.\,]+$").sum()
        numeric_ratio = numeric_count / total_values if total_values > 0 else 0

        # Check for categorical patterns (low unique ratio)
        is_categorical = unique_ratio <= self.max_categorical_ratio

        # Check for free text characteristics
        is_free_text = (
            avg_length >= self.min_text_length
            and avg_word_count >= self.min_word_count
            and numeric_ratio <= self.max_numeric_ratio
            and not is_categorical
            and max_length > 50  # At least one value should be substantial
        )

        # Determine reason
        if avg_length < self.min_text_length:
            reason = f"Average length ({avg_length:.1f}) below threshold ({self.min_text_length})"
        elif avg_word_count < self.min_word_count:
            reason = f"Average word count ({avg_word_count:.1f}) below threshold ({self.min_word_count})"
        elif numeric_ratio > self.max_numeric_ratio:
            reason = f"Too many numeric values ({numeric_ratio:.2%} > {self.max_numeric_ratio:.2%})"
        elif is_categorical:
            reason = f"Appears categorical (unique ratio: {unique_ratio:.2%} <= {self.max_categorical_ratio:.2%})"
        elif max_length <= 50:
            reason = f"Maximum length ({max_length}) too short for free text"
        else:
            reason = "Meets free text criteria"

        return {
            "is_free_text": is_free_text,
            "reason": reason,
            "stats": {
                "total_values": total_values,
                "unique_values": unique_values,
                "unique_ratio": unique_ratio,
                "avg_length": avg_length,
                "avg_word_count": avg_word_count,
                "max_length": max_length,
                "numeric_ratio": numeric_ratio,
                "is_categorical": is_categorical,
            },
        }

    def analyze_pii_in_free_text_columns(
        self, df: pd.DataFrame, free_text_columns: Optional[List[str]] = None
    ) -> Dict[str, Dict]:
        """
        Analyze free text columns for PII using Microsoft Presidio.

        Args:
            df (pd.DataFrame): Input DataFrame
            free_text_columns (List[str], optional): List of columns to analyze.
                                                   If None, will detect automatically.

        Returns:
            Dict containing PII analysis results for each column
        """
        if not PRESIDIO_AVAILABLE or self.analyzer is None:
            logger.error(
                "Microsoft Presidio not available. Cannot perform PII analysis."
            )
            return {}

        # Detect free text columns if not provided
        if free_text_columns is None:
            detection_results = self.detect_free_text_columns(df)
            free_text_columns = [
                col
                for col, result in detection_results.items()
                if result["is_free_text"]
            ]

        if not free_text_columns:
            logger.info("No free text columns found for PII analysis")
            return {}

        results = {}

        for column in free_text_columns:
            logger.info(f"Analyzing PII in column: {column}")

            # Get column data
            series = df[column].dropna().astype(str)

            if len(series) == 0:
                results[column] = {
                    "contains_pii": False,
                    "pii_types": [],
                    "confidence": 0.0,
                    "sample_count": 0,
                    "error": "No data available",
                }
                continue

            # Analyze for PII
            pii_analysis = self._analyze_pii_in_column(series, column)
            results[column] = pii_analysis

        return results

    def _analyze_pii_in_column(self, series: pd.Series, column_name: str) -> Dict:
        """
        Analyze a single column for PII using Presidio.

        Args:
            series (pd.Series): Column data
            column_name (str): Column name

        Returns:
            Dict containing PII analysis results
        """
        try:
            # Sample data if too large (to avoid performance issues)
            max_samples = 100
            if len(series) > max_samples:
                series_sample = series.sample(n=max_samples, random_state=42)
                logger.info(
                    f"Sampled {max_samples} values from column {column_name} for PII analysis"
                )
            else:
                series_sample = series

            # Concatenate all values for analysis
            concatenated_text = " ".join(series_sample.tolist())

            if not concatenated_text.strip():
                return {
                    "contains_pii": False,
                    "pii_types": [],
                    "confidence": 0.0,
                    "sample_count": len(series_sample),
                    "error": "No text content after concatenation",
                }

            # Analyze with Presidio
            if self.analyzer is not None:
                results = self.analyzer.analyze(
                    text=concatenated_text, language=self.language
                )

                # Extract PII types and calculate confidence
                pii_types = list(set([r.entity_type for r in results]))
                confidences = [r.score for r in results]
                avg_confidence = np.mean(confidences) if confidences else 0.0

                # Determine if PII is present
                contains_pii = len(pii_types) > 0 and avg_confidence > 0.3

                return {
                    "contains_pii": contains_pii,
                    "pii_types": pii_types,
                    "confidence": avg_confidence,
                    "sample_count": len(series_sample),
                    "total_detections": len(results),
                    "detection_details": [
                        {
                            "entity_type": r.entity_type,
                            "score": r.score,
                            "start": r.start,
                            "end": r.end,
                            "text": concatenated_text[r.start : r.end],
                        }
                        for r in results
                    ],
                }
            else:
                return {
                    "contains_pii": False,
                    "pii_types": [],
                    "confidence": 0.0,
                    "sample_count": len(series_sample),
                    "error": "Presidio analyzer not available",
                }

        except Exception as e:
            logger.error(f"Error analyzing PII in column {column_name}: {str(e)}")
            return {
                "contains_pii": False,
                "pii_types": [],
                "confidence": 0.0,
                "sample_count": len(series),
                "error": str(e),
            }

    def generate_report(
        self,
        df: pd.DataFrame,
        detection_results: Optional[Dict] = None,
        pii_results: Optional[Dict] = None,
    ) -> str:
        """
        Generate a comprehensive report of the analysis.

        Args:
            df (pd.DataFrame): Input DataFrame
            detection_results (Dict, optional): Results from detect_free_text_columns
            pii_results (Dict, optional): Results from analyze_pii_in_free_text_columns

        Returns:
            String containing the formatted report
        """
        if detection_results is None:
            detection_results = self.detect_free_text_columns(df)

        if pii_results is None and PRESIDIO_AVAILABLE:
            free_text_columns = [
                col
                for col, result in detection_results.items()
                if result["is_free_text"]
            ]
            if free_text_columns:
                pii_results = self.analyze_pii_in_free_text_columns(
                    df, free_text_columns
                )

        # Generate report
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("FREE TEXT DETECTION AND PII ANALYSIS REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"DataFrame shape: {df.shape}")
        report_lines.append(f"Total columns: {len(df.columns)}")
        report_lines.append("")

        # Free text detection results
        report_lines.append("FREE TEXT DETECTION RESULTS:")
        report_lines.append("-" * 40)

        free_text_count = 0
        for column, result in detection_results.items():
            status = "✓ FREE TEXT" if result["is_free_text"] else "✗ NOT FREE TEXT"
            report_lines.append(f"{column}: {status}")
            report_lines.append(f"  Reason: {result['reason']}")

            if result["is_free_text"]:
                free_text_count += 1
                stats = result["stats"]
                report_lines.append(
                    f"  Stats: avg_length={stats['avg_length']:.1f}, "
                    f"avg_words={stats['avg_word_count']:.1f}, "
                    f"unique_ratio={stats['unique_ratio']:.2%}"
                )
            report_lines.append("")

        report_lines.append(f"Total free text columns: {free_text_count}")
        report_lines.append("")

        # PII analysis results
        if pii_results:
            report_lines.append("PII ANALYSIS RESULTS:")
            report_lines.append("-" * 40)

            pii_count = 0
            for column, result in pii_results.items():
                if result.get("error"):
                    report_lines.append(f"{column}: ERROR - {result['error']}")
                else:
                    status = (
                        "⚠ CONTAINS PII"
                        if result["contains_pii"]
                        else "✓ NO PII DETECTED"
                    )
                    report_lines.append(f"{column}: {status}")
                    report_lines.append(f"  Confidence: {result['confidence']:.3f}")
                    pii_types_str = (
                        ", ".join(result["pii_types"])
                        if result["pii_types"]
                        else "None"
                    )
                    report_lines.append(f"  PII Types: {pii_types_str}")
                    report_lines.append(f"  Samples analyzed: {result['sample_count']}")

                    if result["contains_pii"]:
                        pii_count += 1
                report_lines.append("")

            report_lines.append(f"Total columns with PII: {pii_count}")
        else:
            report_lines.append("PII ANALYSIS: Not performed (Presidio not available)")

        report_lines.append("=" * 80)

        return "\n".join(report_lines)


def detect_free_text_and_pii(
    df: pd.DataFrame,
    min_text_length: int = 10,
    min_word_count: int = 3,
    max_numeric_ratio: float = 0.3,
    max_categorical_ratio: float = 0.7,
    sample_size: int = 1000,
    language: str = "en",
) -> Tuple[Dict, Dict, str]:
    """
    Convenience function to detect free text columns and analyze for PII in one call.

    Args:
        df (pd.DataFrame): Input DataFrame
        min_text_length (int): Minimum length for text to be considered free text
        min_word_count (int): Minimum number of words for free text
        max_numeric_ratio (float): Maximum ratio of numeric values to still be considered text
        max_categorical_ratio (float): Maximum ratio of unique values to be categorical
        sample_size (int): Number of rows to sample for analysis
        language (str): Language code for Presidio analysis

    Returns:
        Tuple of (detection_results, pii_results, report)
    """
    detector = FreeTextDetector(
        min_text_length=min_text_length,
        min_word_count=min_word_count,
        max_numeric_ratio=max_numeric_ratio,
        max_categorical_ratio=max_categorical_ratio,
        sample_size=sample_size,
        language=language,
    )

    # Detect free text columns
    detection_results = detector.detect_free_text_columns(df)

    # Analyze for PII
    free_text_columns = [
        col for col, result in detection_results.items() if result["is_free_text"]
    ]

    pii_results = {}
    if free_text_columns and PRESIDIO_AVAILABLE:
        pii_results = detector.analyze_pii_in_free_text_columns(df, free_text_columns)

    # Generate report
    report = detector.generate_report(df, detection_results, pii_results)

    return detection_results, pii_results, report


# Example usage and testing
if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(
        description="Detect free text columns and analyze for PII in CSV/XLSX files"
    )
    parser.add_argument("file_path", help="Path to the CSV or XLSX file to analyze")
    parser.add_argument(
        "--min-text-length",
        type=int,
        default=10,
        help="Minimum length for text to be considered free text (default: 10)",
    )
    parser.add_argument(
        "--min-word-count",
        type=int,
        default=3,
        help="Minimum number of words for free text (default: 3)",
    )
    parser.add_argument(
        "--max-numeric-ratio",
        type=float,
        default=0.3,
        help="Maximum ratio of numeric values to still be considered text (default: 0.3)",
    )
    parser.add_argument(
        "--max-categorical-ratio",
        type=float,
        default=0.7,
        help="Maximum ratio of unique values to be categorical (default: 0.7)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=1000,
        help="Number of rows to sample for analysis (default: 1000)",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Language code for Presidio analysis (default: en)",
    )
    parser.add_argument("--output", help="Output file path for the report (optional)")

    args = parser.parse_args()

    # Check if file exists
    if not os.path.exists(args.file_path):
        print(f"Error: File '{args.file_path}' not found.")
        sys.exit(1)

    # Check file extension
    if not args.file_path.lower().endswith((".csv", ".xlsx")):
        print("Error: File must be a CSV or XLSX file.")
        sys.exit(1)

    print(f"Analyzing file: {args.file_path}")
    print("=" * 50)

    try:
        # Load the data
        df = import_csv_xlsx(args.file_path)
        print(f"Loaded DataFrame with shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print()

        # Run analysis
        detection_results, pii_results, report = detect_free_text_and_pii(
            df,
            min_text_length=args.min_text_length,
            min_word_count=args.min_word_count,
            max_numeric_ratio=args.max_numeric_ratio,
            max_categorical_ratio=args.max_categorical_ratio,
            sample_size=args.sample_size,
            language=args.language,
        )

        # Print results
        print(report)

        # Save report to file if specified
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(report)
            print(f"\nReport saved to: {args.output}")

        # Print detailed results
        print("\nDETAILED RESULTS:")
        print("Detection Results:")
        for col, result in detection_results.items():
            print(f"  {col}: {result['is_free_text']} - {result['reason']}")

        if pii_results:
            print("\nPII Results:")
            for col, result in pii_results.items():
                if result.get("error"):
                    print(f"  {col}: ERROR - {result['error']}")
                else:
                    print(
                        f"  {col}: PII={result['contains_pii']}, "
                        f"Types={result['pii_types']}, "
                        f"Confidence={result['confidence']:.3f}"
                    )

    except Exception as e:
        print(f"Error processing file: {str(e)}")
        sys.exit(1)
