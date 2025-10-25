"""
Data loader for processing CSV, XLSX, and JSON files with country metadata extraction.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Union, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime
from utilities.utils import fetch_country

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Loads and processes datasets from various sources with country metadata extraction.
    """

    def __init__(self, max_records_per_column: int = 20):
        """
        Initialize the data loader.

        Args:
            max_records_per_column (int): Maximum number of records to include per column
        """
        self.max_records_per_column = max_records_per_column
        self.supported_extensions = {".csv", ".xlsx", ".xls", ".json"}

    def load_single_file(self, filepath: Union[str, Path]) -> Dict[str, Any]:
        """
        Load a single file (CSV, XLSX, or JSON).

        Args:
            filepath: Path to the file

        Returns:
            Dict with processed data including metadata
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        if filepath.suffix.lower() not in self.supported_extensions:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")

        logger.info(f"Loading file: {filepath}")

        try:
            if filepath.suffix.lower() == ".json":
                return self._load_json(filepath)
            elif filepath.suffix.lower() == ".csv":
                return self._load_csv(filepath)
            elif filepath.suffix.lower() in [".xlsx", ".xls"]:
                return self._load_excel(filepath)
        except Exception as e:
            logger.error(f"Error loading file {filepath}: {str(e)}")
            raise

    def load_folder(self, folder_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load all supported files from a folder.

        Args:
            folder_path: Path to the folder

        Returns:
            Dict with all processed files
        """
        folder_path = Path(folder_path)

        if not folder_path.exists() or not folder_path.is_dir():
            raise ValueError(f"Folder not found or not a directory: {folder_path}")

        logger.info(f"Loading files from folder: {folder_path}")

        result = {}
        processed_files = 0
        skipped_files = 0

        for file_path in folder_path.iterdir():
            if (
                file_path.is_file()
                and file_path.suffix.lower() in self.supported_extensions
            ):
                try:
                    file_data = self.load_single_file(file_path)
                    result.update(file_data)
                    processed_files += 1
                    logger.info(f"Successfully processed: {file_path.name}")
                except Exception as e:
                    logger.warning(f"Skipping file {file_path.name}: {str(e)}")
                    skipped_files += 1

        logger.info(
            f"Folder processing complete. Processed: {processed_files}, Skipped: {skipped_files}"
        )
        return result

    def load_data(self, input_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load data from file or folder.

        Args:
            input_path: Path to file or folder

        Returns:
            Dict with processed data
        """
        input_path = Path(input_path)

        if input_path.is_file():
            return self.load_single_file(input_path)
        elif input_path.is_dir():
            return self.load_folder(input_path)
        else:
            raise ValueError(f"Input path not found: {input_path}")

    def _load_json(self, filepath: Path) -> Dict[str, Any]:
        """Load JSON file and enhance with country metadata."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        # If it's already in the expected format, enhance metadata
        if isinstance(data, dict):
            enhanced_data = {}
            for table_name, table_data in data.items():
                enhanced_data[table_name] = self._enhance_table_metadata(
                    table_data, table_name, filepath
                )
            return enhanced_data
        else:
            # If it's not in expected format, treat as single table
            table_name = filepath.stem
            return {
                table_name: self._create_table_structure(data, table_name, filepath)
            }

    def _load_csv(self, filepath: Path) -> Dict[str, Any]:
        """Load CSV file and create table structure."""
        try:
            # Try different encodings
            encodings = ["utf-8", "latin1", "cp1252", "iso-8859-1"]
            df = None

            for encoding in encodings:
                try:
                    df = pd.read_csv(filepath, encoding=encoding, low_memory=False)
                    break
                except UnicodeDecodeError:
                    continue

            if df is None:
                raise ValueError("Could not read CSV with any supported encoding")

            table_name = filepath.stem
            table_data = self._dataframe_to_table_structure(df)

            return {
                table_name: self._enhance_table_metadata(
                    table_data, table_name, filepath
                )
            }

        except Exception as e:
            logger.error(f"Error loading CSV {filepath}: {str(e)}")
            raise

    def _load_excel(self, filepath: Path) -> Dict[str, Any]:
        """Load Excel file and create table structure."""
        try:
            # Read all sheets
            excel_file = pd.ExcelFile(filepath)
            result = {}

            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(filepath, sheet_name=sheet_name)

                # Create table name combining file and sheet name
                if len(excel_file.sheet_names) == 1:
                    table_name = filepath.stem
                else:
                    table_name = f"{filepath.stem}_{sheet_name}"

                table_data = self._dataframe_to_table_structure(df)
                result[table_name] = self._enhance_table_metadata(
                    table_data, table_name, filepath
                )

            return result

        except Exception as e:
            logger.error(f"Error loading Excel {filepath}: {str(e)}")
            raise

    def _dataframe_to_table_structure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Convert DataFrame to expected table structure."""
        columns = {}

        for col_name in df.columns:
            column_data = df[col_name].tolist()
            # Take only the first max_records_per_column records
            records = self._convert_to_serializable(
                column_data[: self.max_records_per_column]
            )

            columns[col_name] = {
                "records": records,
                "dtype": str(df[col_name].dtype),
                "null_count": int(df[col_name].isnull().sum()),
                "total_count": len(df),
            }

        return {"metadata": {}, "columns": columns}

    def _enhance_table_metadata(
        self, table_data: Dict[str, Any], table_name: str, filepath: Path
    ) -> Dict[str, Any]:
        """Enhance table data with metadata including country information."""
        metadata_existed = "metadata" in table_data and bool(table_data["metadata"])
        
        if "metadata" not in table_data:
            table_data["metadata"] = {}
        else:
            return table_data

        # Extract country information from filename
        country_info = fetch_country(filepath.name)

        # Add enhanced metadata
        table_data["metadata"].update(
            {
                "country": country_info,
                "filename": filepath.name,
                "filepath": str(filepath),
                "table_name": table_name,
                "file_extension": filepath.suffix.lower(),
                "file_size_bytes": filepath.stat().st_size,
                "processing_timestamp": datetime.now().isoformat(),
                "total_columns": len(table_data.get("columns", {})),
                "max_records_per_column": self.max_records_per_column,
                "metadata_pre_existing": metadata_existed,
            }
        )

        # Add column statistics
        if "columns" in table_data:
            table_data["metadata"]["column_names"] = list(table_data["columns"].keys())
            table_data["metadata"]["column_types"] = {
                col: col_data.get("dtype", "unknown")
                for col, col_data in table_data["columns"].items()
            }

        return table_data

    def _create_table_structure(
        self, data: Any, table_name: str, filepath: Path
    ) -> Dict[str, Any]:
        """Create table structure from generic data."""
        if isinstance(data, dict):
            # Assume it's column-based data
            columns = {}
            for col_name, col_data in data.items():
                if isinstance(col_data, list):
                    records = self._convert_to_serializable(
                        col_data[: self.max_records_per_column]
                    )
                    columns[col_name] = {"records": records}
                else:
                    columns[col_name] = {
                        "records": [self._convert_to_serializable([col_data])[0]]
                    }
        elif isinstance(data, list):
            # Assume it's record-based data
            if data and isinstance(data[0], dict):
                # List of dictionaries
                columns = {}
                for record in data[: self.max_records_per_column]:
                    for key, value in record.items():
                        if key not in columns:
                            columns[key] = {"records": []}
                        columns[key]["records"].append(
                            self._convert_to_serializable([value])[0]
                        )
            else:
                # List of values
                columns = {
                    "data": {
                        "records": self._convert_to_serializable(
                            data[: self.max_records_per_column]
                        )
                    }
                }
        else:
            # Single value
            columns = {"data": {"records": [self._convert_to_serializable([data])[0]]}}

        table_data = {"metadata": {}, "columns": columns}

        return self._enhance_table_metadata(table_data, table_name, filepath)

    def _convert_to_serializable(self, values: List[Any]) -> List[Any]:
        """Convert pandas/numpy data types to JSON serializable types."""
        result = []
        for val in values:
            if isinstance(val, (np.integer, np.floating)):
                result.append(float(val) if isinstance(val, np.floating) else int(val))
            elif isinstance(val, np.bool_):
                result.append(bool(val))
            elif isinstance(val, (pd.Timestamp, np.datetime64)):
                result.append(
                    val.isoformat() if hasattr(val, "isoformat") else str(val)
                )
            elif (
                pd.isna(val)
                if pd
                else (val is None or (isinstance(val, float) and np.isnan(val)))
            ):
                result.append(None)
            else:
                result.append(str(val))
        return result

    def save_data(self, data: Dict[str, Any], output_path: Union[str, Path]) -> None:
        """
        Save processed data to JSON file.

        Args:
            data: Processed data dictionary
            output_path: Path to save the data
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Data saved to: {output_path}")
