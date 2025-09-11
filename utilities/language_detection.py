"""Utility wrapper for language detection."""

from modules.language_detection.detector import LanguageDetector
from modules.language_detection.file_reader import FileReader

__all__ = ["LanguageDetector", "FileReader"]
