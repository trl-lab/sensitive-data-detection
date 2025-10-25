#!/usr/bin/env python3
"""
Example script to run the complete ISP extraction pipeline:
1. Download PDFs from ReliefWeb
2. Convert PDFs to text
3. Extract rules using GPT-4o-mini
4. Save results to JSON
"""

import json
import re
from typing import Dict, List, Optional, Any
from openai import OpenAI
import dotenv
import pycountry
from pathlib import Path
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import os
import time
from collections import Counter


def validate_pdf_file(file_path: str) -> bool:
    """Validate if the file is a PDF and accessible."""
    if not os.path.exists(file_path):
        return False

    if not file_path.lower().endswith(".pdf"):
        return False

    try:
        # Try to open the file
        with open(file_path, "rb") as f:
            header = f.read(4)
            return header == b"%PDF"
    except Exception:
        return False


def find_pdf_files(directory: str) -> List[str]:
    """Find all PDF files in a directory."""
    pdf_files = []

    try:
        path = Path(directory)
        if path.is_dir():
            pdf_files = [str(p) for p in path.glob("*.pdf")]
        elif path.is_file() and path.suffix.lower() == ".pdf":
            pdf_files = [str(path)]
    except Exception:
        pass

    return pdf_files


def create_summary_report(extraction_results: Dict) -> Dict[str, Any]:
    """Create a summary report of extraction results."""

    if (
        isinstance(extraction_results, dict)
        and "sensitivity_rules" in extraction_results
    ):
        # Single file results
        extraction_results = {"single_file": extraction_results}

    summary = {
        "total_files_processed": len(extraction_results),
        "successful_extractions": 0,
        "failed_extractions": 0,
        "total_sensitivity_levels": 0,
        "common_sensitivity_levels": {},
        "average_confidence": 0.0,
        "files_with_issues": [],
    }

    confidences = []
    all_levels = []

    for file_path, results in extraction_results.items():
        if results.get("extraction_successful", False):
            summary["successful_extractions"] += 1

            sensitivity_rules = results.get("sensitivity_rules", {})
            summary["total_sensitivity_levels"] += len(sensitivity_rules)

            # Collect levels for frequency analysis
            all_levels.extend(sensitivity_rules.keys())

            # Collect confidence scores
            metadata = results.get("extraction_metadata", {})
            confidence = metadata.get("confidence", 0.0)
            if confidence > 0:
                confidences.append(confidence)

        else:
            summary["failed_extractions"] += 1
            summary["files_with_issues"].append(
                {"file": file_path, "error": results.get("error", "Unknown error")}
            )

    # Calculate average confidence
    if confidences:
        summary["average_confidence"] = sum(confidences) / len(confidences)

    # Find common sensitivity levels
    level_counts = Counter(all_levels)
    summary["common_sensitivity_levels"] = dict(level_counts.most_common(10))

    return summary


def format_rules_for_display(rules_data: Dict) -> str:
    """Format extracted rules for readable display."""

    if "sensitivity_rules" not in rules_data:
        return "No sensitivity rules found."

    output = []
    sensitivity_rules = rules_data["sensitivity_rules"]

    output.append("EXTRACTED SENSITIVITY RULES")
    output.append("=" * 50)

    for level_name, level_data in sensitivity_rules.items():
        output.append(f"\n{level_name.upper()}")
        output.append("-" * len(level_name))

        if level_data.get("description"):
            output.append(f"Description: {level_data['description']}")

        if level_data.get("rules"):
            output.append("Rules:")
            for rule in level_data["rules"]:
                output.append(f"  • {rule}")

        if level_data.get("criteria"):
            output.append("Classification Criteria:")
            for criterion in level_data["criteria"]:
                output.append(f"  • {criterion}")

        if level_data.get("examples"):
            output.append("Examples:")
            for example in level_data["examples"]:
                output.append(f"  • {example}")

        if level_data.get("handling_requirements"):
            output.append("Handling Requirements:")
            for requirement in level_data["handling_requirements"]:
                output.append(f"  • {requirement}")

    # Add metadata
    if "extraction_metadata" in rules_data:
        metadata = rules_data["extraction_metadata"]
        output.append("\nExtraction Metadata:")
        output.append(f"Confidence: {metadata.get('confidence', 0):.2f}")
        output.append(f"Model: {metadata.get('model_used', 'Unknown')}")

        if metadata.get("extraction_notes"):
            output.append("Notes:")
            for note in metadata["extraction_notes"]:
                output.append(f"  • {note}")

    return "\n".join(output)


class RulesExtractor:
    """
    Extracts sensitivity rules from ISP PDF documents using GPT-4o-mini.
    """

    def __init__(self, model_name: str = "gpt-4o-mini"):
        """
        Initialize the rules extractor.

        Args:
            model_name (str): OpenAI model to use (default: gpt-4o-mini)
        """
        self.model_name = model_name
        self.client = self._setup_openai()

        # Common sensitivity levels to look for
        self.sensitivity_levels = [
            "NON/LOW_SENSITIVE",
            "MODERATE_SENSITIVE",
            "HIGH_SENSITIVE",
            "SEVERE_SENSITIVE",
        ]

    def _setup_openai(self) -> OpenAI:
        """Setup OpenAI client."""
        try:
            dotenv.load_dotenv()
            client = OpenAI()
            return client
        except Exception:
            raise

    def extract_pdf_text(self, pdf_path: str) -> Optional[str]:
        """
        Extract text from PDF file using PyMuPDF (pymupdf).

        Args:
            pdf_path (str): Path to the PDF file

        Returns:
            str: Extracted text or None if failed
        """
        # Check if pymupdf is available
        try:
            import fitz  # PyMuPDF
        except ImportError:
            return None

        # Check if file exists
        if not os.path.exists(pdf_path):
            return None

        try:
            # Extract text content
            text_content = []
            pdf_document = fitz.open(pdf_path)

            for page_num in range(pdf_document.page_count):
                try:
                    page = pdf_document[page_num]
                    text = page.get_text()
                    if text and text.strip():
                        text_content.append(text)

                except Exception:
                    continue

            pdf_document.close()
            extracted_text = "\n\n".join(text_content)

            return extracted_text

        except Exception:
            return None

    def _standardize_country_name(self, country_name: str) -> Dict[str, Any]:
        """
        Standardize country name using pycountry library.

        Args:
            country_name (str): Raw country name extracted from document

        Returns:
            Dict with standardized country information
        """
        if not country_name or not isinstance(country_name, str):
            return {
                "raw_country": country_name,
                "standardized_name": None,
                "alpha_2": None,
                "alpha_3": None,
                "standardization_confidence": 0.0,
            }

        # Clean the country name
        cleaned_name = country_name.strip().title()

        # Try exact match first
        try:
            country = pycountry.countries.lookup(cleaned_name)
            return {
                "raw_country": country_name,
                "standardized_name": country.name,
                "alpha_2": country.alpha_2,
                "alpha_3": country.alpha_3,
                "standardization_confidence": 1.0,
            }
        except LookupError:
            pass

        # Try fuzzy matching by searching in names
        potential_matches = []
        search_terms = [cleaned_name, cleaned_name.lower(), cleaned_name.upper()]

        for term in search_terms:
            for country in pycountry.countries:
                # Check official name
                if term in country.name:
                    confidence = len(term) / len(country.name)
                    potential_matches.append((country, confidence))

                # Check common name if available
                if hasattr(country, "common_name") and term in country.common_name:
                    confidence = len(term) / len(country.common_name)
                    potential_matches.append((country, confidence))

        # Sort by confidence and take the best match
        if potential_matches:
            potential_matches.sort(key=lambda x: x[1], reverse=True)
            best_match, confidence = potential_matches[0]

            # Only accept matches with reasonable confidence
            if confidence >= 0.5:
                return {
                    "raw_country": country_name,
                    "standardized_name": best_match.name,
                    "alpha_2": best_match.alpha_2,
                    "alpha_3": best_match.alpha_3,
                    "standardization_confidence": confidence,
                }

        # No good match found
        return {
            "raw_country": country_name,
            "standardized_name": None,
            "alpha_2": None,
            "alpha_3": None,
            "standardization_confidence": 0.0,
        }

    def extract_rules_from_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract sensitivity rules from an ISP PDF document.

        Args:
            pdf_path (str): Path to the PDF file

        Returns:
            Dict containing extracted rules organized by sensitivity level
        """
        try:
            # Extract text from PDF
            extracted_text = self.extract_pdf_text(pdf_path)

            if not extracted_text:
                return {
                    "error": "Failed to extract text from PDF",
                    "pdf_path": pdf_path,
                    "sensitivity_rules": {},
                }

            # Extract rules using GPT
            rules_data = self._extract_rules_with_gpt(extracted_text)

            # Standardize country name if present
            raw_country = rules_data.get("country", None)
            if raw_country:
                country_info = self._standardize_country_name(raw_country)
            else:
                country_info = self._standardize_country_name("")

            return {
                "pdf_path": pdf_path,
                "extraction_successful": True,
                "country": country_info["standardized_name"],
                "country_info": country_info,
                "sensitivity_rules": rules_data.get("sensitivity_rules", {}),
                "extraction_metadata": {
                    "text_length": len(extracted_text),
                    "model_used": self.model_name,
                    "confidence": rules_data.get("confidence", 0.0),
                    "extraction_notes": rules_data.get("notes", []),
                },
            }

        except Exception as e:
            return {
                "error": str(e),
                "pdf_path": pdf_path,
                "extraction_successful": False,
                "sensitivity_rules": {},
            }

    def _extract_rules_with_gpt(self, text: str) -> Dict[str, Any]:
        """Use GPT-4o-mini to extract and structure sensitivity rules."""

        # Create the prompt
        prompt = self._create_extraction_prompt(text)

        try:
            # Call GPT-4o-mini
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,  # Low temperature for consistent extraction
                max_tokens=2000,
            )

            response_text = response.choices[0].message.content

            # Parse the response
            parsed_rules = self._parse_gpt_response(response_text)
            return parsed_rules

        except Exception as e:
            return {
                "country": None,
                "sensitivity_rules": {},
                "confidence": 0.0,
                "notes": [f"GPT extraction failed: {str(e)}"],
            }

    def _create_extraction_prompt(self, text: str) -> str:
        """Create a comprehensive prompt for rules extraction."""

        # Truncate text if too long to avoid token limits
        max_text_length = 15000  # Leave room for prompt and response
        if len(text) > max_text_length:
            text = text[:max_text_length] + "\n... [TEXT TRUNCATED] ..."

        prompt = f"""I need you to extract sensitivity classification rules from the following ISP (Information Security Policy) document text.

Look for tables, sections, or lists that define different sensitivity levels and their corresponding data and information types.

Common sensitivity levels include:
- NON/LOW_SENSITIVE, MODERATE_SENSITIVE, HIGH_SENSITIVE, SEVERE_SENSITIVE

Document text:
{text}

Please extract only the sensitivity rules data and information type and add the country information in the following JSON format:

{{
  "country": "Region/Country name",
  "sensitivity_rules": {{
    "LOW/NON_SENSITIVE": {{
      "data and information type": [
        "Rule 1 for this level",
        "Rule 2 for this level"
      ]
    }},
    "MODERATE_SENSITIVE": {{
      "data and information type": [ ... ]
    }}
  }}
}}

Return the JSON only.

JSON Response:"""

        return prompt

    def _parse_gpt_response(self, response_text: str) -> Dict[str, Any]:
        """Parse the GPT response and extract structured rules."""

        try:
            # Try to find JSON in the response
            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                rules_data = json.loads(json_str)

                # Validate the structure
                if "sensitivity_rules" in rules_data:
                    return rules_data
                else:
                    pass

            # If JSON parsing fails, try to extract manually
            return self._manual_parse_response(response_text)

        except json.JSONDecodeError:
            return self._manual_parse_response(response_text)
        except Exception as e:
            return {
                "sensitivity_rules": {},
                "confidence": 0.0,
                "notes": [f"Failed to parse response: {str(e)}"],
            }

    def _manual_parse_response(self, response_text: str) -> Dict[str, Any]:
        """Manually parse response if JSON parsing fails."""

        rules = {}
        confidence = 0.3  # Lower confidence for manual parsing
        notes = ["Manual parsing used due to JSON parsing failure"]

        # Look for sensitivity levels mentioned
        for level in self.sensitivity_levels:
            if level.lower() in response_text.lower():
                # Try to extract information about this level
                level_info = {
                    "description": f"Extracted information for {level} level",
                    "rules": [],
                    "criteria": [],
                    "examples": [],
                    "handling_requirements": [],
                }

                # Simple extraction of text around the level mention
                level_pattern = rf"{level}[:\-\s]*([^\.]+\.)"
                matches = re.findall(level_pattern, response_text, re.IGNORECASE)
                if matches:
                    level_info["rules"] = matches[:3]  # Take first 3 matches

                rules[level] = level_info

        return {"sensitivity_rules": rules, "confidence": confidence, "notes": notes}

    def fetch_information_sharing_protocols(
        self, output_dir: str = "downloaded_isps"
    ) -> List[str]:
        """Retrieve ISP PDFs from ReliefWeb search results and return file paths."""

        base_url = (
            "https://reliefweb.int/updates?view=reports&advanced-search=%28F7%29"
            "&search=Information%20Sharing%20Protocol"
        )

        session = requests.Session()
        page = 0
        all_links = []
        downloaded_files = []

        # Create download folder
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)

        while True:
            page_url = f"{base_url}&page={page}"
            try:
                resp = session.get(page_url, timeout=30)
                resp.raise_for_status()
            except Exception:
                break

            soup = BeautifulSoup(resp.text, "html.parser")

            # Extract all h3 titles with the specified class and lang attribute
            h3_titles = soup.find_all("h3", class_="rw-river-article__title", lang="en")

            if not h3_titles:
                break

            for h3 in h3_titles:
                # Find the link associated with this h3 title
                link_element = h3.find_parent("a") or h3.find("a")
                if link_element:
                    link_href = link_element.get("href")
                    if link_href and "protocol" in link_href:
                        full_link = urljoin(base_url, link_href)
                        all_links.append(full_link)

            page += 1

            # Optional: limit to first few pages for testing
            if page >= 5:  # Change this number to scrape more or fewer pages
                break

        # Download PDFs from each link (skip if already exists)
        for i, link in enumerate(all_links, 1):
            # Check if PDF already exists by trying to predict the filename
            # We'll use a simple approach: check if any PDF file exists in the folder
            existing_pdfs = list(output_dir_path.glob("*.pdf"))
            if existing_pdfs:
                # For now, we'll download anyway but this could be enhanced with better filename prediction
                # based on the link content or title
                pass

            downloaded_file = self._download_pdf_from_page(
                link, session, output_dir_path
            )
            if downloaded_file:
                downloaded_files.append(downloaded_file)

            # Add a small delay to be respectful to the server
            time.sleep(1)

        return downloaded_files

    def _find_pdf_links(
        self, soup: BeautifulSoup, page_url: str, session: requests.Session
    ) -> List[str]:
        """Return a list of absolute PDF URLs found on a ReliefWeb HTML page."""

        pdf_links: List[str] = []

        # Direct PDF links (href ends with .pdf)
        for link in soup.find_all("a", href=True):
            href = link.get("href")
            if href and href.lower().endswith(".pdf"):
                pdf_links.append(urljoin(page_url, href))

        # Links that may redirect to a PDF (e.g. download buttons)
        for link in soup.find_all("a", href=True):
            href = link.get("href")
            if not href:
                continue
            link_text = link.get_text(strip=True).lower()
            if "pdf" in link_text or "download" in link_text:
                if not href.lower().endswith(".pdf"):
                    try:
                        head_resp = session.head(href, timeout=10, allow_redirects=True)
                        content_type = head_resp.headers.get("content-type", "")
                        if "application/pdf" in content_type:
                            pdf_links.append(urljoin(page_url, href))
                    except Exception:
                        continue

        return pdf_links

    def _derive_filename(self, pdf_url: str, response: requests.Response) -> str:
        """Determine a safe local filename for the downloaded PDF."""

        filename: Optional[str] = None

        # Try to read from Content-Disposition header
        content_disposition = response.headers.get("content-disposition", "")
        if "filename=" in content_disposition:
            match = re.search(r'filename="?([^";]+)"?', content_disposition)
            if match:
                filename = match.group(1)

        # Fallback: derive from URL
        if not filename:
            filename = os.path.basename(pdf_url.split("/")[-1])

        # Final fallback: generic timestamped name
        if not filename or not filename.lower().endswith(".pdf"):
            filename = f"document_{int(time.time())}.pdf"

        # Sanitize & ensure .pdf extension
        filename = self._sanitize_filename(filename)
        if not filename.lower().endswith(".pdf"):
            filename += ".pdf"

        return filename

    def _get_unique_filepath(self, folder: Path, filename: str) -> Path:
        """Return a Path that does not yet exist inside *folder by appending counters if needed."""

        filepath = folder / filename
        counter = 1
        base, ext = os.path.splitext(filename)
        while filepath.exists():
            filepath = folder / f"{base}_{counter}{ext}"
            counter += 1
        return filepath

    def _download_pdf_from_page(
        self, url: str, session: requests.Session, download_folder: Path
    ) -> Optional[str]:
        """Download the first PDF found on *url* and return the local file path."""

        # 1) Retrieve HTML
        try:
            resp = session.get(url, timeout=30)
            resp.raise_for_status()
        except Exception:
            return None

        soup = BeautifulSoup(resp.text, "html.parser")

        # 2) Locate PDF links
        pdf_links = self._find_pdf_links(soup, url, session)
        if not pdf_links:
            return None

        pdf_url = pdf_links[0]

        # 3) Download the PDF itself
        try:
            pdf_resp = session.get(pdf_url, timeout=60)
            pdf_resp.raise_for_status()
        except Exception:
            return None

        # 4) Work out a safe local filename & path
        filename = self._derive_filename(pdf_url, pdf_resp)
        filepath = self._get_unique_filepath(download_folder, filename)

        # 5) Persist to disk
        try:
            with open(filepath, "wb") as file_obj:
                file_obj.write(pdf_resp.content)
            return str(filepath)
        except Exception:
            return None

    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename by removing invalid characters"""
        # Remove or replace invalid characters
        filename = re.sub(r'[<>:"/\\|?*]', "_", filename)
        # Remove extra spaces and dots
        filename = re.sub(r"\s+", " ", filename).strip()
        filename = filename.strip(".")
        # Limit length
        if len(filename) > 200:
            filename = filename[:200]
        return filename

    def convert_pdfs_to_text(
        self, pdf_paths: List[str], output_dir: str = "converted_texts"
    ) -> Dict[str, str]:
        """Convert PDF files to text and return a dictionary mapping PDF paths to text content."""

        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)

        text_contents = {}

        for pdf_path in pdf_paths:
            try:
                pdf_name = Path(pdf_path).stem
                text_file_path = output_dir_path / f"{pdf_name}.txt"

                # Check if text file already exists
                if text_file_path.exists():
                    with open(text_file_path, "r", encoding="utf-8") as f:
                        extracted_text = f.read()
                    text_contents[pdf_path] = extracted_text
                    continue

                # Extract text from PDF
                extracted_text = self.extract_pdf_text(pdf_path)

                if not extracted_text:
                    continue

                # Save text to file
                with open(text_file_path, "w", encoding="utf-8") as f:
                    f.write(extracted_text)

                text_contents[pdf_path] = extracted_text

            except Exception:
                continue

        return text_contents

    def get_existing_pdfs(self, output_dir: str = "downloaded_isps") -> List[str]:
        """Get list of existing PDF files in the output directory."""
        output_dir_path = Path(output_dir)
        if not output_dir_path.exists():
            return []

        pdf_files = list(output_dir_path.glob("*.pdf"))
        return [str(f) for f in pdf_files]

    def get_existing_text_files(self, output_dir: str = "converted_texts") -> List[str]:
        """Get list of existing text files in the output directory."""
        output_dir_path = Path(output_dir)
        if not output_dir_path.exists():
            return []

        text_files = list(output_dir_path.glob("*.txt"))
        return [str(f) for f in text_files]

    def extract_rules_from_reliefweb(
        self,
        output_dir: str = "downloaded_isps",
        text_output_dir: str = "converted_texts",
        skip_existing: bool = True,
    ) -> Dict[str, Dict]:
        """Fetch ISP PDFs via ReliefWeb, convert to text, and extract rules from each."""

        # Check for existing files
        existing_pdfs = self.get_existing_pdfs(output_dir)

        if existing_pdfs and skip_existing:
            pdf_paths = existing_pdfs
        else:
            # Step 1: Download PDFs
            pdf_paths = self.fetch_information_sharing_protocols(output_dir)

        if not pdf_paths:
            return {}

        # Step 2: Convert PDFs to text
        text_contents = self.convert_pdfs_to_text(pdf_paths, text_output_dir)

        # Step 3: Extract rules from text using GPT
        results = {}

        for pdf_path, text_content in text_contents.items():
            try:
                country = Path(pdf_path).stem.replace("_", " ")

                # Extract rules using GPT from text content
                rules_data = self._extract_rules_with_gpt(text_content)

                # Standardize country name if present
                raw_country = rules_data.get("country", country)
                country_info = self._standardize_country_name(raw_country)

                text_path = str(Path(text_output_dir) / f"{Path(pdf_path).stem}.txt")
                results[country] = {
                    "pdf_path": pdf_path,
                    "text_path": text_path,
                    "extraction_successful": True,
                    "country": country_info["standardized_name"],
                    "country_info": country_info,
                    "sensitivity_rules": rules_data.get("sensitivity_rules", {}),
                    "extraction_metadata": {
                        "text_length": len(text_content),
                        "model_used": self.model_name,
                        "confidence": rules_data.get("confidence", 0.0),
                        "extraction_notes": rules_data.get("notes", []),
                    },
                }

            except Exception as e:
                results[Path(pdf_path).stem] = {
                    "error": str(e),
                    "pdf_path": pdf_path,
                    "extraction_successful": False,
                    "sensitivity_rules": {},
                }

        return results

    def extract_from_multiple_pdfs(self, pdf_paths: List[str]) -> Dict[str, Dict]:
        """Extract rules from multiple PDF files."""
        results = {}

        for pdf_path in pdf_paths:
            results[pdf_path] = self.extract_rules_from_pdf(pdf_path)

        return results

    def save_extracted_rules(self, rules_data: Dict, output_path: str):
        """Save extracted rules to a JSON file."""
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(rules_data, f, indent=2, ensure_ascii=False)
        except Exception:
            pass


def main():
    """Run the complete extraction pipeline."""

    # Initialize the extractor
    extractor = RulesExtractor(model_name="gpt-4o-mini")

    # Check for existing files
    existing_pdfs = extractor.get_existing_pdfs("downloaded_isps")
    existing_texts = extractor.get_existing_text_files("converted_texts")

    # Step 1: Download PDFs and extract rules (skip existing files)
    results = extractor.extract_rules_from_reliefweb(
        output_dir="downloaded_isps",
        text_output_dir="converted_texts",
        skip_existing=True,  # Skip existing files
    )

    if not results:
        return

    # Step 2: Save results
    output_file = "isp_example.json"
    extractor.save_extracted_rules(results, output_file)


if __name__ == "__main__":
    main()
