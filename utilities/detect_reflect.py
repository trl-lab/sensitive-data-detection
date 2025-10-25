from utilities.utils import table_markdown
import json
import re
import os
import sys
# Get root directory
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(os.path.join(root_dir, "utilities/isp_example.json"), "r") as f:
    ISP_DATA = json.load(f)


def get_isp_by_country(table_name):
    table_name = table_name.lower()

    for isp_name, isp_content in ISP_DATA.items():
        print(f"ISP name in get_isp_by_country: {isp_name}")
        location = isp_content.get("country", "").lower()
        if location and re.search(rf"\\b{re.escape(location)}\\b", table_name):
            return isp_name, isp_content

        # Fallback: probeer ook match op de sleutel
        if location in table_name or location.replace(" ", "") in table_name:
            return isp_name, isp_content

    return "default", ISP_DATA["default"]


def detect_non_pii(table_data, generator, fname, method="table"):
    isp_name, isp_used = None, None  # Ensure variables are always defined

    # Initialize metadata if not present
    if not table_data.get("metadata"):
        table_data["metadata"] = {}

    # Get ISP data
    print(f"Fname: {fname}")
    isp_name, isp_used = get_isp_by_country(fname)
    print(f"ISP used: {isp_name}")
    table_data["metadata"]["isp_used"] = isp_name

    # table_md = table_markdown(table_data, pii_model=generator.model_name)
    table_md = table_markdown(table_data)

    if method == "column":
        if "columns" not in table_data:
            table_data["columns"] = {}

        for column_name, col_data in table_data["columns"].items():
            # Initialize column metadata if not present
            if not isinstance(col_data, dict):
                col_data = {}
                table_data["columns"][column_name] = col_data

            reflection_key = f"non_pii_reflection_{generator.model_name}"
            explanation_key = f"non_pii_reflection_{generator.model_name}_explanation"

            # Only process if not already done or if there was an error
            if (
                not col_data.get(reflection_key)
                or col_data.get(reflection_key) == "ERROR_GENERATION"
            ):
                sensitivity, explanation = generator.classify_sensitive_non_pii(
                    column_name, table_md, isp_used
                )
                col_data[reflection_key] = sensitivity
                col_data[explanation_key] = explanation

    elif method == "table":
        # Initialize required metadata keys
        non_pii_key = f"non_pii_{generator.model_name}"
        non_pii_explanation_key = f"non_pii_{generator.model_name}_explanation"
        non_pii_no_isp_key = f"non_pii_no_isp_{generator.model_name}"
        non_pii_no_isp_explanation_key = (
            f"non_pii_no_isp_{generator.model_name}_explanation"
        )

        # Process with ISP if not already done
        if non_pii_key not in table_data["metadata"]:
            sensitivity, explanation = generator.classify_sensitive_non_pii_table(
                table_md, isp=isp_used
            )
            table_data["metadata"][non_pii_key] = sensitivity
            table_data["metadata"][non_pii_explanation_key] = explanation

            # Process without ISP
            sensitivity, explanation = generator.classify_sensitive_non_pii_table(
                table_md, isp=None
            )
            table_data["metadata"][non_pii_no_isp_key] = sensitivity
            table_data["metadata"][non_pii_no_isp_explanation_key] = explanation

    return table_data



def detect_pii(table_data, generator, detect_key=None, k=5, force=False):

    if detect_key:
        pii_detection_key = detect_key
    else:
        pii_detection_key = f"pii_detection_{generator.model_name}"

    for column_name, col_data in table_data["columns"].items():
        sample_values = col_data["records"]
        if all(record is None for record in sample_values):
            col_data[pii_detection_key] = "None"
            continue
        if not col_data.get(pii_detection_key) or force:
            pii_entity = generator.classify_pii(column_name, sample_values, k=k)
            col_data[pii_detection_key] = pii_entity
    
    return table_data


def reflect_pii(table_data, generator, detect_key=None, reflect_key=None, force=False):

    if detect_key:
        pii_detection_key = detect_key
    else:
        pii_detection_key = f"pii_detection_{generator.model_name}"

    if reflect_key:
        pii_reflection_key = reflect_key
    else:
        pii_reflection_key = f"pii_reflection_{generator.model_name}"

    table_md = table_markdown(table_data, pii_key=pii_detection_key)

    # If model name is directory, skip the reflection
    if os.path.isdir(generator.model_name):
        print(f"Model name is directory, skipping reflection for {generator.model_name}")
        return table_data

    for column_name, col_data in table_data["columns"].items():
        pii_entity = col_data[pii_detection_key]

        if not col_data.get(pii_reflection_key) or force: # If not exists or force is True
            if pii_entity == "None": # If no PII entity, set to NON_SENSITIVE automatically
                col_data[pii_reflection_key] = "NON_SENSITIVE"
                continue

            sensitivity = generator.classify_sensitive_pii(
                column_name, table_md, pii_entity
            )
            col_data[pii_reflection_key] = sensitivity

    return table_data