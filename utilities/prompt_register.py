import json

PII_ENTITIES_LIST = [
    "IP_ADDRESS",
    "AGE",
    "CREDIT_CARD_NUMBER",
    "DATE_OF_BIRTH",
    "DISABILITY_GROUP",
    "EDUCATION_LEVEL",
    "EMAIL_ADDRESS",
    "ETHNIC_GROUP",
    "GENDER",
    "GENERIC_IDENTIFIER",
    "GEO_COORDINATES",
    "MARITAL_STATUS",
    "MEDICAL_TERM",
    "OCCUPATION",
    "ORGANIZATION_NAME",
    "PERSON_NAME",
    "PHONE_NUMBER",
    "PROTECTION_GROUP",
    "RELIGIOUS_GROUPS",
    "SEXUALITY",
    "SPOKEN_LANGUAGE",
    "STREET_ADDRESS",
    "URL",
    "IBAN_CODE",
    "PASSPORT",
    "SWIFT_CODE",
    "ZIPCODE",
]

low_sensitivity_columns = [
    "admin_level",
    "population_total",
    "service_coverage",
    "activity_type",
    "sector_name",
    "response_indicator",
    "facility_type",
    "access_level",
    "region_name",
    "organization_presence",
]

sensitive_columns = [
    "aggregated_needs",
    "access_constraints",
    "stock_levels",
    "service_status",
    "incident_counts",
    "movement_estimates",
    "infrastructure_functionality",
    "response_summary",
    "site_name",
    "location_group",
    "household_aggregates",
    "community_vulnerabilities",
    "site_locations",
    "field_plans",
    "hazard_exposure",
    "facility_name",
    "staff_numbers",
    "distribution_sites",
    "area_control",
    "displacement_routes",
    "individual_responses",
    "raw_assessment_data",
    "location_coordinates",
    "group_vulnerability_map",
    "security_reports",
    "movement_tracking",
    "incident_reports",
    "operational_routes",
    "conflict_zone_data",
    "law_violation_records",
    "real_time_access_data",
]


def is_pii(column_name, sample_values, k=5):
    sample_values_prompt = f"Sample values: {sample_values[:k]}"
    if k == 0:
        sample_values_prompt = ""

    prompt = f"""
### INSTRUCTION
You are a PII classification system. Given a column name **AND** sample values, determine if this column contains a specific type of PII.
Choose ONE category from the following list or respond with 'None' if the column doesn't contain PII:

PII entities list: {PII_ENTITIES_LIST}

Return ONLY the entity name or only 'None' with no additional text.

### INPUT
Column name: {column_name}
{sample_values_prompt}

### RESPONSE
"""
    # print(prompt)
    return prompt


def is_sensitive_pii(column_name, table_context, column_pii_entity):
    prompt = f"""
### Instruction:
You are a sensitivity classification system. Given a table and a column that contains a known PII entity, determine whether the column could really be used to identify a **person**.

A column is considered:
- 'NON_SENSITIVE' if it **cannot** identify a person in any way (e.g., general data, aggregate data, location with no identifying features).
- 'MEDIUM_SENSITIVE' if it **could possibly** identify a person (e.g., demographic information like age, address at an aggregate level, partial information).
- 'HIGH_SENSITIVE' if it **definitely** identifies a person (e.g., a full name, SSN, email, phone number).

Return ONLY the classification (NON_SENSITIVE, MEDIUM_SENSITIVE, HIGH_SENSITIVE).

### Input:
Table: {table_context}
Column name: {column_name}
# PII entity: {column_pii_entity}

### Response:
"""
    return prompt


def prompt_non_pii_table(table_context, isp):
    prompt = f"""
### Instruction:
You are a data governance assistant. Your task is to determine the overall sensitivity level of this table, based strictly on the provided Information Sharing Protocols (ISP) and the table's content and state which columns make the table sensitive.

Follow these exact steps

1. Analyze the table schema AND the records of the table.
2. Use the ISP sensitivity levels: NON_SENSITIVE, MODERATE_SENSITIVE, HIGH_SENSITIVE, SEVERE_SENSITIVE. Only assign a sensitivity level if explicitly supported by ISP guidance.
3. Identify ONLY the columns that are sensitive on their own OR that become sensitive in combination with others, DIRECTLY supported by ISP guidance.
4. If multiple sensitivity levels might apply, always choose the highest one explicitly mentioned in the ISP for the relevant data type. We rather be cautious than miss something.

### Input:
ISP Rules
SEVERE_SENSITIVE: {isp['sensitivity_rules']['SEVERE_SENSITIVE']['data and information type']}
HIGH_SENSITIVE: {isp['sensitivity_rules']['HIGH_SENSITIVE']['data and information type']}
MODERATE_SENSITIVE: {isp['sensitivity_rules']['MODERATE_SENSITIVE']['data and information type']}
NON_SENSITIVE: {isp['sensitivity_rules']['LOW/NON_SENSITIVE']['data and information type']}

Table:
{table_context}

### Response Format:
- Sensitivity Classification: <ONE OF: NON_SENSITIVE / MODERATE_SENSITIVE / HIGH_SENSITIVE / SEVERE_SENSITIVE>
- List with ONLY the columns that are sensitive or in combination with other columns are sensitive.
- Cited ISP Rule(s): Quote the specific ISP rule(s) that directly support the classification. If none directly apply, explain briefly why it is NON_SENSITIVE.
"""
    # print(prompt)
    return prompt


def prompt_non_pii_table_no_isp(table_context):
    prompt = f"""
### Instruction:
You are a data governance assistant. Your task is to determine the overall sensitivity level of this table, based strictly on the table's content and state which columns make the table sensitive.

Follow these exact steps

1. Analyze the table schema AND the records of the table.
2. Identify ONLY the columns that are sensitive on their own OR that become sensitive in combination with others.
3. Do look at the records of the table whether they contain sensitive information or not.

NON_SENSITIVE: Information or data that, if disclosed or accessed without proper authorization, are unlikely to cause any harm or negative impacts to affected people and/or humanitarian actors.
MODERATE_SENSITIVE: Information or data that, if disclosed or accessed without proper authorization, are likely to cause minor harm or negative impacts and/or be disadvantageous for affected people and/or humanitarian actors.
HIGH_SENSITIVE: Information or data that, if disclosed or accessed without proper authorization, are likely to cause serious harm or negative impacts to affected people and/or humanitarian actors and/or damage to a response.
SEVERE_SENSITIVE: Information or data that, if disclosed or accessed without proper authorization, are likely to cause severe harm or negative impacts and/or damage to affected people and/or humanitarian actors and/or impede the conduct of the work of a response.

### Input:
Table:
{table_context}

### Response Format:
- Sensitivity Classification: <ONE OF: NON_SENSITIVE / MODERATE_SENSITIVE / HIGH_SENSITIVE / SEVERE_SENSITIVE>
- Explain why you chose the sensitivity level.
"""
    # print(prompt)
    return prompt


def prompt_reflect_non_pii(table_context, isp, classification):
    prompt = f"""
Table:
{table_context}

Information Sharing Protocols (ISP):
Sensitivity Level NON_SENSITIVE: {isp['sensitivity_rules']['LOW/NON_SENSITIVE']}
Sensitivity Level MODERATE_SENSITIVE: {isp['sensitivity_rules']['MODERATE_SENSITIVE']}
Sensitivity Level HIGH_SENSITIVE: {isp['sensitivity_rules']['HIGH_SENSITIVE']}
Sensitivity Level SEVERE_SENSITIVE: {isp['sensitivity_rules']['SEVERE_SENSITIVE']}

Is the table really {classification} according to rules of the ISP? If not, then give the correct sensitivity level.
"""
    return prompt


def prompt_non_pii(column_name, table_context, isp):
    prompt = f"""
### Instruction:
You are a data governance assistant. Your task is to assess which columns in the provided **table** contain sensitive information, using both the **table schema and content** and the provided **Information Sharing Protocols (ISP)**.

Follow these steps in order:

1. Analyze the table schema and content.
2. Analyze each sensitivity level of the ISP: NON_SENSITIVE, MEDIUM_SENSITIVE, HIGH_SENSITIVE, or SEVERE_SENSITIVE.
3. For each column, assess its sensitivity level based on the column's content, its context in the table, and the ISP guidance.

### Input:
**ISP Rules:**
{json.dumps(isp, indent=4)}

**Table:**
{table_context}

**Column to evaluate:**
{column_name}

Return only the following:
- Sensitivity level: <NON_SENSITIVE / MEDIUM_SENSITIVE / HIGH_SENSITIVE / SEVERE_SENSITIVE>
- Justification (Step-by-Step): ...
- Cited ISP Rule(s): Quote or describe relevant ISP guidance, or state explicitly if none apply.

### Response:
"""
    # print(prompt)
    return prompt


def is_microdata(table):
    prompt = f"""
### Instruction:
You are a data governance assistant. A table is considered microdata if it contains individual-level data that could identify a person or entity in a survey or census, such as data about individuals, households, or organizations. Based on the table, determine if this table qualifies as microdata. Microdata is most likely to be found in surveys.

Return ONLY 'YES' or 'NO' with no additional text.

### Input:
Table:
{table}

### Response:
"""
    return prompt


def is_kii(column_name, sample_values):
    prompt = f"""
### Instruction:
You are a data governance assistant. Given the Key Informant Interviews (KII) list, determine if any columns in the table contain potential Key Informant Information (KII) entities. KII entities typically involve sensitive data related to individuals, locations, or organizations involved in interviews or humanitarian assessments. 

KII list: ["gender", "admin", "age", "household_size", "district", "marital_status", "disabled", "occupation", "diseases", "camp", "country", "status", "ethnicity", "location", "governorate", "origin", "region", "income", "level_education", "province", "settlement", "breadwinner", "site", "Sub_district", "language_spoken", "religion", "village"]

Return only the category name or 'None' with no additional text.

### Input:
Column name: {column_name}
Sample values: {sample_values}

### Response:
"""
    return prompt


def is_sensitive_kii(column_name, table_context, column_kii_entity):
    prompt = f"""
### Instruction:
You are a data governance assistant. Based on the table context and the other columns containing potential Key Informant Information (KII), determine if this column is truly sensitive. KII columns typically contain sensitive information that could identify individuals or organizations involved in interviews or assessments. Consider the KII list and any other columns in the table when making your decision.

KII list: ["gender", "admin", "age", "household_size", "district", "marital_status", "disabled", "occupation", "diseases", "camp", "country", "status", "ethnicity", "location", "governorate", "origin", "region", "income", "level_education", "province", "settlement", "breadwinner", "site", "Sub_district", "language_spoken", "religion", "village"]

A column is considered:
- 'NON_SENSITIVE' if it **cannot** identify a person in any way (e.g., general data, aggregate data, location with no identifying features).
- 'MEDIUM_SENSITIVE' if it **could possibly** identify a person (e.g., demographic information like age, address at an aggregate level, partial information like a first name).
- 'HIGH_SENSITIVE' if it **definitely** identifies a person (e.g., a full name, SSN, email, phone number).

Return the classification (**NON_SENSITIVE**, **MEDIUM_SENSITIVE**, **HIGH_SENSITIVE**) and explain **why** or **why not** the column is sensitive.

### Input:
Table: {table_context}
Column name: {column_name}
Column KII entity: {column_kii_entity}

### Response:
"""
    return prompt
