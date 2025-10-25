# generation_utils.py
import os
import torch
from utilities.prompt_register import (
    is_pii,
    PII_ENTITIES_LIST,
    is_sensitive_pii,
    prompt_non_pii,
    prompt_non_pii_table,
    prompt_reflect_non_pii,
    prompt_non_pii_table_no_isp,
)
from llm_model.model import Model


class SensitivityClassifier:
    """
    A unified class for text generation using different models.
    Supports OpenAI models, Hugging Face models, and custom fine-tuned models.
    """

    def __init__(self, model_name):
        """
        Initialize the text generator with the specified model.

        Args:
            model_name (str): Name or path of the model to use.
                Supported models: "gpt-4o-mini", "gemma-2-9b-it", or a path to a fine-tuned model
        """
        self.model_name = model_name
        self.model_instance = Model(model_name)
        self.model, self.tokenizer, self.client, self.model_type = (
            self.model_instance.get_model_components()
        )
        # Use the Model's generate function
        self.generate = self.model_instance.generate

    def _extract_sensitivity_level(self, prediction):
        if "non_" in prediction.split("SENSITIVE")[0].lower():
            return "NON_SENSITIVE"
        elif "moderate_" in prediction.split("SENSITIVE")[0].lower():
            return "MODERATE_SENSITIVE"
        elif "high_" in prediction.split("SENSITIVE")[0].lower():
            return "HIGH_SENSITIVE"
        elif "severe_" in prediction.split("SENSITIVE")[0].lower():
            return "SEVERE_SENSITIVE"
        else:
            return "No match"

    def classify_pii(self, column_name, sample_values, k=5):
        """
        Classify if a column contains PII based on its name and sample values.

        Args:
            column_name (str): The name of the column.
            sample_values (list): Sample values from the column.
            inference_config (dict): Configuration dict with model_name.
            is_pii (function): Function to generate the prompt.
            logging (module): Logging module or object.
            time (module): Time module.

        Returns:
            str: The predicted PII entity or 'None'.
        """
        if not any(
            char.isalpha() or char.isdigit()
            for value in sample_values
            for char in str(value)
        ):
            return "None"

        # Convert sample_values to list if not already
        sample_values_list = list(sample_values)
        prompt = is_pii(column_name, sample_values_list, k=k)
        # try:
        prediction = self.generate(prompt, max_new_tokens=128)

        # if 'gemma' in self.model_name or 'aya' in self.model_name:
        if "none" in prediction.lower():
            return "None"
        else:
            PII_ENTITIES_LIST.remove("AGE")
            # Add age to the back of the list
            PII_ENTITIES_LIST.append("AGE")
            for entity in PII_ENTITIES_LIST:
                if entity.lower() in prediction.lower():
                    return entity
        return prediction

    def classify_sensitive_pii(
        self, column_name, context, pii_entity, max_new_tokens=128
    ):
        """
        Classify if a column contains PII based on its name and sample values.

        Args:
            column_name (str): The name of the column.
            sample_values (list): Sample values from the column.
            inference_config (dict): Configuration dict with model_name.
            is_pii (function): Function to generate the prompt.
            logging (module): Logging module or object.
            time (module): Time module.

        Returns:
            str: The predicted PII entity or 'None'.
        """
        # If sample_values does not contain any letters or numbers, return 'None'
        if pii_entity == "None":
            return "NON_SENSITIVE"

        prompt = is_sensitive_pii(column_name, context, pii_entity)

        prediction = self.generate(prompt, max_new_tokens=max_new_tokens)

        if "non_sensitive" in prediction.lower():
            return "NON_SENSITIVE"
        elif "medium_sensitive" in prediction.lower():
            return "MEDIUM_SENSITIVE"
        elif "high_sensitive" in prediction.lower():
            return "HIGH_SENSITIVE"
        else:
            return prediction

    def classify_sensitive_non_pii(
        self, column_name, table_context, isp, max_new_tokens=256
    ):
        """
        Classify if a column contains PII based on its name and sample values.
        """
        prompt = prompt_non_pii(column_name, table_context, isp)

        try:
            if "gemma" in self.model_name:
                prediction = self.generate(prompt, max_new_tokens=256)
                # print(f"Prediction: {prediction}")
            elif "gemma" not in self.model_name:
                prediction = self.generate(prompt, max_new_tokens=max_new_tokens)
            else:
                prediction = "error"
            explanation = prediction

            if "gemma" in self.model_name:
                # Get the PII entity after 'Response' and before <eos>
                response_index = prediction.find("Response:")
                prediction = prediction[response_index + len("Response:") :].strip()
                explanation = prediction
            elif "aya" in self.model_name:
                # Get the PII entity after 'Response' and before <eos>
                response_index = prediction.find("<|CHATBOT_TOKEN|>")
                prediction = prediction[
                    response_index + len("<|CHATBOT_TOKEN|>") :
                ].strip()
                explanation = prediction

            if "non_sensitive" in prediction.lower():
                return "NON_SENSITIVE", explanation
            elif "medium" in prediction.lower():
                return "MEDIUM_SENSITIVE", explanation
            elif "high" in prediction.lower():
                return "HIGH_SENSITIVE", explanation
            elif "severe_sensitive" in prediction.lower():
                return "SEVERE_SENSITIVE", explanation
            else:
                return "ERROR_GENERATION", "error generating"

        except Exception as e:
            print(f"Error: {e}")
            return "ERROR_GENERATION", prediction

    def classify_sensitive_non_pii_table(
        self, table_context, isp=None, max_new_tokens=512
    ):
        """
        Classify if a column contains PII based on its name and sample values.
        """
        if isp:
            prompt = prompt_non_pii_table(table_context, isp)
        else:
            prompt = prompt_non_pii_table_no_isp(table_context)

        prediction = "<>"

        try:
            prediction = self.generate(prompt, max_new_tokens=max_new_tokens)
            sensitivity_level = self._extract_sensitivity_level(prediction)
            return sensitivity_level, prediction

        except Exception as e:
            print(f"Error: {e}")
            return "ERROR_GENERATION", prediction
