"""A module that contains Translator class that performs text tanslation using LLM"""

import logging
import time
from typing import Callable

from datasets import Dataset
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from src.utils.batching_utils import *


class Translator:
    """
    A class responsible for translating batches of text data using a language model.
    """

    def __init__(
        self,
        system_message: str,
        model_config: dict,
        example_data: dict,
        batch_size: int,
        batch_result_dir: str,
        batch_dir: str,
    ):
        """Initializes the Translator with configuration parameters.

        Args:
            system_message (str): System prompt template for the translation task.
            model_config (dict): Configuration dictionary for model initialization.
            example_data (dict): Example data for prompt construction.
            batch_size (int): Size of processing batches.
            batch_result_dir (str): Directory to save batch results.
            batch_dir (str): Directory containing input batches.
        """
        self.batch_size = batch_size
        self.batch_result_dir = batch_result_dir
        self.batch_dir = batch_dir

        self.model = self._initialize_model(model_config)
        self.chain = self._get_chain(example_data, system_message)

    def _initialize_model(self, config: dict) -> ChatOpenAI:
        """Initializes the ChatOpenAI language model.

        Args:
            config (dict): Model configuration containing:
                - base_url (str): API endpoint URL
                - api_key (str): Authentication key
                - model (str): Model identifier

        Returns:
            ChatOpenAI: Initialized language model instance.
        """
        model = ChatOpenAI(
            base_url=config["base_url"],
            api_key=config["api_key"],
            model=config["model"],
        )
        return model

    def _prepate_system_template(self, examples_data: dict, system_message: str) -> str:
        """
        Prepares the prompt template by combining the system message and examples.

        Args:
            examples_data (dict): Dictionary containing example inputs and outputs.
            system_message (str): Base system instruction text.

        Returns:
            str: Complete system prompt combining instructions and examples.
        """
        system_template = system_message
        examples = examples_data["examples"]
        system_template += "\n".join(
            f"Example Input: {json.dumps(example['input'], ensure_ascii=False, indent=2).replace('{', '{{').replace('}', '}}')}\n"
            f"Example Result: {json.dumps(example['result'], ensure_ascii=False, indent=2).replace('{', '{{').replace('}', '}}')}"
            for example in examples
        )

        return system_template

    def _get_chain(self, examples_data: dict, system_message: str) -> Callable:
        """Creates the processing chain combining prompt, model and parser.

        Args:
            examples_data (dict): Example data for prompt construction.
            system_message (str): System instruction text.

        Returns:
            Callable: Configured processing chain ready for invocation.
        """
        system_template = self._prepate_system_template(
            examples_data, system_message)
        parser = StrOutputParser()
        prompt_template = ChatPromptTemplate.from_messages(
            [("system", system_template), ("user", "{text}")]
        )

        return prompt_template | self.model | parser

    def translate(self, input_dataset: Dataset, ResultSchema: BaseModel) -> list:
        """Processes all batches through the translation pipeline.

        Handles each input with retry logic, validates outputs against schema,
        and manages intermediate saving of results.

        Args:
            input_dataset (Dataset): Dataset containing texts to be translated.
            ResultSchema (BaseModel): A schema to check for the results.

        Returns:
            list: List of all successfully processed translation results.

        Raises:
            ValueError: If response format is invalid after max retries.
            Exception: For unexpected errors during processing.
        """
        batched_input_dataset = split_into_batches(
            input_dataset, batch_dir=self.batch_dir, batch_size=self.batch_size
        )
        list_of_results = []
        for batch_idx, batch in enumerate(batched_input_dataset):
            print(
                f"Processing batch {batch_idx + 1}/{len(batched_input_dataset)}...")

            batch_results = []

            for i, input_example in enumerate(batch):
                user_input = "Input: {0}\nResult:".format(
                    json.dumps(input_example, ensure_ascii=False)
                )
                # получаем id примера или создаем временный
                example_id = input_example.get("id", f"example_{i}")

                success = False
                attempt = 0
                max_retries = 3
                retry_delay = 2

                while not success and attempt < max_retries:
                    try:
                        res = (
                            self.chain.invoke({"text": user_input})
                            .strip("`")
                            .strip("json")
                            .strip()
                        )

                        if not res or not res.startswith("{"):
                            raise ValueError(
                                f"Unexpected response format: {res}")
                        result_json = json.loads(res)

                        # Приведение ID к строке
                        result_json["id"] = str(result_json["id"])

                        # Проверка схемы
                        ResultSchema.parse_obj(result_json)

                        batch_results.append(result_json)
                        success = True
                    except (json.JSONDecodeError, ValueError) as e:
                        attempt += 1
                        logging.warning(
                            f"Error on input #{i} (id: {example_id}): {str(e)}. Attempt {attempt} of {max_retries}. Retrying..."
                        )
                        time.sleep(retry_delay)
                    except Exception as e:
                        attempt += 1
                        logging.error(
                            f"Unexpected error on input #{i} (id: {example_id}): {str(e)}. Attempt {attempt} of {max_retries}. Retrying..."
                        )
                        time.sleep(retry_delay)

            # Добавляем результаты текущего батча к общим результатам
            list_of_results.extend(batch_results)

            # Сохраняем результаты батча только если используется разбиение на батчи
            if self.batch_size > 0:
                batch_result_dir = create_directory_if_not_exists(
                    self.batch_result_dir)
                save_batch_to_json(
                    batch=Dataset.from_list(batch_results),
                    base_filename="model_result",
                    batch_num=batch_idx,
                    save_dir=batch_result_dir,
                )  # Сохраняем промежуточные результаты
        return list_of_results
