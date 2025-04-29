"""A module that contains Translator class that performs text translation using LLM"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Callable, Union, Literal, Type

import requests
from datasets import Dataset
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from src.utils.batching_utils import *


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients"""

    @abstractmethod
    def initialize(self, config: dict) -> None:
        """Initialize the client with configuration"""
        pass

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate response from the model"""
        pass


class OpenAIClient(BaseLLMClient):
    """Client for OpenAI models"""

    def __init__(self):
        self.model = None
        self.chain = None

    def initialize(self, config: dict) -> None:
        """Initialize OpenAI client"""
        self.model = ChatOpenAI(
            base_url=config["base_url"],
            api_key=config["api_key"],
            model=config["model"],
        )

    def set_chain(self, system_template: str) -> None:
        """Set up the processing chain"""
        parser = StrOutputParser()
        prompt_template = ChatPromptTemplate.from_messages(
            [("system", system_template), ("user", "{text}")]
        )
        self.chain = prompt_template | self.model | parser

    def generate(self, prompt: str) -> str:
        """Generate response using OpenAI model"""
        return self.chain.invoke({"text": prompt}).strip("`").strip("json").strip()


class YandexGPTClient(BaseLLMClient):
    """Client for Yandex GPT models"""

    def __init__(self):
        self.config = None
        self.system_message = None

    def initialize(self, config: dict) -> None:
        """Initialize Yandex GPT client"""
        self.config = config

    def set_system_message(self, system_message: str) -> None:
        """Set the system message for Yandex GPT"""
        self.system_message = system_message

    def generate(self, prompt: str) -> str:
        """Generate response using Yandex GPT"""
        request_data = {
            "modelUri": self.config["model_uri"],
            "completionOptions": {
                "stream": False,
                "temperature": 0.6,
                "maxTokens": "20000"
            },
            "messages": [
                {
                    "role": "system",
                    "text": self.system_message
                },
                {
                    "role": "user",
                    "text": prompt
                }
            ]
        }

        response = requests.post(
            self.config["url"],
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Api-Key {self.config['api_key']}"
            },
            json=request_data
        )
        response.raise_for_status()
        result = json.loads(response.text)
        return result["result"]["alternatives"][0]["message"]["text"].strip("`").strip('\n')


class Translator:
    """
    A class responsible for translating batches of text data using different language models.
    """

    def __init__(
        self,
        system_message: str,
        model_config: dict,
        example_data: dict,
        batch_size: int,
        batch_result_dir: str,
        batch_dir: str,
        model_type: Literal["openai", "yandex_gpt"] = "openai"
    ):
        """Initializes the Translator with configuration parameters.

        Args:
            system_message (str): System prompt template for the translation task.
            model_config (dict): Configuration dictionary for model initialization.
            example_data (dict): Example data for prompt construction.
            batch_size (int): Size of processing batches.
            batch_result_dir (str): Directory to save batch results.
            batch_dir (str): Directory containing input batches.
            model_type (str): Type of model to use ("openai" or "yandex_gpt").
        """
        self.batch_size = batch_size
        self.batch_result_dir = batch_result_dir
        self.batch_dir = batch_dir

        self.client = self._initialize_client(model_type, model_config)
        self._prepare_prompt_template(example_data, system_message)

    def _initialize_client(
        self,
        model_type: str,
        config: dict
    ) -> BaseLLMClient:
        """Initializes the appropriate LLM client.

        Args:
            model_type: Type of model ("openai" or "yandex_gpt")
            config: Model configuration dictionary

        Returns:
            Initialized LLM client instance

        Raises:
            ValueError: If unsupported model type is specified
        """
        if model_type == "openai":
            client = OpenAIClient()
        elif model_type == "yandex_gpt":
            client = YandexGPTClient()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        client.initialize(config)
        return client

    def _prepare_prompt_template(
        self,
        examples_data: dict,
        system_message: str
    ) -> None:
        """
        Prepares the prompt template by combining the system message and examples.

        Args:
            examples_data (dict): Dictionary containing example inputs and outputs.
            system_message (str): Base system instruction text.
        """
        system_template = system_message
        examples = examples_data["examples"]
        system_template += "\n".join(
            f"Example Input: {json.dumps(example['input'], ensure_ascii=False, indent=2).replace('{', '{{').replace('}', '}}')}\n"
            f"Example Result: {json.dumps(example['result'], ensure_ascii=False, indent=2).replace('{', '{{').replace('}', '}}')}"
            for example in examples
        )

        if isinstance(self.client, OpenAIClient):
            self.client.set_chain(system_template)
        elif isinstance(self.client, YandexGPTClient):
            self.client.set_system_message(system_template)

    def translate(
        self,
        input_dataset: Dataset,
        ResultSchema: Type[BaseModel],
        max_retries: int = 3,
        retry_delay: int = 2
    ) -> list:
        """Processes all batches through the translation pipeline.

        Handles each input with retry logic, validates outputs against schema,
        and manages intermediate saving of results.

        Args:
            input_dataset (Dataset): Dataset containing texts to be translated.
            ResultSchema (BaseModel): A schema to check for the results.
            max_retries (int): Maximum number of retries to translate.
            retry_delay (int): Time delay between retries.

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

            formatted_inputs = [
                f"{i+1}. Input: {json.dumps(example, ensure_ascii=False)}"
                for i, example in enumerate(batch)
            ]
            user_input = "\n\n".join(
                formatted_inputs) + "\n\nReturn the result as a list of JSON objects, one per input, preserving the order."

            success = False
            attempt = 0
            while not success and attempt < max_retries:
                try:
                    res = self.client.generate(user_input)
                    result_list = json.loads(res)
                    assert isinstance(
                        result_list, list), "Expected a list of JSON results."

                    validated_results = []
                    for result_json in result_list:
                        result_json["id"] = str(result_json["id"])
                        ResultSchema.parse_obj(result_json)
                        validated_results.append(result_json)

                    batch_results = validated_results
                    success = True

                except (json.JSONDecodeError, ValueError, AssertionError, requests.exceptions.RequestException) as e:
                    attempt += 1
                    logging.warning(
                        f"Error on batch #{batch_idx}: {str(e)}. Attempt {attempt} of {max_retries}. Retrying..."
                    )
                    time.sleep(retry_delay)
                except Exception as e:
                    attempt += 1
                    logging.error(
                        f"Unexpected error on batch #{batch_idx}: {str(e)}. Attempt {attempt} of {max_retries}. Retrying..."
                    )
                    time.sleep(retry_delay)

            list_of_results.extend(batch_results)

            if self.batch_size > 0:
                batch_result_dir = create_directory_if_not_exists(
                    self.batch_result_dir)
                save_batch_to_json(
                    batch=Dataset.from_list(batch_results),
                    base_filename="model_result",
                    batch_num=batch_idx,
                    save_dir=batch_result_dir,
                )

        return list_of_results
