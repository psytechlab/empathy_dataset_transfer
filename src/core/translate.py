import sys
import os
import re
current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, '../../'))
sys.path.append(current_dir)
sys.path.append(parent_dir)
sys.path.append(current_dir + '/src')
sys.path.append(current_dir + '/src/core')
sys.path.append(current_dir + '/src/utils')
from typing import Callable
from datetime import datetime
import configparser
from src.utils.batching_utils import *
import logging
import pandas as pd
from pydantic import BaseModel
from datasets import Dataset
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
import time
from pathlib import Path


class ResultSchema(BaseModel):
    """
    Pydantic schema for validating the structure of translation results.

    Attributes:
        id (str): Unique identifier for the translation item.
        text (str): Original input text.
        text_rus (str): Translated text in Russian.
    """
    id: str
    text: str
    text_rus: str


class Translator:
    """
    A class responsible for translating batches of text data using a language model.

    Attributes:
    ----------
        model : ChatOpenAI
            The initialized language model for translation.
        chain : 
            The chained prompt and model processing logic.
        batch_size : int 
            Number of items in each batch.
        batch_result_dir : str
            Directory to save batch translation results.
        batch_dir : str
            Directory containing input batches.
        intermediate_path : str
            File path to save intermediate results.
        list_of_results : list[str] 
            List to hold final translated results.
        batched_input_dataset : list[str]
            List of input data batches for translation.
    """

    def __init__(self, cfg: dict):
        """
        Initializes the Translator object using configuration values.

        Parameters:
        ----------
            cfg : dict
                Configuration dictionary with paths and parameters.
        """
        system_message = Path(cfg['prompt_path']).open().read()
        model_config = json.load(Path(cfg['model_config_path']).open())
        example_data = json.load(Path(cfg['filepath_examples']).open())

        self.intermediate_path = f'int_path_{model_config["model"]}_{datetime.now()}.json'
        self.batch_size = int(cfg['batch_size'])
        self.batch_result_dir = cfg['batch_result_dir']
        self.batch_dir = cfg['batches']

        data = pd.read_csv(cfg['data_path'])
        dataset = {colname: data[colname].tolist()
                   for colname in cfg['cols'].split()}
        input_dataset = Dataset.from_dict(dataset)

        self.model = self._initialize_model(model_config)
        self.chain = self._prepate_system_template(
            example_data, system_message)

        self.batched_input_dataset = split_into_batches(
            input_dataset, batch_dir=self.batch_dir, batch_size=self.batch_size)

    def _initialize_model(self, config: dict) -> ChatOpenAI:
        """
        Initializes and returns the language model using the given config.

        Parameters:
        ----------
        config : dict
            Model configuration including base URL, API key, and model name.

        Returns:
        ----------
        ChatOpenAI
            An instance of the language model.
        """
        model = ChatOpenAI(
            base_url=config["base_url"],
            api_key=config["api_key"],
            model=config["model"]
        )
        return model

    def _prepate_system_template(
            self,
            examples_data: dict,
            system_message: str) -> Callable:
        """
        Prepares the prompt template by combining the system message and examples.

        Parameters:
        ----------
        examples_data : dict
            Dictionary containing input-output examples for the prompt.
        system_message : str
            System instruction text.

        Returns:
        ----------
        Callable
            A LangChain chain combining the prompt, model, and parser.
        """
        parser = StrOutputParser()

        system_template = system_message
        examples = examples_data["examples"]
        system_template += "\n".join(
            f"Example Input: {json.dumps(example['input'], ensure_ascii=False, indent=2).replace('{', '{{').replace('}', '}}')}\n"
            f"Example Result: {json.dumps(example['result'], ensure_ascii=False, indent=2).replace('{', '{{').replace('}', '}}')}"
            for example in examples
        )

        prompt_template = ChatPromptTemplate.from_messages(
            [("system", system_template), ("user", "{text}")]
        )

        return (prompt_template | self.model | parser)

    def _contains_english(self, text: str) -> bool:
        """
        Checks if the given text contains English words.

        Parameters:
        ----------
        text: str 
            Text to check.

        Returns:
        ----------
        bool
            True if English words are found, False otherwise.
        """
        return bool(re.search(r'\b[a-zA-Z]{2,}\b', text))

    def translate(self):
        """
        Translates all batches in the dataset using the model.

        Processes each input in the dataset with retry logic, validates and stores
        results, and saves intermediate outputs to disk.
        """
        self.list_of_results = []
        for batch_idx, batch in enumerate(self.batched_input_dataset):
            print(
                f"Processing batch {batch_idx + 1}/{len(self.batched_input_dataset)}...")

            batch_results = []

            for i, input_example in enumerate(batch):
                user_input = 'Input: {0}\nResult:'.format(
                    json.dumps(input_example, ensure_ascii=False))
                # получаем id примера или создаем временный
                example_id = input_example.get("id", f"example_{i}")

                success = False
                attempt = 0
                max_retries = 3
                retry_delay = 2

                while not success and attempt < max_retries:
                    try:
                        res = self.chain.invoke({"text": user_input}).strip(
                            '`').strip('json').strip()

                        if not res or not res.startswith("{"):
                            raise ValueError(
                                f"Unexpected response format: {res}")
                        result_json = json.loads(res)

                        # Приведение ID к строке
                        result_json["id"] = str(result_json["id"])

                        # Проверка схемы
                        ResultSchema.parse_obj(result_json)

                        if self._contains_english(result_json["text_rus"]):
                            raise ValueError(
                                "Translation contains English words, retrying...")

                        batch_results.append(result_json)
                        success = True
                    except (json.JSONDecodeError, ValueError) as e:
                        attempt += 1
                        logging.warning(
                            f"Error on input #{i} (id: {example_id}): {str(e)}. Attempt {attempt} of {max_retries}. Retrying...")
                        time.sleep(retry_delay)
                    except Exception as e:
                        attempt += 1
                        logging.error(
                            f"Unexpected error on input #{i} (id: {example_id}): {str(e)}. Attempt {attempt} of {max_retries}. Retrying...")
                        time.sleep(retry_delay)

            # Добавляем результаты текущего батча к общим результатам
            self.list_of_results.extend(batch_results)

            # Сохраняем результаты батча только если используется разбиение на батчи
            if self.batch_size > 0:
                batch_result_dir = create_directory_if_not_exists(
                    self.batch_result_dir)
                save_batch_to_json(
                    batch=Dataset.from_list(batch_results),
                    base_filename="model_result",
                    batch_num=batch_idx,
                    save_dir=batch_result_dir)  # Сохраняем промежуточные результаты

        if self.intermediate_path:
            d = Dataset.from_list(self.list_of_results)
            with open(self.intermediate_path, "w", encoding="utf-8") as f:
                json.dump(d.to_list(), f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('conf.yaml')

    tr = Translator(config['DEFAULT'])
    tr.translate()
