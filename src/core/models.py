from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from abc import ABC, abstractmethod
import json
import requests


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
