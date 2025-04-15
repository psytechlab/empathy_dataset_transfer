from pydantic import BaseModel


class GeneralTranslationResultSchema(BaseModel):
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


class RationaleTranslationResultSchema(BaseModel):
    """
    Pydantic schema for validating the structure of rationales translation results.

    Attributes:
        id (str): Unique identifier for the translation item.
        text_eng (str): Original input text.
        text_rus (str): Translated text in Russian.
        rationales_eng (str): Original rationales text.
        rationales_rus (str): Translated rationales in Russian.
    """
    id: str
    text_rus: str
    rationales_eng: str
    rationales_rus: str
    text_eng: str
