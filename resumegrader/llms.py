from typing import Any, Dict

from langchain_cohere import ChatCohere
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from .exceptions import MissingCredentialsError, UnsupportedLLMError


def load_llm(llmconfig: Dict[str, Any]):
    llmconfig["temperature"] = 0
    keys = list(filter(lambda x: "api_key" in x.lower(), llmconfig.keys()))
    if len(keys) == 0:
        raise MissingCredentialsError("Could not find an API key")

    if len(keys) > 1:
        raise ValueError(f"Found {len(keys)} api keys. Please provide only one.")

    match keys[0].upper():
        case "GOOGLE_API_KEY":
            llm = ChatGoogleGenerativeAI(**llmconfig)
        case "OPENAI_API_KEY":
            llm = ChatOpenAI(**llmconfig)
        case "COHERE_API_KEY":
            llm = ChatCohere(**llmconfig)
        case _:
            raise UnsupportedLLMError(
                f"Only {', '.join(['Google', 'OpenAI', 'Cohere'])} are supported in this version. Got `{keys[0].upper()}` instead."
            )

    return llm
