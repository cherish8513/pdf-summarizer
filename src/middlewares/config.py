import os
from typing import Optional, List
from dotenv import load_dotenv
from fastapi import Request, UploadFile
from langchain_core.runnables.config import RunnableConfig

load_dotenv()


class Settings:
    def __init__(self):
        self.langsmith_tracing = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
        self.langsmith_endpoint = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
        self.langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
        self.langsmith_project = os.getenv("LANGSMITH_PROJECT", "pdf-summarizer")

        self.debug = os.getenv("DEBUG", "false").lower() == "true"
        self.log_level = os.getenv("LOG_LEVEL", "INFO").upper()

settings = Settings()