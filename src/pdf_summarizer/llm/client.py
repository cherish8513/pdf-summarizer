import asyncio
from typing import List

from langchain.schema import HumanMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langsmith import traceable

from pdf_summarizer.llm.config import get_settings


class LLMClient:
    def __init__(self):
        settings = get_settings()
        self.llm = ChatOpenAI(
            openai_api_key=settings.groq_api_key,
            openai_api_base=settings.groq_api_base,
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )

    @traceable(name="llm_ask")
    async def ask(self, messages: List[BaseMessage]) -> str:
        try:
            response = await asyncio.to_thread(
                self.llm.invoke,
                messages
            )
            return response.content.strip()
        except Exception as e:
            raise Exception(f"LLM ask failed: {str(e)}") from e

    @traceable(name="llm_ask_simple")
    async def ask_simple(self, prompt: str) -> str:
        message = HumanMessage(content=prompt)
        return await self.ask([message])
