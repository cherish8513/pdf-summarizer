import asyncio
from typing import AsyncGenerator

from llm.pdf_rag_client import PdfRagClient
from llm.pdf_rag_client_factory import PdfRagClientFactory
from middlewares.langsmith_trace import langsmith_trace


class PdfChatService:
    def __init__(self, llm_client: PdfRagClient = None):
        self.llm_client = llm_client or PdfRagClientFactory.create_pdf_rag_client()

    @langsmith_trace(name="Upload Service", tags=["service", "upload"])
    async def upload(self, pdf_file_path: str):
        await self.llm_client.pdf_load(file_path=pdf_file_path)
        return {"message": f"'{pdf_file_path}' uploaded and processed successfully."}

    # @langsmith_trace(name="Ask Stream Service", tags=["service", "ask", "stream"])
    async def question(self, question_text: str) -> AsyncGenerator[str, None]:
        buffer = ""
        async for chunk in self.llm_client.stream(question_text):
            buffer += chunk

            if len(buffer) >= 10 or chunk.endswith(('.', '!', '?', '\n')):
                yield f"data: {buffer}\n\n"
                buffer = ""
                await asyncio.sleep(0.05)  # 50ms 지연

        if buffer:
            yield f"data: {buffer}\n\n"