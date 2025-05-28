from llm.pdf_rag_client_factory import PdfRagClientFactory
from middlewares.langsmith_trace import langsmith_trace


class PdfChatService:
    def __init__(self, llm_client=None):
        self.llm_client = llm_client or PdfRagClientFactory.create_pdf_rag_client()

    @langsmith_trace(name="Upload Service", tags=["service", "upload"])
    async def upload(self, pdf_file_path: str):
        await self.llm_client.pdf_load(file_path=pdf_file_path)

    @langsmith_trace(name="Ask Service", tags=["service", "ask"])
    async def ask(self, question: str) -> dict[str, str]:
        response = await self.llm_client.invoke(question)
        return {"question": question, "answer": response}
