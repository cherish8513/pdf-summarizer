from typing import AsyncGenerator

from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

from llm.config import config
from middlewares.langsmith_trace import langsmith_trace


class PdfNotLoadedError(Exception):
    pass


class PdfProcessingError(Exception):
    pass


class PdfRagClient:
    def __init__(self, llm, pdf_loader, embeddings):
        self.retriever = None
        self.llm = llm
        self.pdf_loader = pdf_loader
        self.embeddings = embeddings

    @langsmith_trace(name="pdf_load", tags=["pdf", "vectorize"])
    async def pdf_load(self, file_path: str):
        self.retriever = None
        try:
            docs = self.pdf_loader.load_and_split_pdf(file_path)

            if not docs:
                raise PdfProcessingError(f"'{file_path}' could not load or split")

            vector_store = FAISS.from_documents(documents=docs, embedding=self.embeddings)
            current_retriever = vector_store.as_retriever(
                search_type=config.search_type,
                search_kwargs={"k": config.default_k}
            )
            self.retriever = current_retriever
        except PdfProcessingError as pe:
            raise pe
        except Exception as e:
            raise PdfProcessingError(
                f"pdf '{file_path}' unexpected error occurred: {e}") from e

    # @langsmith_trace(name="stream", tags=["query", "llm", "stream"])
    async def stream(self, question: str) -> AsyncGenerator[str, None]:
        if self.retriever is None:
            raise PdfNotLoadedError("PDF file not loaded.")

        prompt_template = PromptTemplate.from_template(
            """You are an assistant for question-answering tasks.\n
            Use the following pieces of retrieved context to answer the question.\n
            If you don't know the answer, just say that you don't know. \n
            Answer in Korean.\n

            #Question: 
            {question} 
            #Context: 
            {context}

            #Answer: 
            """
        )

        chain = (
                {"context": self.retriever, "question": RunnablePassthrough()}
                | prompt_template
                | self.llm
                | StrOutputParser()
        )

        async for chunk in chain.astream(question):
            yield chunk