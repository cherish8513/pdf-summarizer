from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langsmith import traceable

from llm.config import config


class PdfRagClient:
    def __init__(self, llm, pdf_loader, embeddings):
        self.retriever = None
        self.llm = llm
        self.pdf_loader = pdf_loader
        self.embeddings = embeddings

    @traceable(name="pdf_load", tags=["pdf", "vectorize"])
    async def pdf_load(self, file_path: str):
        docs = self.pdf_loader.load_and_split_pdf(file_path)
        vector_store = FAISS.from_documents(documents=docs, embedding=self.embeddings)
        self.retriever = vector_store.as_retriever(
            search_type=config.search_type,
            search_kwargs={"k": config.default_k}
        )

    @traceable(name="invoke", tags=["query", "llm"])
    async def invoke(self, question: str) -> str:
        prompt = PromptTemplate.from_template(
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
                | prompt
                | self.llm
                | StrOutputParser()
        )

        result = await chain.ainvoke(question)
        return result
