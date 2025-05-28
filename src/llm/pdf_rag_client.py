import time

from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import traceable

from llm.config import config
from llm.pdf_loader import PdfLoader


class PdfSummarizerClient:
    def __init__(self):
        self.retriever = None
        self.llm = ChatOpenAI(
            openai_api_key=config.groq_api_key,
            openai_api_base=config.groq_api_base,
            model=config.llm_model,
            temperature=config.llm_temperature,
            max_tokens=config.llm_max_tokens,
        )
        self.pdf_loader = PdfLoader(
            text_splitter=RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
        )

        print(f"임베딩 모델 로딩 중: {config.embedding_model}")
        start_time = time.time()

        # 임베딩 모델 최적화 설정
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.embedding_model,
            model_kwargs={
                'device': 'cpu',
                'trust_remote_code': True,
                'torch_dtype': 'float32',
            },
            encode_kwargs={
                'normalize_embeddings': True,
                'batch_size': 8,
                'show_progress_bar': True,
                'convert_to_numpy': True,
            }
        )

        print(f"임베딩 모델 로딩 완료 ({time.time() - start_time:.2f}초)")

    @traceable("pdf_load")
    async def pdf_load(self, file_path: str):
        docs = self.pdf_loader.load_and_split_pdf(file_path)
        vector_store = FAISS.from_documents(documents=docs, embedding=self.embeddings)
        self.retriever = vector_store.as_retriever(
            search_type=config.search_type,
            search_kwargs={"k": config.default_k}
        )

    @traceable("invoke")
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
