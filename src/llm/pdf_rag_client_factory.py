import time
import threading
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

from llm.config import config
from llm.pdf_loader import PdfLoader
from llm.pdf_rag_client import PdfRagClient


class PdfRagClientFactory:
    _instance: 'PdfRagClient' = None
    _lock: threading.Lock = threading.Lock()  # 인스턴스 생성을 위한 락

    @staticmethod
    def create_pdf_rag_client() -> PdfRagClient:
        if PdfRagClientFactory._instance is None:
            with PdfRagClientFactory._lock:
                if PdfRagClientFactory._instance is None:
                    print(f"임베딩 모델 로딩 중: {config.embedding_model}")
                    start_time = time.time()

                    llm = ChatOpenAI(
                        openai_api_key=config.groq_api_key,
                        openai_api_base=config.groq_api_base,
                        model_name=config.llm_model,
                        temperature=config.llm_temperature,
                        max_tokens=config.llm_max_tokens,
                    )

                    embeddings = HuggingFaceEmbeddings(
                        model_name=config.embedding_model,
                        model_kwargs={'device': 'cpu', 'trust_remote_code': True},
                        encode_kwargs={'normalize_embeddings': True, 'batch_size': 8},
                        show_progress=True,
                    )

                    pdf_loader = PdfLoader(
                        text_splitter=RecursiveCharacterTextSplitter(
                            chunk_size=config.chunk_size,
                            chunk_overlap=config.chunk_overlap,
                            separators=["\n\n", "\n", ". ", " ", ""],
                            length_function=len,
                            is_separator_regex=False,
                        )
                    )

                    PdfRagClientFactory._instance = PdfRagClient(llm=llm, embeddings=embeddings, pdf_loader=pdf_loader)
                    print(f"임베딩 모델 로딩 완료 ({time.time() - start_time:.2f}초)")
                else:
                    print("다른 스레드에 의해 PdfRagClient 인스턴스가 이미 생성되었습니다.")
        else:
            print("PdfRagClient는 이미 생성되어 있는 싱글턴 인스턴스를 반환합니다.")
        return PdfRagClientFactory._instance
