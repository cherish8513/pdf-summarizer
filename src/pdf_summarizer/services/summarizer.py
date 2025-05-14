import os
import re

from dotenv import load_dotenv
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from src.pdf_summarizer.models.schemas import SummaryResponse

load_dotenv()

llm = ChatOpenAI(
    temperature=os.getenv("OPENAI_TEMPERATURE"),
    model_name=os.getenv("OPENAI_MODEL_NAME"),  # 모델명
)

embeddings = OpenAIEmbeddings()


def pdf_summarize(file_path: str) -> SummaryResponse:
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    chunks = text_splitter.split_documents(pages)

    vector_store = FAISS.from_documents(chunks, embeddings)

    map_prompt_template = """다음은 긴 문서의 일부입니다:
    {text}

    이 부분의 핵심 내용을 100단어 이내로 요약해주세요.
    """

    map_prompt = PromptTemplate(
        template=map_prompt_template,
        input_variables=["text"]
    )

    combine_prompt_template = """다음은 문서의 각 부분에 대한 요약입니다:
    {text}

    이 요약들을 바탕으로 전체 문서의 내용을 500단어 이내로 요약하고, 문서의 제목을 추정하여 제공해주세요.
    또한 문서의 주요 키워드 5개를 추출해주세요.

    다음 형식으로 응답해주세요:
    제목: [문서의 추정 제목]
    요약: [문서의 요약 내용]
    키워드: [키워드1, 키워드2, 키워드3, 키워드4, 키워드5]
    """

    combine_prompt = PromptTemplate(
        template=combine_prompt_template,
        input_variables=["text"]
    )

    chain = load_summarize_chain(
        llm,
        chain_type="map_reduce",
        map_prompt=map_prompt,
        combine_prompt=combine_prompt,
        verbose=False
    )

    result = chain.run(chunks)

    title_match = re.search(r'제목:\s*(.*?)(?=\n요약:|$)', result, re.DOTALL)
    summary_match = re.search(r'요약:\s*(.*?)(?=\n키워드:|$)', result, re.DOTALL)
    keywords_match = re.search(r'키워드:\s*(.*?)$', result, re.DOTALL)

    title = title_match.group(1).strip() if title_match else "요약 문서"
    summary = summary_match.group(1).strip() if summary_match else result

    keywords_str = keywords_match.group(1).strip() if keywords_match else ""
    keywords_str = re.sub(r'[\[\]]', '', keywords_str)
    keywords = [k.strip() for k in keywords_str.split(',')]

    return SummaryResponse(
        title=title,
        summary=summary,
        keywords=keywords
    )
