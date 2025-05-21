import os
import re
from dotenv import load_dotenv
from langchain.schema import HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI

from pdf_summarizer.models.schemas import SummaryResponse

load_dotenv()

llm = ChatOpenAI(
    openai_api_key=os.getenv("GROQ_API_KEY"),
    openai_api_base="https://api.groq.com/openai/v1",
    model="llama-3.3-70b-versatile",
    temperature=0.7,
    max_tokens=500,
)


def pdf_summarize(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    if not pages or all(not page.page_content.strip() for page in pages):
        return SummaryResponse(
            title="빈 문서",
            summary="PDF 파일이 비어있거나 텍스트를 추출할 수 없습니다.",
            keywords=["빈 문서"]
        )

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(pages)

    if not chunks:
        return SummaryResponse(
            title="처리 불가능한 문서",
            summary="PDF에서 분석 가능한 텍스트를 추출할 수 없습니다.",
            keywords=["처리 불가"]
        )

    summaries = []
    for chunk in chunks:
        chunk_text = chunk.page_content
        if not chunk_text.strip():
            continue

        prompt = f"[INST] 다음 내용을 요약해줘:\n{chunk_text}\n[/INST]"
        response = llm.invoke([HumanMessage(content=prompt)])
        summaries.append(response.content.strip())

    if not summaries:
        return SummaryResponse(
            title="요약 실패",
            summary="문서 내용을 요약하는 데 실패했습니다.",
            keywords=["요약 실패"]
        )

    combined_summary = "\n\n".join([f"부분 {i + 1} 요약:\n{summary}" for i, summary in enumerate(summaries)])

    final_prompt = f"[INST] 다음은 문서의 각 부분 요약입니다:\n{combined_summary}\n\n전체 문서의 내용을 500단어 이내로 요약하고, 제목을 추정하고, 주요 키워드 5개를 추출해줘.\n응답 형식:\n제목: [제목]\n요약: [요약 내용]\n키워드: [키워드1, 키워드2, 키워드3, 키워드4, 키워드5] [/INST]"

    final_response = llm.invoke([HumanMessage(content=final_prompt)])
    final_text = final_response.content.strip()

    title_match = re.search(r'제목:\s*(.*?)(?=\n요약:|$)', final_text, re.DOTALL)
    summary_match = re.search(r'요약:\s*(.*?)(?=\n키워드:|$)', final_text, re.DOTALL)
    keywords_match = re.search(r'키워드:\s*(.*?)\s*(?:\[/INST\])?$', final_text, re.DOTALL)  # [/INST] 태그 제거

    title = title_match.group(1).strip() if title_match else "제목 없음"
    summary = summary_match.group(1).strip() if summary_match else final_text

    keywords_str = keywords_match.group(1).strip() if keywords_match else ""
    keywords_str = re.sub(r'[\[\]/INST]', '', keywords_str)
    keywords = [kw.strip() for kw in keywords_str.split(',') if kw.strip()]

    return SummaryResponse(
        title=title,
        summary=summary,
        keywords=keywords
    )