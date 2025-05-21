import os
import re

from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from pdf_summarizer.models.schemas import SummaryResponse
from pdf_summarizer.models.type import Type

load_dotenv()

model_id = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_id, token=os.getenv("HUGGINGFACE_ACCESS_TOKEN"))
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", token=os.getenv("HUGGINGFACE_ACCESS_TOKEN"))
hf_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=300)
llm = HuggingFacePipeline(pipeline=hf_pipeline)

def pdf_summarize(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(pages)

    prompt_template = f"""[INST] 다음 내용을 요약해줘:\n{Type.TEXT.value}\n[/INST]"""
    prompt = PromptTemplate(input_variables=[Type.TEXT.value], template=prompt_template)

    summaries = []
    for chunk in chunks:
        chunk_text = chunk.page_content
        formatted_prompt = prompt.format(text=chunk_text)
        summary = llm(formatted_prompt)
        summaries.append(summary.strip())

    # 전체 요약 결합
    combined_summary = "\n\n".join([f"부분 {i+1} 요약:\n{summary}" for i, summary in enumerate(summaries)])

    # 전체 문서 요약 생성
    final_prompt_template = f"""[INST] 다음은 문서의 각 부분 요약입니다:\n{Type.TEXT.value}\n\n전체 문서의 내용을 500단어 이내로 요약하고, 제목을 추정하고, 주요 키워드 5개를 추출해줘.\n응답 형식:\n제목: [제목]\n요약: [요약 내용]\n키워드: [키워드1, 키워드2, 키워드3, 키워드4, 키워드5] [/INST]"""
    final_prompt = PromptTemplate(input_variables=[Type.TEXT.value], template=final_prompt_template)
    formatted_final_prompt = final_prompt.format(text=combined_summary)

    final_output = llm(formatted_final_prompt)
    final_text = final_output.strip()

    title_match = re.search(r'제목:\s*(.*?)(?=\n요약:|$)', final_text, re.DOTALL)
    summary_match = re.search(r'요약:\s*(.*?)(?=\n키워드:|$)', final_text, re.DOTALL)
    keywords_match = re.search(r'키워드:\s*(.*?)$', final_text, re.DOTALL)

    title = title_match.group(1).strip() if title_match else "제목 없음"
    summary = summary_match.group(1).strip() if summary_match else final_text
    keywords = [kw.strip() for kw in re.sub(r'[\[\]]', '', keywords_match.group(1)).split(',')] if keywords_match else []

    return SummaryResponse(
        title=title,
        summary=summary,
        keywords=keywords
    )
