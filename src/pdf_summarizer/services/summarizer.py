import asyncio
import re
from pathlib import Path
from typing import List, Optional, Dict, Any, Coroutine

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnableSerializable, RunnablePassthrough
)

from pdf_summarizer.llm.client import LLMClient
from pdf_summarizer.models.schemas import SummaryResponse


class PDFSummarizer:
    def __init__(self,
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 max_chunks: int = 50):

        self.llm_client = LLMClient()
        self.max_chunks = max_chunks
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    class PDFSummarizerError(Exception):
        pass

    class SummaryOutputParser(BaseOutputParser[SummaryResponse]):
        def parse(self, text: str) -> SummaryResponse:
            title_pattern = r'제목\s*[:：]\s*([^\n]+)'
            summary_pattern = r'요약\s*[:：]\s*(.*?)(?=\n\s*키워드\s*[:：]|$)'
            keywords_pattern = r'키워드\s*[:：]\s*([^\n\[\]]+)'

            title_match = re.search(title_pattern, text, re.IGNORECASE | re.DOTALL)
            summary_match = re.search(summary_pattern, text, re.IGNORECASE | re.DOTALL)
            keywords_match = re.search(keywords_pattern, text, re.IGNORECASE | re.DOTALL)

            title = "제목 없음"
            if title_match:
                title = title_match.group(1).strip()
                title = re.sub(r'[^\w\s가-힣]', '', title)[:100]

            summary = text.strip()
            if summary_match:
                summary = summary_match.group(1).strip()
                summary = re.sub(r'\[/?INST\]', '', summary)
                summary = re.sub(r'^\s*[-*]\s*', '', summary, flags=re.MULTILINE)

            keywords = []
            if keywords_match:
                keywords_str = keywords_match.group(1).strip()
                keywords_str = re.sub(r'[\[\]"\']', '', keywords_str)
                keywords = [
                               kw.strip()
                               for kw in re.split(r'[,、]', keywords_str)
                               if kw.strip() and len(kw.strip()) > 1
                           ][:10]

            if not keywords:
                keywords = ["문서 요약"]

            return SummaryResponse(
                title=title or "제목 없음",
                summary=summary or "요약을 생성할 수 없습니다.",
                keywords=keywords
            )

    async def _call_llm(self, prompt_or_text: Any) -> str:
        try:
            if hasattr(prompt_or_text, 'text'):
                text = prompt_or_text.text
            elif isinstance(prompt_or_text, str):
                text = prompt_or_text
            else:
                text = str(prompt_or_text)

            result = await asyncio.wait_for(
                self.llm_client.invoke(text),
                timeout=120
            )

            if not result or not result.strip():
                raise self.PDFSummarizerError("LLM returned empty response")

            return result.strip()

        except asyncio.TimeoutError:
            raise self.PDFSummarizerError("Timeout while calling LLM")
        except Exception as e:
            raise self.PDFSummarizerError(f"Failed to call LLM: {str(e)}") from e

    async def _load_pdf_documents(self, file_path: str) -> List[Document]:
        try:
            pdf_path = Path(file_path)
            if not pdf_path.exists():
                raise self.PDFSummarizerError(f"PDF file not found: {file_path}")

            if not pdf_path.suffix.lower() == '.pdf':
                raise self.PDFSummarizerError(f"File is not a PDF: {file_path}")

            loader = PyPDFLoader(file_path)
            pages = await asyncio.to_thread(loader.load)

            if not pages:
                raise self.PDFSummarizerError("PDF loaded but contains no pages")

            chunks = self.text_splitter.split_documents(pages)

            valid_chunks = [
                chunk for chunk in chunks
                if chunk.page_content.strip() and len(chunk.page_content.strip()) > 50
            ]

            if not valid_chunks:
                raise self.PDFSummarizerError("No valid content found in PDF after filtering")

            if len(valid_chunks) > self.max_chunks:
                valid_chunks = valid_chunks[:self.max_chunks]

            return valid_chunks

        except self.PDFSummarizerError:
            raise
        except Exception as e:
            raise self.PDFSummarizerError(f"Failed to load PDF: {str(e)}") from e

    async def _generate_chunk_summaries(self, chunks: List[Document]) -> List[str]:
        chunk_summary_prompt = PromptTemplate(
            input_variables=["text"],
            template="[INST] 다음 내용을 요약해주세요: {text}\n\n요약은 핵심 내용만 간결하게 작성해주세요. [/INST]"
        )

        try:
            # 각 청크에 대한 요약 체인을 생성
            chunk_chains = []
            for chunk in chunks:
                chunk_chain = (
                        chunk_summary_prompt.partial(text=chunk.page_content)
                        | RunnableLambda(self._call_llm)
                )
                chunk_chains.append(chunk_chain)

            # 병렬 실행을 위한 딕셔너리 생성
            parallel_dict = {f"chunk_{i}": chain for i, chain in enumerate(chunk_chains)}
            parallel_chain = RunnableParallel(parallel_dict)

            # 병렬 실행
            results = await asyncio.wait_for(
                parallel_chain.ainvoke({}),
                timeout=300
            )

            # 결과 정리
            valid_summaries = []
            for key in sorted(results.keys(), key=lambda x: int(x.split('_')[1])):
                summary = results[key]
                if summary and isinstance(summary, str) and summary.strip():
                    valid_summaries.append(summary.strip())

            if not valid_summaries:
                raise self.PDFSummarizerError("Failed to generate any valid chunk summaries")

            return valid_summaries

        except asyncio.TimeoutError:
            raise self.PDFSummarizerError("Timeout while generating chunk summaries")
        except Exception as e:
            raise self.PDFSummarizerError(f"Failed to generate chunk summaries: {str(e)}") from e

    async def _create_final_summary_chain(self, summaries: List[str]) -> str:
        final_summary_prompt = PromptTemplate(
            input_variables=["combined_summaries"],
            template="""[INST] 다음은 문서의 각 부분 요약입니다: 
            
            {combined_summaries}
            
            전체 문서의 내용을 바탕으로 다음 형식으로 응답해주세요:
            제목: [문서의 주제를 나타내는 제목]
            요약: [전체 문서의 핵심 내용을 500단어 이내로 요약]  
            키워드: [주요 키워드 5개를 쉼표로 구분]
            
            응답은 반드시 위 형식을 정확히 따라주세요. [/INST]"""
        )

        chain = (
                {"combined_summaries": RunnablePassthrough()}
                | final_summary_prompt
                | RunnableLambda(self._call_llm)
        )
        combined_summaries = "\n\n".join([
            f"섹션 {i + 1}:\n{summary}"
            for i, summary in enumerate(summaries)
        ])

        return await chain.ainvoke(combined_summaries)

    async def summarize(self, file_path: str, config: Optional[Dict[str, Any]] = None) -> SummaryResponse:
        try:
            summarization_chain = (
                    RunnableLambda(self._load_pdf_documents)
                    | RunnableLambda(self._generate_chunk_summaries)
                    | RunnableLambda(self._create_final_summary_chain)
                    | self.SummaryOutputParser()
            )

            return await summarization_chain.ainvoke(file_path, config=config)

        except self.PDFSummarizerError:
            raise
        except Exception as e:
            raise self.PDFSummarizerError(f"Unexpected error during summarization: {str(e)}") from e


__summarizer_instance: Optional[PDFSummarizer] = None


def get_summarizer() -> PDFSummarizer:
    global __summarizer_instance
    if __summarizer_instance is None:
        __summarizer_instance = PDFSummarizer()
    return __summarizer_instance


async def pdf_summarize(file_path: str, config: Optional[Dict[str, Any]] = None) -> SummaryResponse:
    summarizer = get_summarizer()
    return await summarizer.summarize(file_path, config=config)
