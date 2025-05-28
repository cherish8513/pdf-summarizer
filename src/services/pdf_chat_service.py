from typing import Optional, Dict, Any

from llm.pdf_loader import PdfLoader
from llm.pdf_rag_client import PdfRagClient
from llm.summarize_response_parser import SummaryResponseParser
from models.schemas import SummaryResponse


class PDFSummarizer:
    def __init__(self,
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 max_chunks: int = 50):

        self.llm_client = PdfRagClient()
        self.document_processor = PdfLoader(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            max_chunks=max_chunks
        )
        self.response_parser = SummaryResponseParser()

    class PDFSummarizerError(Exception):
        pass

    async def summarize(self, file_path: str, config: Optional[Dict[str, Any]] = None) -> SummaryResponse:
        """PDF 문서를 요약합니다."""
        try:
            # 1. 문서 로드 및 청크 분할
            chunks = await self.document_processor.load_and_split_pdf(file_path)

            # 2. 청크별 요약 생성
            chunk_summaries = await self.llm_client.generate_chunk_summaries(chunks)

            # 3. 최종 요약 생성
            final_summary_text = await self.llm_client.generate_final_summary(chunk_summaries)

            # 4. 응답 파싱
            return self.response_parser.parse(final_summary_text)

        except Exception as e:
            if isinstance(e, self.PDFSummarizerError):
                raise
            raise self.PDFSummarizerError(f"Unexpected error during summarization: {str(e)}") from e


_summarizer_instance: Optional[PDFSummarizer] = None


def get_summarizer() -> PDFSummarizer:
    global _summarizer_instance
    if _summarizer_instance is None:
        _summarizer_instance = PDFSummarizer()
    return _summarizer_instance


async def pdf_summarize(file_path: str, config: Optional[Dict[str, Any]] = None) -> SummaryResponse:
    summarizer = get_summarizer()
    return await summarizer.summarize(file_path, config=config)