import asyncio
from pathlib import Path
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


class DocumentProcessor:
    def __init__(self,
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 max_chunks: int = 50):
        self.max_chunks = max_chunks
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    async def load_and_split_pdf(self, file_path: str) -> List[str]:
        """PDF를 로드하고 청크로 분할하여 텍스트 리스트 반환"""
        try:
            pdf_path = Path(file_path)
            if not pdf_path.exists():
                raise Exception(f"PDF file not found: {file_path}")

            if not pdf_path.suffix.lower() == '.pdf':
                raise Exception(f"File is not a PDF: {file_path}")

            loader = PyPDFLoader(file_path)
            pages = await asyncio.to_thread(loader.load)

            if not pages:
                raise Exception("PDF loaded but contains no pages")

            chunks = self.text_splitter.split_documents(pages)

            valid_chunks = [
                chunk.page_content.strip() for chunk in chunks
                if chunk.page_content.strip() and len(chunk.page_content.strip()) > 50
            ]

            if not valid_chunks:
                raise Exception("No valid content found in PDF after filtering")

            if len(valid_chunks) > self.max_chunks:
                valid_chunks = valid_chunks[:self.max_chunks]

            return valid_chunks

        except Exception as e:
            raise Exception(f"Failed to load PDF: {str(e)}") from e