from pathlib import Path
from typing import List, Callable

from langchain_core.documents import Document
from langchain.text_splitter import TextSplitter
from langchain_community.document_loaders import PyPDFLoader


class PdfLoader:
    def __init__(
        self,
        text_splitter: TextSplitter,
        max_chunks: int = 50,
    ):
        self.text_splitter = text_splitter
        self.max_chunks = max_chunks

    def load_and_split_pdf(self, file_path: str) -> List[Document]:
        path = validate_pdf_file(file_path)
        loader = PyPDFLoader(str(path))
        pages = loader.load()

        if not pages:
            raise ValueError("PDF loaded but contains no pages")

        chunks = self.text_splitter.split_documents(pages)

        valid_chunks = [
            chunk.page_content.strip()
            for chunk in chunks
            if chunk.page_content.strip() and len(chunk.page_content.strip()) > 50
        ]

        if not valid_chunks:
            raise ValueError("No valid content found in PDF after filtering")

        if len(valid_chunks) > self.max_chunks:
            valid_chunks = valid_chunks[:self.max_chunks]

        return [Document(page_content=chunk) for chunk in valid_chunks]

def validate_pdf_file(file_path: str) -> Path:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Invalid file type: {file_path}")
    return path
