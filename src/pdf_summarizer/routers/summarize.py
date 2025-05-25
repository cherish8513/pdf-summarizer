import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException, Request

from pdf_summarizer.models.schemas import SummaryResponse, ErrorResponse
from pdf_summarizer.services.summarizer import pdf_summarize
from pdf_summarizer.middlewares.config import settings

router = APIRouter()


@asynccontextmanager
async def temporary_pdf_file(file_content: bytes):
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_content)
            temp_path = tmp.name
        yield temp_path
    finally:
        if temp_path and Path(temp_path).exists():
            try:
                os.remove(temp_path)
            except OSError as e:
                print(f"Warning: Failed to delete temp file {temp_path}: {e}")


def validate_pdf_file(file: UploadFile) -> None:
    if not file.filename or not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail={"error": "Invalid file type", "details": "Only PDF files are supported"}
        )


def validate_file_content(contents: bytes) -> None:
    if not contents:
        raise HTTPException(
            status_code=400,
            detail={"error": "Empty file", "details": "Uploaded file is empty"}
        )


@router.post(
    "/summarize",
    response_model=SummaryResponse,
    responses={
        500: {"model": ErrorResponse, "description": "Internal server error"},
        400: {"model": ErrorResponse, "description": "Bad request"}
    }
)
async def summarize(request: Request, file: UploadFile = File(...)):
    # 파일 유효성 검사
    validate_pdf_file(file)

    # 파일 내용 읽기 및 검사
    contents = await file.read()
    validate_file_content(contents)

    # 설정 생성
    config = settings.create_runnable_config(
        request=request,
        file=file,
        additional_tags=["api-upload"],
        additional_metadata={"source": "file_upload"}
    )

    # 임시 파일을 사용하여 PDF 처리
    async with temporary_pdf_file(contents) as temp_path:
        try:
            result = await pdf_summarize(temp_path, config=config)
            return result
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail={"error": "Failed to process PDF", "details": str(e)}
            )