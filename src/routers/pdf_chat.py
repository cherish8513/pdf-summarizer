import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException, Body

from middlewares.langsmith_trace import langsmith_trace
from routers.request.question_request import QuestionRequest
from services.pdf_chat_service import PdfChatService

router = APIRouter()
chat_service = PdfChatService()


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


@router.post("/upload", )
@langsmith_trace(name="Upload Router", tags=["router", "upload"])
async def upload(file: UploadFile = File(...)):
    validate_pdf_file(file)

    contents = await file.read()
    validate_file_content(contents)

    async with temporary_pdf_file(contents) as temp_path:
        try:
            await chat_service.upload(pdf_file_path=temp_path)
            return {"message": "PDF uploaded and processed successfully."}
        except Exception as e:
            raise HTTPException(status_code=500, detail={"error": "Unexpected error occurred", "details": str(e)})


@router.post("/ask", response_model=dict[str, str])
@langsmith_trace(name="Ask Router", tags=["router", "ask"])
async def ask(question_request: QuestionRequest) -> dict[str, str]:
    return await chat_service.ask(question_request.question)
