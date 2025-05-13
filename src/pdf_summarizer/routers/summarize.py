from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
import os
import tempfile

from src.pdf_summarizer.models.schemas import SummaryResponse

router = APIRouter()


@router.post("/summarize", response_model=SummaryResponse)
async def summarize(file: UploadFile = File(requried = True)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            contents = await file.read()
            tmp.write(contents)
            temp_path = tmp.name
        # TODO: pdf to text
        # TODO: sLLM 연동
        return {
            "title": "요약 결과",
            "summary": "이 문서는 학습 노트 생성을 위한 테스트 문서입니다.",
            "keywords": ["요약", "노트", "FastAPI"]
        }

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)