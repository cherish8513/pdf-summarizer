from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
import os
import tempfile

from src.pdf_summarizer.models.schemas import SummaryResponse
from src.pdf_summarizer.services.summarizer import pdf_summarize

router = APIRouter()


@router.post("/summarize", response_model=SummaryResponse)
async def summarize(file: UploadFile = File(requried = True)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            contents = await file.read()
            tmp.write(contents)
            temp_path = tmp.name

        return pdf_summarize(temp_path)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)