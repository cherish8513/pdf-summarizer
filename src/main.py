import uvicorn
from fastapi import FastAPI

from pdf_summarizer.middlewares.config import settings
from pdf_summarizer.middlewares.langsmith import create_langsmith_middleware
from src.pdf_summarizer.routers import summarize

app = FastAPI(
    title="PDF Summarizer API",
    description="API for summarizing PDF documents with LangSmith tracing",
    version="1.0.0"
)

LangSmithMiddleware = create_langsmith_middleware(
    langsmith_enabled=settings.langsmith_tracing,
    project_name=settings.langsmith_project
)
app.add_middleware(LangSmithMiddleware)

app.include_router(summarize.router, prefix="/api")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
