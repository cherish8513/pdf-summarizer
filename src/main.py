import uvicorn
from fastapi import FastAPI

from middlewares.config import settings
from middlewares.langsmith_tracing_middleware import LangSmithTracingMiddleware
from routers import pdf_chat

app = FastAPI(
    title="PDF Summarizer API",
    description="API for summarizing PDF documents with LangSmith tracing",
    version="1.0.0"
)

app.add_middleware(
    LangSmithTracingMiddleware,
    langsmith_enabled=settings.langsmith_tracing,
    project_name=settings.langsmith_project
)

app.include_router(pdf_chat.router, prefix="/api")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)