from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from pdf_summarizer.middlewares.config import settings


def create_langsmith_middleware(langsmith_enabled: bool, project_name: str):
    class LangSmithTracingMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            request.state.langsmith_enabled = langsmith_enabled
            request.state.langsmith_project = project_name

            if langsmith_enabled:
                if not all([settings.langsmith_api_key, settings.langsmith_endpoint]):
                    print("Warning: LangSmith is enabled but API key or endpoint is missing")
                    request.state.langsmith_enabled = False

            try:
                response = await call_next(request)
                return response
            except Exception as e:
                raise e

    return LangSmithTracingMiddleware
