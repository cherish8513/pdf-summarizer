from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from middlewares.config import settings
from middlewares.langsmith_context import set_langsmith_context


class LangSmithTracingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        enabled = settings.langsmith_enabled
        project = settings.langsmith_project
        metadata = {"path": request.url.path, "method": request.method}

        set_langsmith_context(enabled, project, metadata)

        return await call_next(request)
