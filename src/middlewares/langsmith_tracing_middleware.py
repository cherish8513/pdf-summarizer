from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response
from typing import Dict, Any

from middlewares.langsmith_context import langsmith_context, set_langsmith_context


class LangSmithTracingMiddleware(BaseHTTPMiddleware):
    def __init__(
            self,
            app,
            langsmith_enabled: bool,
            project_name: str,
    ):
        super().__init__(app)
        self.langsmith_enabled = langsmith_enabled
        self.project_name = project_name

    async def dispatch(
            self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        token = langsmith_context.set({})

        try:
            base_metadata: Dict[str, Any] = {
                "path": request.url.path,
                "method": request.method,
                "client_host": request.client.host if request.client else None,
                "query_params": str(request.query_params),
                "headers": dict(request.headers),
            }

            set_langsmith_context(
                enabled=self.langsmith_enabled,
                project=self.project_name,
                metadata=base_metadata
            )

            response = await call_next(request)
        finally:
            langsmith_context.reset(token)

        return response
