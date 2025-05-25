import os
from typing import Optional, List
from dotenv import load_dotenv
from fastapi import Request, UploadFile
from langchain_core.runnables.config import RunnableConfig

load_dotenv()


class Settings:
    def __init__(self):
        self.langsmith_tracing = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
        self.langsmith_endpoint = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
        self.langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
        self.langsmith_project = os.getenv("LANGSMITH_PROJECT", "pdf-summarizer")

        self.debug = os.getenv("DEBUG", "false").lower() == "true"
        self.log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    def create_runnable_config(
            self,
            request: Request,
            file: Optional[UploadFile] = None,
            additional_tags: Optional[List[str]] = None,
            additional_metadata: Optional[dict] = None
    ) -> Optional[RunnableConfig]:

        if not getattr(request.state, "langsmith_enabled", False):
            return None

        metadata = {
            "user_agent": request.headers.get("user-agent", "unknown"),
            "client_ip": request.client.host if request.client else "unknown",
            "endpoint": str(request.url),
            "method": request.method,
            "project": self.langsmith_project
        }

        if file:
            metadata.update({
                "filename": file.filename,
                "content_type": file.content_type,
                "file_size": getattr(file, 'size', 'unknown'),
            })

        custom_headers = {
            "x-user-id": "user_id",
            "x-session-id": "session_id",
            "x-request-id": "request_id",
            "x-organization-id": "organization_id"
        }

        for header_key, metadata_key in custom_headers.items():
            if header_key in request.headers:
                metadata[metadata_key] = request.headers[header_key]

        if additional_metadata:
            metadata.update(additional_metadata)

        tags = ["pdf-summarization", "api-request"]

        if file and file.filename:
            file_ext = file.filename.split('.')[-1].lower()
            tags.append(f"file-{file_ext}")

        if "user_id" in metadata:
            tags.append(f"user-{metadata['user_id']}")

        if "organization_id" in metadata:
            tags.append(f"org-{metadata['organization_id']}")

        endpoint_path = request.url.path.replace("/", "-").strip("-")
        if endpoint_path:
            tags.append(f"endpoint-{endpoint_path}")

        if additional_tags:
            tags.extend(additional_tags)

        run_name_parts = ["pdf_summarize"]
        if file and file.filename:
            clean_filename = "".join(c for c in file.filename if c.isalnum() or c in ".-_")
            run_name_parts.append(clean_filename[:50])  # 길이 제한

        if "user_id" in metadata:
            run_name_parts.append(f"user_{metadata['user_id']}")

        run_name = "_".join(run_name_parts)

        return RunnableConfig(
            tags=tags,
            metadata=metadata,
            run_name=run_name
        )

    def create_chain_config(
            self,
            operation_name: str,
            additional_tags: Optional[List[str]] = None,
            additional_metadata: Optional[dict] = None
    ) -> Optional[RunnableConfig]:

        if not self.langsmith_tracing:
            return None

        metadata = {
            "operation": operation_name,
            "project": self.langsmith_project
        }

        if additional_metadata:
            metadata.update(additional_metadata)

        tags = ["pdf-summarization", f"operation-{operation_name}"]

        if additional_tags:
            tags.extend(additional_tags)

        return RunnableConfig(
            tags=tags,
            metadata=metadata,
            run_name=f"pdf_chain_{operation_name}"
        )


settings = Settings()