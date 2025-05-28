import contextvars
from typing import Dict, Any

langsmith_context: contextvars.ContextVar[Dict[str, Any]] = \
    contextvars.ContextVar("langsmith_context", default={})


def set_langsmith_context(enabled: bool, project: str, metadata: Dict[str, Any] = None):
    current_context = langsmith_context.get()
    new_context_data = {
        "enabled": enabled,
        "project": project,  # Langsmith 클라이언트 설정에 사용될 수 있음
        "metadata": {**current_context.get("metadata", {}), **(metadata or {})}
    }
    langsmith_context.set(new_context_data)


def get_langsmith_context() -> Dict[str, Any]:
    return langsmith_context.get()


def update_langsmith_metadata(metadata_update: Dict[str, Any]):
    context = langsmith_context.get()
    if context:
        existing_metadata = context.get("metadata", {})
        existing_metadata.update(metadata_update)
        context["metadata"] = existing_metadata
        langsmith_context.set(context)
