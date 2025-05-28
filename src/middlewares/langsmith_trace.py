from functools import wraps
from typing import Optional, List, Callable, Any, Dict
from langsmith import traceable

from middlewares.langsmith_context import get_langsmith_context, update_langsmith_metadata


def langsmith_trace(name: Optional[str] = None, tags: Optional[List[str]] = None, extra_metadata: Optional[Dict[str, Any]] = None):
    """
    Langsmith 추적을 위한 비동기 함수 데코레이터.

    컨텍스트에서 Langsmith가 활성화된 경우에만 `traceable`로 함수를 감쌉니다.
    미들웨어에서 설정된 기본 메타데이터에 추가적인 메타데이터를 병합할 수 있습니다.

    Args:
        name (Optional[str]): 트레이스의 이름. 기본값은 함수 이름입니다.
        tags (Optional[List[str]]): 트레이스에 추가할 태그 리스트.
        extra_metadata (Optional[Dict[str, Any]]): 이 특정 트레이스에 추가할 메타데이터.
    """
    def decorator(func: Callable[..., Any]):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            context = get_langsmith_context()

            # 데코레이터 인자로 받은 추가 메타데이터를 컨텍스트에 업데이트
            if extra_metadata:
                update_langsmith_metadata(extra_metadata)
                # get_langsmith_context를 다시 호출하여 업데이트된 컨텍스트 사용
                context = get_langsmith_context()


            if context.get("enabled"):
                trace_name = name or func.__name__
                trace_tags = tags
                trace_metadata = context.get("metadata", {})

                # langsmith의 traceable 데코레이터를 동적으로 적용
                return await traceable(
                    name=trace_name,
                    tags=trace_tags,
                    metadata=trace_metadata,
                    # run_type="chain" # 필요에 따라 run_type 지정 가능
                )(func)(*args, **kwargs)
            else:
                return await func(*args, **kwargs)
        return wrapper
    return decorator