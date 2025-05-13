from pydantic import BaseModel
from typing import List

class SummaryResponse(BaseModel):
    title: str
    summary: str
    keywords: List[str]
