from pydantic import BaseModel, Field
from typing import List

class SummaryResponse(BaseModel):
    title: str = Field(..., description="The estimated title of the document")
    summary: str = Field(..., description="Summary of the document content")
    keywords: List[str] = Field(default_factory=list, description="Key terms from the document")

class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error message")
    details: str = Field(default="", description="Additional error details")