"""Schemas for advisory_agent."""

from typing import Any, Literal

from pydantic import BaseModel, Field

# Tamil ("ta") and Hindi ("hi") are planned; widen this Literal when they land.
Language = Literal["en"]
LANGUAGE_NAMES: dict[str, str] = {"en": "English"}


class AdvisoryDraft(BaseModel):
    """Structured output requested from the LLM."""

    headline: str = Field(description="One plain sentence stating the outcome")
    explanation: str = Field(description="Three to six short plain sentences explaining why")


class AdvisoryNote(BaseModel):
    language: Language
    headline: str
    explanation: str
    generated_by: Literal["llm", "template"] = Field(
        description="template when the LLM text failed the number check"
    )
    facts: dict[str, Any] = Field(description="The facts the explanation was written from")
    notes: list[str] = Field(default_factory=list)
