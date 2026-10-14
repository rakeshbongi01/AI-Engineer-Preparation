"""Pydantic contract for the financial snapshot. Field descriptions are sent to the model."""
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

_NULL_STRINGS = {"", "null", "none", "n/a"}


class Metric(BaseModel):
    # field ORDER matters: quote first, then page, then value -> the value is grounded in the quote
    evidence: Optional[str] = Field(
        description="Verbatim quote (max 25 words) from the cited page that contains the number. "
                    "null if not found.")
    source_page: Optional[int] = Field(
        description="Page number from the [PAGE n] marker. null if not found.")
    value: Optional[float] = Field(
        description="Number exactly as printed, commas removed, in the snapshot unit. "
                    "Do not convert or compute. null if not found.")

    @field_validator("evidence", mode="before")
    @classmethod
    def _blank_to_none(cls, value: object) -> object:
        """Models sometimes write the word 'null'; store a real null instead."""
        if isinstance(value, str) and value.strip().lower() in _NULL_STRINGS:
            return None
        return value


class FinancialSnapshot(BaseModel):
    company: str
    fiscal_year: str = Field(description="Fiscal year the figures belong to, format FY2025-26")
    currency: Literal["INR", "USD"]
    unit: Literal["crore", "lakh", "million", "billion"] = Field(
        description="Unit the report states for these figures")
    basis: Literal["standalone", "consolidated", "unknown"]
    revenue_from_operations: Metric
    profit_after_tax: Metric
    order_book: Metric


METRICS = ("revenue_from_operations", "profit_after_tax", "order_book")