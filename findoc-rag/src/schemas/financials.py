from typing import Literal, Optional
from pydantic import BaseModel, Field

class Metric(BaseModel):
    # field ORDER matters: quote first, then page, then value -> value is grounded in the quote
    evidence: Optional[str] = Field(description = "Verbatim quote (max 25 words) from the cited page that"
                                    "contains the number . null if not found.")
    source_page: Optional[int] = Field(description = "Page number from the [PAGE n] marker. null if not found.")

    value: Optional[float] = Field(description = "Number exactly as printed , comas removed, in the snapshot unit"
                                   "Do not conert or compute. null if not found.")


class FinancialSnapshot(BaseModel):
    company: str
    fiscal_year: str = Field(description = "Fiscal year the figure belongs to, format FY2025-26")
    currency: Literal["USD", "INR"]
    unit: Literal["crore", "lakh", "million"] = Field(description = "unit the report states for these figures")
    basis: Literal["standalone", "consolidated", "unknown"]
    revenue_from_operations: Metric
    profit_after_tax: Metric
    order_book: Metric

METRICS = ["revenue_from_operations", "profit_after_tax", "order_book"]

