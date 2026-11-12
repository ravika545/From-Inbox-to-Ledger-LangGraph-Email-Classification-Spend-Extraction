from __future__ import annotations
from typing import TypedDict, Annotated, Optional, Literal
from pydantic import BaseModel, Field
from operator import add

Category = Literal["Work", "Online Orders", "Subscriptions", "Miscellaneous"]


class Classification(BaseModel):
    category: Category
    confidence: float = Field(..., ge=0.0, le=1.0)
    reason: str


class Transaction(BaseModel):
    amount: float
    currency: str = "INR"
    merchant: str
    date: str
    category: str
    payment_method: Optional[str] = None
    source_email_id: str = ""
    dedup_key: str = ""


class TransactionList(BaseModel):
    """Wrapper because with_structured_output in langchain-openai 0.2.x
    does not accept list[Model] directly."""
    transactions: list[Transaction] = Field(default_factory=list)


class Email(BaseModel):
    id: str
    from_: str = Field(..., alias="from")
    subject: str
    date: str
    body: str
    model_config = {"populate_by_name": True}


class GraphState(TypedDict):
    emails: list[Email]
    classifications: Annotated[list[dict], add]
    transactions:     Annotated[list[dict], add]
    review_queue:     Annotated[list[dict], add]
    review_decisions: Annotated[dict, lambda a, b: {**(a or {}), **(b or {})}]
    summary: dict
    errors:           Annotated[list[str], add]