from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field, RootModel


class SkymithStriderInput(BaseModel):
    product_query: str = Field(
        description="product description to research using Skymith agents. It must be short and to the point."
    )
    domains: list[str] = Field(description="domains or websites to deep research.")
    product_schema: dict = Field(
        description="schema in natural language defining what characteristics must be retrieved for the product."
    )


class SkymithHolmesInput(BaseModel):
    query: str = Field(description="query for the queries to search with Skymith.")


class SkymithNewsInput(BaseModel):
    unique_id: UUID | None = Field(
        description="ID of the news to retrieve. If None, the last day is retrieved."
    )


class SkymithConceptInput(BaseModel):
    pass


class SkymithFinancialNewsTitles(BaseModel):
    pass


class CompareDomainsOut(BaseModel):
    """Model to return the result of a domain research comparison."""

    domain: str = Field(description="domain where the information comes from")
    result: list[dict] | None = Field(
        default=None, description="the information the user requested per domain"
    )
    references: list[str] = Field(
        description="list of URLs where the information can be found more in detail"
    )


class BaseAgentResponse(BaseModel):
    """Model to define the base response parameters."""

    follow_up_questions: list[str] | None = None
    references: list[str] | None = None
    keywords: list[str] | None = None


class DataAgentResponse(BaseAgentResponse):
    """Model to define the speed response parameters."""

    type: str
    body: str


class QualityDataAgentResponse(DataAgentResponse):
    """Model to define the quality data response parameters."""

    subqueries_answered: list[str]
    subqueries_responses: list[str]


QualitySpeedDataAgentResponse = RootModel[QualityDataAgentResponse | DataAgentResponse]


class ContentPresentedOut(BaseAgentResponse):
    """Model to define an educational concept."""

    title: str = Field(description="title of the content")
    content_presented: str = Field(
        description="content sent by the API",
    )
    timestamp: datetime = Field(description="when the content presented was generated")
    unique_id: UUID | None = Field(default=None, description="UUID for the content, if any")


class NewsTitleOut(BaseModel):
    """Model to define a news title."""

    unique_id: UUID = Field(description="UUID of the news")
    title: str = Field(description="title of the news")


NewsTitlesOut = RootModel[list[NewsTitleOut]]
