from unittest.mock import AsyncMock, patch

import httpx
import pytest
from langchain.tools import StructuredTool

from skymith.sdk.buffet import DailyFinancialConcept, FinancialNews, FinancialNewsTitles
from skymith.sdk.models import ContentPresentedOut, NewsTitlesOut

CONTENT_PRESENTED_RES: dict = {
    "content_presented": "my content",
    "follow_up_questions": ["string"],
    "references": ["string"],
    "keywords": [],
    "title": "string",
    "timestamp": "2025-02-17T08:28:42.218Z",
    "unique_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
}
NEWS_TITLES_RES: list[dict] = [
    {"unique_id": "f52ce0c8-7160-40e0-9bcb-39605ea3f21b", "title": "News 1"},
    {"unique_id": "249608ac-eefb-4df5-a53e-5fdc29aad99b", "title": "News 2"},
]


@patch.object(
    httpx._client.AsyncClient,
    "get",
    AsyncMock(
        return_value=httpx.Response(
            200, json=CONTENT_PRESENTED_RES, request=httpx.Request(method="GET", url="https://...")
        )
    ),
)
@pytest.mark.asyncio
async def test_daily_financial_concept_str():
    concept_tool = DailyFinancialConcept.create()
    assert isinstance(concept_tool, StructuredTool)
    res = await concept_tool.ainvoke({})
    assert isinstance(res, str)
    assert "my content" in res


@patch.object(
    httpx._client.AsyncClient,
    "get",
    AsyncMock(
        return_value=httpx.Response(
            200, json=CONTENT_PRESENTED_RES, request=httpx.Request(method="GET", url="https://...")
        )
    ),
)
@pytest.mark.asyncio
async def test_daily_financial_concept_pydantic():
    concept_tool = DailyFinancialConcept.create(return_pydantic=True)
    assert isinstance(concept_tool, StructuredTool)
    res = await concept_tool.ainvoke({})
    assert isinstance(res, ContentPresentedOut)
    assert res.content_presented == "my content"


@patch.object(
    httpx._client.AsyncClient,
    "get",
    AsyncMock(
        return_value=httpx.Response(
            200, json=CONTENT_PRESENTED_RES, request=httpx.Request(method="GET", url="https://...")
        )
    ),
)
@pytest.mark.asyncio
async def test_financial_news_str():
    financial_news = FinancialNews.create()
    assert isinstance(financial_news, StructuredTool)
    res = await financial_news.ainvoke({"unique_id": None})
    assert isinstance(res, str)
    assert "my content" in res


@patch.object(
    httpx._client.AsyncClient,
    "get",
    AsyncMock(
        return_value=httpx.Response(
            200, json=CONTENT_PRESENTED_RES, request=httpx.Request(method="GET", url="https://...")
        )
    ),
)
@pytest.mark.asyncio
async def test_financial_news_pydantic():
    financial_news = FinancialNews.create(return_pydantic=True)
    assert isinstance(financial_news, StructuredTool)
    res = await financial_news.ainvoke({"unique_id": None})
    assert isinstance(res, ContentPresentedOut)
    assert res.content_presented == "my content"


@patch.object(
    httpx._client.AsyncClient,
    "get",
    AsyncMock(
        return_value=httpx.Response(
            200, json=NEWS_TITLES_RES, request=httpx.Request(method="GET", url="https://...")
        )
    ),
)
@pytest.mark.asyncio
async def test_news_titles_str():
    financial_news_titles = FinancialNewsTitles.create()
    assert isinstance(financial_news_titles, StructuredTool)
    res = await financial_news_titles.ainvoke({})
    assert isinstance(res, str)
    assert "News 1" in res
    assert "News 2" in res


@patch.object(
    httpx._client.AsyncClient,
    "get",
    AsyncMock(
        return_value=httpx.Response(
            200, json=NEWS_TITLES_RES, request=httpx.Request(method="GET", url="https://...")
        )
    ),
)
@pytest.mark.asyncio
async def test_news_titles_pydantic():
    financial_news_titles = FinancialNewsTitles.create(return_pydantic=True)
    assert isinstance(financial_news_titles, StructuredTool)
    res = await financial_news_titles.ainvoke({})
    assert isinstance(res, NewsTitlesOut)
    assert len(res.root) == 2
