from functools import partial
from uuid import UUID

import httpx
from langchain.tools import StructuredTool

from ..constants import SKYMITH_API_KEY
from ..endpoints import (
    SKYMITH_FINANCIAL_CONCEPT_URI,
    SKYMITH_FINANCIAL_NEWS_URI,
    SKYMITH_FINANCIAL_TITLES_URI,
)
from ..models import (
    ContentPresentedOut,
    NewsTitlesOut,
    SkymithConceptInput,
    SkymithFinancialNewsTitles,
    SkymithNewsInput,
)


def format_content_presented_out_json(
    content_presented_out_json: dict, return_pydantic: bool
) -> str | ContentPresentedOut:
    content_presented_data: ContentPresentedOut = ContentPresentedOut.model_validate(
        content_presented_out_json
    )
    if return_pydantic:
        return content_presented_data
    references: str = "\n".join(f"- {x}" for x in content_presented_data.references)
    return f"Title: {content_presented_data.title}\nContent: {content_presented_data.content_presented}\nReferences:\n{references}"


class FinancialNews(StructuredTool):
    """StructuredTool that provides financial news."""

    @classmethod
    async def get_news(
        cls, unique_id: UUID | None, timeout: float = 60, return_pydantic: bool = False
    ) -> ContentPresentedOut:
        """Provides daily financial news. For past news, you must first retrieve the UUID using the titles endpoint.

        Args:
            unique_id (`str | None`): ID of the news to retrieve. If None, the last day is retrieved.
            timeout (`float`): how many seconds to wait the server's response before raising an error.
            return_pydantic (`bool`): whether to return the complete Pydantic model or only the main content.

        Returns:
            `ContentPresentedOut`: the result for the comparison for a specific domain and URL.
        """

        async with httpx.AsyncClient(timeout=timeout) as client:
            params = (
                {"token": SKYMITH_API_KEY}
                if unique_id is None
                else {"unique_id": unique_id, "token": SKYMITH_API_KEY}
            )
            res = await client.get(SKYMITH_FINANCIAL_NEWS_URI, params=params)
        res.raise_for_status()
        return format_content_presented_out_json(
            content_presented_out_json=res.json(), return_pydantic=return_pydantic
        )

    @classmethod
    def create(
        cls,
        name: str = "SkymithFinancialNews",
        description: str = "Useful to get daily news in finance",
        return_direct: bool = False,
        timeout: float = 60,
        return_pydantic: bool = False,
    ) -> StructuredTool:
        """Creates a LangChain tool to interact with Skymith API.

        Args:
            name (`str`): tool name for LangChain.
            description (`str`): the purpose of the tool, according to LangChain's documentation.
            return_direct (`bool`): Whether to return the result directly or as a callback.
            timeout (`float`): how many seconds to wait the server's response before raising an error.
            return_pydantic (`bool`): whether to return the complete Pydantic model or only the main content.

        Returns:
            `StructuredTool`: the tool."""

        return cls.from_function(
            func=None,
            coroutine=partial(cls.get_news, timeout=timeout, return_pydantic=return_pydantic),
            name=name,
            description=description,
            args_schema=SkymithNewsInput,
            return_direct=return_direct,
        )


class DailyFinancialConcept(StructuredTool):
    """StructuredTool that provides a daily financial concept to learn."""

    @classmethod
    async def get_concept(
        cls, timeout: float = 60, return_pydantic: bool = False
    ) -> str | ContentPresentedOut:
        """Provides a daily financial concept to learn.

        Args:
            token (`str | None`): Skymith token to translate the news to your account's language. If None, they are returned in English.
            timeout (`float`): how many seconds to wait the server's response before raising an error.
            return_pydantic (`bool`): whether to return the complete Pydantic model or only the main content.

        Returns:
            `str | ContentPresentedOut`: the result for the comparison for a specific domain and URL.
        """

        async with httpx.AsyncClient(timeout=timeout) as client:
            res = await client.get(
                SKYMITH_FINANCIAL_CONCEPT_URI, params={"token": SKYMITH_API_KEY}
            )
        res.raise_for_status()
        return format_content_presented_out_json(
            content_presented_out_json=res.json(), return_pydantic=return_pydantic
        )

    @classmethod
    def create(
        cls,
        name: str = "SkymithFinancialConcept",
        description: str = "Useful to learn a daily concept in finance",
        return_direct: bool = False,
        timeout: float = 60,
        return_pydantic: bool = False,
    ) -> StructuredTool:
        """Creates a LangChain tool to interact with Skymith API.

        Args:
            name (`str`): tool name for LangChain.
            description (`str`): the purpose of the tool, according to LangChain's documentation.
            return_direct (`bool`): Whether to return the result directly or as a callback.
            timeout (`float`): how many seconds to wait the server's response before raising an error.
            return_pydantic (`bool`): whether to return the complete Pydantic model or only the main content.

        Returns:
            `StructuredTool`: the tool."""

        return cls.from_function(
            func=None,
            coroutine=partial(cls.get_concept, timeout=timeout, return_pydantic=return_pydantic),
            name=name,
            description=description,
            args_schema=SkymithConceptInput,
            return_direct=return_direct,
        )


class FinancialNewsTitles(StructuredTool):
    """StructuredTool that provides the titles and UUIDs of the past daily financial news."""

    @classmethod
    async def get_news_titles(
        cls, timedelta_days: int = 7, timeout: float = 60, return_pydantic: bool = False
    ) -> str | NewsTitlesOut:
        """Provides daily financial news titles and UUIDs up to the last days (max. 100).

        Args:
            token (`str | None`): Skymith token to translate the news to your account's language. If None, they are returned in English.
            timedelta_days (`int`): no. days in the past to retrieve their titles for. Defaults to 7.
            timeout (`float`): how many seconds to wait the server's response before raising an error.
            return_pydantic (`bool`): whether to return the complete Pydantic model or only the titles.

        Returns:
            `str | NewsTitles`: the news titles in str or Pydantic format.
        """

        async with httpx.AsyncClient(timeout=timeout) as client:
            res = await client.get(
                SKYMITH_FINANCIAL_TITLES_URI,
                params={"token": SKYMITH_API_KEY, "timedelta_days": timedelta_days},
            )
        res.raise_for_status()
        news_titles_data: NewsTitlesOut = NewsTitlesOut.model_validate(res.json())
        if return_pydantic:
            return news_titles_data
        return "\n".join(
            [
                f"- News Title {idx}: {news_title.title}"
                for idx, news_title in enumerate(news_titles_data.root)
            ]
        )

    @classmethod
    def create(
        cls,
        name: str = "SkymithFinancialNewsTitles",
        description: str = "Useful to get titles of financial news for the past days",
        return_direct: bool = False,
        timedelta_days: int = 7,
        timeout: float = 60,
        return_pydantic: bool = False,
    ) -> StructuredTool:
        """Creates a LangChain tool to interact with Skymith API.

        Args:
            name (`str`): tool name for LangChain.
            description (`str`): the purpose of the tool, according to LangChain's documentation.
            return_direct (`bool`): Whether to return the result directly or as a callback.
            timedelta_days (`int`): no. days in the past to retrieve their titles for. Defaults to 7.
            timeout (`float`): how many seconds to wait the server's response before raising an error.
            return_pydantic (`bool`): whether to return the complete Pydantic model or only the titles.

        Returns:
            `StructuredTool`: the tool."""

        return cls.from_function(
            func=None,
            coroutine=partial(
                cls.get_news_titles,
                timedelta_days=timedelta_days,
                timeout=timeout,
                return_pydantic=return_pydantic,
            ),
            name=name,
            description=description,
            args_schema=SkymithFinancialNewsTitles,
            return_direct=return_direct,
        )
