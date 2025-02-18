import json
import urllib.parse
from functools import partial

from langchain.tools import StructuredTool
from websockets.asyncio.client import connect

from ..constants import SKYMITH_API_KEY
from ..endpoints import SKYMITH_STRIDER_URI
from ..models import CompareDomainsOut, SkymithStriderInput


class Strider(StructuredTool):
    """StructuredTool that compares a given product across different websites / domains."""

    @classmethod
    async def compare(
        cls,
        product_query: str,
        domains: list[str],
        product_schema: dict,
        return_pydantic: bool = False,
    ) -> str | list[CompareDomainsOut]:
        """Deep researches given domains to provide a product comparison.

        Args:
            product_query (`str`): product description to research using Skymith agents. It must be short and to the point.
            domains (`list[str]`): domains or websites to deep research.
            product_schema (`dict`): schema in natural language defining what characteristics must be retrieved for the product.
            return_pydantic (`bool`): whether to return the complete Pydantic model or only the main content.

        Returns:
            `CompareDomainsOut`: the result for the comparison for a specific domain and URL.
        """

        params: dict = {"token": SKYMITH_API_KEY}
        websocket_uri: str = f"{SKYMITH_STRIDER_URI}?{urllib.parse.urlencode(params)}"
        total_comparison: list[CompareDomainsOut] = []
        async with connect(websocket_uri) as websocket:
            await websocket.send(
                json.dumps(
                    {"query": product_query, "domains": domains, "product_schema": product_schema}
                )
            )
            async for mith_res in websocket:
                total_comparison.append(CompareDomainsOut.model_validate_json(mith_res))
        if return_pydantic:
            return total_comparison
        return "\n\n".join(
            [
                f"Domain: {compare_out.domain}\nResult: {json.dumps(compare_out.result)}"
                for compare_out in total_comparison
            ]
        )

    @classmethod
    def create(
        cls,
        name: str = "SkymithStrider",
        description: str = "Useful to compare products across Internet platforms",
        return_direct: bool = False,
        return_pydantic: bool = False,
    ) -> StructuredTool:
        """Creates a LangChain tool to interact with Skymith API.

        Args:
            name (`str`): tool name for LangChain.
            description (`str`): the purpose of the tool, according to LangChain's documentation.
            return_direct (`bool`): Whether to return the result directly or as a callback.
            return_pydantic (`bool`): whether to return the complete Pydantic model or only the main content.

        Returns:
            `StructuredTool`: the tool."""

        return cls.from_function(
            func=None,
            coroutine=partial(cls.compare, return_pydantic=return_pydantic),
            name=name,
            description=description,
            args_schema=SkymithStriderInput,
            return_direct=return_direct,
        )
