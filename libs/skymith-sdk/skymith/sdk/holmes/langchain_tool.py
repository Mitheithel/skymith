import json
import urllib.parse
from functools import partial

from langchain.tools import StructuredTool
from websockets.asyncio.client import connect

from ..constants import SKYMITH_API_KEY
from ..endpoints import SKYMITH_SEARCH_URI
from ..models import QualitySpeedDataAgentResponse, SkymithHolmesInput


class Holmes(StructuredTool):
    """StructuredTool that searches the Internet."""

    @classmethod
    async def search(
        cls,
        query: str,
        use_quality: bool = False,
        timelimit: str = "w",
        return_pydantic: bool = False,
    ) -> str | QualitySpeedDataAgentResponse:
        """Deep researches the Internet to answer the given query.

        Args:
            query (`str`): what to search using Skymith agents. It works better with full requests in natural language, instead of keywords.
            use_quality (`bool`): whether to use quality or speed mode when searching.
            timelimit (`str`): how far into the past our search engine can look for data. One of d, w, m, y. Defaults to one week old.
            return_pydantic (`bool`): whether to return the complete Pydantic model or only the main content.

        Returns:
            `str | dict`: the complete JSON from Skymith API (`dict`) or only its content (`str`), depending on `return_json`.
        """

        params: dict = {
            "token": SKYMITH_API_KEY,
            "use_quality": use_quality,
            "timelimit": timelimit,
        }
        websocket_uri: str = f"{SKYMITH_SEARCH_URI}?{urllib.parse.urlencode(params)}"
        async with connect(websocket_uri) as websocket:
            await websocket.send(json.dumps({"query": query}))
            mith_res: str = await websocket.recv()
        mith_res_data: QualitySpeedDataAgentResponse = (
            QualitySpeedDataAgentResponse.model_validate_json(mith_res)
        )
        if return_pydantic:
            return mith_res_data.root
        references: str = (
            "\n".join(f"- {x}" for x in mith_res_data.root.references)
            if mith_res_data.root.references
            else "No references available."
        )
        return f"Content: {mith_res_data.root.body}\nReferences:\n{references}"

    @classmethod
    def create(
        cls,
        name: str = "SkymithHolmes",
        description: str = "Useful to deep research anything on the Internet",
        return_direct: bool = False,
        use_quality: bool = False,
        timelimit: str = "w",
        return_pydantic: bool = False,
    ) -> StructuredTool:
        """Creates a LangChain tool to interact with Skymith API.

        Args:
            name (`str`): tool name for LangChain.
            description (`str`): the purpose of the tool, according to LangChain's documentation.
            return_direct (`bool`): Whether to return the result directly or as a callback.
            use_quality (`bool`): whether to use quality or speed mode when searching.
            timelimit (`str`): how far into the past our search engine can look for data. One of d, w, m, y. Defaults to one week old.
            return_pydantic (`bool`): whether to return the complete Pydantic model or only the main content.

        Returns:
            `StructuredTool`: the tool."""

        return cls.from_function(
            func=None,
            coroutine=partial(
                cls.search,
                use_quality=use_quality,
                timelimit=timelimit,
                return_pydantic=return_pydantic,
            ),
            name=name,
            description=description,
            args_schema=SkymithHolmesInput,
            return_direct=return_direct,
        )
