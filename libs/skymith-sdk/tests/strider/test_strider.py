from unittest.mock import patch

import pytest
from langchain.tools import StructuredTool
from websockets.asyncio.client import connect


@pytest.mark.asyncio
async def test_strider():
    with patch.object(connect, "create_connection") as mock_connect:
        from skymith.sdk.strider import Strider

        strider = Strider.create()
        assert isinstance(strider, StructuredTool)


# TODO: test async streaming
