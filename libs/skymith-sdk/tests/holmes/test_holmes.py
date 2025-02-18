from unittest.mock import patch

import pytest
from langchain.tools import StructuredTool
from websockets.asyncio.client import connect

from skymith.sdk.models import QualitySpeedDataAgentResponse


@pytest.mark.asyncio
async def test_holmes():
    with patch.object(connect, "create_connection") as mock_connect:
        from skymith.sdk.holmes import Holmes

        holmes = Holmes.create()
        assert isinstance(holmes, StructuredTool)
        with patch.object(
            QualitySpeedDataAgentResponse,
            "model_validate_json",
            return_value=QualitySpeedDataAgentResponse.model_validate(
                {
                    "type": "string",
                    "body": "string",
                    "follow_up_questions": ["string"],
                    "references": ["string"],
                    "intermediate_steps": "string",
                    "keywords": ["string"],
                    "subqueries_answered": ["string"],
                    "subqueries_responses": ["string"],
                }
            ),
        ):
            res = await holmes.ainvoke("skymith")
            assert isinstance(res, str)
            assert "References:" in res
