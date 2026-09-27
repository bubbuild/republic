import base64
import json

import pytest
from pydantic import ValidationError

from republic import (
    FilePart,
    Message,
    ReasoningPart,
    Request,
    RequestOptions,
    Response,
    TextPart,
    Tool,
    ToolCallPart,
    ToolChoice,
    ToolResultPart,
    Usage,
)


def test_request_and_response_round_trip_without_runtime_objects() -> None:
    image = FilePart.from_bytes(b"\xfb\xff\x00", media_type="image/png", filename="sample.png")
    assert image.data == "+/8A"
    assert base64.b64decode(image.data) == b"\xfb\xff\x00"
    assistant = Message(
        role="assistant",
        parts=[
            ReasoningPart(text="", provider_metadata={"anthropic": {"signature": "opaque", "redacted": True}}),
            TextPart(text="Checking", provider_metadata={"example": {"annotations": [1, None]}}),
            ToolCallPart(tool_call_id="call-1", tool_name="weather", tool_args='{"city":"上海"}'),
        ],
        provider_metadata={"openai": {"item_id": "msg-1"}},
    )
    original = Request(
        model="fake",
        messages=[
            Message(role="system", parts=[TextPart(text="Be brief.")]),
            Message(
                role="user", parts=[image, FilePart(data="https://example.com/doc.pdf", media_type="application/pdf")]
            ),
            assistant,
            Message(
                role="tool", parts=[ToolResultPart(tool_call_id="call-1", tool_name="weather", result={"temp": 20})]
            ),
        ],
        tools=[Tool(name="weather", parameters={"type": "object", "properties": {"city": {"type": "string"}}})],
        options=RequestOptions(
            temperature=0,
            top_p=0.9,
            max_output_tokens=100,
            stop=["done"],
            tool_choice=ToolChoice(name="weather"),
            parallel_tool_calls=False,
            provider_options={"example": {"reasoning": "low"}},
        ),
    )
    assert Request.model_validate_json(original.model_dump_json()) == original
    result = Response(
        message=assistant,
        usage=Usage(input_tokens=10, output_tokens=4, raw={"detail": 1}),
        response_id="response-1",
        response_model="resolved-model",
        finish_reason="tool_call",
    )
    assert Response.model_validate_json(result.model_dump_json()) == result
    assert json.loads(result.model_dump_json())["response_id"] == "response-1"
    assert result.message.text == "Checking"


@pytest.mark.parametrize("field", ["client", "replay", "turn_id", "hooks"])
def test_unknown_runtime_or_agent_fields_are_rejected(field: str) -> None:
    with pytest.raises(ValidationError):
        Message.model_validate({"role": "user", "parts": [], field: "runtime"})


def test_metadata_results_and_tools_cannot_contain_runtime_objects() -> None:
    client = object()
    with pytest.raises(ValidationError):
        TextPart.model_validate({"text": "x", "provider_metadata": {"provider": {"client": client}}})
    with pytest.raises(ValidationError):
        ToolResultPart.model_validate({"tool_call_id": "c", "tool_name": "f", "result": client})
    with pytest.raises(ValidationError):
        Tool.model_validate({"name": "f", "parameters": {}, "execute": lambda: None})


def test_missing_usage_is_unknown_and_zero_is_preserved() -> None:
    assert Usage().input_tokens is None
    assert Usage(input_tokens=0).total_tokens is None
    assert Usage(input_tokens=0, output_tokens=0).total_tokens == 0
    assert Usage(input_tokens=7, output_tokens=3).total_tokens == 10
    with pytest.raises(ValidationError):
        Usage(output_tokens=-1)


def test_partial_tool_json_and_error_result_are_data() -> None:
    call = ToolCallPart(tool_call_id="c", tool_name="f", tool_args='{"value":')
    assert ToolCallPart.model_validate_json(call.model_dump_json()).tool_args == '{"value":'
    result = ToolResultPart(tool_call_id="c", tool_name="f", result="caller rejected", is_error=True)
    assert ToolResultPart.model_validate_json(result.model_dump_json()) == result
