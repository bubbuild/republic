from __future__ import annotations

from pathlib import Path

import pytest

import republic
from republic import Document, Message, ToolCall
from republic.errors import UnsupportedFeatureError
from tests.conftest import FakeService

PDF = b"%PDF"
PDF_DATA_URL = "data:application/pdf;base64,JVBERg=="
CHAT_REPLY = {"choices": [{"message": {"content": "read"}}]}
MESSAGES_REPLY = {"content": [{"type": "text", "text": "read"}], "usage": {"input_tokens": 1, "output_tokens": 1}}


def test_document_loader_keeps_the_filename(tmp_path: Path) -> None:
    path = tmp_path / "report.pdf"
    path.write_bytes(PDF)

    assert republic.document(path) == Document("application/pdf", data=PDF, filename="report.pdf")
    assert republic.document("https://example.com/files/report.pdf?v=2") == Document(
        "application/pdf", url="https://example.com/files/report.pdf?v=2", filename="report.pdf"
    )
    assert republic.document(PDF_DATA_URL).filename is None
    assert republic.document(PDF, media_type="application/pdf", filename="a.pdf").filename == "a.pdf"


def test_document_bytes_require_a_media_type() -> None:
    with pytest.raises(ValueError, match="media_type"):
        republic.document(PDF)


def test_document_round_trips_through_dict() -> None:
    message = Message("user", (Document("application/pdf", data=PDF, filename="report.pdf"),))

    data = message.to_dict()

    assert data["content"] == [
        {"type": "document", "media_type": "application/pdf", "data": "JVBERg==", "filename": "report.pdf"}
    ]
    assert Message.from_dict(data) == message


async def test_chat_sends_a_file_part(service: FakeService) -> None:
    service.reply_json(CHAT_REPLY)
    model = republic.get_model("openai:test", api_format="chat", http_client=service.client())

    await model.chat(republic.user("Summarize", republic.document(PDF, media_type="application/pdf")))

    assert service.body()["messages"][0]["content"] == [
        {"type": "text", "text": "Summarize"},
        {"type": "file", "file": {"filename": "document.pdf", "file_data": PDF_DATA_URL}},
    ]


@pytest.mark.parametrize(
    "document,expected",
    [
        (
            Document("application/pdf", data=PDF, filename="report.pdf"),
            {"type": "input_file", "filename": "report.pdf", "file_data": PDF_DATA_URL},
        ),
        (
            Document("application/pdf", url="https://example.com/report.pdf"),
            {"type": "input_file", "file_url": "https://example.com/report.pdf"},
        ),
    ],
)
async def test_responses_sends_an_input_file(service: FakeService, document: Document, expected: dict) -> None:
    service.reply_json({"output": []})
    model = republic.get_model("openai:test", api_key="key", http_client=service.client())

    await model.chat(republic.user(document))

    assert service.body()["input"][0]["content"] == [expected]


@pytest.mark.parametrize(
    "document,expected",
    [
        (
            Document("application/pdf", data=PDF, filename="report.pdf"),
            {
                "type": "document",
                "source": {"type": "base64", "media_type": "application/pdf", "data": "JVBERg=="},
                "title": "report.pdf",
            },
        ),
        (
            Document("text/plain", data="héllo".encode()),
            {"type": "document", "source": {"type": "text", "media_type": "text/plain", "data": "héllo"}},
        ),
        (
            Document("application/pdf", url="https://example.com/report.pdf"),
            {"type": "document", "source": {"type": "url", "url": "https://example.com/report.pdf"}},
        ),
    ],
)
async def test_messages_sends_a_document_block(service: FakeService, document: Document, expected: dict) -> None:
    service.reply_json(MESSAGES_REPLY)
    model = republic.get_model("anthropic:test", api_key="key", http_client=service.client())

    await model.chat(republic.user(document))

    assert service.body()["messages"][0]["content"] == [expected]


async def test_messages_accepts_documents_in_tool_results(service: FakeService) -> None:
    service.reply_json(MESSAGES_REPLY)
    model = republic.get_model("anthropic:test", api_key="key", http_client=service.client())
    call = ToolCall("call_1", "fetch", "{}")

    await model.chat([
        "Fetch the report",
        republic.assistant(tool_calls=[call]),
        republic.tool(call, Document("application/pdf", data=PDF)),
    ])

    result = service.body()["messages"][-1]["content"][0]
    assert result["content"] == [
        {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": "JVBERg=="}}
    ]


async def test_messages_rejects_unsupported_inline_documents() -> None:
    model = republic.get_model("anthropic:test", api_key="key")

    with pytest.raises(UnsupportedFeatureError, match="text/csv"):
        await model.chat(republic.user(Document("text/csv", data=b"a,b")))


@pytest.mark.parametrize("source", [PDF, "gs://bucket/report.pdf"])
async def test_gemini_encodes_documents(service: FakeService, source: bytes | str) -> None:
    service.reply_json({"candidates": [{"content": {"parts": [{"text": "read"}]}, "finishReason": "STOP"}]})
    model = republic.get_model("google:test", api_key="key", http_client=service.client())

    await model.chat(republic.user(republic.document(source, media_type="application/pdf")))

    expected = (
        {"inlineData": {"mimeType": "application/pdf", "data": "JVBERg=="}}
        if isinstance(source, bytes)
        else {"fileData": {"mimeType": "application/pdf", "fileUri": source}}
    )
    assert service.body()["contents"][0]["parts"] == [expected]
