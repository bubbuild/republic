from __future__ import annotations

import ast
import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType

import pytest

PYTHON_BLOCK = re.compile(r"```python\n(.*?)```", re.DOTALL)


class ExampleLoadError(AssertionError):
    def __init__(self, path: Path) -> None:
        super().__init__(f"Could not load example: {path}")


def _python_blocks(text: str) -> list[str]:
    return [match.strip() for match in PYTHON_BLOCK.findall(text)]


def _normalize_block(block: str) -> str:
    lines = []
    for line in block.splitlines():
        if line.strip().startswith("# test:skip"):
            continue
        lines.append(line)
    return "\n".join(lines)


@pytest.mark.parametrize("path", [Path("README.md"), *sorted(Path("docs").rglob("*.md"))])
def test_markdown_python_blocks_are_valid_python(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    blocks = _python_blocks(text)
    for block in blocks:
        ast.parse(_normalize_block(block))


@pytest.mark.parametrize("path", sorted(Path("examples").glob("*.py")))
def test_examples_are_valid_python(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    ast.parse(source)


def _load_example(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ExampleLoadError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def test_durable_tape_stream_example_runs_without_network() -> None:
    module = _load_example(Path("examples/08_durable_tape_stream.py"))

    result = module.run_example()

    assert result["visible"]["payloads"] == [
        {"event": "created", "id": "ticket-1"},
        {"event": "labeled", "label": "database"},
    ]
    assert result["visible"]["closed"] is True
    assert result["anchors"] == ["checkpoint", "close"]


def test_otel_tape_view_example_builds_context_from_telemetry() -> None:
    module = _load_example(Path("examples/09_otel_tape_view.py"))

    view = module.build_view()
    events = module.record_incident_trace().events()
    anchor_event = events[0]

    assert view["otel_event_names"] == [
        "anchor",
        "debug",
        "message",
        "tool_call",
        "tool_result",
        "message",
    ]
    assert view["tapes"] == ["ops"]
    assert view["query_kinds"] == ["message", "message"]
    assert anchor_event.attributes[module.TAPE_ANCHOR_NAME_KEY] == "incident_42"
    assert view["messages"] == [
        {"role": "system", "content": "owner=tier1"},
        {"role": "user", "content": "Investigate DB timeout."},
        {"role": "assistant", "content": "Check pool saturation and slow queries."},
    ]
