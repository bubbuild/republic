"""The explicit JSON file operations shared by concrete OAuth helpers."""

import json
import os
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Any


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=destination.parent, delete=False) as handle:
            temporary = Path(handle.name)
            os.chmod(temporary, 0o600)
            json.dump(payload, handle)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            with suppress(OSError):
                temporary.unlink(missing_ok=True)
