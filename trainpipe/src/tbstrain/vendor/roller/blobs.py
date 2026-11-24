# SPDX-License-Identifier: Apache-2.0
# Adapted from Roller, revision cd4c7f2. See NOTICE and LICENSE.
import base64
import re
from pathlib import Path
from typing import Any

_REFERENCE = re.compile(r"(data:[\w.+-]+/[\w.+-]+;base64,)roller-blob:([0-9a-f]{64}\.[\w.+-]+)")


def load_blobs(value: Any, directory: Path) -> Any:
    """``value`` with each blob reference replaced by its original data URI."""

    if isinstance(value, dict):
        return {key: load_blobs(item, directory) for key, item in value.items()}
    if isinstance(value, list):
        return [load_blobs(item, directory) for item in value]
    if isinstance(value, str) and (match := _REFERENCE.fullmatch(value)):
        data = (directory / match[2]).read_bytes()
        return match[1] + base64.b64encode(data).decode()
    return value
