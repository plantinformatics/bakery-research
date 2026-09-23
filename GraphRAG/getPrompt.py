"""Load prompt text for use in :mod:`GraphRAG.Query`.

Example::

    from getPrompt import getPrompt

    prompt = getPrompt("answer.txt") + f"\n\nQuestion: {question}"
"""

import os
from pathlib import Path

_promptCache = {}


def getPrompt(promptName):
    """Return and cache ``promptName`` from the configured prompts directory."""
    if promptName in _promptCache:
        return _promptCache[promptName]

    promptsDir = os.getenv("promptsDir") or "prompts"
    if not promptsDir:
        raise RuntimeError("The promptsDir environment variable is not set")

    promptText = (Path(promptsDir) / promptName).read_text(encoding="utf-8")
    _promptCache[promptName] = promptText
    return promptText
