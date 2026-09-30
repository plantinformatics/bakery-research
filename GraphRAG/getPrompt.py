"""Load prompt text for use in :mod:`GraphRAG.Query`.

Example::

    from getPrompt import getPrompt

    prompt = getPrompt("answer.txt") + f"\n\nQuestion: {question}"

Note that when changing prompts,
uvicorn needs to be loaded because the hot reload doesn't extend to the prompts
"""

import os
from pathlib import Path

_promptCache = {}


def getPrompt(promptName):
    """Return and cache ``promptName`` from the configured prompts directory.

    ``promptsDir`` overrides the location. Otherwise prompts are read from the
    repository ``prompts/`` directory next to ``GraphRAG/``, so the lookup does
    not depend on the process working directory.
    """
    if promptName in _promptCache:
        return _promptCache[promptName]

    promptsDir = os.getenv("promptsDir")
    if promptsDir:
        promptsPath = Path(promptsDir)
    else:
        promptsPath = Path(__file__).resolve().parent.parent / "prompts"

    promptText = (promptsPath / promptName).read_text(encoding="utf-8")
    _promptCache[promptName] = promptText
    return promptText
