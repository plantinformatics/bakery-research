import logging
import uuid
from typing import Any, AsyncGenerator, Optional, Tuple

from ag_ui.core import (
    ReasoningEndEvent,
    ReasoningMessageContentEvent,
    ReasoningMessageEndEvent,
    ReasoningMessageStartEvent,
    ReasoningStartEvent,
    RunAgentInput,
    RunErrorEvent,
    RunFinishedEvent,
    RunStartedEvent,
    StateSnapshotEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
    TextMessageStartEvent,
)
from ag_ui.encoder import EventEncoder
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from Query import (
    ANSWER_THINKING_LEVEL,
    AVAILABLE_MODELS,
    AVAILABLE_REASONING_LEVELS,
    GEMINI_MODEL,
    ErrorEvent,
    PlantBioRAG,
    ReasoningEvent,
    ResultEvent,
    StageChangeEvent,
    TextEvent,
)

logger = logging.getLogger(__name__)

app = FastAPI()

# Allow the local Next.js UI (`frontend/`, typically :3000) to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared across requests: holds the Neo4j/Gemini clients only, no per-run state.
rag = PlantBioRAG()


def _forwarded_model_selection(
    input: RunAgentInput,
) -> Tuple[Optional[str], Optional[str]]:
    """Reads the frontend's model/reasoning selection out of `forwardedProps`.

    `frontend/hooks/use-model-config.ts` publishes the selection as
    `ModelContext.config.{modelName,reasoningEffort}`, which
    `AgUiThreadRuntimeCore.buildRunInput` (in `@assistant-ui/react-ag-ui`)
    spreads into `RunAgentInput.forwardedProps` on every run. `PlantBioRAG.
    query()` re-validates whatever comes back here against
    `AVAILABLE_MODELS`/`AVAILABLE_REASONING_LEVELS`: no selection at all
    (older frontend build) falls back to its defaults, but a stale or
    invalid explicit selection raises `UnavailableModelSelectionError`,
    which the `except Exception` below turns into a `RunErrorEvent`
    instead of silently answering with a different model than requested.
    """
    forwarded: Any = input.forwarded_props
    if not isinstance(forwarded, dict):
        return None, None
    model_name = forwarded.get("modelName")
    reasoning_level = forwarded.get("reasoningEffort")
    return (
        model_name if isinstance(model_name, str) else None,
        reasoning_level if isinstance(reasoning_level, str) else None,
    )


def _latest_user_message(input: RunAgentInput) -> str:
    """Single-turn only: drive the pipeline off the latest user message,
    per the plan's scope boundary. Ignores prior conversation history."""
    for message in reversed(input.messages):
        if message.role != "user":
            continue
        content = message.content
        if isinstance(content, str):
            return content
        return "".join(
            part.text for part in content if getattr(part, "type", None) == "text"
        )
    return ""


async def _run_agui_events(input: RunAgentInput) -> AsyncGenerator[str, None]:
    """Drives `PlantBioRAG.query()` and maps its internal events onto
    `ag_ui.core` events, encoded as SSE strings."""
    encoder = EventEncoder()
    yield encoder.encode(
        RunStartedEvent(thread_id=input.thread_id, run_id=input.run_id)
    )

    message_id: Optional[str] = None
    reasoning_message_id: Optional[str] = None

    def _close_reasoning():
        nonlocal reasoning_message_id
        if reasoning_message_id is None:
            return []
        events = [
            encoder.encode(ReasoningMessageEndEvent(message_id=reasoning_message_id)),
            encoder.encode(ReasoningEndEvent(message_id=reasoning_message_id)),
        ]
        reasoning_message_id = None
        return events

    model_name, reasoning_level = _forwarded_model_selection(input)
    try:
        async for event in rag.query(
            _latest_user_message(input),
            model_name=model_name,
            reasoning_level=reasoning_level,
        ):
            if isinstance(event, StageChangeEvent):
                yield encoder.encode(
                    StateSnapshotEvent(snapshot=event.state.model_dump(mode="json"))
                )
            elif isinstance(event, ReasoningEvent):
                if reasoning_message_id is None:
                    reasoning_message_id = str(uuid.uuid4())
                    yield encoder.encode(
                        ReasoningStartEvent(message_id=reasoning_message_id)
                    )
                    yield encoder.encode(
                        ReasoningMessageStartEvent(
                            message_id=reasoning_message_id, role="reasoning"
                        )
                    )
                yield encoder.encode(
                    ReasoningMessageContentEvent(
                        message_id=reasoning_message_id, delta=event.text
                    )
                )
            elif isinstance(event, TextEvent):
                for e in _close_reasoning():
                    yield e
                if message_id is None:
                    message_id = str(uuid.uuid4())
                    yield encoder.encode(
                        TextMessageStartEvent(message_id=message_id, role="assistant")
                    )
                yield encoder.encode(
                    TextMessageContentEvent(message_id=message_id, delta=event.text)
                )
            elif isinstance(event, ResultEvent):
                for e in _close_reasoning():
                    yield e
                if message_id is not None:
                    yield encoder.encode(TextMessageEndEvent(message_id=message_id))
                yield encoder.encode(
                    StateSnapshotEvent(snapshot=event.state.model_dump(mode="json"))
                )
                yield encoder.encode(
                    RunFinishedEvent(thread_id=input.thread_id, run_id=input.run_id)
                )
            elif isinstance(event, ErrorEvent):
                for e in _close_reasoning():
                    yield e
                if message_id is not None:
                    yield encoder.encode(TextMessageEndEvent(message_id=message_id))
                yield encoder.encode(
                    RunErrorEvent(message=event.state.error or "Unknown error")
                )
    except Exception as e:
        logger.exception("Unhandled error while streaming AG-UI events: %s", e)
        for ev in _close_reasoning():
            yield ev
        if message_id is not None:
            yield encoder.encode(TextMessageEndEvent(message_id=message_id))
        yield encoder.encode(RunErrorEvent(message=str(e)))


@app.post("/agent")
async def run_agent(input: RunAgentInput) -> StreamingResponse:
    encoder = EventEncoder()
    return StreamingResponse(
        _run_agui_events(input), media_type=encoder.get_content_type()
    )


@app.get("/options")
async def get_options() -> dict:
    """Model/reasoning choices for the frontend's selectors
    (`frontend/components/model-selector.tsx`), kept in sync with what
    `PlantBioRAG.query()` actually accepts (`frontend/config/models.json`,
    loaded by `Query.py`)."""
    return {
        "models": AVAILABLE_MODELS,
        "defaultModel": GEMINI_MODEL,
        "reasoningLevels": AVAILABLE_REASONING_LEVELS,
        "defaultReasoningLevel": ANSWER_THINKING_LEVEL,
    }
