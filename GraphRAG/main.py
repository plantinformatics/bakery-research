import logging
import uuid
from typing import AsyncGenerator, Optional

from ag_ui.core import (
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
from fastapi.responses import StreamingResponse

from Query import ErrorEvent, PlantBioRAG, ResultEvent, StageChangeEvent, TextEvent

logger = logging.getLogger(__name__)

app = FastAPI()

# Shared across requests: holds the Neo4j/Gemini clients only, no per-run state.
rag = PlantBioRAG()


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
    try:
        async for event in rag.query(_latest_user_message(input)):
            if isinstance(event, StageChangeEvent):
                yield encoder.encode(
                    StateSnapshotEvent(snapshot=event.state.model_dump(mode="json"))
                )
            elif isinstance(event, TextEvent):
                if message_id is None:
                    message_id = str(uuid.uuid4())
                    yield encoder.encode(
                        TextMessageStartEvent(message_id=message_id, role="assistant")
                    )
                yield encoder.encode(
                    TextMessageContentEvent(message_id=message_id, delta=event.text)
                )
            elif isinstance(event, ResultEvent):
                if message_id is not None:
                    yield encoder.encode(TextMessageEndEvent(message_id=message_id))
                yield encoder.encode(
                    StateSnapshotEvent(snapshot=event.state.model_dump(mode="json"))
                )
                yield encoder.encode(
                    RunFinishedEvent(thread_id=input.thread_id, run_id=input.run_id)
                )
            elif isinstance(event, ErrorEvent):
                if message_id is not None:
                    yield encoder.encode(TextMessageEndEvent(message_id=message_id))
                yield encoder.encode(
                    RunErrorEvent(message=event.state.error or "Unknown error")
                )
    except Exception as e:
        logger.exception("Unhandled error while streaming AG-UI events: %s", e)
        if message_id is not None:
            yield encoder.encode(TextMessageEndEvent(message_id=message_id))
        yield encoder.encode(RunErrorEvent(message=str(e)))


@app.post("/agent")
async def run_agent(input: RunAgentInput) -> StreamingResponse:
    encoder = EventEncoder()
    return StreamingResponse(
        _run_agui_events(input), media_type=encoder.get_content_type()
    )
