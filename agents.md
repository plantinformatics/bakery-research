
## General instructions

This a pre-production prototype of a RAG 

Avoid the use of wrapper functions to avoid the changing of existing function names.

Avoid killing and starting processes. The development enviroment is already set up. If it isn't ask me to start them up. 

When a step doesn't need my input, keep going. Put status notes in the
same message as your next action.
Stop and ask only when you can't continue without me, or before anything
destructive: deleting data, force-pushing, or changing anything outside
this repository.

End every run with three headings: Blocked on me, Changed, Found

## Backend(GraphRAG) specific

When trying to run things locally, use the version of python found in .venv first.

## Frontend specific

### shadcn

This project uses shadcn for other general purpose UI components.

When instructed to build a new UI component check the shadcn skill to see if there are any existing components that can be used.

### assistant-ui

This project uses assistant-ui for chat interfaces.

Documentation: https://www.assistant-ui.com/llms-full.txt

Key patterns:
- Use AssistantRuntimeProvider at the app root
- Thread component for full chat interface
- AssistantModal for floating chat widget
- useChatRuntime hook with AI SDK transport