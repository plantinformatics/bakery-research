"use client";

import { makeAssistantDataUI, useAuiState } from "@assistant-ui/react";
import type { ThreadMessage } from "@assistant-ui/react";
import { DownloadIcon, ExternalLinkIcon, MailIcon, MapIcon } from "lucide-react";
import { Button } from "@/components/ui/button";

/**
 * Renderers for the named `data` parts `GraphRAG/Query.py` attaches to an
 * assistant message via `UiDataEvent` (sent as AG-UI CUSTOM events by
 * `GraphRAG/main.py`). Mount once inside `AssistantRuntimeProvider`.
 */

// Long mailto: URLs get cut off or rejected by some mail clients, so the
// emailed copy is capped; "Download chat" always has the full export.
const MAILTO_TRANSCRIPT_MAX_CHARS = 1500;

function messageText(message: ThreadMessage): string {
  return message.content
    .flatMap((part) => (part.type === "text" ? [part.text] : []))
    .join("")
    .trim();
}

function chatTranscript(messages: readonly ThreadMessage[]): string {
  return messages
    .flatMap((message) => {
      const text = messageText(message);
      if (!text) return [];
      return [`**${message.role === "user" ? "You" : "Assistant"}:** ${text}`];
    })
    .join("\n\n");
}

function downloadTranscript(transcript: string) {
  const blob = new Blob([`# Bakery Research chat\n\n${transcript}\n`], {
    type: "text/markdown",
  });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `bakery-research-chat-${new Date().toISOString().slice(0, 10)}.md`;
  link.click();
  URL.revokeObjectURL(url);
}

function mailtoHref(email: string, transcript: string): string {
  const truncated = transcript.length > MAILTO_TRANSCRIPT_MAX_CHARS;
  const body = [
    "Hi, I asked the Bakery Research assistant a question it could not answer.",
    "",
    "Chat so far:",
    "",
    truncated
      ? `${transcript.slice(0, MAILTO_TRANSCRIPT_MAX_CHARS)}\n\n[Chat truncated - use "Download chat" in the assistant and attach the file for the full chat.]`
      : transcript,
  ].join("\n");
  const params = new URLSearchParams({
    subject: "Bakery Research assistant: question outside its scope",
    body,
  });
  // URLSearchParams encodes spaces as "+", which mail clients show literally.
  return `mailto:${email}?${params.toString().replaceAll("+", "%20")}`;
}

type OutOfScopeData = { contactEmail?: string };

function OutOfScopeContact({ data }: { data: OutOfScopeData }) {
  const messages = useAuiState((s) => s.thread.messages);
  const transcript = chatTranscript(messages);
  const email = data.contactEmail?.trim();

  return (
    <div
      data-slot="out-of-scope-contact"
      className="border-border bg-muted/40 my-3 flex flex-col gap-2 rounded-lg border p-3"
    >
      <p className="text-muted-foreground text-xs">
        Contact the team with a copy of this chat:
      </p>
      <div className="flex flex-wrap gap-2">
        {email ? (
          <Button asChild size="sm">
            <a href={mailtoHref(email, transcript)}>
              <MailIcon />
              Email {email}
            </a>
          </Button>
        ) : null}
        <Button
          size="sm"
          variant="outline"
          onClick={() => downloadTranscript(transcript)}
        >
          <DownloadIcon />
          Download chat
        </Button>
      </div>
    </div>
  );
}

type PretzelQuestionData = { docsUrl?: string };

function PretzelQuestion({ data }: { data: PretzelQuestionData }) {
  return (
    <a
      data-slot="pretzel-question"
      href={data.docsUrl}
      target="_blank"
      rel="noreferrer"
      className="my-3 flex items-center gap-3 rounded-lg border border-sky-200 bg-sky-100 p-3 text-sky-950 no-underline transition-colors hover:border-sky-300 hover:bg-sky-200 dark:border-sky-900 dark:bg-sky-950 dark:text-sky-50 dark:hover:border-sky-700 dark:hover:bg-sky-900"
    >
      <MapIcon className="size-5 shrink-0 text-sky-600 dark:text-sky-300" />
      <span className="flex min-w-0 flex-1 flex-col">
        <span className="text-sm font-medium">Pretzel how-to question</span>
        <span className="text-xs text-sky-800 dark:text-sky-200">
          This answer draws on the Pretzel documentation. Open the docs for
          more detail.
        </span>
      </span>
      <ExternalLinkIcon className="size-4 shrink-0 text-sky-600 dark:text-sky-300" />
    </a>
  );
}

export const OutOfScopeContactUI = makeAssistantDataUI<OutOfScopeData>({
  name: "out_of_scope",
  render: OutOfScopeContact,
});

export const PretzelQuestionUI = makeAssistantDataUI<PretzelQuestionData>({
  name: "pretzel_question",
  render: PretzelQuestion,
});

export function MessageDataBoxes() {
  return (
    <>
      <OutOfScopeContactUI />
      <PretzelQuestionUI />
    </>
  );
}
