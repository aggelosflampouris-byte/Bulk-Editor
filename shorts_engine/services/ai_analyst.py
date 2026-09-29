"""
services/ai_analyst.py — AI Strategy Analyst (Qwen 2.5 72B / OpenRouter).

Provides a stateful, streaming chat interface backed by Qwen2.5-72B-Instruct
via the OpenRouter API (OpenAI-compatible endpoint).

Responsibilities:
  1. Maintain multi-turn conversation history with role=system analytics injection.
  2. Stream responses token-by-token for a fluid chat UX.
  3. Enforce a strict system persona focused on evidence-based Greek YouTube strategy.
  4. Sanitize and validate inputs to prevent prompt injection.

Usage:
    analyst = AIAnalyst(api_key="or-...", analytics_context=ctx)
    for chunk in analyst.stream_response(user_message="..."):
        print(chunk, end="", flush=True)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Generator

import httpx

logger = logging.getLogger(__name__)

# OpenRouter Qwen model identifier
QWEN_MODEL: str = "qwen/qwen3-30b-a3b"

# Limits
MAX_HISTORY_TURNS: int = 20      # Keep at most 20 back-and-forth pairs
MAX_PROMPT_CHARS: int = 6000     # Guard against runaway user input
STREAM_TIMEOUT_SECONDS: float = 120.0

_SYSTEM_PERSONA: str = """\
You are **Greek Shorts Strategy Analyst**, an elite AI consultant specializing in \
@DianismaNews — a Greek YouTube Shorts channel focused on investigative politics, \
economics, and current affairs.

You have been given a complete analytics snapshot of the channel's production history. \
Your role is to analyze this data and give **concrete, evidence-based recommendations** on:

1. **Which videos to upload next** — based on niche performance and virality patterns.
2. **When to upload** — based on Greek audience traffic windows and optimal scheduling slots.
3. **Editing changes** — caption style, hook structure, clip pacing, and subtitle positioning \
   to improve retention.
4. **Metadata changes** — title optimization, description structure, tag strategy, and \
   thumbnail directives to improve CTR.
5. **Content gaps** — identify underexplored niches or hook patterns that could unlock growth.

**Rules:**
- Always cite evidence from the analytics data (specific video titles, scores, niches).
- Be direct, specific, and actionable. Never give vague advice.
- Respond in fluent English unless the user explicitly asks for Greek.
- When asked about upload times, always refer to Greek timezone (Europe/Athens) data.
- Never fabricate statistics. If data is unavailable, say so explicitly.

**Analytics Data (ground truth — treat as authoritative):**
{analytics_markdown}
"""


@dataclass
class ChatMessage:
    """A single message in the conversation history."""
    role: str   # "system" | "user" | "assistant"
    content: str


@dataclass
class AIAnalyst:
    """
    Stateful streaming AI analyst backed by Qwen2.5 72B via OpenRouter.
    """
    api_key: str
    analytics_markdown: str
    model: str = QWEN_MODEL
    history: list[ChatMessage] = field(default_factory=list)

    def _system_message(self) -> ChatMessage:
        """Construct the system message with injected analytics."""
        return ChatMessage(
            role="system",
            content=_SYSTEM_PERSONA.format(analytics_markdown=self.analytics_markdown),
        )

    def _build_messages(self, user_input: str) -> list[dict[str, str]]:
        """Build the full messages list for the API call."""
        messages: list[dict[str, str]] = [
            {"role": self._system_message().role, "content": self._system_message().content}
        ]
        # Include trimmed history (last MAX_HISTORY_TURNS turns)
        trimmed = self.history[-(MAX_HISTORY_TURNS * 2):]
        for msg in trimmed:
            messages.append({"role": msg.role, "content": msg.content})
        # Append the new user message
        messages.append({"role": "user", "content": user_input[:MAX_PROMPT_CHARS]})
        return messages

    def stream_response(self, user_input: str) -> Generator[str, None, None]:
        """
        Stream the AI analyst's response token by token.

        Appends the user message to history before calling the API,
        then appends the completed assistant response after streaming.

        Args:
            user_input: The user's raw message text.

        Yields:
            Individual text delta chunks from the API stream.

        Raises:
            RuntimeError: On API errors or connection failures.
        """
        sanitized_input = user_input.strip()
        if not sanitized_input:
            return

        messages = self._build_messages(sanitized_input)

        # Record user message before streaming
        self.history.append(ChatMessage(role="user", content=sanitized_input))

        collected_response: list[str] = []

        try:
            with httpx.Client(timeout=STREAM_TIMEOUT_SECONDS) as client:
                with client.stream(
                    "POST",
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://github.com/aggelosflampouris-byte/Bulk-Editor",
                        "X-Title": "Greek Shorts Engine AI Analyst",
                    },
                    json={
                        "model": self.model,
                        "messages": messages,
                        "stream": True,
                        "temperature": 0.35,
                        "max_tokens": 2048,
                    },
                ) as response:
                    if response.status_code != 200:
                        body = response.read().decode("utf-8", errors="replace")
                        raise RuntimeError(
                            f"OpenRouter API error {response.status_code}: {body[:400]}"
                        )

                    for line in response.iter_lines():
                        if not line or not line.startswith("data: "):
                            continue
                        payload = line[6:]
                        if payload.strip() == "[DONE]":
                            break
                        try:
                            import json as _json
                            data = _json.loads(payload)
                            delta = (
                                data.get("choices", [{}])[0]
                                .get("delta", {})
                                .get("content", "")
                            )
                            if delta:
                                collected_response.append(delta)
                                yield delta
                        except (ValueError, KeyError, IndexError):
                            continue

        except httpx.TimeoutException as exc:
            raise RuntimeError(
                f"AI Analyst request timed out after {STREAM_TIMEOUT_SECONDS}s: {exc}"
            ) from exc
        except httpx.RequestError as exc:
            raise RuntimeError(f"AI Analyst connection error: {exc}") from exc

        # Record the full assistant response
        full_response = "".join(collected_response)
        if full_response:
            self.history.append(ChatMessage(role="assistant", content=full_response))

    def clear_history(self) -> None:
        """Reset conversation history while preserving analytics context."""
        self.history.clear()

    def refresh_analytics(self, new_analytics_markdown: str) -> None:
        """Hot-swap the analytics context without clearing history."""
        self.analytics_markdown = new_analytics_markdown
