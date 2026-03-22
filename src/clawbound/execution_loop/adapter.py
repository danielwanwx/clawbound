"""DeterministicAdapter — scripted response sequence for testing.

Each send() call returns the next scripted response. If exhausted,
returns a final_answer with "no more scripted responses".
"""

from __future__ import annotations

from clawbound.contracts.types import (
    FinalAnswer,
    ModelError,
    ModelRequest,
    ToolCalls,
)


class DeterministicAdapter:
    def __init__(self, responses: list[FinalAnswer | ToolCalls | ModelError]) -> None:
        self._responses = list(responses)
        self._index = 0
        self.request_log: list[ModelRequest] = []

    async def send(self, request: ModelRequest) -> FinalAnswer | ToolCalls | ModelError:
        self.request_log.append(request)

        if self._index < len(self._responses):
            response = self._responses[self._index]
            self._index += 1
            return response

        return FinalAnswer(content="[DeterministicAdapter] No more scripted responses.")
