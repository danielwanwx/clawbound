"""ActionGate — L1 static policy gate.

Pre-execution check: "should I even attempt this tool call?"
Separate from ToolBroker's authorization.
"""

from __future__ import annotations

from clawbound.contracts.types import (
    ActionAllowed,
    ActionDenied,
    ActionGateDecision,
    RuntimePolicy,
)


class ActionGateImpl:
    def check(
        self,
        tool_name: str,
        _args: dict[str, object],
        policy: RuntimePolicy,
    ) -> ActionGateDecision:
        tool_profile = policy.tool_profile

        if tool_name in tool_profile.denied_tools:
            return ActionDenied(
                reason=f'Tool "{tool_name}" is explicitly denied by profile "{tool_profile.profile_name}".',
            )

        if tool_profile.allowed_tools and tool_name not in tool_profile.allowed_tools:
            return ActionDenied(
                reason=f'Tool "{tool_name}" is not in allowed list for profile "{tool_profile.profile_name}".',
            )

        return ActionAllowed()
