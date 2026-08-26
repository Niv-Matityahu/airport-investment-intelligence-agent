"""
The conversational agent: an Anthropic tool-use loop over the deterministic
airport tools. The LLM plans and narrates; every airport fact comes from a tool.

Public surface:
    agent = AirportAgent()
    result = agent.chat("Compare LA and Santa Ana congestion")
    # result = {"reply": str, "trace": [ {tool, input, result_preview}, ... ]}

State (the message history) lives on the instance, so follow-up questions work.
"""
from __future__ import annotations

import json

import anthropic

from . import llm
from .tools import TOOL_SPECS, dispatch

MAX_ROUNDS = 8  # hard cap on tool rounds per user turn

SYSTEM_PROMPT = """\
You are the Airport Investment Intelligence Agent for a firm that invests in \
US airport modernization and terminal-expansion projects. You help analysts \
find airports where expansion capital will be most productive — i.e. where \
demand is large and growing but the airport is already capacity-constrained, so \
added capacity converts into more served flights and passengers.

## How you work
- Every airport NUMBER or FACT must come from a tool call. Never state a \
passenger count, delay rate, growth figure, long-haul %, or score from memory. \
If a tool doesn't have it, say so plainly.
- When the user names an airport by city or an ambiguous term (\"LA\", \
\"New York\", \"Santa Ana\"), call resolve_airport FIRST, then use the returned \
IATA code(s). If a metro has several airports, mention the alternatives.
- Prefer one well-chosen tool call over many. You may call several tools in one \
turn when they're independent (e.g. resolve two airports, then compare).
- When ranking EXPANSION CANDIDATES, exclude very small airports by default \
(pass min_enplanements around 500000) — tiny airports post volatile growth % on \
a small base and rank misleadingly high. Mention you applied a size floor, and \
offer to include smaller ones. Always call out any Low/Medium data_confidence \
entries.

## The Expansion Opportunity Score (EOS)
A 0-100 score = weighted blend of four national-percentile pillars: \
Congestion 35% (delays + cancellations — is it constrained now?), \
Growth 30% (YoY passenger growth — is demand rising?), \
Scale 20% (passenger volume, log — how big is the payoff?), \
Utilization 15% (departures per runway — are the runways worked hard?). \
Higher = stronger expansion candidate. When you rank or recommend, briefly \
explain WHICH pillars drove the result, using the tool's breakdown.

## Communicating honestly (required)
- State assumptions and scope: the data is US commercial airports only; \
congestion and long-haul come from the BTS On-Time domestic dataset (sampled \
months, domestic mainline flights — international long-haul is NOT included); \
passengers are FAA CY2024 enplanements. Surface these caveats when they matter \
to the answer (especially long-haul % and unmet-demand questions).
- Flag uncertainty: if an airport has low data_confidence or imputed pillars, \
say the estimate is soft. \"Unmet demand\" is a transparent proxy, not a \
forecast — pass along its caveats.
- Don't overclaim. This tool surfaces candidates and reasoning; it is not \
investment advice.

## Style
Concise analyst voice. Lead with the answer, then the reasoning. Use short \
tables or bullet lists for rankings/comparisons. Offer a relevant follow-up \
when useful. You cannot render live charts — describe the numbers.\
"""


def _preview(obj, limit: int = 600) -> str:
    s = json.dumps(obj, ensure_ascii=False)
    return s if len(s) <= limit else s[:limit] + "…"


class AirportAgent:
    def __init__(self):
        self.client = llm.get_client()
        self.model = llm.MODEL
        self.messages: list[dict] = []

    def reset(self):
        self.messages = []

    def chat(self, user_text: str) -> dict:
        """Run one user turn to completion. Returns {reply, trace, error?}."""
        self.messages.append({"role": "user", "content": user_text})
        trace: list[dict] = []

        try:
            for _ in range(MAX_ROUNDS):
                resp = self.client.messages.create(
                    model=self.model,
                    max_tokens=llm.MAX_TOKENS,
                    system=SYSTEM_PROMPT,
                    tools=TOOL_SPECS,
                    messages=self.messages,
                )
                # preserve the full assistant turn (incl. tool_use blocks) in history
                self.messages.append({"role": "assistant", "content": resp.content})

                if resp.stop_reason != "tool_use":
                    return {"reply": _final_text(resp), "trace": trace}

                # execute every tool_use block, return all results in one user turn
                tool_results = []
                for block in resp.content:
                    if block.type != "tool_use":
                        continue
                    result = dispatch(block.name, block.input)
                    trace.append({"tool": block.name, "input": block.input,
                                  "result_preview": _preview(result)})
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result, ensure_ascii=False),
                        "is_error": isinstance(result, dict) and "error" in result,
                    })
                self.messages.append({"role": "user", "content": tool_results})

            # ran out of rounds — ask the model for a best-effort wrap-up
            return {"reply": "I reached my tool-call limit for this question. "
                    "Could you narrow it down a little?", "trace": trace}

        except anthropic.AuthenticationError:
            return {"reply": "", "trace": trace,
                    "error": "Authentication failed — check ANTHROPIC_API_KEY."}
        except anthropic.RateLimitError:
            return {"reply": "", "trace": trace,
                    "error": "Rate limited by the API. Please retry in a moment."}
        except anthropic.APIError as e:
            return {"reply": "", "trace": trace, "error": f"API error: {e}"}


def _final_text(resp) -> str:
    return "\n".join(b.text for b in resp.content if getattr(b, "type", None) == "text").strip()
