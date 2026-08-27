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
from .tools import METRIC_TOOLS, TOOL_SPECS, dispatch, needs_grounding

MAX_ROUNDS = 8  # hard cap on tool rounds per user turn

# Shared by both agents: shown when a reply states airport figures but no metric
# tool was called that turn (i.e. the numbers were fabricated from memory).
GROUNDING_NUDGE = (
    "GROUNDING CHECK: your reply states airport figures (a score, percentile, "
    "passenger count, or growth number) but you did not call any data tool this "
    "turn. Never state such figures from memory — they must come from a tool. "
    "Call the right tool now (airport_report for one airport; rank_airports, "
    "compare_airports, unmet_demand, or long_haul_profile as appropriate) and "
    "answer ONLY from its result."
)

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
- NEVER accept a superlative or ranking claim as given — \"highest\", \"lowest\", \
\"#1\", \"the best/worst\", \"most/least X in the country/region\". Whether the user \
ASKS it or PRESUPPOSES it (e.g. \"since X is #1, explain why\"), first VERIFY with \
rank_airports (or compare_airports) before affirming or explaining it, and correct \
it if it's false. Do not explain \"why X is #1\" until you've confirmed X actually is.
- Prefer one well-chosen tool call over many. You may call several tools in one \
turn when they're independent (e.g. resolve two airports, then compare).
- When ranking EXPANSION CANDIDATES, exclude very small airports by default \
(pass min_enplanements around 500000) — tiny airports post volatile growth % on \
a small base and rank misleadingly high. Mention you applied a size floor, and \
offer to include smaller ones. Always call out any Low/Medium data_confidence \
entries.

## Which tool for which question
- Passenger VOLUME / \"how many passengers\" / \"how big is X\" → `airport_report` \
and quote `enplanements_2024`. That is the headline passenger figure (FAA, all \
boardings — domestic AND international). Also use `airport_report` for a single \
airport's growth, congestion, choke, score, long-haul %, and runways.
- `bts_traffic` is a LIVE cross-check of BTS T-100 **domestic** segments ONLY. It \
undercounts total passengers at international gateways (JFK, SFO, LAX, EWR, MIA) — \
often by half — so NEVER present it as the total passenger count. Use it only when \
the user explicitly asks for live/official *domestic-segment* figures or a cross-check.
- `live_traffic` (OpenSky) = real-time aircraft near an airport — a coarse \"right \
now\" activity snapshot, not capacity or scheduled traffic.

## The investment thesis (use this to reason)
The fund profits by BUILDING capacity where it is choked. A good target needs \
THREE things at once: (1) capacity CHOKE — the airport is maxed out and spills \
flights it can't serve; (2) DEMAND — traffic is growing; (3) SCALE — big enough \
to justify major capex. High demand ALONE is not enough: if an airport has room, \
it just absorbs the demand and no build is needed. Choke is the differentiator.

## The Expansion Opportunity Score (EOS)
A 0-100 score = (Choke 45% + Demand 30% + Scale 25%, as national percentiles) \
× a CHOKE GATE. The gate multiplies the score down when choke is low, so a big, \
growing, but un-choked airport can NOT rank high — the score requires a \
bottleneck by construction. Choke combines delays, cancellations, runway \
utilization, and FAA slot-control. Two authoritative capacity flags matter: \
`slot_controlled` (FAA legally caps flights — JFK/LGA/DCA/EWR — the strongest \
choke signal) and `faa_core30` (FAA congestion-tracked hub). When you rank or \
recommend, explain WHICH pillars drove it using the tool's `eos_breakdown` \
(pillars + base + choke_gate), and call out slot-control when present.

## Communicating honestly (required)
- State assumptions and scope: the data is US commercial airports only; \
congestion and long-haul come from the BTS On-Time domestic dataset (sampled \
months, domestic mainline flights — international long-haul is NOT included); \
passengers are FAA CY2024 enplanements; geography, runways and metro grouping \
come from OurAirports. All sources are PUBLIC (FAA, BTS, OurAirports; plus the \
live OpenSky and BTS T-100 APIs). Surface these caveats when they matter to the \
answer (especially long-haul % and unmet-demand questions).
- Flag uncertainty: if an airport has low data_confidence or imputed pillars, \
say the estimate is soft. \"Unmet demand\" is a transparent proxy, not a \
forecast — pass along its caveats.
- Don't overclaim. This tool surfaces candidates and reasoning; it is not \
investment advice.
- EOS ranks where adding capacity is most PRODUCTIVE (a precondition for profit) \
— it does NOT model construction cost, revenue, airline fees, or ROI. If asked \
which airport is \"most profitable\" or for a buy/sell call, name the strongest \
capacity-need candidate with its evidence, but clarify that dollar ROI needs \
separate financial diligence (capex + revenue) beyond this screen.

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
        self.client = llm.get_anthropic_client()
        self.model = llm.ANTHROPIC_MODEL
        self.messages: list[dict] = []

    def reset(self):
        self.messages = []

    def chat(self, user_text: str) -> dict:
        """Run one user turn to completion. Returns {reply, trace, error?}."""
        self.messages.append({"role": "user", "content": user_text})
        trace: list[dict] = []
        nudged = False

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
                    text = _final_text(resp)
                    called_metric = any(e.get("tool") in METRIC_TOOLS for e in trace)
                    if not called_metric and not nudged and needs_grounding(text):
                        nudged = True
                        self.messages.append({"role": "user", "content": GROUNDING_NUDGE})
                        continue
                    return {"reply": text, "trace": trace}

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


def make_agent():
    """Return the agent for the active provider (llm.provider())."""
    if llm.provider() == "gemini":
        from .agent_gemini import GeminiAirportAgent
        return GeminiAirportAgent()
    return AirportAgent()
