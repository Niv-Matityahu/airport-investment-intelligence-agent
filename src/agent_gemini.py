"""
Gemini backend for the airport agent — a manual function-calling loop over the
same tools, system prompt, and deterministic dispatcher as the Anthropic agent.
Only the per-turn model call and message/function-response shapes differ.

Same public interface as AirportAgent: .chat(text) -> {reply, trace, error?}, .reset().
"""
from __future__ import annotations

from google.genai import types

from . import llm
from .agent import GROUNDING_NUDGE, MAX_ROUNDS, SYSTEM_PROMPT, _preview
from .tools import METRIC_TOOLS, TOOL_SPECS, dispatch, needs_grounding


def _declarations() -> list[dict]:
    """Anthropic tool schema -> Gemini function declaration (input_schema -> parameters)."""
    return [
        {
            "name": t["name"],
            "description": t["description"],
            "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
        }
        for t in TOOL_SPECS
    ]


class GeminiAirportAgent:
    def __init__(self):
        self.client = llm.get_gemini_client()
        self.model = llm.GEMINI_MODEL
        self.decls = _declarations()
        self.messages: list = []

    def reset(self):
        self.messages = []

    def _config(self, thinking: bool):
        cfg = {
            "system_instruction": SYSTEM_PROMPT,
            "tools": [{"function_declarations": self.decls}],
            "temperature": 0.1,
            "max_output_tokens": llm.MAX_OUTPUT_TOKENS,
        }
        if thinking:
            cfg["thinking_config"] = {"thinking_budget": llm.GEMINI_THINKING_BUDGET}
        return cfg

    def _generate(self):
        # thinking_config is unsupported on some model/SDK combos — degrade gracefully.
        use_thinking = llm.GEMINI_THINKING_BUDGET > 0
        try:
            return self.client.models.generate_content(
                model=self.model, contents=self.messages, config=self._config(use_thinking))
        except Exception as e:  # noqa: BLE001
            if use_thinking and "thinking" in str(e).lower():
                return self.client.models.generate_content(
                    model=self.model, contents=self.messages, config=self._config(False))
            raise

    def chat(self, user_text: str) -> dict:
        self.messages.append({"role": "user", "parts": [types.Part.from_text(text=user_text)]})
        trace: list[dict] = []
        nudged = False
        try:
            for _ in range(MAX_ROUNDS):
                resp = self._generate()
                cand = (resp.candidates or [None])[0]
                if cand is None or cand.content is None or not cand.content.parts:
                    fr = getattr(cand, "finish_reason", None) if cand else None
                    return {"reply": "", "trace": trace,
                            "error": f"Gemini returned no content (finish_reason={fr}). "
                                     "The prompt may have been blocked."}
                parts = cand.content.parts
                fcs = [p.function_call for p in parts
                       if getattr(p, "function_call", None) is not None]

                # echo the model turn (function_call parts) into history
                self.messages.append({"role": "model", "parts": parts})

                if not fcs:
                    text = "".join(
                        p.text for p in parts
                        if getattr(p, "text", None) and not getattr(p, "thought", False)
                    ).strip()
                    # Grounding guard: figures stated without any metric tool = fabricated.
                    called_metric = any(e.get("tool") in METRIC_TOOLS for e in trace)
                    if not called_metric and not nudged and needs_grounding(text):
                        nudged = True
                        self.messages.append({"role": "user",
                                              "parts": [types.Part.from_text(text=GROUNDING_NUDGE)]})
                        continue
                    return {"reply": text or "(no text in response)", "trace": trace}

                responses = []
                for fc in fcs:
                    args = dict(fc.args) if fc.args else {}
                    result = dispatch(fc.name, args)
                    trace.append({"tool": fc.name, "input": args,
                                  "result_preview": _preview(result)})
                    responses.append(types.Part.from_function_response(
                        name=fc.name,
                        response=result if isinstance(result, dict) else {"result": result},
                    ))
                self.messages.append({"role": "user", "parts": responses})

            return {"reply": "I reached my tool-call limit for this question. "
                    "Could you narrow it down a little?", "trace": trace}

        except Exception as e:  # noqa: BLE001 - surface any provider error to the UI
            return {"reply": "", "trace": trace,
                    "error": f"Gemini error: {type(e).__name__}: {e}"}
