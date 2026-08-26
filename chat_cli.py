"""
Terminal chat with the Airport Investment Intelligence Agent — no UI needed.

    export ANTHROPIC_API_KEY=sk-ant-...
    python chat_cli.py                       # interactive
    python chat_cli.py "Compare LAX and SNA congestion"   # one-shot

Type 'reset' to clear history, 'exit'/Ctrl-D to quit.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import llm  # noqa: E402


def _print_result(res: dict):
    if res.get("error"):
        print(f"\n[error] {res['error']}\n")
        return
    if res.get("trace"):
        for t in res["trace"]:
            print(f"  · {t['tool']}({t['input']})")
    print(f"\n{res['reply']}\n")


def main():
    if not llm.have_credentials():
        print("No ANTHROPIC_API_KEY found. Set it (see .env.example) and retry.")
        sys.exit(1)

    from src.agent import AirportAgent
    agent = AirportAgent()
    print(f"Airport Investment Intelligence Agent (model: {llm.MODEL})")

    if len(sys.argv) > 1:  # one-shot mode
        _print_result(agent.chat(" ".join(sys.argv[1:])))
        return

    print("Ask about US airport expansion opportunities. 'reset' to clear, 'exit' to quit.\n")
    while True:
        try:
            q = input("you › ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            break
        if q.lower() == "reset":
            agent.reset()
            print("(history cleared)\n")
            continue
        _print_result(agent.chat(q))


if __name__ == "__main__":
    main()
