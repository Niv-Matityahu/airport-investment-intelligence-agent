"""
Terminal chat with the Airport Investment Intelligence Agent — no UI needed.

    export ANTHROPIC_API_KEY=sk-ant-...
    python chat_cli.py                       # interactive
    python chat_cli.py "Compare LAX and SNA congestion"   # one-shot

Type 'reset' to clear history, 'exit'/Ctrl-D to quit.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def _load_env():
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


_load_env()

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
        key = "GEMINI_API_KEY" if llm.provider() == "gemini" else "ANTHROPIC_API_KEY"
        print(f"No {key} found. Set it (see .env.example) and retry.")
        sys.exit(1)

    from src.agent import make_agent
    agent = make_agent()
    print(f"Airport Investment Intelligence Agent "
          f"(provider: {llm.provider()}, model: {llm.active_model()})")

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
