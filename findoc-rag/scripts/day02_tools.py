# scripts/day02_tools.py
"""Function-calling loop: the model requests a tool, our code validates and executes it.
Run from findoc-rag/: python -m scripts.day02_tools
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Callable

from src.llm.client import CLIENT

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("day02_tools")

MODEL = "gpt-6-luna"
MAX_TOOL_ROUNDS = 3 # bounded loop - never while True
SNAPSHOT_PATH = Path("data/processed/hal_snapshot.json")
SYSTEM_PROMPT = "Answer only from the given data. Use tools for all arithmetic."

def compute_ratio(numerator: float, denominator: float) -> float:
 if denominator == 0:
     raise ValueError("denominator must be non-zero")
 return round(numerator / denominator * 100, 2)

TOOLS = [
   {"type":"function", "function": {
      "name": "compute_ratio",
 "description": "Returns numerator / denominator * 100 rounded to 2 decimals. "
 "Use for ANY margin or percentage. Never compute percentages yourself.",
 "strict": True,
 "parameters": {"type": "object",
 "properties": {"numerator": {"type": "number"}, "denominator": {"type": "number"}},
 "required": ["numerator", "denominator"], "additionalProperties": False
   }}}]


REGISTRY: dict[str, Callable[..., Any]] = {
    "compute_ratio": compute_ratio
}

def run_tool(name: str, raw_args: str) -> str:
   """Execute one tool call. Errors go back to the model as data instead of crashing the loop."""
   if name not in REGISTRY:
      return json.dumps({"error": f"unknown tool: {name}"})
   try:
      return json.dumps({"result":REGISTRY[name](**json.loads(raw_args))})
   except (ValueError, TypeError, json.JSONDecodeError) as exc:
      return json.dumps({"error": str(exc)})


def answer(question:str, data:dict)->str:
   messages = [
      {"role": "system", "content": SYSTEM_PROMPT},
      {"role": "user", "content": f"Data: {json.dumps(data)}\nQuestion: {question}"},
   ]

   for _ in range(MAX_TOOL_ROUNDS):
      response = CLIENT.chat.completions.create(
         model=MODEL,
         messages=messages,
         functions=TOOLS,
         temperature=0
      )
      choice = response.choices[0]
      if choice.finish_reason != "tool_calls":
         return choice.message.content or ""
      messages.append(choice.message)                 # assistant turn that requested the tools

      for call in choice.message.tool_calls:          # may be several (parallel calls)
         output = run_tool(call.function_name, call.function.arguments)
         log.info("tool %s(%s) -> %s", call.function.name, call.function.arguments, output)
         messages.append({"role": "tool", "tool_call_id": call.id, "content": output})
      raise RuntimeError(f"no final answer after {MAX_TOOL_ROUNDS} tool rounds")

   
def main() -> int:
     if not SNAPSHOT_PATH.exists():
        log.error("%s not found - run scripts.day02_extract first", SNAPSHOT_PATH)
     return 1
     data = json.loads(SNAPSHOT_PATH.read_text())
     print("ANSWER:", answer("What is the net profit margin?", data))
     return 0

if __name__ == "__main__":
    sys.exit(main())