"""Instrumented LLM client: one place for model params, retries, and cost/latency logging."""
import json
import time
import uuid
from pathlib import Path

import yaml
from dotenv import load_dotenv
from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential

load_dotenv()

CFG = yaml.safe_load(Path("config/models.yaml").read_text())
LOG_PATH = Path("logs/llm_calls.jsonl")
LOG_PATH.parent.mkdir(exist_ok=True)

CLIENT = OpenAI(max_retries=0)          # reads OPENAI_API_KEY; tenacity owns retries (no double retry)
RETRYABLE = (RateLimitError, APITimeoutError, APIConnectionError)
DEFAULT_REASONING_EFFORT = "low"


def build_params(model: str, temperature: float, max_tokens: int,
                 reasoning_effort: str | None = None) -> dict:
    """Model-specific request parameters, driven by config/models.yaml.

    - Reasoning models take reasoning_effort instead of temperature.
    - max_completion_tokens works for every model; on reasoning models it must also
      cover the hidden reasoning tokens, so size it generously.
    - reasoning_effort overrides the configured default (e.g. "none" for tool calls).
    """
    spec = CFG["models"].get(model, {})
    params: dict = {"model": model, "max_completion_tokens": max_tokens}
    if spec.get("reasoning"):
        params["reasoning_effort"] = reasoning_effort or spec.get(
            "reasoning_effort", DEFAULT_REASONING_EFFORT)
    else:
        params["temperature"] = temperature
    return params


def _cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    price = CFG["models"].get(model, {})
    return (prompt_tokens * price.get("input_per_1m", 0)
            + completion_tokens * price.get("output_per_1m", 0)) / 1_000_000


def _log(record: dict) -> None:
    """One JSON line per call. Metrics only - never raw prompts (PII / confidential data)."""
    with LOG_PATH.open("a") as f:
        f.write(json.dumps(record) + "\n")


def _reasoning_tokens(usage) -> int:
    details = getattr(usage, "completion_tokens_details", None)
    return (getattr(details, "reasoning_tokens", 0) or 0) if details else 0


@retry(retry=retry_if_exception_type(RETRYABLE),
       wait=wait_random_exponential(min=1, max=30), stop=stop_after_attempt(5))
def chat(messages: list[dict], model: str | None = None, temperature: float = 0.0,
         max_tokens: int = 2000, stream: bool = False, tag: str = "",
         reasoning_effort: str | None = None) -> tuple[str, dict]:
    """Send a chat request; return (text, log record)."""
    model = model or CFG["default_model"]
    kwargs = dict(messages=messages, **build_params(model, temperature, max_tokens, reasoning_effort))
    started = time.perf_counter()
    ttft, parts, usage, finish = None, [], None, None

    if stream:
        response = CLIENT.chat.completions.create(**kwargs, stream=True,
                                                  stream_options={"include_usage": True})
        for chunk in response:
            if chunk.choices:
                delta = chunk.choices[0].delta.content
                if delta:
                    if ttft is None:
                        ttft = time.perf_counter() - started
                    parts.append(delta)
                    print(delta, end="", flush=True)
                finish = chunk.choices[0].finish_reason or finish
            if chunk.usage:                      # the final chunk carries usage
                usage = chunk.usage
        print()
    else:
        response = CLIENT.chat.completions.create(**kwargs)
        parts.append(response.choices[0].message.content or "")
        usage, finish = response.usage, response.choices[0].finish_reason

    total = time.perf_counter() - started
    prompt_tokens = usage.prompt_tokens if usage else 0
    completion_tokens = usage.completion_tokens if usage else 0
    generation_s = total - (ttft or 0)
    record = {
        "id": str(uuid.uuid4()), "ts": time.time(), "tag": tag, "model": model, "stream": stream,
        "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
        "reasoning_tokens": _reasoning_tokens(usage) if usage else 0,
        "ttft_s": round(ttft, 3) if ttft else None, "total_s": round(total, 3),
        "tps": round(completion_tokens / generation_s, 1) if completion_tokens and generation_s > 0 else None,
        "finish_reason": finish,
        "cost_usd": round(_cost(model, prompt_tokens, completion_tokens), 6),
    }
    _log(record)
    if finish == "length":
        print(f"[warn] output truncated at max_completion_tokens={max_tokens}")
    return "".join(parts), record