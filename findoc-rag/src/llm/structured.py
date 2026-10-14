"""Structured Outputs + business validation + feedback retry, with cost logging."""
import logging
import time
import uuid
from typing import Callable, TypeVar

from openai import LengthFinishReasonError
from pydantic import BaseModel, ValidationError

from src.llm.client import CFG, CLIENT, _cost, _log, build_params

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)
Validator = Callable[[BaseModel], list[str]]


class ExtractionError(Exception):
    """Expected extraction failure (refusal, truncation, validation) - callers quarantine it."""


def _log_call(model: str, response, attempt: int, started: float, tag: str, prompt_version: str) -> None:
    usage, choice = response.usage, response.choices[0]
    details = getattr(usage, "completion_tokens_details", None)
    _log({"id": str(uuid.uuid4()), "ts": time.time(), "tag": tag, "model": model,
          "prompt_version": prompt_version, "attempt": attempt,
          "prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens,
          "reasoning_tokens": getattr(details, "reasoning_tokens", 0) if details else 0,
          "total_s": round(time.perf_counter() - started, 3), "finish_reason": choice.finish_reason,
          "cost_usd": round(_cost(model, usage.prompt_tokens, usage.completion_tokens), 6)})


def extract(messages: list[dict], schema: type[T], validate: Validator | None = None,
            model: str | None = None, max_attempts: int = 3, max_tokens: int = 2000,
            tag: str = "", prompt_version: str = "") -> T:
    """Return a validated schema instance or raise ExtractionError."""
    model = model or CFG["default_model"]
    msgs = list(messages)
    for attempt in range(1, max_attempts + 1):
        started = time.perf_counter()
        try:
            response = CLIENT.chat.completions.parse(
                messages=msgs, response_format=schema,
                **build_params(model, temperature=0, max_tokens=max_tokens))
        except LengthFinishReasonError as exc:        # parse() raises this on finish_reason="length"
            raise ExtractionError("output truncated - raise max_tokens or shrink the input") from exc
        except ValidationError as exc:                # rare with strict mode, but handle it
            msgs.append({"role": "user", "content": f"Output failed schema validation: {exc}. Fix it."})
            continue
        _log_call(model, response, attempt, started, tag, prompt_version)

        message = response.choices[0].message
        if message.refusal:                           # safety refusal: content is None
            raise ExtractionError(f"model refused: {message.refusal}")

        result = message.parsed
        errors = validate(result) if validate else []
        if not errors:
            return result
        log.warning("attempt %d failed validation: %s", attempt, errors)
        feedback = ("Your output failed these checks:\n- " + "\n- ".join(errors)
                    + "\nReturn corrected output. Use null if a value is not stated.")
        msgs += [{"role": "assistant", "content": message.content},
                 {"role": "user", "content": feedback}]
    raise ExtractionError(f"failed validation after {max_attempts} attempts")