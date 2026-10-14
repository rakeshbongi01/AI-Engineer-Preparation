import time, uuid
from openai import LengthFinishReasonError
from pydantic import ValidationError
from src.llm.client import CLIENT, CFG, _cost, _log


class ExtractionError(Exception): ...

def extract(messages, schema, validate=None, model=None, max_attempts=3, max_tokens=1500, tag="", prompt_version= ""):
    """Structured Outputs + business validation + feedback retry. Returns a schema instance."""
    model = model or CFG["default_model"]
    msgs = list(messages)
    for attempt in range(1, max_attempts+1):
        t0 = time.perf_counter()
        try:
            resp = CLIENT.chat.completions.parse(
                model = model, messages = msgs, response_format = schema, temperature = 0, max_tokens = max_tokens
            )
        except LengthFinishReasonError as e:                       # parse() raises this on finish_reason="length"
            raise ExtractionError("output truncated - raise max_tokens or shrink the input")
        except ValidationError as e:
            msgs.append({"role":"user", "content": f"Output failed schema validation: {e}. Fix it."})
            continue
        choice, u = resp.choices[0], resp.usage
        _log({
            "id": str(uuid.uuid4()), "ts": time.time(), "tag":tag, "model":model, "prompt_version": prompt_version, "attempt": attempt, 
            "prompt_tokens": u.prompt_tokens, "completion_tokens":u.completion_tokens, "total_s": round(time.perf_counter()-t0, 3), 
            "finish_reason": choice.finish_reason, "cost_usd": round(_cost(model, u.prompt_tokens, u.completion_tokens), 6)
        })

        if choice.message.refusal:                                  # safety refusal: content is None
            raise ExtractionError(f"model refused to answer: {choice.message.refusal}")      

        obj = choice.message.parsed                                 # typed Pydantic object
        errors = validate(obj) if validate else []
        if not errors:
            return obj
        print(f"[attempt {attempt}] validation errors: {errors}.")
        msgs.append({"role":"assistant", "content": choice.message.content})
        feedback = ("Your output failed these checks:\n- " + "\n- ".join(errors) + "\nReturn corrected output. Use null if a value is not stated.")  
        msgs.append({"role":"user", "content": feedback})
    raise ExtractionError(f"failed after {max_attempts} attempts")