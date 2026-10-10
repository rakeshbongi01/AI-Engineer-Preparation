import json, os, time, uuid, yaml
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APITimeoutError, APIConnectionError
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type

load_dotenv()

CFG = yaml.safe_load(Path("config/models.yaml").read_text())
LOG = Path("logs/llm_calls.jsonl"); LOG.parent.mkdir(exist_ok=True, parents=True)

CLIENT = OpenAI() # reads OPENAI_API_KEY from the environment

def _cost(model, p_tok, c_tok):
 price = CFG["models"].get(model, {"input_per_1m": 0, "output_per_1m": 0})
 return (p_tok * price["input_per_1m"] + c_tok * price["output_per_1m"]) / 1_000_000

def _log(rec):
 with LOG.open("a") as f:
  f.write(json.dumps(rec) + "\n")

RETRYABLE = (RateLimitError, APITimeoutError, APIConnectionError)

@retry(retry=retry_if_exception_type(RETRYABLE), wait=wait_random_exponential(min=1, max=30), stop=stop_after_attempt(6))

def chat(messages, model=None, temperature=0.0, max_tokens=512, stream=False, tag=""):
    model = model or CFG["default_model"]
    t0 = time.perf_counter(); ttft = None; text = []; usage =None; Finish = None
    kwargs = dict(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)

    if stream:
        resp = CLIENT.chat.completions.create(**kwargs,stream=True, stream_options= {"include_usage": True})
        for chunk in resp:
            if chunk.choices:
                delta = chunk.choices[0].delta.content
                if delta:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                        text.append(delta); print(delta, end="", flush=True)
                    finish = chunk.choices[0].finish_reason or finish
            if chunk.usage:
                usage = chunk.usage # final chunk carries usage
        print()

    else:
       resp = CLIENT.chat.completions.create(**kwargs)
       text.append(resp.choices[0].message.content)
       usage, finish = resp.usage, resp.choices[0].finish_reason
    total = time.perf_counter() - t0
    p_tok, c_tok = usage.prompt_tokens, usage.completion_tokens
    rec = {"id": str(uuid.uuid4()), "ts": time.time(), "tag": tag, "model": model,
    "stream": stream, "temperature": temperature,
    "prompt_tokens": p_tok, "completion_tokens": c_tok,
    "ttft_s": round(ttft, 3) if ttft else None, "total_s": round(total, 3),
    "tps": round(c_tok / (total - (ttft or 0)), 1) if c_tok and total > (ttft or 0) else None,
    "finish_reason": finish, "cost_usd": round(_cost(model, p_tok, c_tok), 6)}

    _log(rec)

    if finish == "length":
        print(f"Warning: model {model} hit max_tokens limit ({max_tokens})")
    return "".join(text), rec