# scripts/day01_tokens.py (run from repo root: python -m scripts.day01_tokens /Users/rakesh/Documents/Learning/AI-Engineer-Preparation/findoc-rag/data/raw/HAL_AR-2025-26.pdf)
import sys, json, statistics, tiktoken
from pypdf import PdfReader
from src.llm.client import chat, _cost

enc = tiktoken.get_encoding("o200k_base")
reader = PdfReader(sys.argv[1])
pages = [p.extract_text() for p in reader.pages]

# 1) Token profile of the document
counts = [len(enc.encode(p)) for p in pages]
ratios = [len(p) / max(1,n) for p,n in zip(pages, counts) if n > 50 ]

print(f"pages={len(pages)} total_tokens={sum(counts):,} "
      f"avg chars/token={statistics.mean(ratios):.2f}")
worst = sorted([i for i in range(len(pages)) if counts[i] > 200],
               key=lambda i: len(pages[i]) / counts[i])[:3]

print("lowest chars/token pages (likely tables):", [w + 1 for w in worst])
print("cost to send WHOLE report as input once: $",
 round(_cost("gpt-4o-mini", sum(counts), 0), 4), "(mini) vs $",
 round(_cost("gpt-4o", sum(counts), 0), 4), "(4o)")


# 2) Sampling: same question, 3 runs each at temp 0 and 1.0
ctx = pages[worst[0]][:6000]
q = [{"role": "system", "content": "Answer only from the context. Be concise."},
 {"role": "user", "content": f"Context:\n{ctx}\n\nWhat are the 3 key figures on this page?"}]

for t in (0.0, 1.0):
 outs = [chat(q, temperature=t, max_tokens=150, tag=f"temp{t}")[0] for _ in range(3)]
 print(f"\ntemp={t}: distinct answers = {len(set(outs))}/3")

 # 3) Streaming vs non-streaming latency
_, a = chat(q, max_tokens=300, stream=False, tag="nostream")
_, b = chat(q, max_tokens=300, stream=True, tag="stream")
print(json.dumps({"non_stream_total": a["total_s"], "stream_ttft": b["ttft_s"],
 "stream_total": b["total_s"], "tps": b["tps"]}, indent=2))