import argparse
import logging
import re
import sys
from functools import partial
from pathlib import Path
from pypdf import PdfReader
from src.llm.structured import ExtractionError, extract
from src.schemas.financials import METRICS, FinancialSnapshot

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("day02_extract")

PROMPT_VERSION = "extract_financials.v1"
KEYWORDS = ("revenue from operations", "profit after tax", "order book")
TOP_K_PAGES = 4
MAX_CHARS_PER_PAGE = 6000
OUTPUT_PATH = Path("data/processed/hal_snapshot.json")
QUARANTINE_PATH = Path("data/quarantine/hal_snapshot.err")

SYSTEM_PROMPT = """You are a financial data extraction engine.
Rules:
- Use ONLY the text inside <context>. Never use prior knowledge.
- If a value is not explicitly stated, return null for evidence, source_page and value.
- Copy numbers exactly as printed (commas removed) in the report's unit. Do not convert or compute.
- evidence must be copied verbatim from the cited page.
- Prefer standalone figures; if only consolidated are given, use them and set basis accordingly."""

_WHITESPACE = re.compile(r"\s+")

def normalize(text:str) -> str:
    return _WHITESPACE.sub(" ",text).lower()

def load_pages(pdf_path:Path) -> list[str]:
    """Return the text of every page; image-only pages become empty strings."""
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    return [page.extract_text() or "" for page in PdfReader(pdf_path).pages]

def keyword_score(page_text: str) -> int:
 text = normalize(page_text) # normalise once per page
 return sum(text.count(keyword) for keyword in KEYWORDS)

def select_pages(pages: list[str], k: int = TOP_K_PAGES) -> list[int]:
    """Naive retrieval: 0-based indexes of the k pages with most keyword hits, in page order.Replaced by vector / hybrid search in later days."""
    scored = [(keyword_score(text), idx) for idx, text in enumerate(pages)]
    scored.sort(reverse=True)
    return sorted (idx for score, idx in scored[:k] if score > 0)

def build_context(pages: list[str], page_indexes:list[int])->str:
   blocks = [f"[PAGE {idx+1}]\n{pages[idx][:MAX_CHARS_PER_PAGE]}" for idx in page_indexes]
   return "\n\n".join(blocks)

def _quote_on_page(quote: str, page_text:str)->bool:
   """Verbatim check that tolerates whitespace, case and thousands-separator differences."""
   return normalize(quote).replace(",", "") in page_text.replace(",","")

def _value_on_page(value: float, page_text:str) -> bool:
   """True if the number appears on the page in a common print format (commas removed)."""
   digits_only = page_text.replace(",", "")
   candidates = {f"{value: .2f}", f"{value:.1f}", f"{value:g}"}
   return any(candidate in digits_only for candidate in candidates)

def grounding_errors(snapshot: FinancialSnapshot, pages: list[str], allowed: set[int])-> list[str]:
   """Business validation: every reported value must be traceable to its cited page."""
   errors : list[str] = []
   for name in METRICS:
      metric = getattr(snapshot, name)
      if metric.value is None:
         continue
      page_idx = (metric.source_page or 0 ) - 1

      if page_idx not in allowed:
         errors.append(f"{name}: source_page {metric.source_page} was not in the context")
         continue
      page_text = normalize(pages[page_idx])
      if not metric.evidence or not _quote_on_page(metric.evidence, page_text):
         errors.append(f"{name}: evidence is not a verbatim quote from page {metric.source_page}")
      if _value_on_page(metric.value, page_text):
         errors.append(f"{name}: value {metric.value} does not appear on page {metric.source_page}")

   revenue = snapshot.revenue_from_operations.value
   pat = snapshot.profit_after_tax.value

   if revenue is not None and pat is not None and pat>=revenue:
      errors.append("profit_after_tax must be smaller than revenue_from_operations")
   return errors

def build_messages(context: str) -> list[dict]:
   user = (F"<context>\n{context}\n</context>\n\n"
           "Extract the financial snapshot for the latest fiscal year in this report.")
   return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content":user}]

def write_file(path: Path, content: str) -> None:
   path.parent.mkdir(parents=True, exist_ok=True)
   path.write_text(content)



def main() -> int:
   parser = argparse.ArgumentParser(description="Extract a grounded financial report")
   parser.add_argument("pdf", type=Path, help=" Path to the Annual report PDF")
   args = parser.parse_args()

   pages = load_pages(args.pdf)
   selected = select_pages(pages)
   if not selected:
      log.error(" no page mathed the keywords - nothing to extract")
      return 1
   log. info ("pages sent: %s", [idx + 1 for idx in selected])

   try:
      snapshot = extract(
        build_messages(build_context (pages, selected)) ,
        FinancialSnapshot,
        validate=partial(grounding_errors, pages=pages, allowed=set (selected)),
        tag="day02_extract", 
        prompt_version=PROMPT_VERSION,
      )
   except ExtractionError as exc:                          # never write unvalidated data
      write_file(QUARANTINE_PATH, str (exc))
      log.error ("quarantined: %s", exc)
      return 2

   write_file(OUTPUT_PATH, snapshot.model_dump_json(indent=2))
   log.info("saved %s", OUTPUT_PATH)
   print (snapshot.model_dump_json(indent=2))
   return 0

if __name__ == "__main__":
   sys.exit(main())