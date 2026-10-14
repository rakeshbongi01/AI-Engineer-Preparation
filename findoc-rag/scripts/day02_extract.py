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

def select_pages(pages: list[str], k: int = TOP_K_PAGES) -> list[int]:
    """Naive retrieval: 0-based indexes of the k pages with most keyword hits, in page order.Replaced by vector / hybrid search in later days."""
    scored = [(keyword_score(text), )]
