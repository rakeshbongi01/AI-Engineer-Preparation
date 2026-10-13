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