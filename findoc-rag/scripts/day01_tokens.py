# scripts/day01_tokens.py (run from repo root: python -m scripts.day01_tokens data/raw/<file>.pdf)
import sys, json, statistics, tiktoken
from pypdf import PdfReader
from src.llm.client import chat, _cost

enc = tiktoken.get_encoding("o200k_base")
reader = PdfReader(sys.argv[1])
pages =