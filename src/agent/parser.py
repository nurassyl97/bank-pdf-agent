import gc
import re
from datetime import datetime
from typing import Iterable, List, Optional, Tuple

import pdfplumber
from dateutil import parser as date_parser

from .models import StatementSummary, Transaction

DATE_REGEX = re.compile(r"\b(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})\b")
# Allow digits with spaces/commas for thousands and comma/dot for decimals.
AMOUNT_REGEX = re.compile(
    r"(?P<sign>[-+])?\(?\s*(?P<num>\d[\d\s,\.]*\d)\s*\)?\s*(?P<tag>CR|DR)?",
    re.IGNORECASE,
)


def _parse_date(text: str) -> Optional[datetime]:
    """Parse date string, ensuring result is within reasonable bounds (2000-2100)."""
    try:
        dt = date_parser.parse(text, dayfirst=True, yearfirst=False, fuzzy=True)
        # Validate: year must be between 2000 and 2100 (reject misparsed dates like 3505)
        if dt.year < 2000 or dt.year > 2100:
            return None
        return dt
    except (ValueError, OverflowError, TypeError):
        return None


def _parse_amount(text: str) -> Optional[float]:
    if not text:
        return None
    match = AMOUNT_REGEX.search(text)
    if not match:
        return None
    num_raw = match.group("num")
    # Normalize spaces (used as thousands separators in KZT, e.g. "16 313,00")
    num_raw = num_raw.replace(" ", "")
    # If we only have commas (no dot), treat last comma as decimal separator.
    if "," in num_raw and "." not in num_raw:
        num_raw = num_raw.replace(".", "")
        # turn all commas into dots, float() will read decimal correctly
        num_raw = num_raw.replace(",", ".")
    else:
        # Standard western format: remove thousands commas, keep decimal dot
        num_raw = num_raw.replace(",", "")
    try:
        value = float(num_raw)
    except ValueError:
        return None

    sign = match.group("sign") or ""
    tag = (match.group("tag") or "").upper()
    negative = "-" in sign or text.strip().startswith("(") or tag == "DR"
    if tag == "CR":
        negative = False
    return -value if negative else value


def _extract_table_transactions(rows: Iterable[Iterable[str]]) -> List[Transaction]:
    transactions: List[Transaction] = []
    for row in rows:
        cells = [c.strip() for c in row if c and c.strip()]
        if not cells:
            continue
        date_candidate = None
        for cell in cells:
            date_candidate = _parse_date(cell)
            if date_candidate:
                break
        if not date_candidate:
            continue

        amount_candidate: Optional[float] = None
        balance_candidate: Optional[float] = None
        for cell in reversed(cells):
            amt = _parse_amount(cell)
            if amt is not None and amount_candidate is None:
                amount_candidate = amt
                continue
            if amt is not None and balance_candidate is None:
                balance_candidate = amt
        if amount_candidate is None:
            continue

        description_parts = [
            c for c in cells if _parse_date(c) is None and _parse_amount(c) is None
        ]
        description = " ".join(description_parts).strip() or "N/A"
        transactions.append(
            Transaction(
                date=date_candidate,
                description=description,
                amount=amount_candidate,
                balance=balance_candidate,
            )
        )
    return transactions


def _extract_text_transactions(lines: Iterable[str]) -> List[Transaction]:
    transactions: List[Transaction] = []
    for line in lines:
        line_clean = " ".join(line.split())
        if len(line_clean) < 10:
            continue
        date_match = DATE_REGEX.search(line_clean)
        if not date_match:
            continue
        date_str = date_match.group(1)
        date_val = _parse_date(date_str)
        if not date_val:
            continue

        trailing = line_clean[date_match.end() :].strip()
        amount_val = _parse_amount(trailing)
        if amount_val is None:
            # try last token
            tokens = trailing.split()
            if tokens:
                amount_val = _parse_amount(tokens[-1])
        if amount_val is None:
            continue

        desc = trailing
        if desc.endswith(tokens[-1]) and amount_val is not None and tokens:
            desc = " ".join(tokens[:-1])
        desc = desc.strip() or "N/A"

        transactions.append(
            Transaction(
                date=date_val,
                description=desc,
                amount=amount_val,
                balance=None,
            )
        )
    return transactions


def extract_transactions(pdf_path: str, max_pages: int = 100) -> List[Transaction]:
    """Parse transactions from a PDF bank statement.
    
    Args:
        pdf_path: Path to PDF file
        max_pages: Maximum number of pages to process (to limit memory usage)
    """
    transactions: List[Transaction] = []
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        pages_to_process = min(total_pages, max_pages)
        
        for i, page in enumerate(pdf.pages[:pages_to_process]):
            # Process tables first (more reliable for Kaspi)
            tables = page.extract_tables() or []
            for table in tables:
                transactions.extend(_extract_table_transactions(table))

            # Fallback to text extraction if no tables
            if not tables:
                text = page.extract_text() or ""
                lines = text.splitlines()
                transactions.extend(_extract_text_transactions(lines))
            
            # Explicit cleanup after each page to free memory
            del tables
            del page
            # Force GC every 5 pages to be more aggressive
            if (i + 1) % 5 == 0:
                gc.collect()

    transactions.sort(key=lambda t: t.date)
    return transactions


def analyze_statement(pdf_path: str, currency: str = "unknown") -> StatementSummary:
    txns = extract_transactions(pdf_path)
    return StatementSummary(transactions=txns, currency=currency)
