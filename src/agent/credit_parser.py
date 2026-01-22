"""Credit statement parser - extracts loan issuances, repayments, and debt structure."""
import re
from datetime import datetime
from typing import List, Optional

import pdfplumber
from dateutil import parser as date_parser

from .models import CreditTransaction

# Patterns to identify credit-related transactions
LOAN_ISSUANCE_PATTERNS = [
    r"\bкредит\s*(?:наличными|для|на)\b",
    r"\bвыдан\s*кредит\b",
    r"\bполучен\s*кредит\b",
    r"\bloan\s*issued\b",
    r"\bкредит\s*для\s*ип\b",
    r"\bкредит\s*наличными\b",
]

LOAN_REPAYMENT_PATTERNS = [
    r"\bпогашен\s*кредит\b",
    r"\bоплат[аы]\s*kaspi\s*кредит\b",
    r"\bпогашение\s*кредит\b",
    r"\brepayment\b",
    r"\bкредит\s*погашен\b",
]

INSTALLMENT_PATTERNS = [
    r"\bрассрочк\b",
    r"\binstallment\b",
    r"\bkaspi\s*red\b",
    r"\bрассрочка\b",
]

CREDIT_CARD_PATTERNS = [
    r"\bкредитная\s*карт\b",
    r"\bcredit\s*card\b",
    r"\bкарта\s*кредит\b",
]


def _parse_date(text: str) -> Optional[datetime]:
    """Parse date string, ensuring result is within reasonable bounds (2000-2100)."""
    try:
        dt = date_parser.parse(text, dayfirst=True, yearfirst=False, fuzzy=True)
        if dt.year < 2000 or dt.year > 2100:
            return None
        return dt
    except (ValueError, OverflowError, TypeError):
        return None


def _parse_amount(text: str) -> Optional[float]:
    """Parse amount from text, handling KZT format with spaces and commas."""
    if not text:
        return None
    # Match numbers with spaces/commas
    match = re.search(r"([-+]?)\s*(\d[\d\s,\.]*\d)", text)
    if not match:
        return None
    sign = match.group(1) or ""
    num_raw = match.group(2)
    num_raw = num_raw.replace(" ", "")
    if "," in num_raw and "." not in num_raw:
        num_raw = num_raw.replace(",", ".")
    else:
        num_raw = num_raw.replace(",", "")
    try:
        value = float(num_raw)
        return -value if sign == "-" else value
    except ValueError:
        return None


def _classify_loan_type(description: str) -> str:
    """Classify loan type from description."""
    desc_lower = description.lower()
    
    for pattern in CREDIT_CARD_PATTERNS:
        if re.search(pattern, desc_lower, re.IGNORECASE):
            return "credit_card"
    
    for pattern in INSTALLMENT_PATTERNS:
        if re.search(pattern, desc_lower, re.IGNORECASE):
            return "installment"
    
    if re.search(r"\b(?:ип|бизнес|business)\b", desc_lower, re.IGNORECASE):
        return "business_loan"
    
    return "cash_loan"  # default


def extract_credit_transactions(pdf_path: str, max_pages: int = 30) -> List[CreditTransaction]:
    """
    Extract credit-related transactions from a credit statement PDF.
    
    Looks for:
    - Loan issuances (positive amounts with loan keywords)
    - Loan repayments (negative amounts with repayment keywords)
    - Installment purchases
    - Credit card transactions
    """
    credit_txns: List[CreditTransaction] = []
    
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        pages_to_process = min(total_pages, max_pages)
        
        for page in pdf.pages:
            tables = page.extract_tables() or []
            text = page.extract_text() or ""
            lines = text.splitlines()
            
            # Process tables
            for table in tables:
                for row in table:
                    cells = [c.strip() if c else "" for c in row if c and c.strip()]
                    if not cells:
                        continue
                    
                    # Find date
                    date_val = None
                    for cell in cells:
                        date_val = _parse_date(cell)
                        if date_val:
                            break
                    if not date_val:
                        continue
                    
                    # Find amount
                    amount_val = None
                    for cell in reversed(cells):
                        amount_val = _parse_amount(cell)
                        if amount_val is not None:
                            break
                    if amount_val is None:
                        continue
                    
                    # Check if credit-related
                    description = " ".join(cells).lower()
                    is_loan_issuance = any(re.search(p, description, re.IGNORECASE) for p in LOAN_ISSUANCE_PATTERNS)
                    is_repayment = any(re.search(p, description, re.IGNORECASE) for p in LOAN_REPAYMENT_PATTERNS)
                    is_credit_related = is_loan_issuance or is_repayment or any(
                        re.search(p, description, re.IGNORECASE) 
                        for p in INSTALLMENT_PATTERNS + CREDIT_CARD_PATTERNS
                    )
                    
                    if is_credit_related:
                        loan_type = _classify_loan_type(" ".join(cells))
                        credit_txns.append(
                            CreditTransaction(
                                date=date_val,
                                description=" ".join(cells),
                                amount=amount_val,
                                loan_type=loan_type,
                            )
                        )
            
            # Process text lines as fallback
            for line in lines:
                line_clean = " ".join(line.split())
                if len(line_clean) < 10:
                    continue
                
                date_match = re.search(r"\b(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})\b", line_clean)
                if not date_match:
                    continue
                
                date_val = _parse_date(date_match.group(1))
                if not date_val:
                    continue
                
                amount_val = _parse_amount(line_clean[date_match.end():])
                if amount_val is None:
                    continue
                
                # Check if credit-related
                is_loan_issuance = any(re.search(p, line_clean, re.IGNORECASE) for p in LOAN_ISSUANCE_PATTERNS)
                is_repayment = any(re.search(p, line_clean, re.IGNORECASE) for p in LOAN_REPAYMENT_PATTERNS)
                is_credit_related = is_loan_issuance or is_repayment or any(
                    re.search(p, line_clean, re.IGNORECASE) 
                    for p in INSTALLMENT_PATTERNS + CREDIT_CARD_PATTERNS
                )
                
                if is_credit_related:
                    loan_type = _classify_loan_type(line_clean)
                    credit_txns.append(
                        CreditTransaction(
                            date=date_val,
                            description=line_clean,
                            amount=amount_val,
                            loan_type=loan_type,
                        )
                    )
    
    credit_txns.sort(key=lambda t: t.date)
    return credit_txns
