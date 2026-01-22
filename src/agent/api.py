from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .analytics import build_analysis
from .combined_analysis import build_combined_analysis
from .credit_parser import extract_credit_transactions
from .models import QuestionnaireAnswers
from .parser import extract_transactions

MAX_BYTES = 1_000_000  # 1MB

app = FastAPI(title="Bank PDF Statement Analyzer", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    bank: str = "kaspi",
    currency: str = "KZT",
    questionnaire: Optional[str] = Form(None),  # JSON string
    credit_statement: Optional[UploadFile] = File(None),
) -> dict:
    """
    Analyze bank statement(s) with optional questionnaire and credit statement.
    
    - file: Regular bank statement (required)
    - credit_statement: Credit/loan statement (optional)
    - questionnaire: JSON string with user answers (optional)
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are supported")

    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 1MB)")

    # Parse questionnaire if provided
    questionnaire_answers = None
    if questionnaire:
        try:
            q_dict = json.loads(questionnaire)
            questionnaire_answers = QuestionnaireAnswers(**q_dict)
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            raise HTTPException(status_code=400, detail=f"Invalid questionnaire format: {e}") from e

    # Parse regular statement
    suffix = Path(file.filename).suffix or ".pdf"
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / f"upload{suffix}"
        tmp_path.write_bytes(data)

        try:
            txns = extract_transactions(str(tmp_path), max_pages=30)
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Failed to parse PDF: {e}") from e

        # Parse credit statement if provided
        credit_txns = None
        if credit_statement:
            if not credit_statement.filename.lower().endswith(".pdf"):
                raise HTTPException(status_code=400, detail="Credit statement must be PDF")
            
            credit_data = await credit_statement.read()
            if len(credit_data) > MAX_BYTES:
                raise HTTPException(status_code=413, detail="Credit statement too large (max 1MB)")
            
            credit_path = Path(tmpdir) / f"credit{suffix}"
            credit_path.write_bytes(credit_data)
            
            try:
                credit_txns = extract_credit_transactions(str(credit_path), max_pages=30)
            except Exception as e:
                raise HTTPException(status_code=422, detail=f"Failed to parse credit statement: {e}") from e

    # Questionnaire is now mandatory for accurate analysis
    # Use combined analysis if questionnaire provided
    if questionnaire_answers:
        return build_combined_analysis(
            txns,
            currency=currency,
            bank=bank,
            questionnaire=questionnaire_answers,
            credit_transactions=credit_txns,
        )
    
    # Fallback to regular analysis (but warn that it's incomplete)
    result = build_analysis(txns, currency=currency, bank=bank)
    result["warning"] = "Questionnaire not provided. Analysis is incomplete. Please provide questionnaire for accurate financial assessment."
    return result


def main() -> None:
    """
    Local dev runner. Prefer: `uvicorn agent.api:app --reload --port 8000`
    """
    os.execvp("uvicorn", ["uvicorn", "agent.api:app", "--port", "8000"])

