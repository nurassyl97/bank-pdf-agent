from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .analytics import build_analysis
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
) -> dict:
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are supported")

    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 1MB)")

    suffix = Path(file.filename).suffix or ".pdf"
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / f"upload{suffix}"
        tmp_path.write_bytes(data)

        try:
            txns = extract_transactions(str(tmp_path))
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Failed to parse PDF: {e}") from e

    return build_analysis(txns, currency=currency, bank=bank)


def main() -> None:
    """
    Local dev runner. Prefer: `uvicorn agent.api:app --reload --port 8000`
    """
    os.execvp("uvicorn", ["uvicorn", "agent.api:app", "--port", "8000"])

