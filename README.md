# Bank PDF Statement Agent

Lightweight Python agent to extract transactions from bank statement PDFs and summarize balances/flows. Designed for quick experimentation; swap in your own heuristics as you collect sample statements.

## Features
- Parse PDFs with `pdfplumber`, scanning tables first then raw text as fallback.
- Heuristic transaction extraction: date + description + debit/credit detection.
- Analytics: spending/earning totals, category breakdown, balances, trends, anomalies, weekly and monthly summaries.
- CLI for batch runs with JSON output and optional plaintext preview.
- FastAPI endpoint for your website upload flow (1MB limit).

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m agent.cli path/to/statement.pdf --preview
```

## Run as an API (for your website)
```bash
uvicorn agent.api:app --reload --port 8000
```

Then POST a PDF (max 1MB) to `/analyze` as multipart form-data under the field name `file`.

## Implementation notes
- `agent/parser.py`: PDF loading and transaction extraction heuristics.
- `agent/categories.py`: Kaspi-oriented category rules (RU/KZ keywords).
- `agent/analytics.py`: weekly/monthly summaries, trends, anomaly flags.
- `agent/cli.py`: CLI entrypoint; prints JSON + optional preview table.
- `agent/api.py`: FastAPI service (`/analyze`).
- Heuristics favor robustness: ignores short/noisy lines, attempts multiple date formats, detects amounts via regex.

## Next steps
- Tune regexes and column detection with real Kaspi statement samples (especially table headers/column order).
- Add strict PII controls (retention, encryption, redaction) and bank-specific profiles.
- Add persistent storage (optional) for dashboards and user history.
