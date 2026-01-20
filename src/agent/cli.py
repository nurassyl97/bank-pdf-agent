import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .analytics import build_analysis
from .parser import analyze_statement

app = typer.Typer(help="Analyze PDF bank statements and extract transactions.")
console = Console()


def _print_preview(statement) -> None:
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Date", width=12)
    table.add_column("Description", width=50)
    table.add_column("Amount", justify="right")
    table.add_column("Balance", justify="right")

    for txn in statement.transactions[:100]:
        table.add_row(
            txn.date.strftime("%Y-%m-%d"),
            txn.description[:48],
            f"{txn.amount:,.2f}",
            "" if txn.balance is None else f"{txn.balance:,.2f}",
        )

    console.print(table)
    console.print(
        f"Credits: {statement.total_credit:,.2f} | Debits: {statement.total_debit:,.2f} | Closing balance: {statement.closing_balance}"
    )


@app.command()
def analyze(
    pdf: Path = typer.Argument(..., help="Path to a PDF bank statement"),
    currency: str = typer.Option("unknown", help="Currency code for outputs"),
    preview: bool = typer.Option(True, help="Print a human-readable preview table"),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Optional path to save JSON"
    ),
) -> None:
    """Extract transactions from a PDF and emit JSON."""
    if not pdf.exists():
        typer.echo(f"File not found: {pdf}")
        raise typer.Exit(code=1)

    statement = analyze_statement(str(pdf), currency=currency)

    data = build_analysis(statement.transactions, currency=currency, bank="kaspi")

    json_output = json.dumps(data, indent=2)
    if output:
        output.write_text(json_output, encoding="utf-8")
        typer.echo(f"Wrote {output}")
    else:
        typer.echo(json_output)

    if preview:
        _print_preview(statement)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
