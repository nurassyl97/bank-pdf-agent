from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from statistics import median
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .categories import categorize, categories_version
from .models import Transaction


def _to_df(transactions: List[Transaction]) -> pd.DataFrame:
    rows = []
    for t in transactions:
        rows.append(
            {
                "date": t.date,
                "description": t.description,
                "amount": float(t.amount),
                "balance": None if t.balance is None else float(t.balance),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["date", "description", "amount", "balance"])
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def _weekly_monthly_summaries(df: pd.DataFrame) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if df.empty:
        return [], []

    d = df.copy()
    d["week"] = d["date"].dt.to_period("W-MON").astype(str)  # week starting Monday
    d["month"] = d["date"].dt.to_period("M").astype(str)
    d["income"] = d["amount"].clip(lower=0.0)
    d["spend"] = (-d["amount"]).clip(lower=0.0)

    weekly = (
        d.groupby("week", as_index=False)
        .agg(
            income=("income", "sum"),
            spending=("spend", "sum"),
            net=("amount", "sum"),
            transactions=("amount", "count"),
        )
        .sort_values("week")
        .to_dict(orient="records")
    )

    monthly = (
        d.groupby("month", as_index=False)
        .agg(
            income=("income", "sum"),
            spending=("spend", "sum"),
            net=("amount", "sum"),
            transactions=("amount", "count"),
        )
        .sort_values("month")
        .to_dict(orient="records")
    )
    return weekly, monthly


def _category_breakdown(df: pd.DataFrame) -> List[Dict[str, Any]]:
    if df.empty:
        return []
    d = df.copy()
    d["category"] = d["description"].apply(categorize)
    d["income"] = d["amount"].clip(lower=0.0)
    d["spend"] = (-d["amount"]).clip(lower=0.0)
    out = (
        d.groupby("category", as_index=False)
        .agg(
            income=("income", "sum"),
            spending=("spend", "sum"),
            transactions=("amount", "count"),
        )
        .sort_values(["spending", "income"], ascending=[False, False])
        .to_dict(orient="records")
    )
    return out


def _trends(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {"daily_net": []}
    d = df.copy()
    d["day"] = d["date"].dt.date.astype(str)
    daily = (
        d.groupby("day", as_index=False)
        .agg(net=("amount", "sum"), transactions=("amount", "count"))
        .sort_values("day")
        .to_dict(orient="records")
    )
    return {"daily_net": daily}


def _anomalies(df: pd.DataFrame, top_n: int = 10) -> List[Dict[str, Any]]:
    """
    Simple anomaly heuristic:
    - flag unusually large debits relative to typical debit size (median-based)
    - always include top-N absolute debits
    """
    if df.empty:
        return []

    debits = df[df["amount"] < 0].copy()
    if debits.empty:
        return []

    debit_sizes = (-debits["amount"]).tolist()
    typical = median(debit_sizes) if debit_sizes else 0.0
    threshold = typical * 4.0 if typical > 0 else float("inf")

    debits["abs_debit"] = -debits["amount"]
    debits["reason"] = debits["abs_debit"].apply(
        lambda x: "large_debit" if x >= threshold else "top_spend"
    )

    top = debits.sort_values("abs_debit", ascending=False).head(top_n)
    out = []
    for _, r in top.iterrows():
        out.append(
            {
                "date": r["date"].isoformat(),
                "description": r["description"],
                "amount": float(r["amount"]),
                "balance": None if pd.isna(r["balance"]) else float(r["balance"]),
                "category": categorize(r["description"]),
                "reason": r["reason"],
            }
        )
    return out


def build_analysis(
    transactions: List[Transaction],
    currency: str = "KZT",
    bank: str = "kaspi",
) -> Dict[str, Any]:
    df = _to_df(transactions)
    df["category"] = df["description"].apply(categorize) if not df.empty else []

    total_income = float(df[df["amount"] > 0]["amount"].sum()) if not df.empty else 0.0
    total_spending = float((-df[df["amount"] < 0]["amount"]).sum()) if not df.empty else 0.0
    net = float(df["amount"].sum()) if not df.empty else 0.0

    opening_balance: Optional[float] = None
    closing_balance: Optional[float] = None
    if not df.empty and df["balance"].notna().any():
        first = df["balance"].dropna().iloc[0]
        last = df["balance"].dropna().iloc[-1]
        opening_balance = float(first)
        closing_balance = float(last)

    weekly, monthly = _weekly_monthly_summaries(df)

    return {
        "meta": {
            "bank": bank,
            "currency": currency,
            "categories_version": categories_version(),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "transactions": int(len(transactions)),
        },
        "totals": {
            "income": total_income,
            "spending": total_spending,
            "net": net,
        },
        "balances": {
            "opening": opening_balance,
            "closing": closing_balance,
        },
        "category_breakdown": _category_breakdown(df),
        "weekly_summary": weekly,
        "monthly_summary": monthly,
        "trends": _trends(df),
        "anomalies": _anomalies(df),
        "transactions": [
            {
                "date": t.date.isoformat(),
                "description": t.description,
                "amount": t.amount,
                "balance": t.balance,
                "category": categorize(t.description),
            }
            for t in transactions
        ],
    }

