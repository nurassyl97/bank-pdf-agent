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
        # Skip transactions with invalid dates (shouldn't happen, but safety check)
        if t.date is None:
            continue
        # Double-check date is reasonable (2000-2100)
        if t.date.year < 2000 or t.date.year > 2100:
            continue
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
    # Use errors='coerce' to handle any edge cases gracefully
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    # Drop rows where date conversion failed
    df = df.dropna(subset=["date"])
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


def _detect_credits(df: pd.DataFrame) -> Dict[str, Any]:
    """Detect credit-related transactions and calculate monthly credit payments."""
    if df.empty:
        return {
            "total_monthly": 0.0,
            "percentage_of_expenses": 0.0,
            "recurring_payments": [],
            "credit_transactions": [],
            "warning_level": "low",
        }
    
    # Keywords for credit/loan/installment detection
    credit_keywords = [
        r"\bкредит\b", r"\bcredit\b", r"\bloan\b", r"\bзайм\b",
        r"\bkaspi\s*кредит\b", r"\bkaspi\s*red\b", r"\bрассрочк", r"\binstallment\b",
        r"\bоплат[аы]\s*kaspi\s*кредит", r"\bпогашен", r"\brepayment\b",
        r"\bкредит\s*для\s*ип\b", r"\bкредит\s*наличными\b",
    ]
    
    import re
    credit_pattern = re.compile("|".join(credit_keywords), re.IGNORECASE)
    
    debits = df[df["amount"] < 0].copy()
    if debits.empty:
        return {
            "total_monthly": 0.0,
            "percentage_of_expenses": 0.0,
            "recurring_payments": [],
            "credit_transactions": [],
            "warning_level": "low",
        }
    
    # Find credit transactions
    debits["is_credit"] = debits["description"].apply(
        lambda x: bool(credit_pattern.search(str(x).lower()))
    )
    credit_txns = debits[debits["is_credit"]].copy()
    
    if credit_txns.empty:
        return {
            "total_monthly": 0.0,
            "percentage_of_expenses": 0.0,
            "recurring_payments": [],
            "credit_transactions": [],
            "warning_level": "low",
        }
    
    total_credit_spending = float((-credit_txns["amount"]).sum())
    total_expenses = float((-debits["amount"]).sum())
    credit_percentage = (total_credit_spending / total_expenses * 100) if total_expenses > 0 else 0.0
    
    # Detect recurring monthly payments (same merchant, similar amount, monthly pattern)
    credit_txns["month"] = credit_txns["date"].dt.to_period("M").astype(str)
    monthly_credits = credit_txns.groupby("month", as_index=False).agg(
        total=("amount", "sum"),
        count=("amount", "count"),
    )
    monthly_credits["total"] = -monthly_credits["total"]  # Make positive
    
    # Calculate average monthly credit payment
    avg_monthly = float(monthly_credits["total"].mean()) if not monthly_credits.empty else 0.0
    
    # Find recurring patterns (merchants that appear multiple times with similar amounts)
    recurring = []
    for merchant in credit_txns["description"].unique():
        merchant_txns = credit_txns[credit_txns["description"] == merchant]
        if len(merchant_txns) >= 2:  # At least 2 occurrences
            amounts = (-merchant_txns["amount"]).tolist()
            avg_amount = sum(amounts) / len(amounts)
            recurring.append({
                "merchant": merchant,
                "monthly_amount": avg_amount,
                "frequency": len(merchant_txns),
            })
    
    recurring.sort(key=lambda x: x["monthly_amount"], reverse=True)
    
    # Determine warning level
    if credit_percentage > 40:
        warning = "high"
    elif credit_percentage > 25:
        warning = "medium"
    else:
        warning = "low"
    
    return {
        "total_monthly": avg_monthly,
        "total_period": total_credit_spending,
        "percentage_of_expenses": credit_percentage,
        "recurring_payments": recurring[:5],  # Top 5
        "credit_transactions": [
            {
                "date": r["date"].isoformat(),
                "description": r["description"],
                "amount": float(r["amount"]),
            }
            for _, r in credit_txns.head(10).iterrows()
        ],
        "warning_level": warning,
    }


def _detect_money_leaks(df: pd.DataFrame) -> Dict[str, Any]:
    """Detect frequent small transactions and repeating merchants that add up."""
    if df.empty:
        return {
            "total_monthly": 0.0,
            "leak_sources": [],
            "insight": "",
        }
    
    debits = df[df["amount"] < 0].copy()
    if debits.empty:
        return {
            "total_monthly": 0.0,
            "leak_sources": [],
            "insight": "",
        }
    
    # Small transactions (under 5000 KZT) that happen frequently
    small_threshold = 5000.0
    small_txns = debits[(-debits["amount"]) <= small_threshold].copy()
    
    if small_txns.empty:
        return {
            "total_monthly": 0.0,
            "leak_sources": [],
            "insight": "",
        }
    
    # Group by merchant/description
    small_txns["abs_amount"] = -small_txns["amount"]
    leaks = (
        small_txns.groupby("description", as_index=False)
        .agg(
            total=("abs_amount", "sum"),
            count=("abs_amount", "count"),
            avg=("abs_amount", "mean"),
        )
        .sort_values("total", ascending=False)
    )
    
    # Filter: at least 3 occurrences and total > 10000 KZT
    significant_leaks = leaks[(leaks["count"] >= 3) & (leaks["total"] >= 10000)].head(10)
    
    leak_sources = [
        {
            "merchant": row["description"],
            "total": float(row["total"]),
            "count": int(row["count"]),
            "avg": float(row["avg"]),
        }
        for _, row in significant_leaks.iterrows()
    ]
    
    # Calculate monthly average
    small_txns["month"] = small_txns["date"].dt.to_period("M").astype(str)
    monthly_small = small_txns.groupby("month", as_index=False).agg(total=("abs_amount", "sum"))
    monthly_avg = float(monthly_small["total"].mean()) if not monthly_small.empty else 0.0
    
    return {
        "total_monthly": monthly_avg,
        "leak_sources": leak_sources,
        "insight": f"Обнаружено {len(leak_sources)} источников незаметных трат" if leak_sources else "",
    }


def _generate_recommendations(
    totals: Dict[str, float],
    credit_analysis: Dict[str, Any],
    leak_analysis: Dict[str, Any],
    category_breakdown: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Generate practical recommendations based on the analysis."""
    recommendations = []
    
    # Recommendation 1: Credit load
    if credit_analysis["percentage_of_expenses"] > 25:
        monthly_credit = credit_analysis["total_monthly"]
        recommendations.append({
            "title": "Снизить кредитную нагрузку",
            "description": f"Кредиты составляют {credit_analysis['percentage_of_expenses']:.1f}% ваших расходов. Рассмотрите рефинансирование или досрочное погашение.",
            "monthly_savings": monthly_credit * 0.2,  # 20% reduction estimate
            "impact": "Высокий",
        })
    
    # Recommendation 2: Money leaks
    if leak_analysis["total_monthly"] > 20000:
        top_leak = leak_analysis["leak_sources"][0] if leak_analysis["leak_sources"] else None
        if top_leak:
            recommendations.append({
                "title": "Сократить незаметные траты",
                "description": f"На '{top_leak['merchant']}' уходит {top_leak['total']:,.0f} KZT в месяц. Пересмотрите необходимость этих покупок.",
                "monthly_savings": top_leak["total"] * 0.5,  # 50% reduction estimate
                "impact": "Средний",
            })
    
    # Recommendation 3: Large category spending
    if category_breakdown:
        top_category = category_breakdown[0]
        if top_category["spending"] > totals["spending"] * 0.3:  # More than 30% of spending
            cat_name = top_category["category"]
            recommendations.append({
                "title": f"Оптимизировать траты на {cat_name}",
                "description": f"На эту категорию уходит {top_category['spending'] / totals['spending'] * 100:.1f}% всех расходов. Ищите более выгодные варианты.",
                "monthly_savings": top_category["spending"] * 0.15,  # 15% reduction estimate
                "impact": "Средний",
            })
    
    # Recommendation 4: Net result
    if totals["net"] < 0:
        recommendations.append({
            "title": "Увеличить доходы или сократить расходы",
            "description": f"Ваши расходы превышают доходы на {abs(totals['net']):,.0f} KZT. Нужно либо увеличить доходы, либо сократить расходы минимум на эту сумму.",
            "monthly_savings": abs(totals["net"]),
            "impact": "Критический",
        })
    
    # Recommendation 5: Transfers (if too high)
    transfers_cat = next((c for c in category_breakdown if c["category"] == "transfers"), None)
    if transfers_cat and transfers_cat["spending"] > totals["spending"] * 0.4:
        recommendations.append({
            "title": "Проверить переводы",
            "description": f"Переводы составляют {transfers_cat['spending'] / totals['spending'] * 100:.1f}% расходов. Убедитесь, что все переводы необходимы.",
            "monthly_savings": transfers_cat["spending"] * 0.1,
            "impact": "Средний",
        })
    
    # Ensure we have at least 3 recommendations
    while len(recommendations) < 3:
        recommendations.append({
            "title": "Вести учет расходов",
            "description": "Регулярно отслеживайте свои траты, чтобы видеть, куда уходят деньги.",
            "monthly_savings": 0.0,
            "impact": "Низкий",
        })
    
    return recommendations[:5]  # Max 5 recommendations


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
    category_breakdown = _category_breakdown(df)
    credit_analysis = _detect_credits(df)
    leak_analysis = _detect_money_leaks(df)
    recommendations = _generate_recommendations(
        {"income": total_income, "spending": total_spending, "net": net},
        credit_analysis,
        leak_analysis,
        category_breakdown,
    )

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
        "category_breakdown": category_breakdown,
        "weekly_summary": weekly,
        "monthly_summary": monthly,
        "trends": _trends(df),
        "anomalies": _anomalies(df),
        "credit_analysis": credit_analysis,
        "money_leaks": leak_analysis,
        "recommendations": recommendations,
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

