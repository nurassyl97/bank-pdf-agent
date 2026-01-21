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


def _calculate_health_score(
    totals: Dict[str, float],
    credit_analysis: Dict[str, Any],
    leak_analysis: Dict[str, Any],
    balances: Dict[str, Optional[float]],
) -> Dict[str, Any]:
    """Calculate Financial Health Score (0-100) based on multiple factors."""
    score = 100.0
    factors = []
    
    income = totals.get("income", 0.0)
    spending = totals.get("spending", 0.0)
    net = totals.get("net", 0.0)
    
    # Factor 1: Expenses-to-income ratio (0-30 points)
    if income > 0:
        expense_ratio = (spending / income) * 100
        if expense_ratio <= 70:
            exp_score = 30
            factors.append({"name": "Соотношение расходов к доходам", "score": exp_score, "max": 30, "note": "Отличное соотношение"})
        elif expense_ratio <= 85:
            exp_score = 20
            factors.append({"name": "Соотношение расходов к доходам", "score": exp_score, "max": 30, "note": "Хорошее соотношение"})
        elif expense_ratio <= 100:
            exp_score = 10
            factors.append({"name": "Соотношение расходов к доходам", "score": exp_score, "max": 30, "note": "Расходы равны доходам"})
        else:
            exp_score = 0
            factors.append({"name": "Соотношение расходов к доходам", "score": exp_score, "max": 30, "note": "Расходы превышают доходы"})
        score = score - (30 - exp_score)
    else:
        factors.append({"name": "Соотношение расходов к доходам", "score": 0, "max": 30, "note": "Нет доходов"})
        score -= 30
    
    # Factor 2: Credit load (0-25 points)
    credit_pct = credit_analysis.get("percentage_of_expenses", 0.0)
    if credit_pct <= 10:
        credit_score = 25
        factors.append({"name": "Кредитная нагрузка", "score": credit_score, "max": 25, "note": "Низкая нагрузка"})
    elif credit_pct <= 20:
        credit_score = 20
        factors.append({"name": "Кредитная нагрузка", "score": credit_score, "max": 25, "note": "Умеренная нагрузка"})
    elif credit_pct <= 30:
        credit_score = 10
        factors.append({"name": "Кредитная нагрузка", "score": credit_score, "max": 25, "note": "Высокая нагрузка"})
    else:
        credit_score = 0
        factors.append({"name": "Кредитная нагрузка", "score": credit_score, "max": 25, "note": "Критическая нагрузка"})
    score = score - (25 - credit_score)
    
    # Factor 3: Savings buffer (0-20 points)
    closing_balance = balances.get("closing", 0.0) or 0.0
    months_in_period = totals.get("months", 1)
    if isinstance(months_in_period, list):
        months_in_period = len(months_in_period)
    monthly_expenses = spending / max(1, float(months_in_period)) if spending > 0 else spending
    # Approximate months from data period
    months_covered = (closing_balance / monthly_expenses) if monthly_expenses > 0 else 0
    if months_covered >= 6:
        buffer_score = 20
        factors.append({"name": "Финансовая подушка", "score": buffer_score, "max": 20, "note": f"Хватит на {months_covered:.1f} месяцев"})
    elif months_covered >= 3:
        buffer_score = 15
        factors.append({"name": "Финансовая подушка", "score": buffer_score, "max": 20, "note": f"Хватит на {months_covered:.1f} месяцев"})
    elif months_covered >= 1:
        buffer_score = 8
        factors.append({"name": "Финансовая подушка", "score": buffer_score, "max": 20, "note": f"Хватит только на {months_covered:.1f} месяц"})
    else:
        buffer_score = 0
        factors.append({"name": "Финансовая подушка", "score": buffer_score, "max": 20, "note": "Нет финансовой подушки"})
    score = score - (20 - buffer_score)
    
    # Factor 4: Money leaks (0-15 points)
    leak_total = leak_analysis.get("total_monthly", 0.0)
    leak_pct = (leak_total / spending * 100) if spending > 0 else 0
    if leak_pct <= 5:
        leak_score = 15
        factors.append({"name": "Незаметные траты", "score": leak_score, "max": 15, "note": "Минимальные незаметные траты"})
    elif leak_pct <= 10:
        leak_score = 10
        factors.append({"name": "Незаметные траты", "score": leak_score, "max": 15, "note": "Умеренные незаметные траты"})
    elif leak_pct <= 20:
        leak_score = 5
        factors.append({"name": "Незаметные траты", "score": leak_score, "max": 15, "note": "Высокие незаметные траты"})
    else:
        leak_score = 0
        factors.append({"name": "Незаметные траты", "score": leak_score, "max": 15, "note": "Критический уровень незаметных трат"})
    score = score - (15 - leak_score)
    
    # Factor 5: Net result (0-10 points)
    if net > 0:
        net_score = 10
        factors.append({"name": "Итоговый результат", "score": net_score, "max": 10, "note": "Положительный баланс"})
    else:
        net_score = 0
        factors.append({"name": "Итоговый результат", "score": net_score, "max": 10, "note": "Отрицательный баланс"})
    score = score - (10 - net_score)
    
    score = max(0, min(100, score))  # Clamp to 0-100
    
    if score >= 80:
        status = "Отлично"
        status_color = "positive"
    elif score >= 60:
        status = "Хорошо"
        status_color = "positive"
    elif score >= 40:
        status = "Риск"
        status_color = "warning"
    else:
        status = "Опасность"
        status_color = "negative"
    
    return {
        "score": round(score, 1),
        "status": status,
        "status_color": status_color,
        "factors": factors,
        "explanation": _get_health_explanation(score, status),
    }


def _get_health_explanation(score: float, status: str) -> str:
    """Generate human-readable explanation for health score."""
    if score >= 80:
        return "Ваши финансы в отличном состоянии. Вы контролируете расходы, кредитная нагрузка приемлема, и есть финансовая подушка. Продолжайте в том же духе."
    elif score >= 60:
        return "Ваши финансы в порядке, но есть области для улучшения. Обратите внимание на кредитную нагрузку и незаметные траты — они могут подорвать стабильность."
    elif score >= 40:
        return "Ваши финансы находятся в зоне риска. Расходы слишком высоки относительно доходов, или кредитная нагрузка критична. Нужны срочные меры для стабилизации."
    else:
        return "Ваши финансы в критическом состоянии. Расходы превышают доходы, высокие кредитные обязательства, нет финансовой подушки. Необходимы радикальные изменения."


def _calculate_safety_buffer(
    spending: float,
    closing_balance: Optional[float],
    monthly_summary: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Calculate how many months user can survive without income."""
    if closing_balance is None or closing_balance < 0:
        return {
            "months": 0.0,
            "status": "Нет подушки",
            "status_color": "negative",
            "explanation": "У вас нет финансовой подушки. В случае потери дохода вы окажетесь в сложной ситуации.",
            "monthly_expenses": spending / max(1, len(monthly_summary)) if monthly_summary else spending,
        }
    
    avg_monthly_spending = spending / max(1, len(monthly_summary)) if monthly_summary else spending
    if avg_monthly_spending <= 0:
        months = 0.0
    else:
        months = closing_balance / avg_monthly_spending
    
    if months >= 6:
        status = "Безопасно"
        status_color = "positive"
        explanation = f"Ваша финансовая подушка покрывает {months:.1f} месяцев расходов. Это отличный показатель финансовой безопасности."
    elif months >= 3:
        status = "Приемлемо"
        status_color = "positive"
        explanation = f"Ваша финансовая подушка покрывает {months:.1f} месяцев расходов. Это хороший уровень, но можно увеличить до 6 месяцев."
    elif months >= 1:
        status = "Слабо"
        status_color = "warning"
        explanation = f"Ваша финансовая подушка покрывает только {months:.1f} месяца расходов. Это рискованно. Старайтесь накопить минимум 3 месяца расходов."
    else:
        status = "Критично"
        status_color = "negative"
        explanation = "У вас нет финансовой подушки безопасности. В случае непредвиденных обстоятельств вы можете оказаться в долгах. Немедленно начните откладывать."
    
    return {
        "months": round(months, 1),
        "status": status,
        "status_color": status_color,
        "explanation": explanation,
        "monthly_expenses": avg_monthly_spending,
        "buffer_amount": closing_balance,
    }


def _generate_future_scenarios(
    totals: Dict[str, float],
    credit_analysis: Dict[str, Any],
    recommendations: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Generate 3 financial forecasting scenarios."""
    current_income = totals.get("income", 0.0)
    current_spending = totals.get("spending", 0.0)
    current_net = totals.get("net", 0.0)
    
    # Approximate months from period (assuming 1 month if we don't know)
    months_in_period = 1.0  # Default assumption
    
    monthly_income = current_income / months_in_period
    monthly_spending = current_spending / months_in_period
    monthly_net = current_net / months_in_period
    
    # Scenario 1: If nothing changes
    scenario1 = {
        "title": "Если ничего не менять",
        "description": "Текущая ситуация сохранится без изменений",
        "monthly_balance": monthly_net,
        "6_month_outcome": monthly_net * 6,
        "12_month_outcome": monthly_net * 12,
        "risk_level": "high" if monthly_net < 0 else "medium",
        "summary": f"Через 6 месяцев: {monthly_net * 6:+,.0f} KZT. Через год: {monthly_net * 12:+,.0f} KZT." if monthly_net != 0 else "Баланс не изменится.",
    }
    
    # Scenario 2: If recommendations are followed
    total_savings = sum(r.get("monthly_savings", 0.0) for r in recommendations)
    scenario2_monthly = monthly_net + total_savings
    scenario2 = {
        "title": "Если следовать рекомендациям",
        "description": f"Реализуете все рекомендации (экономия ~{total_savings:,.0f} KZT/месяц)",
        "monthly_balance": scenario2_monthly,
        "6_month_outcome": scenario2_monthly * 6,
        "12_month_outcome": scenario2_monthly * 12,
        "risk_level": "low" if scenario2_monthly > 0 else "medium",
        "summary": f"Через 6 месяцев: {scenario2_monthly * 6:+,.0f} KZT. Через год: {scenario2_monthly * 12:+,.0f} KZT. Экономия: {total_savings:,.0f} KZT/месяц.",
    }
    
    # Scenario 3: If credit load is optimized
    current_credit_monthly = credit_analysis.get("total_monthly", 0.0)
    credit_reduction = current_credit_monthly * 0.3 if credit_analysis.get("percentage_of_expenses", 0) > 25 else 0.0
    scenario3_monthly = monthly_net + credit_reduction
    scenario3 = {
        "title": "Если оптимизировать кредиты",
        "description": f"Снизить кредитную нагрузку на 30% (экономия ~{credit_reduction:,.0f} KZT/месяц)" if credit_reduction > 0 else "Кредитная нагрузка уже оптимальна",
        "monthly_balance": scenario3_monthly,
        "6_month_outcome": scenario3_monthly * 6,
        "12_month_outcome": scenario3_monthly * 12,
        "risk_level": "low" if scenario3_monthly > 0 else "medium",
        "summary": f"Через 6 месяцев: {scenario3_monthly * 6:+,.0f} KZT. Через год: {scenario3_monthly * 12:+,.0f} KZT." if credit_reduction > 0 else "Изменения минимальны.",
    }
    
    return [scenario1, scenario2, scenario3]


def _generate_action_plan(
    recommendations: List[Dict[str, Any]],
    credit_analysis: Dict[str, Any],
    leak_analysis: Dict[str, Any],
) -> List[Dict[str, str]]:
    """Generate a concrete 30-day action plan."""
    actions = []
    
    # Action 1: Review top transfers/credits
    if credit_analysis.get("percentage_of_expenses", 0) > 25:
        actions.append({
            "day": "День 1-3",
            "action": "Проанализируйте все кредитные платежи и рассрочки",
            "how": "Откройте раздел 'Кредиты и рассрочки' и проверьте каждый платеж. Найдите возможности рефинансирования или досрочного погашения.",
        })
    
    # Action 2: Address money leaks
    top_leak = leak_analysis.get("leak_sources", [{}])[0] if leak_analysis.get("leak_sources") else None
    if top_leak:
        actions.append({
            "day": "День 4-7",
            "action": f"Сократите траты на '{top_leak.get('merchant', 'определенный магазин')}'",
            "how": f"Эта категория трат составляет {top_leak.get('total', 0):,.0f} KZT в месяц. Определите, какие из этих покупок действительно необходимы.",
        })
    
    # Action 3: Set spending limits
    if leak_analysis.get("total_monthly", 0) > 20000:
        actions.append({
            "day": "День 8-14",
            "action": "Установите лимиты на часто используемые категории",
            "how": "В приложении Kaspi установите дневные или месячные лимиты на доставку еды, развлечения и другие категории с высокими незаметными тратами.",
        })
    
    # Action 4: Review recurring payments
    recurring = credit_analysis.get("recurring_payments", [])
    if recurring:
        actions.append({
            "day": "День 15-21",
            "action": "Проверьте регулярные платежи и подписки",
            "how": f"У вас {len(recurring)} регулярных платежа. Проверьте, используете ли вы все эти услуги. Отмените неиспользуемые подписки.",
        })
    
    # Action 5: Create savings plan
    actions.append({
        "day": "День 22-30",
        "action": "Начните откладывать на финансовую подушку",
        "how": "Определите сумму, которую можете откладывать ежемесячно (даже 10,000 KZT — хорошее начало). Настройте автоматический перевод на накопительный счет.",
    })
    
    return actions[:5]  # Max 5 actions


def _calculate_before_after(
    totals: Dict[str, float],
    credit_analysis: Dict[str, Any],
    leak_analysis: Dict[str, Any],
    recommendations: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Calculate before/after comparison after implementing recommendations."""
    current_net = totals.get("net", 0.0)
    current_credit_pct = credit_analysis.get("percentage_of_expenses", 0.0)
    current_leaks = leak_analysis.get("total_monthly", 0.0)
    
    total_savings = sum(r.get("monthly_savings", 0.0) for r in recommendations)
    future_net = current_net + total_savings
    
    # Estimate future credit percentage (assume 20% reduction in credit spending)
    credit_reduction = credit_analysis.get("total_monthly", 0.0) * 0.2
    future_credit_spending = credit_analysis.get("total_monthly", 0.0) - credit_reduction
    future_total_spending = totals.get("spending", 0.0) - total_savings
    future_credit_pct = (future_credit_spending / future_total_spending * 100) if future_total_spending > 0 else 0
    
    # Estimate leak reduction (assume 50% reduction)
    future_leaks = current_leaks * 0.5
    
    return {
        "current": {
            "net_balance": current_net,
            "credit_percentage": current_credit_pct,
            "money_leaks": current_leaks,
            "monthly_savings_potential": 0.0,
        },
        "after": {
            "net_balance": future_net,
            "credit_percentage": future_credit_pct,
            "money_leaks": future_leaks,
            "monthly_savings_potential": total_savings,
        },
        "improvement": {
            "net_change": future_net - current_net,
            "credit_reduction": current_credit_pct - future_credit_pct,
            "leak_reduction": current_leaks - future_leaks,
        },
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
    
    totals_dict = {"income": total_income, "spending": total_spending, "net": net, "months": len(monthly)}
    balances_dict = {"opening": opening_balance, "closing": closing_balance}
    
    recommendations = _generate_recommendations(
        totals_dict,
        credit_analysis,
        leak_analysis,
        category_breakdown,
    )
    
    health_score = _calculate_health_score(
        totals_dict,
        credit_analysis,
        leak_analysis,
        balances_dict,
    )
    
    safety_buffer = _calculate_safety_buffer(
        total_spending,
        closing_balance,
        monthly,
    )
    
    future_scenarios = _generate_future_scenarios(
        totals_dict,
        credit_analysis,
        recommendations,
    )
    
    action_plan = _generate_action_plan(
        recommendations,
        credit_analysis,
        leak_analysis,
    )
    
    before_after = _calculate_before_after(
        totals_dict,
        credit_analysis,
        leak_analysis,
        recommendations,
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
        "balances": balances_dict,
        "category_breakdown": category_breakdown,
        "weekly_summary": weekly,
        "monthly_summary": monthly,
        "trends": _trends(df),
        "anomalies": _anomalies(df),
        "credit_analysis": credit_analysis,
        "money_leaks": leak_analysis,
        "recommendations": recommendations,
        "health_score": health_score,
        "safety_buffer": safety_buffer,
        "future_scenarios": future_scenarios,
        "action_plan": action_plan,
        "before_after": before_after,
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

