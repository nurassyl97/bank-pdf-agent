"""Combined financial analysis merging questionnaire, regular statement, and credit statement."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

from .analytics import (
    _calculate_before_after,
    _calculate_health_score,
    _calculate_safety_buffer,
    _category_breakdown,
    _detect_credits,
    _detect_money_leaks,
    _generate_action_plan,
    _generate_future_scenarios,
    _generate_recommendations,
    _to_df,
    _trends,
    _weekly_monthly_summaries,
    categorize,
    categories_version,
)
from .models import CreditTransaction, QuestionnaireAnswers, Transaction


def _analyze_credit_statement(credit_txns: List[CreditTransaction]) -> Dict[str, Any]:
    """Analyze credit statement to extract debt structure and behavior."""
    if not credit_txns:
        return {
            "total_loans_issued": 0.0,
            "total_repayments": 0.0,
            "estimated_active_debt": 0.0,
            "loan_frequency": 0,
            "refinancing_detected": False,
            "credit_spiral_risk": "none",
            "loan_types": {},
        }
    
    loans_issued = [t for t in credit_txns if t.amount > 0]
    repayments = [t for t in credit_txns if t.amount < 0]
    
    total_loans = sum(t.amount for t in loans_issued)
    total_repayments = sum(-t.amount for t in repayments)
    
    # Estimate active debt (loans issued - repayments, simplified)
    estimated_active_debt = max(0, total_loans - total_repayments)
    
    # Detect refinancing pattern: loan → repayment → new loan within short period
    refinancing_detected = False
    if len(loans_issued) >= 2:
        # Check if loans are followed by repayments and then new loans
        for i in range(len(loans_issued) - 1):
            loan1 = loans_issued[i]
            loan2 = loans_issued[i + 1]
            days_between = (loan2.date - loan1.date).days
            # If new loan within 30 days of previous, might be refinancing
            if days_between <= 30:
                # Check if there was a repayment between them
                repayments_between = [
                    r for r in repayments
                    if loan1.date < r.date < loan2.date
                ]
                if repayments_between:
                    refinancing_detected = True
                    break
    
    # Count loans by type
    loan_types = {}
    for loan in loans_issued:
        loan_type = loan.loan_type or "unknown"
        loan_types[loan_type] = loan_types.get(loan_type, 0) + 1
    
    # Detect credit spiral: frequent new loans with increasing amounts
    credit_spiral_risk = "none"
    if len(loans_issued) >= 3:
        amounts = [t.amount for t in loans_issued]
        # Check if loan amounts are increasing
        increasing = all(amounts[i] <= amounts[i + 1] for i in range(len(amounts) - 1))
        frequent = len(loans_issued) >= 4  # 4+ loans in period
        if frequent and increasing:
            credit_spiral_risk = "high"
        elif frequent:
            credit_spiral_risk = "medium"
    
    return {
        "total_loans_issued": total_loans,
        "total_repayments": total_repayments,
        "estimated_active_debt": estimated_active_debt,
        "loan_frequency": len(loans_issued),
        "refinancing_detected": refinancing_detected,
        "credit_spiral_risk": credit_spiral_risk,
        "loan_types": loan_types,
        "loans": [
            {
                "date": t.date.isoformat(),
                "description": t.description,
                "amount": t.amount,
                "loan_type": t.loan_type,
            }
            for t in loans_issued
        ],
        "repayments": [
            {
                "date": t.date.isoformat(),
                "description": t.description,
                "amount": -t.amount,  # Make positive
            }
            for t in repayments
        ],
    }


def _calculate_real_income(
    transactions: List[Transaction],
    credit_analysis: Dict[str, Any],
) -> Dict[str, Any]:
    """Calculate REAL income excluding credit inflows."""
    df = _to_df(transactions)
    if df.empty:
        return {
            "total_income": 0.0,
            "credit_inflows": 0.0,
            "real_income": 0.0,
            "credit_dependency_ratio": 0.0,
        }
    
    # Identify credit inflows (transactions that look like loans/credits)
    credit_keywords = [
        r"\bкредит\b",
        r"\bloan\b",
        r"\bзайм\b",
        r"\bкредит\s*наличными\b",
        r"\bкредит\s*для\s*ип\b",
    ]
    
    import re
    credit_pattern = re.compile("|".join(credit_keywords), re.IGNORECASE)
    
    income_txns = df[df["amount"] > 0].copy()
    credit_inflows = income_txns[
        income_txns["description"].str.lower().str.contains(credit_pattern.pattern, regex=True, na=False)
    ]
    
    total_income = float(income_txns["amount"].sum())
    credit_inflow_amount = float(credit_inflows["amount"].sum()) if not credit_inflows.empty else 0.0
    real_income = total_income - credit_inflow_amount
    
    credit_dependency = (credit_inflow_amount / total_income * 100) if total_income > 0 else 0.0
    
    return {
        "total_income": total_income,
        "credit_inflows": credit_inflow_amount,
        "real_income": real_income,
        "credit_dependency_ratio": credit_dependency,
    }


def _calculate_credit_risk_index(
    credit_analysis: Dict[str, Any],
    credit_statement_analysis: Dict[str, Any],
    real_income: Dict[str, Any],
    questionnaire: Optional[QuestionnaireAnswers],
) -> Dict[str, Any]:
    """Calculate comprehensive credit risk index."""
    risk_factors = []
    risk_score = 0  # 0-100, higher = more risky
    
    # Factor 1: Credit load percentage from regular statement
    credit_pct = credit_analysis.get("percentage_of_expenses", 0.0)
    if credit_pct > 40:
        risk_score += 30
        risk_factors.append("Кредиты составляют более 40% расходов")
    elif credit_pct > 25:
        risk_score += 20
        risk_factors.append("Кредиты составляют 25-40% расходов")
    elif credit_pct > 10:
        risk_score += 10
        risk_factors.append("Кредиты составляют 10-25% расходов")
    
    # Factor 2: Credit dependency (credit inflows vs real income)
    credit_dep = real_income.get("credit_dependency_ratio", 0.0)
    if credit_dep > 30:
        risk_score += 25
        risk_factors.append("Более 30% дохода — это кредитные поступления")
    elif credit_dep > 15:
        risk_score += 15
        risk_factors.append("15-30% дохода — это кредитные поступления")
    elif credit_dep > 5:
        risk_score += 5
        risk_factors.append("5-15% дохода — это кредитные поступления")
    
    # Factor 3: Refinancing detected
    if credit_statement_analysis.get("refinancing_detected", False):
        risk_score += 20
        risk_factors.append("Обнаружены признаки рефинансирования (новые кредиты для погашения старых)")
    
    # Factor 4: Credit spiral
    spiral_risk = credit_statement_analysis.get("credit_spiral_risk", "none")
    if spiral_risk == "high":
        risk_score += 25
        risk_factors.append("КРИТИЧЕСКИЙ РИСК: Обнаружена кредитная спираль (частые новые кредиты)")
    elif spiral_risk == "medium":
        risk_score += 15
        risk_factors.append("Высокий риск: Частые новые кредиты")
    
    # Factor 5: Loan frequency
    loan_freq = credit_statement_analysis.get("loan_frequency", 0)
    if loan_freq >= 5:
        risk_score += 15
        risk_factors.append(f"Очень высокая частота кредитов: {loan_freq} за период")
    elif loan_freq >= 3:
        risk_score += 10
        risk_factors.append(f"Высокая частота кредитов: {loan_freq} за период")
    
    # Factor 6: Questionnaire perception vs reality
    if questionnaire:
        perception = questionnaire.credit_load_perception
        actual_pct = credit_pct
        if perception == "low" and actual_pct > 25:
            risk_score += 10
            risk_factors.append("Несоответствие: вы считаете нагрузку низкой, но она высокая")
        elif perception == "medium" and actual_pct > 40:
            risk_score += 10
            risk_factors.append("Несоответствие: реальная нагрузка выше вашей оценки")
    
    risk_score = min(100, risk_score)  # Cap at 100
    
    if risk_score >= 70:
        level = "critical"
        label = "Критический риск"
        explanation = "Ваша финансовая ситуация критична. Высокая зависимость от кредитов, возможна кредитная спираль. Требуются срочные меры."
    elif risk_score >= 50:
        level = "dangerous"
        label = "Опасный уровень"
        explanation = "Ваша кредитная нагрузка опасна. Вы зависите от кредитов, есть признаки рефинансирования. Необходимы изменения."
    elif risk_score >= 30:
        level = "risky"
        label = "Рискованный уровень"
        explanation = "Ваша кредитная нагрузка рискованна. Стоит обратить внимание и начать снижать зависимость от кредитов."
    else:
        level = "safe"
        label = "Безопасный уровень"
        explanation = "Ваша кредитная нагрузка в пределах нормы. Продолжайте контролировать ситуацию."
    
    return {
        "score": risk_score,
        "level": level,
        "label": label,
        "explanation": explanation,
        "risk_factors": risk_factors,
    }


def build_combined_analysis(
    transactions: List[Transaction],
    currency: str = "KZT",
    bank: str = "kaspi",
    questionnaire: Optional[QuestionnaireAnswers] = None,
    credit_transactions: Optional[List[CreditTransaction]] = None,
) -> Dict[str, Any]:
    """
    Build comprehensive financial analysis combining:
    - Regular bank statement
    - Credit statement (if provided)
    - Questionnaire answers (if provided)
    """
    df = _to_df(transactions)
    if df.empty:
        return {"error": "No transactions found"}
    
    df["category"] = df["description"].apply(categorize) if not df.empty else []
    
    # Basic totals
    total_income = float(df[df["amount"] > 0]["amount"].sum()) if not df.empty else 0.0
    total_spending = float((-df[df["amount"] < 0]["amount"]).sum()) if not df.empty else 0.0
    net = float(df["amount"].sum()) if not df.empty else 0.0
    
    # Balances
    opening_balance: Optional[float] = None
    closing_balance: Optional[float] = None
    if not df.empty and df["balance"].notna().any():
        first = df["balance"].dropna().iloc[0]
        last = df["balance"].dropna().iloc[-1]
        opening_balance = float(first)
        closing_balance = float(last)
    
    # Regular analysis components
    weekly, monthly = _weekly_monthly_summaries(df)
    category_breakdown = _category_breakdown(df)
    credit_analysis = _detect_credits(df)
    leak_analysis = _detect_money_leaks(df)
    
    # Credit statement analysis
    credit_statement_analysis = {}
    if credit_transactions:
        credit_statement_analysis = _analyze_credit_statement(credit_transactions)
    
    # Calculate REAL income (excluding credit inflows)
    real_income_analysis = _calculate_real_income(transactions, credit_analysis)
    
    # Calculate credit risk index
    credit_risk = _calculate_credit_risk_index(
        credit_analysis,
        credit_statement_analysis,
        real_income_analysis,
        questionnaire,
    )
    
    # Update totals with real income
    totals_dict = {
        "income": total_income,
        "real_income": real_income_analysis.get("real_income", total_income),
        "spending": total_spending,
        "net": net,
        "months": len(monthly),
    }
    
    balances_dict = {"opening": opening_balance, "closing": closing_balance}
    
    # Generate recommendations (will use questionnaire goals if provided)
    recommendations = _generate_recommendations(
        totals_dict,
        credit_analysis,
        leak_analysis,
        category_breakdown,
    )
    
    # Reorder recommendations based on questionnaire goal if provided
    if questionnaire and questionnaire.primary_goal:
        goal = questionnaire.primary_goal
        if goal == "reduce_debt":
            # Prioritize debt reduction recommendations
            recommendations.sort(key=lambda r: "кредит" in r.get("title", "").lower(), reverse=True)
        elif goal == "save":
            # Prioritize savings recommendations
            recommendations.sort(key=lambda r: r.get("monthly_savings", 0), reverse=True)
    
    # Calculate health score (updated with real income)
    health_score = _calculate_health_score(
        totals_dict,
        credit_analysis,
        leak_analysis,
        balances_dict,
    )
    
    # Safety buffer
    safety_buffer = _calculate_safety_buffer(
        total_spending,
        closing_balance,
        monthly,
    )
    
    # Future scenarios
    future_scenarios = _generate_future_scenarios(
        totals_dict,
        credit_analysis,
        recommendations,
    )
    
    # Action plan
    action_plan = _generate_action_plan(
        recommendations,
        credit_analysis,
        leak_analysis,
    )
    
    # Before/after
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
            "has_questionnaire": questionnaire is not None,
            "has_credit_statement": credit_transactions is not None,
        },
        "totals": {
            "income": total_income,
            "real_income": real_income_analysis.get("real_income", total_income),
            "credit_inflows": real_income_analysis.get("credit_inflows", 0.0),
            "spending": total_spending,
            "net": net,
        },
        "real_income_analysis": real_income_analysis,
        "balances": balances_dict,
        "category_breakdown": category_breakdown,
        "weekly_summary": weekly,
        "monthly_summary": monthly,
        "trends": _trends(df),
        "credit_analysis": credit_analysis,
        "credit_statement_analysis": credit_statement_analysis,
        "credit_risk_index": credit_risk,
        "money_leaks": leak_analysis,
        "recommendations": recommendations,
        "health_score": health_score,
        "safety_buffer": safety_buffer,
        "future_scenarios": future_scenarios,
        "action_plan": action_plan,
        "before_after": before_after,
        "questionnaire_summary": {
            "primary_goal": questionnaire.primary_goal if questionnaire else None,
            "credit_load_perception": questionnaire.credit_load_perception if questionnaire else None,
            "has_savings": questionnaire.has_savings if questionnaire else None,
        } if questionnaire else None,
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
