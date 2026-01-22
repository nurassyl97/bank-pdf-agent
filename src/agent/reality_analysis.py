"""Financial reality analysis - comparing declared vs detected values."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .analytics import _to_df, categorize
from .models import QuestionnaireAnswers, Transaction


def compare_declared_vs_detected(
    transactions: List[Transaction],
    questionnaire: QuestionnaireAnswers,
) -> Dict[str, Any]:
    """
    Compare user-declared financial data with what transactions actually show.
    
    Returns discrepancies and warnings.
    """
    df = _to_df(transactions)
    if df.empty:
        return {
            "income_comparison": {},
            "expense_comparison": {},
            "discrepancies": [],
            "warnings": [],
        }
    
    # Calculate detected values from transactions
    income_txns = df[df["amount"] > 0]
    detected_monthly_income = float(income_txns["amount"].sum()) if not income_txns.empty else 0.0
    
    # Approximate months in period (simplified - assume 1 month if we can't determine)
    months_in_period = 1.0  # TODO: Calculate from date range
    detected_monthly_income = detected_monthly_income / months_in_period
    
    expense_txns = df[df["amount"] < 0]
    detected_monthly_expenses = float((-expense_txns["amount"]).sum()) if not expense_txns.empty else 0.0
    detected_monthly_expenses = detected_monthly_expenses / months_in_period
    
    # Compare declared vs detected
    declared_income = questionnaire.monthly_income
    declared_expenses = questionnaire.monthly_living_expenses
    
    income_diff = detected_monthly_income - declared_income
    income_diff_pct = (income_diff / declared_income * 100) if declared_income > 0 else 0
    
    expense_diff = detected_monthly_expenses - declared_expenses
    expense_diff_pct = (expense_diff / declared_expenses * 100) if declared_expenses > 0 else 0
    
    # Build discrepancies list
    discrepancies = []
    warnings = []
    
    # Income discrepancy
    if abs(income_diff_pct) > 20:  # More than 20% difference
        if income_diff > 0:
            discrepancies.append({
                "type": "income_overestimate",
                "message": f"Вы заявили доход {declared_income:,.0f} KZT/месяц, но в выписке обнаружено {detected_monthly_income:,.0f} KZT/месяц. Разница: {income_diff:+,.0f} KZT ({income_diff_pct:+.1f}%).",
                "severity": "medium",
            })
        else:
            discrepancies.append({
                "type": "income_underestimate",
                "message": f"Вы заявили доход {declared_income:,.0f} KZT/месяц, но в выписке обнаружено {detected_monthly_income:,.0f} KZT/месяц. Разница: {income_diff:+,.0f} KZT ({income_diff_pct:+.1f}%). Возможно, часть дохода не проходит через этот счет.",
                "severity": "low",
            })
    
    # Expense discrepancy
    if abs(expense_diff_pct) > 20:  # More than 20% difference
        if expense_diff > 0:
            discrepancies.append({
                "type": "expense_underestimate",
                "message": f"Вы заявили расходы {declared_expenses:,.0f} KZT/месяц, но в выписке обнаружено {detected_monthly_expenses:,.0f} KZT/месяц. Разница: {expense_diff:+,.0f} KZT ({expense_diff_pct:+.1f}%). Вы тратите больше, чем думаете.",
                "severity": "high",
            })
            warnings.append("Вы недооцениваете свои расходы. Это может привести к финансовым проблемам.")
        else:
            discrepancies.append({
                "type": "expense_overestimate",
                "message": f"Вы заявили расходы {declared_expenses:,.0f} KZT/месяц, но в выписке обнаружено {detected_monthly_expenses:,.0f} KZT/месяц. Разница: {expense_diff:+,.0f} KZT ({expense_diff_pct:+.1f}%). Возможно, часть расходов проходит через другие счета.",
                "severity": "low",
            })
    
    return {
        "income_comparison": {
            "declared": declared_income,
            "detected": detected_monthly_income,
            "difference": income_diff,
            "difference_percent": income_diff_pct,
        },
        "expense_comparison": {
            "declared": declared_expenses,
            "detected": detected_monthly_expenses,
            "difference": expense_diff,
            "difference_percent": expense_diff_pct,
        },
        "discrepancies": discrepancies,
        "warnings": warnings,
    }


def build_financial_reality_summary(
    questionnaire: QuestionnaireAnswers,
    detected_income: float,
    detected_expenses: float,
    credit_behavior: Dict[str, Any],
    discrepancies: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Build the Financial Reality Summary with three sections:
    1. What you think your situation is (declared)
    2. What transactions show (detected)
    3. What this means (advisor interpretation)
    """
    
    # Section 1: What user thinks
    user_view = {
        "monthly_income": questionnaire.monthly_income,
        "monthly_expenses": questionnaire.monthly_living_expenses,
        "monthly_credit_payments": questionnaire.monthly_credit_payments,
        "total_debt": questionnaire.total_outstanding_debt,
        "safety_buffer": questionnaire.financial_safety_months,
        "income_stability": questionnaire.income_stability,
    }
    
    # Section 2: What transactions show
    detected_view = {
        "monthly_income": detected_income,
        "monthly_expenses": detected_expenses,
        "credit_behavior": {
            "loans_issued": credit_behavior.get("total_loans_issued", 0),
            "repayments": credit_behavior.get("total_repayments", 0),
            "net_flow": credit_behavior.get("net_credit_flow", 0),
            "frequency": credit_behavior.get("loan_frequency", 0),
            "refinancing": credit_behavior.get("refinancing_detected", False),
        },
    }
    
    # Section 3: What this means (interpretation)
    interpretation = []
    
    # Income interpretation
    if detected_income > questionnaire.monthly_income * 1.2:
        interpretation.append({
            "category": "income",
            "message": "Ваш фактический доход выше заявленного. Возможно, вы не учитываете все источники дохода, или часть дохода — это кредитные поступления.",
            "severity": "info",
        })
    elif detected_income < questionnaire.monthly_income * 0.8:
        interpretation.append({
            "category": "income",
            "message": "Ваш фактический доход ниже заявленного. Возможно, часть дохода проходит через другие счета, или доход нестабилен.",
            "severity": "warning",
        })
    
    # Expense interpretation
    if detected_expenses > questionnaire.monthly_living_expenses * 1.2:
        interpretation.append({
            "category": "expenses",
            "message": "Вы тратите больше, чем думаете. Это серьезная проблема — недооценка расходов ведет к долгам и финансовым проблемам.",
            "severity": "critical",
        })
    elif detected_expenses < questionnaire.monthly_living_expenses * 0.8:
        interpretation.append({
            "category": "expenses",
            "message": "Ваши фактические расходы ниже заявленных. Возможно, часть расходов проходит через другие счета или карты.",
            "severity": "info",
        })
    
    # Credit behavior interpretation
    if credit_behavior.get("refinancing_detected", False):
        interpretation.append({
            "category": "credit_behavior",
            "message": "Обнаружены признаки рефинансирования: вы берете новые кредиты для погашения старых. Это опасная практика.",
            "severity": "critical",
        })
    
    if credit_behavior.get("credit_spiral_risk") in ["high", "medium"]:
        interpretation.append({
            "category": "credit_behavior",
            "message": "Ваше кредитное поведение указывает на высокий риск кредитной спирали. Частые новые кредиты — это красный флаг.",
            "severity": "critical",
        })
    
    # Debt load interpretation
    if questionnaire.total_outstanding_debt:
        debt_to_income_ratio = (questionnaire.total_outstanding_debt / questionnaire.monthly_income) if questionnaire.monthly_income > 0 else 0
        if debt_to_income_ratio > 12:  # More than 1 year of income
            interpretation.append({
                "category": "debt_load",
                "message": f"Ваш долг составляет {debt_to_income_ratio:.1f} месячных доходов. Это критически высокий уровень долговой нагрузки.",
                "severity": "critical",
            })
        elif debt_to_income_ratio > 6:
            interpretation.append({
                "category": "debt_load",
                "message": f"Ваш долг составляет {debt_to_income_ratio:.1f} месячных доходов. Это высокий уровень долговой нагрузки.",
                "severity": "warning",
            })
    
    return {
        "what_you_think": user_view,
        "what_transactions_show": detected_view,
        "what_this_means": interpretation,
        "discrepancies": discrepancies,
    }
