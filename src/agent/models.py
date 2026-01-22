from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional


@dataclass
class Transaction:
    date: datetime
    description: str
    amount: float  # positive = credit, negative = debit
    balance: Optional[float] = None


@dataclass
class CreditTransaction:
    """Credit-specific transaction (loan issuance, repayment, etc.)"""
    date: datetime
    description: str
    amount: float  # positive = loan issued, negative = repayment
    loan_type: Optional[str] = None  # "cash_loan", "business_loan", "installment", "credit_card"
    remaining_balance: Optional[float] = None


@dataclass
class QuestionnaireAnswers:
    """User onboarding questionnaire responses - MANDATORY declared financial data"""
    # Required financial numbers
    monthly_income: float  # Average monthly income in KZT
    income_stability: str  # "stable", "fluctuating", "unstable"
    monthly_living_expenses: float  # Monthly expenses excluding credits, in KZT
    monthly_credit_payments: float  # Monthly payments for all credits/installments, in KZT
    total_outstanding_debt: Optional[float] = None  # Total current debt (can be None if unknown)
    total_debt_range: Optional[str] = None  # "0-100k", "100k-500k", "500k-1m", "1m-3m", "3m+"
    financial_safety_months: str = "0"  # "0", "<1", "1-3", "3+"
    primary_goal: str = "understand_reality"  # "reduce_debt", "save", "stabilize", "understand_reality"


@dataclass
class StatementSummary:
    transactions: List[Transaction] = field(default_factory=list)
    currency: str = "unknown"

    @property
    def total_credit(self) -> float:
        return sum(t.amount for t in self.transactions if t.amount > 0)

    @property
    def total_debit(self) -> float:
        return sum(-t.amount for t in self.transactions if t.amount < 0)

    @property
    def closing_balance(self) -> Optional[float]:
        if not self.transactions:
            return None
        last_with_balance = next(
            (t.balance for t in reversed(self.transactions) if t.balance is not None),
            None,
        )
        return last_with_balance
