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
    """User onboarding questionnaire responses"""
    has_credits: bool = False
    credit_load_perception: str = "low"  # "low", "medium", "high", "very_high"
    financial_safety_months: Optional[int] = None  # months can survive without income
    has_savings: bool = False
    savings_amount: Optional[float] = None
    financial_behavior: str = "reactive"  # "planned", "reactive", "mixed"
    primary_goal: str = "stabilize"  # "reduce_debt", "save", "stabilize"
    ready_for_real_picture: bool = True


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
