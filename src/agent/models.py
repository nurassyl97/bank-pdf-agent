from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass
class Transaction:
    date: datetime
    description: str
    amount: float  # positive = credit, negative = debit
    balance: Optional[float] = None


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
