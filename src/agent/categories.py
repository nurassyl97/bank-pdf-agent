import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class CategoryRule:
    category: str
    patterns: Tuple[re.Pattern, ...]

    def matches(self, description: str) -> bool:
        d = description.lower()
        return any(p.search(d) for p in self.patterns)


def _p(*patterns: str) -> Tuple[re.Pattern, ...]:
    return tuple(re.compile(p, re.IGNORECASE) for p in patterns)


# Kaspi (KZ) statements tend to include RU/KZ merchant descriptors; rules are heuristic.
KASPI_RULES: List[CategoryRule] = [
    CategoryRule("transfers", _p(r"\bперевод\b", r"\bперечислен", r"\bkaspi\s*перевод\b", r"\btransfer\b")),
    CategoryRule("cash_withdrawal", _p(r"\bснятие\b", r"\batm\b", r"\bбанкомат\b")),
    CategoryRule("salary", _p(r"\bзарплат", r"\bоклад\b", r"\bsalary\b")),
    CategoryRule("fees", _p(r"\bкомисси", r"\bfee\b", r"\bservice\s*charge\b")),
    CategoryRule("taxes", _p(r"\bналог\b", r"\btax\b")),
    CategoryRule("utilities", _p(r"\bкоммун", r"\bэлектро", r"\bвода\b", r"\bгаз\b", r"\binternet\b", r"\bтелефон\b")),
    CategoryRule("groceries", _p(r"\bmagnum\b", r"\bsmall\b", r"\bstore\b", r"\bmarket\b", r"\bсупермаркет\b", r"\bпродукт")),
    CategoryRule("restaurants", _p(r"\bcafe\b", r"\bкофе\b", r"\bресторан\b", r"\bfast\s*food\b", r"\bburger\b")),
    CategoryRule("transport", _p(r"\btaxi\b", r"\byandex\b", r"\buber\b", r"\btransport\b", r"\bпроезд\b", r"\bавтобус\b", r"\bметро\b")),
    CategoryRule("fuel", _p(r"\bазс\b", r"\bfuel\b", r"\bpetrol\b", r"\bbenz\b", r"\bgas\s*station\b")),
    CategoryRule("health", _p(r"\bаптека\b", r"\bpharm", r"\bclinic\b", r"\bмед\b")),
    CategoryRule("shopping", _p(r"\bkaspi\s*магазин\b", r"\bмагазин\b", r"\bshop\b", r"\bmarketplace\b")),
    CategoryRule("subscriptions", _p(r"\bnetflix\b", r"\bspotify\b", r"\bsubscription\b", r"\bподписк")),
    CategoryRule("education", _p(r"\bкурс\b", r"\bшкол", r"\bуниверситет\b", r"\bedu\b")),
]


def categorize(description: str, rules: Optional[List[CategoryRule]] = None) -> str:
    rules = rules or KASPI_RULES
    for rule in rules:
        if rule.matches(description):
            return rule.category
    return "other"


def categories_version() -> str:
    # bump when you change rules materially; helps you invalidate cached results
    return "kaspi-v1"
