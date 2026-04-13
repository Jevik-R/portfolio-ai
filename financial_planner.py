"""
financial_planner.py — Personal Finance Planning Module
────────────────────────────────────────────────────────
Standalone module: no external API calls, pure Python math and logic.
Can be imported and tested independently of the rest of PortfolioAI.

Classes:
    FinancialProfile     — stores user financial inputs with validation
    FinancialAnalyzer    — calculates derived metrics from a profile
    RiskProfiler         — 5-question risk tolerance questionnaire
    AssetAllocator       — maps risk profile → equity/debt/gold/cash split
    FinancialPlanGenerator — combines all the above into one plan dict
"""

from __future__ import annotations
from dataclasses import dataclass


# ══════════════════════════════════════════════════════════════════════════════
#  CLASS 1 — FinancialProfile
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class FinancialProfile:
    """Stores all user financial inputs with validation."""

    monthly_income:           float   # take-home salary after tax
    fixed_expenses:           float   # rent, EMI, subscriptions
    variable_expenses:        float   # food, transport, shopping
    existing_savings:         float   # FD + savings account balance
    age:                      int
    dependents:               int     # 0 = single, 1+ = family

    # optional fields with sensible defaults
    existing_investments:     float = 0.0   # stocks / MF already held
    monthly_sip:              float = 0.0   # existing SIPs already running
    insurance_premium_annual: float = 0.0   # sum of all insurance premiums/yr
    has_emergency_fund:       bool  = False
    loan_emi:                 float = 0.0   # monthly loan EMI (home/car/personal)

    def __post_init__(self) -> None:
        self._validate()

    # ── Validation ────────────────────────────────────────────────────────────

    def _validate(self) -> None:
        if self.monthly_income <= 0:
            raise ValueError("Monthly income must be greater than ₹0.")

        total_core_exp = self.fixed_expenses + self.variable_expenses + self.loan_emi
        if total_core_exp > self.monthly_income * 0.95:
            raise ValueError(
                f"Total expenses (₹{total_core_exp:,.0f}/month) exceed 95% of your "
                f"income (₹{self.monthly_income * 0.95:,.0f}). "
                "Please review your expense figures."
            )

        if not (18 <= self.age <= 70):
            raise ValueError(
                f"Age must be between 18 and 70. Got: {self.age}"
            )

        non_negative_fields = {
            "Monthly income":        self.monthly_income,
            "Fixed expenses":        self.fixed_expenses,
            "Variable expenses":     self.variable_expenses,
            "Existing savings":      self.existing_savings,
            "Existing investments":  self.existing_investments,
            "Monthly SIP":           self.monthly_sip,
            "Insurance premium":     self.insurance_premium_annual,
            "Loan EMI":              self.loan_emi,
        }
        for field_name, val in non_negative_fields.items():
            if val < 0:
                raise ValueError(f"{field_name} cannot be negative. Got: {val}")

        if self.dependents < 0:
            raise ValueError("Number of dependents cannot be negative.")


# ══════════════════════════════════════════════════════════════════════════════
#  CLASS 2 — FinancialAnalyzer
# ══════════════════════════════════════════════════════════════════════════════

class FinancialAnalyzer:
    """Calculates all derived financial metrics from a FinancialProfile."""

    def __init__(self, profile: FinancialProfile) -> None:
        self.p = profile

    # ── Core metrics ──────────────────────────────────────────────────────────

    def total_expenses(self) -> float:
        """Fixed + variable + loan EMI + monthly share of insurance."""
        return (
            self.p.fixed_expenses
            + self.p.variable_expenses
            + self.p.loan_emi
            + (self.p.insurance_premium_annual / 12)
        )

    def net_disposable_income(self) -> float:
        """What is left after all monthly expenses."""
        return self.p.monthly_income - self.total_expenses()

    def savings_rate(self) -> float:
        """Savings as a percentage of income."""
        return (self.net_disposable_income() / self.p.monthly_income) * 100

    # ── Emergency fund ────────────────────────────────────────────────────────

    def emergency_fund_required(self) -> float:
        """6 months of total expenses — standard thumb rule."""
        return self.total_expenses() * 6

    def emergency_fund_gap(self) -> float:
        """How much more is needed to complete the emergency fund."""
        return max(0.0, self.emergency_fund_required() - self.p.existing_savings)

    def months_to_fill_emergency_fund(self, monthly_allocation: float) -> float:
        """Months required to close the emergency fund gap."""
        gap = self.emergency_fund_gap()
        if gap <= 0 or monthly_allocation <= 0:
            return 0.0
        return gap / monthly_allocation

    # ── Investment capacity ───────────────────────────────────────────────────

    def safe_monthly_investment(self) -> tuple[float, float]:
        """
        Returns (emergency_monthly_allocation, safe_investment_amount).

        Logic
        ─────
        1. Start with net disposable income.
        2. If emergency fund gap exists → 40 % goes to emergency fund,
           60 % is the investable base.
           Else → 100 % is the investable base.
        3. Apply a 10 % safety buffer on the investable base
           (keep as cash for unexpected expenses).
        """
        net = self.net_disposable_income()

        if self.emergency_fund_gap() > 0:
            emergency_allocation = net * 0.40
            investable_base      = net * 0.60
        else:
            emergency_allocation = 0.0
            investable_base      = net

        safe_investment = investable_base * 0.90   # 10 % safety buffer
        return emergency_allocation, safe_investment

    # ── Health score ──────────────────────────────────────────────────────────

    def financial_health_score(self) -> tuple[int, str]:
        """
        Returns (score_out_of_100, label_with_colour).

        Scoring breakdown
        ─────────────────
        Savings rate > 20 %            → +20 pts
        Savings rate > 30 %            → +10 pts (bonus, stacks)
        Emergency fund complete         → +20 pts
        Has insurance                   → +15 pts
        No loan EMI                     → +15 pts
        Age-appropriate savings         → +10 pts
        Expense ratio < 50 %           → +10 pts
        ──────────────────────────────
        Max                            → 100 pts

        Labels: 80+ Excellent 🟢 | 60+ Good 🟡 | 40+ Needs Attention 🟠 | <40 Critical 🔴
        """
        score = 0
        sr    = self.savings_rate()

        if sr > 20:
            score += 20
        if sr > 30:
            score += 10   # stacks with the above

        if self.emergency_fund_gap() == 0:
            score += 20

        if self.p.insurance_premium_annual > 0:
            score += 15

        if self.p.loan_emi == 0:
            score += 15

        # Age-appropriate savings heuristic:
        # By (age - 25) years of earning, should have that many years × income saved
        # Simplified: total assets ≥ max(0, (age-25)) × annual_income / 12
        age_target = max(0, self.p.age - 25) * (self.p.monthly_income / 12)
        total_assets = self.p.existing_savings + self.p.existing_investments
        if self.p.age <= 25 or total_assets >= age_target:
            score += 10

        expense_ratio = self.total_expenses() / self.p.monthly_income
        if expense_ratio < 0.50:
            score += 10

        score = min(100, max(0, score))

        if score >= 80:
            label = "Excellent 🟢"
        elif score >= 60:
            label = "Good 🟡"
        elif score >= 40:
            label = "Needs Attention 🟠"
        else:
            label = "Critical 🔴"

        return score, label

    # ── Insights ──────────────────────────────────────────────────────────────

    def generate_insights(self) -> list[str]:
        """Returns 3–5 personalised insight strings with actual rupee numbers."""
        insights: list[str] = []
        p = self.p

        # Insight 1: Expense ratio
        expense_ratio = self.total_expenses() / p.monthly_income * 100
        if expense_ratio < 50:
            insights.append(
                f"Your expense ratio is {expense_ratio:.0f}% of income — excellent discipline! "
                "Keep it under 50% to maximise wealth creation."
            )
        else:
            savings_needed = (expense_ratio - 50) / 100 * p.monthly_income
            insights.append(
                f"Your expense ratio is {expense_ratio:.0f}% — aim for under 50%. "
                f"Cutting ₹{savings_needed:,.0f}/month in spending would make a big difference."
            )

        # Insight 2: Emergency fund
        gap = self.emergency_fund_gap()
        em_alloc, _ = self.safe_monthly_investment()
        if gap > 0:
            months = self.months_to_fill_emergency_fund(em_alloc)
            insights.append(
                f"You need ₹{gap:,.0f} more to complete your emergency fund "
                f"(target: ₹{self.emergency_fund_required():,.0f}). "
                f"At ₹{em_alloc:,.0f}/month you'll be done in "
                f"{months:.0f} month{'s' if months != 1 else ''}."
            )
        else:
            insights.append(
                "Your emergency fund is fully funded — great foundation! "
                "You can invest with confidence knowing you have a safety net."
            )

        # Insight 3: Savings rate
        sr = self.savings_rate()
        if sr < 20:
            extra_needed = (0.20 - sr / 100) * p.monthly_income
            insights.append(
                f"Your savings rate is {sr:.0f}% — try to reach 20% "
                f"by saving ₹{extra_needed:,.0f} more each month."
            )
        elif sr < 30:
            extra_needed = (0.30 - sr / 100) * p.monthly_income
            insights.append(
                f"Good savings rate of {sr:.0f}%! Pushing to 30% would only need "
                f"₹{extra_needed:,.0f}/month more — well worth targeting."
            )
        else:
            insights.append(
                f"Excellent savings rate of {sr:.0f}%! "
                "You're on track for early financial independence."
            )

        # Insight 4: Insurance
        if p.insurance_premium_annual == 0:
            insights.append(
                "No insurance detected — consider a term life plan and health cover. "
                "A ₹1 Crore term plan typically costs ₹8,000–₹15,000/year at your age."
            )
        else:
            monthly_ins = p.insurance_premium_annual / 12
            insights.append(
                f"Insurance costs ₹{monthly_ins:,.0f}/month. "
                "Ensure you have both adequate life cover and health insurance."
            )

        # Insight 5: SIP (optional, only if running)
        if p.monthly_sip > 0:
            insights.append(
                f"You already invest ₹{p.monthly_sip:,.0f}/month via SIPs. "
                "Consistency beats timing — keep them running through market cycles."
            )

        return insights[:5]


# ══════════════════════════════════════════════════════════════════════════════
#  CLASS 3 — RiskProfiler
# ══════════════════════════════════════════════════════════════════════════════

RISK_QUESTIONS: list[dict] = [
    {
        "id":   "age_group",
        "text": "What is your age group?",
        "options": [
            ("A) 18–30 years — long investment horizon", 4),
            ("B) 31–40 years",                          3),
            ("C) 41–50 years",                          2),
            ("D) 51+ years — approaching retirement",   1),
        ],
    },
    {
        "id":   "income_stability",
        "text": "How stable is your income?",
        "options": [
            ("A) Government job / very stable",          4),
            ("B) Private job, established company",      3),
            ("C) Private job, startup / small firm",     2),
            ("D) Self-employed / freelancer",            1),
        ],
    },
    {
        "id":   "investment_horizon",
        "text": "How long can you stay invested without needing this money?",
        "options": [
            ("A) More than 7 years", 4),
            ("B) 5–7 years",         3),
            ("C) 3–5 years",         2),
            ("D) Less than 3 years", 1),
        ],
    },
    {
        "id":   "market_fall_reaction",
        "text": "If your portfolio drops 20% in 3 months, you would:",
        "options": [
            ("A) Buy more — great opportunity!",   4),
            ("B) Stay calm and hold",              3),
            ("C) Feel worried but hold",           2),
            ("D) Sell immediately to stop losses", 1),
        ],
    },
    {
        "id":   "financial_dependents",
        "text": "How many people depend on your income?",
        "options": [
            ("A) None — only myself",        4),
            ("B) Spouse (both earning)",      3),
            ("C) Spouse + 1 child",          2),
            ("D) Spouse + 2 or more children", 1),
        ],
    },
]

_RISK_TIERS: list[tuple[int, str, str, str]] = [
    # (min_score, label, emoji, description)
    (
        17, "Aggressive", "🔴",
        "You have a high risk appetite and a long investment horizon. "
        "You can handle significant volatility in exchange for potentially higher returns. "
        "Small and mid cap stocks, sectoral funds, and high-growth opportunities suit you well.",
    ),
    (
        13, "Moderate", "🟡",
        "You seek a balance between growth and stability. "
        "You can tolerate moderate market swings but prefer a diversified mix. "
        "A blend of large-cap equities, debt funds, and some mid caps works best.",
    ),
    (
        9, "Conservative", "🟢",
        "You prioritise capital preservation over aggressive growth. "
        "Steady, predictable returns matter more than high but volatile gains. "
        "Large-cap index funds, PPF, debt mutual funds, and gold are your core instruments.",
    ),
    (
        0, "Very Conservative", "🔵",
        "You prefer maximum safety and minimal market exposure. "
        "FDs, liquid funds, sovereign gold bonds, and debt instruments suit you best. "
        "Equity exposure only through large-cap index funds in small doses.",
    ),
]


class RiskProfiler:
    """Determines investment risk tolerance from five questionnaire answers."""

    @staticmethod
    def questions() -> list[dict]:
        return RISK_QUESTIONS

    @staticmethod
    def score(answers: list[int]) -> tuple[int, str, str, str]:
        """
        Parameters
        ----------
        answers : list of 5 point values (1–4 each), matching RISK_QUESTIONS order

        Returns
        -------
        (total_score, label, emoji, description)
        """
        if len(answers) != 5:
            raise ValueError(f"Exactly 5 answers required, got {len(answers)}.")
        if not all(1 <= a <= 4 for a in answers):
            raise ValueError("Each answer must be between 1 and 4.")

        total = sum(answers)

        for min_score, label, emoji, description in _RISK_TIERS:
            if total >= min_score:
                return total, label, emoji, description

        # Fallback (should never reach here given valid input)
        return total, "Very Conservative", "🔵", _RISK_TIERS[-1][3]


# ══════════════════════════════════════════════════════════════════════════════
#  CLASS 4 — AssetAllocator
# ══════════════════════════════════════════════════════════════════════════════

_ALLOCATION_TABLE: dict[str, dict] = {
    "Very Conservative": {
        "equity": 0.20, "debt": 0.50, "gold": 0.15, "cash": 0.15,
        "equity_breakdown": {"largecap": 1.00, "midcap": 0.00, "smallcap": 0.00},
        "products": {
            "equity": "Nifty 50 Index Fund (large cap only)",
            "debt":   "FD + Debt Mutual Funds",
            "gold":   "Sovereign Gold Bonds (SGB)",
            "cash":   "Liquid Fund / Savings Account",
        },
    },
    "Conservative": {
        "equity": 0.35, "debt": 0.40, "gold": 0.15, "cash": 0.10,
        "equity_breakdown": {"largecap": 0.80, "midcap": 0.20, "smallcap": 0.00},
        "products": {
            "equity": "Large Cap Fund + Nifty Next 50",
            "debt":   "Debt Mutual Funds + PPF",
            "gold":   "Gold ETF / Sovereign Gold Bond",
            "cash":   "Liquid Fund",
        },
    },
    "Moderate": {
        "equity": 0.55, "debt": 0.25, "gold": 0.10, "cash": 0.10,
        "equity_breakdown": {"largecap": 0.60, "midcap": 0.30, "smallcap": 0.10},
        "products": {
            "equity": "NSE Stocks via PortfolioAI (Large + Mid Cap)",
            "debt":   "Debt Mutual Funds + PPF",
            "gold":   "Gold ETF",
            "cash":   "Liquid Fund",
        },
    },
    "Aggressive": {
        "equity": 0.75, "debt": 0.10, "gold": 0.10, "cash": 0.05,
        "equity_breakdown": {"largecap": 0.50, "midcap": 0.30, "smallcap": 0.20},
        "products": {
            "equity": "NSE Stocks via PortfolioAI (Large + Mid + Small Cap)",
            "debt":   "Short-Duration Debt Fund",
            "gold":   "Gold ETF",
            "cash":   "Liquid Fund (minimal cash drag)",
        },
    },
}


class AssetAllocator:
    """
    Maps a risk profile label to concrete rupee amounts across
    equity / debt / gold / cash, given a monthly investable amount.
    """

    @staticmethod
    def get_allocation(risk_label: str, monthly_amount: float) -> dict:
        """
        Parameters
        ----------
        risk_label    : "Very Conservative", "Conservative", "Moderate", or "Aggressive"
                        (emoji suffix stripped automatically)
        monthly_amount: total investable rupees per month

        Returns
        -------
        dict with equity_pct, equity_amount, debt_pct, debt_amount,
             gold_pct, gold_amount, cash_pct, cash_amount,
             equity_breakdown {largecap, midcap, smallcap},
             nse_stock_amount (= equity_amount → feeds BL optimizer),
             products (recommended instrument names),
             risk_label (clean, no emoji)
        """
        # Strip trailing emoji if present
        for emoji in (" 🔴", " 🟡", " 🟢", " 🔵"):
            risk_label = risk_label.replace(emoji, "")
        risk_label = risk_label.strip()

        if risk_label not in _ALLOCATION_TABLE:
            risk_label = "Moderate"

        tbl = _ALLOCATION_TABLE[risk_label]

        eq_amt  = monthly_amount * tbl["equity"]
        dbt_amt = monthly_amount * tbl["debt"]
        gld_amt = monthly_amount * tbl["gold"]
        csh_amt = monthly_amount * tbl["cash"]
        bkdn    = tbl["equity_breakdown"]

        return {
            "equity_pct":    tbl["equity"],
            "equity_amount": eq_amt,
            "debt_pct":      tbl["debt"],
            "debt_amount":   dbt_amt,
            "gold_pct":      tbl["gold"],
            "gold_amount":   gld_amt,
            "cash_pct":      tbl["cash"],
            "cash_amount":   csh_amt,
            "equity_breakdown": {
                "largecap":  eq_amt * bkdn["largecap"],
                "midcap":    eq_amt * bkdn["midcap"],
                "smallcap":  eq_amt * bkdn["smallcap"],
            },
            "nse_stock_amount": eq_amt,   # feeds directly into BL optimizer
            "products":    tbl["products"],
            "risk_label":  risk_label,
        }


# ══════════════════════════════════════════════════════════════════════════════
#  CLASS 5 — FinancialPlanGenerator
# ══════════════════════════════════════════════════════════════════════════════

class FinancialPlanGenerator:
    """Assembles a complete financial plan from profile + risk answers."""

    @staticmethod
    def generate_plan(
        profile:      FinancialProfile,
        risk_answers: list[int],
    ) -> dict:
        """
        Parameters
        ----------
        profile      : validated FinancialProfile
        risk_answers : list of 5 integer scores (1–4 each), one per RISK_QUESTIONS entry

        Returns
        -------
        Complete plan dict — see docstring for key structure.
        """
        az = FinancialAnalyzer(profile)

        # Financial health
        score, health_label = az.financial_health_score()
        insights            = az.generate_insights()

        # Budget metrics
        total_exp   = az.total_expenses()
        net_disp    = az.net_disposable_income()
        sav_rate    = az.savings_rate()
        ef_req      = az.emergency_fund_required()
        ef_gap      = az.emergency_fund_gap()
        em_alloc, inv_amt = az.safe_monthly_investment()
        months_ef   = az.months_to_fill_emergency_fund(em_alloc)

        # Risk profile
        risk_total, risk_lbl, risk_emoji, risk_desc = RiskProfiler.score(risk_answers)
        full_risk_label = f"{risk_lbl} {risk_emoji}"

        # Asset allocation
        allocation = AssetAllocator.get_allocation(risk_lbl, inv_amt)

        # Action plan (numbered, step-by-step)
        action_plan: list[str] = []
        step = 1

        if ef_gap > 0:
            action_plan.append(
                f"Step {step}: Build emergency fund — put ₹{em_alloc:,.0f}/month "
                f"in a liquid fund until you accumulate ₹{ef_req:,.0f} "
                f"(~{months_ef:.0f} more months to go)."
            )
            step += 1

        if allocation["debt_amount"] > 0:
            action_plan.append(
                f"Step {step}: Start SIP of ₹{allocation['debt_amount']:,.0f}/month "
                f"in {allocation['products']['debt']}."
            )
            step += 1

        if allocation["nse_stock_amount"] > 0:
            action_plan.append(
                f"Step {step}: Invest ₹{allocation['nse_stock_amount']:,.0f}/month "
                f"in NSE stocks via PortfolioAI "
                f"({allocation['products']['equity']})."
            )
            step += 1

        if allocation["gold_amount"] > 0:
            action_plan.append(
                f"Step {step}: Buy ₹{allocation['gold_amount']:,.0f} of "
                f"{allocation['products']['gold']} monthly."
            )
            step += 1

        if allocation["cash_amount"] > 0:
            action_plan.append(
                f"Step {step}: Keep ₹{allocation['cash_amount']:,.0f}/month "
                f"in {allocation['products']['cash']} as a liquidity buffer."
            )

        return {
            "health_score": score,
            "health_label": health_label,
            "insights":     insights,
            "emergency_fund": {
                "required":           ef_req,
                "existing":           profile.existing_savings,
                "gap":                ef_gap,
                "monthly_allocation": em_alloc,
                "months_to_complete": months_ef,
            },
            "monthly_budget": {
                "income":               profile.monthly_income,
                "total_expenses":       total_exp,
                "net_disposable":       net_disp,
                "savings_rate":         sav_rate,
                "emergency_allocation": em_alloc,
                "investment_amount":    inv_amt,
            },
            "risk_profile": {
                "score":       risk_total,
                "label":       full_risk_label,
                "clean_label": risk_lbl,
                "emoji":       risk_emoji,
                "description": risk_desc,
            },
            "asset_allocation": allocation,
            "action_plan":      action_plan,
        }


# ══════════════════════════════════════════════════════════════════════════════
#  STANDALONE SMOKE TEST — run:  python financial_planner.py
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # ── Sample data from spec ─────────────────────────────────────────────────
    profile = FinancialProfile(
        monthly_income=80_000,
        fixed_expenses=25_000,
        variable_expenses=15_000,
        existing_savings=200_000,
        age=28,
        dependents=0,
        existing_investments=0,
        monthly_sip=0,
        insurance_premium_annual=12_000,
        has_emergency_fund=False,
        loan_emi=0,
    )

    az = FinancialAnalyzer(profile)

    print()
    print("=" * 60)
    print("  STEP 1 OUTPUT — FinancialAnalyzer")
    print("=" * 60)
    print(f"  Monthly income          : ₹{profile.monthly_income:>10,.0f}")
    print(f"  Total expenses/month    : ₹{az.total_expenses():>10,.0f}")
    print(f"  Net disposable income   : ₹{az.net_disposable_income():>10,.0f}")
    print(f"  Savings rate            : {az.savings_rate():>9.1f}%")
    print(f"  Emergency fund required : ₹{az.emergency_fund_required():>10,.0f}")
    print(f"  Emergency fund gap      : ₹{az.emergency_fund_gap():>10,.0f}")

    em_alloc, inv_amt = az.safe_monthly_investment()
    print(f"  Emergency allocation/mo : ₹{em_alloc:>10,.0f}")
    print(f"  Safe to invest/month    : ₹{inv_amt:>10,.0f}")

    score, label = az.financial_health_score()
    print(f"  Financial health score  : {score}/100  ({label})")

    print()
    print("  Personalised insights:")
    for i, insight in enumerate(az.generate_insights(), 1):
        print(f"    {i}. {insight}")

    print()
    print("─" * 60)
    print("  Risk Profile (sample: Moderate answers)")
    print("─" * 60)
    sample_answers = [4, 3, 3, 3, 4]   # Moderate-to-Aggressive
    total, lbl, emoji, desc = RiskProfiler.score(sample_answers)
    print(f"  Score: {total}/20  →  {lbl} {emoji}")
    print(f"  {desc}")

    print()
    print("─" * 60)
    print("  Asset Allocation")
    print("─" * 60)
    alloc = AssetAllocator.get_allocation(lbl, inv_amt)
    print(f"  Equity  {alloc['equity_pct']*100:.0f}%  →  ₹{alloc['equity_amount']:>9,.0f}/month")
    print(f"    Large cap  ₹{alloc['equity_breakdown']['largecap']:>9,.0f}")
    print(f"    Mid cap    ₹{alloc['equity_breakdown']['midcap']:>9,.0f}")
    print(f"    Small cap  ₹{alloc['equity_breakdown']['smallcap']:>9,.0f}")
    print(f"  Debt    {alloc['debt_pct']*100:.0f}%  →  ₹{alloc['debt_amount']:>9,.0f}/month")
    print(f"  Gold    {alloc['gold_pct']*100:.0f}%  →  ₹{alloc['gold_amount']:>9,.0f}/month")
    print(f"  Cash    {alloc['cash_pct']*100:.0f}%  →  ₹{alloc['cash_amount']:>9,.0f}/month")
    print(f"  NSE stock amount (→ BL optimizer): ₹{alloc['nse_stock_amount']:,.0f}")

    print()
    print("─" * 60)
    print("  Complete Action Plan")
    print("─" * 60)
    plan = FinancialPlanGenerator.generate_plan(profile, sample_answers)
    for step in plan["action_plan"]:
        print(f"  {step}")
    print("=" * 60)
    print()
