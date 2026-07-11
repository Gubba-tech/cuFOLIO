"""Explain the relationship between the existing Mean-CVaR LP and unified QP."""

from cufolio import QPParameters, QuadraticPortfolioOptimizer
from cufolio.cvar_parameters import CvarParameters


def main() -> None:
    print("status: overview")
    print("objective: not_solved")
    print("max_constraint_violation: n/a")
    print("existing_workflow: cuFOLIO Mean-CVaR LP, scenario-based tail-risk")
    print("new_workflow: PortOpt unified QP, variance/Sharpe/regularization")
    print("relationship: complementary workflows, neither replaces the other")
    print(f"qp_api: {QPParameters.__name__}, {QuadraticPortfolioOptimizer.__name__}")
    print(f"cvar_api: {CvarParameters.__name__}")
    print("speedup_claim: none")


if __name__ == "__main__":
    main()
