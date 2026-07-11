# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""cuFOLIO-native optimizer for unified QP portfolio formulations."""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
import pandas as pd

from .base_optimizer import BaseOptimizer
from .portfolio import Portfolio
from .qp_backend import (
    solve_compiled_qp_cuopt,
    solve_compiled_qp_osqp,
)
from .qp_formulations import compile_portfolio_qp
from .qp_parameters import QPParameters
from .settings import ApiSettings


class QuadraticPortfolioOptimizer(BaseOptimizer):
    """PortOpt-style QP optimizer that returns cuFOLIO result objects."""

    def __init__(
        self,
        returns_dict: dict,
        qp_params: QPParameters,
        api_settings: Optional[ApiSettings] = None,
        existing_portfolio: Optional[Portfolio] = None,
    ):
        if api_settings is None:
            api_settings = ApiSettings(
                api="cuopt_python" if qp_params.backend == "cuopt" else "cvxpy",
                scale_risk_aversion=False,
            )
        super().__init__(
            returns_dict,
            qp_params,
            api_settings=api_settings,
            existing_portfolio=existing_portfolio,
            risk_measure="variance",
        )
        self._result_columns = [
            "regime",
            "solver",
            "objective",
            "status",
            "raw_status",
            "solve_time",
            "total_time",
            "objective_value",
            "c_scale",
            "expected_return",
            "excess_return",
            "variance",
            "volatility",
            "sharpe",
            "lambda_l1",
            "lambda_l2",
            "short_budget",
            "turnover",
            "max_constraint_violation",
            "factor_weights",
            "recovered_weights",
            "stock_weights",
            "mapping_matrix_shape",
        ]
        start = time.time()
        self.compiled_qp = compile_portfolio_qp(self.returns_dict, self.params)
        self.set_up_time = time.time() - start

    def solve_optimization_problem(
        self, solver_settings: dict = None, print_results: bool = True
    ):
        """Solve the compiled QP with the explicitly selected backend."""

        if self.params.backend == "osqp":
            solution = solve_compiled_qp_osqp(self.compiled_qp)
        elif self.params.backend == "cuopt":
            solution = solve_compiled_qp_cuopt(
                self.compiled_qp,
                solver_settings=solver_settings,
            )
        else:
            raise ValueError(f"Unsupported QP backend: {self.params.backend}")

        stock_weights = self.compiled_qp.recover_stock_weights(solution.x)
        factor_weights = self.compiled_qp.recover_factor_weights(solution.x)
        cash = 0.0
        portfolio = Portfolio(
            name=f"{solution.solver}_qp_optimal",
            tickers=self.tickers,
            weights=stock_weights,
            cash=cash,
            time_range=self.regime_range,
        )
        result_row = self._build_result_row(solution, stock_weights, factor_weights)

        if print_results:
            self._print_results(result_row, portfolio)
        return result_row, portfolio

    def _build_result_row(self, solution, stock_weights, factor_weights) -> pd.Series:
        expected_return = float(self.compiled_qp.mean @ stock_weights)
        variance = float(stock_weights @ self.compiled_qp.covariance @ stock_weights)
        volatility = float(np.sqrt(max(variance, 0.0)))
        excess_return = expected_return - self.params.risk_free_rate
        sharpe = excess_return / volatility if volatility > 0 else np.nan
        c_scale = (
            self.compiled_qp.recover_scale(solution.x)
            if self.params.objective == "max_sharpe"
            else np.nan
        )

        previous = self.params.previous_weights
        turnover = (
            float(np.sum(np.abs(stock_weights - np.asarray(previous, dtype=float))))
            if previous is not None
            else np.nan
        )

        row = pd.Series(
            [
                self.regime_name,
                solution.solver,
                self.params.objective,
                solution.status,
                solution.raw_status,
                solution.solve_time,
                solution.total_time,
                solution.objective_value,
                c_scale,
                expected_return,
                excess_return,
                variance,
                volatility,
                sharpe,
                self.params.lambda_l1,
                self.params.lambda_l2,
                self.params.short_budget,
                turnover,
                solution.max_constraint_violation,
                factor_weights,
                stock_weights,
                stock_weights,
                tuple(self.compiled_qp.stock_mapping.shape),
            ],
            index=self._result_columns,
        )
        return row

    def _print_results(self, result_row: pd.Series, portfolio: Portfolio):
        print("\n" + "=" * 60)
        print("UNIFIED QP PORTFOLIO OPTIMIZATION RESULTS")
        print("=" * 60)
        print(f"Solver:          {result_row['solver']}")
        print(f"Objective:       {result_row['objective']}")
        print(f"Status:          {result_row['status']}")
        print(f"Expected Return: {result_row['expected_return']:.6f}")
        print(f"Variance:        {result_row['variance']:.6f}")
        print(f"Sharpe:          {result_row['sharpe']:.6f}")
        print(f"Objective Value: {result_row['objective_value']:.6f}")
        print(f"Solve Time:      {result_row['solve_time']}")
        print(f"Setup Time:      {self.set_up_time:.6f}")
        portfolio.print_clean(verbose=True, min_percentage=1)
        print("=" * 60 + "\n")
