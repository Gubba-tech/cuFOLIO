# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parameters for unified QP portfolio optimization."""

from typing import Literal, Optional, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator


class QPParameters(BaseModel):
    """User-facing parameters for PortOpt-style quadratic portfolios."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    objective: Literal[
        "min_variance",
        "mean_variance",
        "target_return",
        "max_sharpe",
    ] = "min_variance"
    risk_aversion: float = 1.0
    target_return: Optional[float] = None
    risk_free_rate: float = 0.0

    w_min: Optional[Union[np.ndarray, dict, float]] = 0.0
    w_max: Optional[Union[np.ndarray, dict, float]] = 1.0
    short_budget: Optional[float] = None

    lambda_l1: float = 0.0
    lambda_l2: float = 0.0
    lambda_tracking_error: float = 0.0

    turnover_budget: Optional[float] = None
    previous_weights: Optional[np.ndarray] = None
    benchmark_weights: Optional[np.ndarray] = None
    benchmark_l1_budget: Optional[float] = None

    factor_exposure_matrix: Optional[np.ndarray] = None
    factor_exposure_lower: Optional[np.ndarray] = None
    factor_exposure_upper: Optional[np.ndarray] = None

    V: Optional[np.ndarray] = None
    backend: Literal["cuopt", "osqp"] = "cuopt"

    @field_validator("risk_aversion")
    @classmethod
    def validate_risk_aversion(cls, value: float) -> float:
        if value < 0:
            raise ValueError("risk_aversion must be non-negative.")
        return value

    @field_validator("lambda_l1", "lambda_l2", "lambda_tracking_error")
    @classmethod
    def validate_penalty(cls, value: float) -> float:
        if value < 0:
            raise ValueError("regularization penalties must be non-negative.")
        return value

    @field_validator("short_budget")
    @classmethod
    def validate_short_budget(cls, value: Optional[float]) -> Optional[float]:
        if value is not None and value < 0:
            raise ValueError("short_budget must be non-negative.")
        return value

    @field_validator("turnover_budget", "benchmark_l1_budget")
    @classmethod
    def validate_budget(cls, value: Optional[float]) -> Optional[float]:
        if value is not None and value < 0:
            raise ValueError("budget values must be non-negative.")
        return value
