# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small factor-data adapters for the unified QP compiler.

This module intentionally stops at external factor returns and a deterministic
PCA construction. RP-PCA, IPCA, and AP-Trees estimators remain upstream model
responsibilities and can provide their outputs through the external adapter.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _as_2d(value, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 2 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite two-dimensional array.")
    return array


def _sample_covariance(returns: np.ndarray, name: str) -> np.ndarray:
    if returns.shape[0] < 2:
        raise ValueError(f"{name} requires at least two observations.")
    covariance = np.asarray(np.cov(returns, rowvar=False), dtype=float)
    if returns.shape[1] == 1:
        covariance = covariance.reshape(1, 1)
    return covariance


@dataclass
class FactorModelQPData:
    """Factor returns, factor covariance, and stock exposure mapping."""

    factor_mean: np.ndarray
    factor_covariance: np.ndarray
    stock_mapping: np.ndarray
    factor_returns: np.ndarray | None = None
    stock_returns: np.ndarray | None = None
    tickers: list[str] | None = None
    factor_names: list[str] | None = None
    model_name: str = "external"
    stock_covariance: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.factor_mean = np.asarray(self.factor_mean, dtype=float).reshape(-1)
        self.factor_covariance = np.asarray(self.factor_covariance, dtype=float)
        self.stock_mapping = _as_2d(self.stock_mapping, "stock_mapping")
        if (
            self.factor_mean.size == 0
            or not np.all(np.isfinite(self.factor_mean))
            or self.factor_covariance.shape
            != (self.factor_mean.size, self.factor_mean.size)
            or not np.all(np.isfinite(self.factor_covariance))
            or self.stock_mapping.shape[1] != self.factor_mean.size
        ):
            raise ValueError(
                "factor_mean, factor_covariance, and stock_mapping dimensions "
                "are inconsistent."
            )
        if self.factor_returns is not None:
            self.factor_returns = _as_2d(self.factor_returns, "factor_returns")
            if self.factor_returns.shape[1] != self.factor_mean.size:
                raise ValueError("factor_returns columns must match factor_mean.")
        if self.stock_returns is not None:
            self.stock_returns = _as_2d(self.stock_returns, "stock_returns")
            if self.stock_returns.shape[1] != self.stock_mapping.shape[0]:
                raise ValueError("stock_returns columns must match stock_mapping rows.")
        if self.stock_covariance is not None:
            self.stock_covariance = np.asarray(self.stock_covariance, dtype=float)
            n_assets = self.stock_mapping.shape[0]
            if (
                self.stock_covariance.shape != (n_assets, n_assets)
                or not np.all(np.isfinite(self.stock_covariance))
            ):
                raise ValueError("stock_covariance must have shape (n_assets, n_assets).")
        if self.tickers is not None and len(self.tickers) != self.stock_mapping.shape[0]:
            raise ValueError("tickers length must match stock_mapping rows.")
        if self.factor_names is not None and len(self.factor_names) != self.factor_mean.size:
            raise ValueError("factor_names length must match factor_mean.")

    def to_returns_dict(self) -> dict:
        """Return explicit factor-space inputs for ``compile_portfolio_qp``."""
        returns_dict = {
            "mean": self.factor_mean.copy(),
            "covariance": self.factor_covariance.copy(),
            "factor_mean": self.factor_mean.copy(),
            "factor_covariance": self.factor_covariance.copy(),
            "stock_mapping": self.stock_mapping.copy(),
            "factor_returns": self.factor_returns,
            "stock_returns": self.stock_returns,
            "tickers": self.tickers,
            "factor_names": self.factor_names,
            "model_name": self.model_name,
        }
        if self.stock_covariance is not None:
            returns_dict["stock_covariance"] = self.stock_covariance.copy()
        return returns_dict


def build_external_factor_qp_data(
    factor_returns,
    stock_mapping,
    stock_returns=None,
    tickers: list[str] | None = None,
    factor_names: list[str] | None = None,
    model_name: str = "external",
) -> FactorModelQPData:
    """Build factor QP data from externally estimated factors and mapping ``V``."""
    factors = _as_2d(factor_returns, "factor_returns")
    mapping = _as_2d(stock_mapping, "stock_mapping")
    if factors.shape[1] != mapping.shape[1]:
        raise ValueError("factor_returns columns must match stock_mapping factors.")
    stock = None if stock_returns is None else _as_2d(stock_returns, "stock_returns")
    stock_covariance = None
    if stock is not None:
        if stock.shape[0] != factors.shape[0] or stock.shape[1] != mapping.shape[0]:
            raise ValueError("stock_returns must align with factor_returns and mapping.")
        stock_covariance = _sample_covariance(stock, "stock_returns")
    return FactorModelQPData(
        factor_mean=factors.mean(axis=0),
        factor_covariance=_sample_covariance(factors, "factor_returns"),
        stock_mapping=mapping,
        factor_returns=factors,
        stock_returns=stock,
        tickers=tickers,
        factor_names=factor_names,
        model_name=model_name,
        stock_covariance=stock_covariance,
    )


def build_pca_factor_qp_data(
    stock_returns,
    n_components: int,
    center: bool = True,
    tickers: list[str] | None = None,
) -> FactorModelQPData:
    """Build deterministic PCA factors from a public or synthetic return matrix."""
    original = _as_2d(stock_returns, "stock_returns")
    n_observations, n_assets = original.shape
    if not isinstance(n_components, int) or not 1 <= n_components <= min(
        n_observations, n_assets
    ):
        raise ValueError("n_components must be between 1 and min(T, N).")
    if tickers is None and hasattr(stock_returns, "columns"):
        tickers = [str(value) for value in stock_returns.columns]

    centered = original - original.mean(axis=0) if center else original.copy()
    _, _, right_singular_vectors = np.linalg.svd(centered, full_matrices=False)
    mapping = right_singular_vectors[:n_components].T.copy()
    for factor_idx in range(n_components):
        pivot = int(np.argmax(np.abs(mapping[:, factor_idx])))
        if mapping[pivot, factor_idx] < 0:
            mapping[:, factor_idx] *= -1.0
    factor_returns = centered @ mapping
    return FactorModelQPData(
        factor_mean=factor_returns.mean(axis=0),
        factor_covariance=_sample_covariance(factor_returns, "factor_returns"),
        stock_mapping=mapping,
        factor_returns=factor_returns,
        stock_returns=original,
        tickers=tickers,
        factor_names=[f"pca_{idx}" for idx in range(n_components)],
        model_name="pca",
        stock_covariance=_sample_covariance(original, "stock_returns"),
    )
