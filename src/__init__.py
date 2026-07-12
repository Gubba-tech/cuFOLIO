# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.  # noqa
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""cufolio: GPU-accelerated portfolio optimization with cuOpt."""

version = "1.0.0"

from .base_parameters import BaseParameters
from .cvar_data import CvarData
from .cvar_parameters import CvarParameters
from .exceptions import GPUBackendUnavailable, QPCompilationError, QPSolveError
from .mean_variance_parameters import MeanVarianceParameters
from .qp_factor_workflows import (
    FactorModelQPData,
    build_external_factor_qp_data,
    build_pca_factor_qp_data,
)
from .qp_optimizer import QuadraticPortfolioOptimizer
from .qp_paper_replay import (
    PaperReplayWindow,
    ReplaySolveResult,
    compute_replay_diagnostics,
    load_replay_window,
    run_replay_directory,
    save_replay_window,
    solve_replay_window,
)
from .qp_parameters import QPParameters
from .settings import (
    ApiSettings,
    KDESettings,
    ReturnsComputeSettings,
    ScenarioGenerationSettings,
)

__all__ = [
    "BaseParameters",
    "CvarData",
    "CvarParameters",
    "MeanVarianceParameters",
    "QPParameters",
    "QuadraticPortfolioOptimizer",
    "PaperReplayWindow",
    "ReplaySolveResult",
    "load_replay_window",
    "save_replay_window",
    "solve_replay_window",
    "compute_replay_diagnostics",
    "run_replay_directory",
    "GPUBackendUnavailable",
    "QPCompilationError",
    "QPSolveError",
    "FactorModelQPData",
    "build_external_factor_qp_data",
    "build_pca_factor_qp_data",
    "ApiSettings",
    "KDESettings",
    "ReturnsComputeSettings",
    "ScenarioGenerationSettings",
    "version",
]
