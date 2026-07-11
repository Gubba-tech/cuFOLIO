# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Project-specific exceptions for cuFOLIO extensions."""


class GPUBackendUnavailable(RuntimeError):
    """Raised when a requested GPU/cuOpt backend is not available."""


class QPCompilationError(ValueError):
    """Raised when portfolio inputs cannot be compiled into a valid QP."""


class QPSolveError(RuntimeError):
    """Raised when a compiled QP backend cannot produce a valid solution."""
