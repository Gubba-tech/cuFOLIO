def test_qp_public_api_exports():
    from cufolio import (
        FactorModelQPData,
        GPUBackendUnavailable,
        QPCompilationError,
        QPParameters,
        QPSolveError,
        QuadraticPortfolioOptimizer,
        build_external_factor_qp_data,
        build_pca_factor_qp_data,
    )

    assert QPParameters is not None
    assert QuadraticPortfolioOptimizer is not None
    assert FactorModelQPData is not None
    assert build_external_factor_qp_data is not None
    assert build_pca_factor_qp_data is not None
    assert GPUBackendUnavailable is not None
    assert QPCompilationError is not None
    assert QPSolveError is not None
