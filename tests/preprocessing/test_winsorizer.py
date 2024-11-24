import numpy as np
import pytest

from ps3.preprocessing import Winsorizer

# TODO: Test your implementation of a simple Winsorizer
@pytest.mark.parametrize(
    "lower_quantile, upper_quantile", [(0, 1), (0.05, 0.95), (0.5, 0.5)]
)
def test_winsorizer(lower_quantile, upper_quantile):

    X = np.random.normal(0, 1, 1000)

    # Initialize and fit Winsorizer
    winsorizer = Winsorizer(lower_quantile, upper_quantile).fit(X)
    
    # Compute expected bounds
    lower_expected = np.percentile(X, lower_quantile * 100)
    upper_expected = np.percentile(X, upper_quantile * 100)
    
    # Assert quantiles
    assert np.isclose(winsorizer.lower_quantile_, lower_expected)
    assert np.isclose(winsorizer.upper_quantile_, upper_expected)
    
    # Assert transformed data is clipped correctly
    X_transformed = winsorizer.transform(X)
    assert np.all((X_transformed >= lower_expected) & (X_transformed <= upper_expected))
