import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

# TODO: Write a simple Winsorizer transformer which takes a lower and upper quantile and cuts the
# data accordingly
class Winsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, lower_quantile=0.05, upper_quantile=0.95):
        """
        Initialize the Winsorizer with the given quantiles.
        
        Parameters:
        - lower_quantile: float, default=0.05
          The lower quantile to use for clipping.
        - upper_quantile: float, default=0.95
          The upper quantile to use for clipping.
        """
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X, y=None):
        """
        Compute the lower and upper quantiles for the input data.
        
        Parameters:
        - X: array-like, shape (n_samples, n_features)
          The data to compute the quantiles.
        - y: Ignored
        
        Returns:
        - self: object
        """
        X = np.asarray(X)
        self.lower_quantile_ = np.percentile(X, self.lower_quantile * 100, axis=0)
        self.upper_quantile_ = np.percentile(X, self.upper_quantile * 100, axis=0)
        return self

    def transform(self, X):
        """
        Clip the data to the computed quantiles.
        
        Parameters:
        - X: array-like, shape (n_samples, n_features)
          The data to transform.
        
        Returns:
        - X_clipped: array-like, shape (n_samples, n_features)
          The clipped data.
        """
        X = np.asarray(X)
        X_clipped = np.clip(X, self.lower_quantile_, self.upper_quantile_)
        return X_clipped
