"""
Time Series Splitting Module - TimeSeriesSplitter

This module provides the TimeSeriesSplitter class for proper train-test splitting
in time-series data, preventing data leakage by ensuring no future data is used
in training sets.

Author: Data Engineering Team
Date: 2024
"""

import logging
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class TimeSeriesSplitterError(Exception):
    """Custom exception for TimeSeriesSplitter errors."""
    pass


class TimeSeriesSplitter:
    """
    Time-series aware cross-validation splitter.
    
    Ensures proper temporal ordering in train-test splits to prevent data leakage.
    Unlike standard cross-validation, this splitter guarantees that training data
    always comes before test data temporally.
    
    Attributes:
        n_splits: Number of cross-validation folds
        test_size: Size of test set as fraction of total data
    """
    
    def __init__(self, n_splits: int = 3, test_size: Optional[float] = None):
        """
        Initialize TimeSeriesSplitter.
        
        Args:
            n_splits: Number of cross-validation folds (default: 3)
            test_size: Size of test set as fraction of total data.
                      If None, calculated as 1/(n_splits+1)
        
        Raises:
            TimeSeriesSplitterError: If parameters are invalid
        """
        if n_splits < 1:
            raise TimeSeriesSplitterError("n_splits must be >= 1")
        
        self.n_splits = n_splits
        
        if test_size is None:
            self.test_size = 1.0 / (n_splits + 1)
        else:
            self.test_size = test_size
        
        if self.test_size <= 0 or self.test_size >= 1:
            raise TimeSeriesSplitterError("test_size must be between 0 and 1")
        
        logger.info(
            f"TimeSeriesSplitter initialized with n_splits={n_splits}, "
            f"test_size={self.test_size:.2%}"
        )
    
    def split(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate time-series cross-validation splits.
        
        Ensures temporal ordering: training set always contains data before test set.
        No future data leaks into training set.
        
        Args:
            X: Input features DataFrame with temporal ordering
            y: Target variable (optional, not used but kept for sklearn compatibility)
        
        Returns:
            List of (train_indices, test_indices) tuples
        
        Raises:
            TimeSeriesSplitterError: If X is invalid
        """
        if X is None or len(X) == 0:
            raise TimeSeriesSplitterError("X cannot be None or empty")
        
        n_samples = len(X)
        fold_size = int(n_samples / (self.n_splits + 1))
        
        if fold_size < 1:
            raise TimeSeriesSplitterError(
                f"Not enough samples ({n_samples}) for {self.n_splits} splits"
            )
        
        splits = []
        
        for i in range(self.n_splits):
            # Training set: from start to (i+1)*fold_size
            train_end = fold_size * (i + 1)
            
            # Test set: from train_end to train_end + fold_size
            test_end = train_end + fold_size
            
            train_idx = np.arange(0, train_end)
            test_idx = np.arange(train_end, min(test_end, n_samples))
            
            splits.append((train_idx, test_idx))
            
            logger.debug(
                f"Split {i+1}/{self.n_splits}: "
                f"train=[0:{train_end}], test=[{train_end}:{test_end}]"
            )
        
        logger.info(f"Generated {len(splits)} time-series cross-validation splits")
        return splits
    
    def get_train_test_split(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        test_size: Optional[float] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.Series], Optional[pd.Series]]:
        """
        Get a single train-test split respecting temporal ordering.
        
        Args:
            X: Input features DataFrame
            y: Target variable (optional)
            test_size: Override test size for this split (default: use instance test_size)
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
            y_train and y_test are None if y is None
        
        Raises:
            TimeSeriesSplitterError: If inputs are invalid
        """
        if X is None or len(X) == 0:
            raise TimeSeriesSplitterError("X cannot be None or empty")
        
        test_size_to_use = test_size if test_size is not None else self.test_size
        
        if test_size_to_use <= 0 or test_size_to_use >= 1:
            raise TimeSeriesSplitterError("test_size must be between 0 and 1")
        
        n_samples = len(X)
        split_point = int(n_samples * (1 - test_size_to_use))
        
        X_train = X.iloc[:split_point]
        X_test = X.iloc[split_point:]
        
        y_train = None
        y_test = None
        
        if y is not None:
            y_train = y.iloc[:split_point]
            y_test = y.iloc[split_point:]
        
        logger.info(
            f"Train-test split: train={len(X_train)}, test={len(X_test)}, "
            f"test_size={len(X_test)/n_samples:.2%}"
        )
        
        return X_train, X_test, y_train, y_test
    
    def validate_no_leakage(
        self,
        X: pd.DataFrame,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        timestamp_col: str = 'timestamp'
    ) -> bool:
        """
        Validate that no future data leaks into training set.
        
        Checks that the maximum timestamp in training set is less than
        the minimum timestamp in test set.
        
        Args:
            X: Input DataFrame with timestamp column
            train_idx: Training set indices
            test_idx: Test set indices
            timestamp_col: Name of timestamp column (default: 'timestamp')
        
        Returns:
            True if no leakage detected, False otherwise
        
        Raises:
            TimeSeriesSplitterError: If timestamp column not found
        """
        if timestamp_col not in X.columns:
            raise TimeSeriesSplitterError(
                f"Timestamp column '{timestamp_col}' not found in X"
            )
        
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        
        max_train_time = X_train[timestamp_col].max()
        min_test_time = X_test[timestamp_col].min()
        
        no_leakage = max_train_time < min_test_time
        
        if no_leakage:
            logger.debug(
                f"No data leakage detected: "
                f"max_train_time={max_train_time} < min_test_time={min_test_time}"
            )
        else:
            logger.warning(
                f"Data leakage detected: "
                f"max_train_time={max_train_time} >= min_test_time={min_test_time}"
            )
        
        return no_leakage
    
    def get_n_splits(self) -> int:
        """
        Get number of splits.
        
        Returns:
            Number of cross-validation folds
        """
        return self.n_splits
