"""
Hysteresis models and parameter estimation for flow control systems.

This module provides tools for:
- Multi-level hysteresis modeling with separate ascending/descending thresholds
- Parameter estimation from empirical data
- Data cleaning and outlier removal for hysteresis fitting

The hysteresis model is commonly used to represent control systems with different
response thresholds depending on whether the input is increasing or decreasing,
e.g., water flow rates that respond differently to rising vs falling magnet power.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def multi_level_hysteresis(
    x: np.ndarray,
    thresholds: List[Tuple[float, float]],
    low_values: List[float],
    high_values: List[float],
) -> np.ndarray:
    """
    Apply multi-level hysteresis model to input signal.

    Models n-level hysteresis where each level has separate ascending and
    descending thresholds. The output depends on both current input value
    and the direction of change (history-dependent behavior).

    Parameters
    ----------
    x : np.ndarray
        Input signal (e.g., power values in MW)
    thresholds : List[Tuple[float, float]]
        List of (ascending_threshold, descending_threshold) pairs for each level.
        Must be ordered from lowest to highest level.
        Each descending_threshold must be < corresponding ascending_threshold.
    low_values : List[float]
        Output values for the low state at each threshold level.
        Must have same length as thresholds.
    high_values : List[float]
        Output values for the high state at each threshold level.
        Must have same length as thresholds.

    Returns
    -------
    np.ndarray
        Output signal with hysteresis effect applied

    Raises
    ------
    ValueError
        If parameter lengths don't match or thresholds are not properly ordered

    Examples
    --------
    >>> import numpy as np
    >>> # Three-level hysteresis
    >>> x = np.array([0, 5, 10, 15, 10, 5, 0])
    >>> thresholds = [(3, 2), (8, 6), (12, 10)]
    >>> low_values = [100, 200, 300, 400]
    >>> high_values = [100, 250, 350, 450]
    >>> y = multi_level_hysteresis(x, thresholds, low_values, high_values)
    
    Notes
    -----
    The model tracks an "active level" that changes based on threshold crossings:
    - When x increases above ascending_threshold[i], level increases to i
    - When x decreases below descending_threshold[i], level may decrease
    - Different thresholds for ascending/descending create hysteresis loops
    """
    if len(thresholds) != len(low_values) or len(thresholds) != len(high_values):
        raise ValueError(
            f"thresholds, low_values, and high_values must have same length. "
            f"Got {len(thresholds)}, {len(low_values)}, {len(high_values)}"
        )

    # Handle empty input
    if len(x) == 0:
        return np.array([])

    # Extract and validate thresholds
    ascending_thresholds = [t[0] for t in thresholds]
    descending_thresholds = [t[1] for t in thresholds]

    if len(ascending_thresholds) > 1:
        if not all(
            a < b for a, b in zip(ascending_thresholds[:-1], ascending_thresholds[1:])
        ):
            raise ValueError("ascending thresholds must be in ascending order")

    if len(descending_thresholds) > 1:
        if not all(
            a < b for a, b in zip(descending_thresholds[:-1], descending_thresholds[1:])
        ):
            raise ValueError("descending thresholds must be in ascending order")

    if not all(d < a for d, a in zip(descending_thresholds, ascending_thresholds)):
        raise ValueError(
            "Each descending threshold must be less than its corresponding ascending threshold"
        )

    output = np.zeros_like(x, dtype=float)
    active_level = -1  # Start with no active level

    # Determine initial level based on first value
    for i in range(len(thresholds)):
        if x[0] > ascending_thresholds[i]:
            active_level = i

    # Set initial output
    if active_level >= 0:
        output[0] = high_values[active_level]
    else:
        output[0] = low_values[0] if low_values else 0.0

    # Process remaining points
    for i in range(1, len(x)):
        current_value = x[i]
        previous_level = active_level

        if active_level >= 0:
            # Already in a high state - check for level changes
            
            # Check for upward transitions to higher levels
            for j in range(active_level + 1, len(thresholds)):
                if current_value > ascending_thresholds[j]:
                    active_level = j

            # Check for downward transition
            if current_value < descending_thresholds[active_level]:
                # Find highest level that's still active
                new_level = -1
                for j in range(active_level):
                    if current_value > descending_thresholds[j]:
                        new_level = j
                active_level = new_level
        else:
            # In lowest state - check for upward transitions
            for j in range(len(thresholds)):
                if current_value > ascending_thresholds[j]:
                    active_level = j

        # Set output based on current level
        if active_level != previous_level:
            # Level changed
            if active_level >= 0:
                output[i] = high_values[active_level]
            else:
                output[i] = low_values[0] if low_values else 0.0
        else:
            # No level change - maintain previous output
            output[i] = output[i - 1]

    return output


def estimate_hysteresis_parameters(
    df: pd.DataFrame,
    x_col: str = "x",
    y_col: str = "y",
    n_levels: Optional[int] = None,
    verbose: bool = False,
) -> Dict:
    """
    Estimate hysteresis parameters from empirical time-series data.

    Analyzes transitions in x-y data to identify:
    - Distinct output levels
    - Ascending thresholds (x values where y increases during x increase)
    - Descending thresholds (x values where y decreases during x decrease)

    Parameters
    ----------
    df : pd.DataFrame
        Data with x and y columns (e.g., power and flow rate over time)
    x_col : str, default="x"
        Name of the input column (e.g., "Pmagnet")
    y_col : str, default="y"
        Name of the output column (e.g., "debitbrut")
    n_levels : int, optional
        If provided, cluster y values into this many discrete levels using KMeans.
        If None, auto-detect distinct levels (best for already-discrete data).
    verbose : bool, default=False
        If True, print diagnostic information

    Returns
    -------
    dict
        Dictionary with keys:
        - 'thresholds': List[Tuple[float, float]] - (ascending, descending) pairs
        - 'low_values': List[float] - Output values for low state at each level
        - 'high_values': List[float] - Output values for high state at each level
        - 'diagnostics': dict - Transition counts and level information

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> # Create synthetic data with hysteresis
    >>> t = np.linspace(0, 4*np.pi, 100)
    >>> power = 10 + 8 * np.sin(t)
    >>> # ... (compute flow with hysteresis) ...
    >>> df = pd.DataFrame({'power': power, 'flow': flow})
    >>> result = estimate_hysteresis_parameters(df, x_col='power', y_col='flow', n_levels=3)
    >>> thresholds = result['thresholds']
    >>> low_values = result['low_values']
    >>> high_values = result['high_values']

    Notes
    -----
    - Requires sequential time-series data (not randomly sampled)
    - Works best with clear, discrete output levels
    - Use n_levels parameter if output has continuous variation
    - Consider cleaning outliers with remove_low_x_outliers() first
    """
    x = df[x_col].values
    y = df[y_col].values.copy()

    # Cluster y values if requested
    if n_levels is not None:
        try:
            from sklearn.cluster import KMeans

            km = KMeans(n_clusters=n_levels, random_state=42, n_init=10)
            y_clusters = km.fit_predict(y.reshape(-1, 1))
            y = km.cluster_centers_[y_clusters].flatten()
            if verbose:
                logger.info(f"Clustered y into {n_levels} levels")
        except ImportError:
            logger.warning(
                "sklearn not available. Using data-driven level detection instead."
            )

    # Compute derivatives to identify direction of change
    dx = np.diff(x, prepend=np.nan)
    dy = np.diff(y, prepend=np.nan)

    # Classify direction of x change
    is_increasing = dx[1:] > 1e-10
    is_decreasing = dx[1:] < -1e-10

    # Find transitions (significant y changes)
    transitions = np.where(np.abs(dy[1:]) > 1e-10)[0] + 1

    # Identify distinct output levels
    unique_y = sorted(set(np.round(y, decimals=8)))
    n_levels_found = len(unique_y)

    if verbose:
        logger.info(f"Found {n_levels_found} distinct output levels")
        logger.info(f"Found {len(transitions)} transitions")

    # Collect threshold observations
    ascending_observations: Dict[int, List[float]] = {}
    descending_observations: Dict[int, List[float]] = {}

    for t_idx in transitions:
        if t_idx == 0:
            continue

        y_from = np.round(y[t_idx - 1], decimals=8)
        y_to = np.round(y[t_idx], decimals=8)
        x_at_transition = float(x[t_idx])

        try:
            level_from = unique_y.index(y_from)
            level_to = unique_y.index(y_to)
        except ValueError:
            continue

        # Ascending: x increasing, y moving to higher level
        if level_to > level_from and is_increasing[t_idx - 1]:
            if level_to not in ascending_observations:
                ascending_observations[level_to] = []
            ascending_observations[level_to].append(x_at_transition)

        # Descending: x decreasing, y moving to lower level
        if level_to < level_from and is_decreasing[t_idx - 1]:
            if level_from not in descending_observations:
                descending_observations[level_from] = []
            descending_observations[level_from].append(x_at_transition)

    # Compute mean thresholds
    ascending_thresholds = []
    descending_thresholds = []

    for i in range(n_levels_found):
        asc_threshold = (
            float(np.mean(ascending_observations[i]))
            if i in ascending_observations
            else None
        )
        desc_threshold = (
            float(np.mean(descending_observations[i]))
            if i in descending_observations
            else None
        )

        ascending_thresholds.append(asc_threshold)
        descending_thresholds.append(desc_threshold)

        if verbose:
            asc_str = f"{asc_threshold:.4f}" if asc_threshold is not None else "N/A"
            desc_str = f"{desc_threshold:.4f}" if desc_threshold is not None else "N/A"
            logger.info(
                f"Level {i} (y={unique_y[i]:g}): "
                f"asc_threshold={asc_str}, desc_threshold={desc_str}"
            )

    # Build final threshold list, filtering incomplete levels
    thresholds = []
    valid_indices = []
    for i, (asc, desc) in enumerate(zip(ascending_thresholds, descending_thresholds)):
        if asc is not None and desc is not None:
            thresholds.append((asc, desc))
            valid_indices.append(i)

    # Extract output values for valid levels
    high_values = [float(unique_y[i]) for i in valid_indices]
    low_values = [float(unique_y[max(0, i - 1)]) for i in valid_indices]

    # Sort by low_values to ensure ascending order
    sorted_items = sorted(zip(low_values, high_values, thresholds))
    low_values = [item[0] for item in sorted_items]
    high_values = [item[1] for item in sorted_items]
    thresholds = [item[2] for item in sorted_items]

    diagnostics = {
        "n_transitions": len(transitions),
        "n_levels_found": n_levels_found,
        "n_valid_levels": len(valid_indices),
        "ascending_obs": {k: len(v) for k, v in ascending_observations.items()},
        "descending_obs": {k: len(v) for k, v in descending_observations.items()},
    }

    return {
        "thresholds": thresholds,
        "low_values": low_values,
        "high_values": high_values,
        "diagnostics": diagnostics,
    }


def remove_low_x_outliers(
    df: pd.DataFrame,
    x_col: str = "x",
    y_col: str = "y",
    x_percentile: float = 25,
    method: str = "iqr",
    threshold: float = 1.5,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Remove outliers specifically from the low-x region of data.

    Useful for cleaning hysteresis data where low-input (e.g., low power) regions
    may contain more noise or measurement artifacts.

    Parameters
    ----------
    df : pd.DataFrame
        Data with x and y columns
    x_col : str, default="x"
        Name of x column
    y_col : str, default="y"
        Name of y column
    x_percentile : float, default=25
        Define "low x" as values below this percentile (0-100)
    method : str, default="iqr"
        Outlier detection method:
        - 'iqr': Interquartile range on y in low-x region (robust)
        - 'zscore': Z-score on y in low-x region (statistical)
        - 'both_dims': IQR on both x and y in low-x region
    threshold : float, default=1.5
        For 'iqr'/'both_dims': IQR multiplier (typical: 1.5-2.0)
        For 'zscore': z-score cutoff (typical: 2-3)
    verbose : bool, default=False
        Print diagnostic information

    Returns
    -------
    pd.DataFrame
        DataFrame with outliers removed, index reset

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.read_csv("magnet_data.csv")
    >>> df_clean = remove_low_x_outliers(
    ...     df, x_col="Pmagnet", y_col="debitbrut",
    ...     x_percentile=25, method="iqr", threshold=1.5, verbose=True
    ... )
    """
    df_clean = df.copy()
    
    # Handle empty dataframe
    if len(df_clean) == 0:
        return df_clean
    
    x = df_clean[x_col].values
    y = df_clean[y_col].values

    # Define low-x region
    x_cutoff = np.percentile(x, x_percentile)
    low_x_mask = x <= x_cutoff
    n_low_x = np.sum(low_x_mask)

    if verbose:
        logger.info(
            f"Low-x region: x <= {x_cutoff:.4f} "
            f"({n_low_x} points, {100*n_low_x/len(df):.1f}%)"
        )

    # Start with all points as inliers
    inlier_mask = np.ones(len(df_clean), dtype=bool)

    if method == "iqr":
        if n_low_x > 0:
            y_low = y[low_x_mask]
            Q1 = np.percentile(y_low, 25)
            Q3 = np.percentile(y_low, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR

            low_x_outliers = low_x_mask & ((y < lower_bound) | (y > upper_bound))
            inlier_mask = ~low_x_outliers

            if verbose:
                n_removed = np.sum(low_x_outliers)
                logger.info(
                    f"  IQR on low-x y: [{lower_bound:.4f}, {upper_bound:.4f}]"
                )
                logger.info(f"  Removed {n_removed} outliers from low-x region")

    elif method == "zscore":
        if n_low_x > 0:
            y_low = y[low_x_mask]
            mean_y = np.mean(y_low)
            std_y = np.std(y_low)
            if std_y > 0:
                z_scores = np.abs((y[low_x_mask] - mean_y) / std_y)
                low_x_outliers_bool = z_scores > threshold

                low_x_outliers = np.zeros(len(df_clean), dtype=bool)
                low_x_indices = np.where(low_x_mask)[0]
                low_x_outliers[low_x_indices[low_x_outliers_bool]] = True

                inlier_mask = ~low_x_outliers

                if verbose:
                    n_removed = np.sum(low_x_outliers)
                    logger.info(f"  Z-score on low-x y (threshold={threshold})")
                    logger.info(f"  Removed {n_removed} outliers from low-x region")

    elif method == "both_dims":
        if n_low_x > 0:
            x_low = x[low_x_mask]
            y_low = y[low_x_mask]

            # IQR on x in low-x region
            Q1_x = np.percentile(x_low, 25)
            Q3_x = np.percentile(x_low, 75)
            IQR_x = Q3_x - Q1_x
            x_lower = Q1_x - threshold * IQR_x
            x_upper = Q3_x + threshold * IQR_x

            # IQR on y in low-x region
            Q1_y = np.percentile(y_low, 25)
            Q3_y = np.percentile(y_low, 75)
            IQR_y = Q3_y - Q1_y
            y_lower = Q1_y - threshold * IQR_y
            y_upper = Q3_y + threshold * IQR_y

            low_x_outliers = low_x_mask & (
                (x < x_lower) | (x > x_upper) | (y < y_lower) | (y > y_upper)
            )
            inlier_mask = ~low_x_outliers

            if verbose:
                n_removed = np.sum(low_x_outliers)
                logger.info("  IQR on low-x region (both dims):")
                logger.info(f"    x in [{x_lower:.4f}, {x_upper:.4f}]")
                logger.info(f"    y in [{y_lower:.4f}, {y_upper:.4f}]")
                logger.info(f"  Removed {n_removed} outliers from low-x region")

    else:
        raise ValueError(
            f"Unknown method: {method}. Choose from: 'iqr', 'zscore', 'both_dims'"
        )

    return df_clean[inlier_mask].reset_index(drop=True)


def remove_outliers(
    df: pd.DataFrame,
    x_col: str = "x",
    y_col: str = "y",
    method: str = "iqr",
    threshold: float = 1.5,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Remove outliers from x-y data using various methods.

    Parameters
    ----------
    df : pd.DataFrame
        Data with x and y columns
    x_col : str, default="x"
        Name of x column
    y_col : str, default="y"
        Name of y column
    method : str, default="iqr"
        Outlier detection method:
        - 'iqr': Interquartile range (robust, recommended)
        - 'zscore': Z-score method (statistical)
        - 'mad': Median Absolute Deviation (very robust)
    threshold : float, default=1.5
        For 'iqr'/'mad': multiplier on IQR/MAD (typical: 1.0-2.0)
        For 'zscore': z-score cutoff (typical: 2-3)
    verbose : bool, default=False
        Print outlier information

    Returns
    -------
    pd.DataFrame
        DataFrame with outliers removed, index reset

    Examples
    --------
    >>> df_clean = remove_outliers(df, x_col="power", y_col="flow",
    ...                            method="iqr", threshold=1.5, verbose=True)
    """
    df_clean = df.copy()
    x = df_clean[x_col].values
    y = df_clean[y_col].values

    # Handle empty dataframe
    if len(df_clean) == 0:
        return df_clean

    if method == "iqr":
        # IQR on x
        Q1_x = np.percentile(x, 25)
        Q3_x = np.percentile(x, 75)
        IQR_x = Q3_x - Q1_x
        lower_x = Q1_x - threshold * IQR_x
        upper_x = Q3_x + threshold * IQR_x

        # IQR on y
        Q1_y = np.percentile(y, 25)
        Q3_y = np.percentile(y, 75)
        IQR_y = Q3_y - Q1_y
        lower_y = Q1_y - threshold * IQR_y
        upper_y = Q3_y + threshold * IQR_y

        mask = (
            (x >= lower_x) & (x <= upper_x) & (y >= lower_y) & (y <= upper_y)
        )

        if verbose:
            n_removed = len(df_clean) - np.sum(mask)
            logger.info(f"IQR method (threshold={threshold}): Removed {n_removed} outliers")
            logger.info(f"  x range: [{lower_x:.4f}, {upper_x:.4f}]")
            logger.info(f"  y range: [{lower_y:.4f}, {upper_y:.4f}]")

    elif method == "zscore":
        z_x = np.abs((x - np.mean(x)) / np.std(x))
        z_y = np.abs((y - np.mean(y)) / np.std(y))
        mask = (z_x < threshold) & (z_y < threshold)

        if verbose:
            n_removed = len(df_clean) - np.sum(mask)
            logger.info(
                f"Z-score method (threshold={threshold}): Removed {n_removed} outliers"
            )

    elif method == "mad":
        # Median Absolute Deviation (robust)
        median_x = np.median(x)
        mad_x = np.median(np.abs(x - median_x))

        median_y = np.median(y)
        mad_y = np.median(np.abs(y - median_y))

        lower_x = median_x - threshold * mad_x
        upper_x = median_x + threshold * mad_x
        lower_y = median_y - threshold * mad_y
        upper_y = median_y + threshold * mad_y

        mask = (
            (x >= lower_x) & (x <= upper_x) & (y >= lower_y) & (y <= upper_y)
        )

        if verbose:
            n_removed = len(df_clean) - np.sum(mask)
            logger.info(f"MAD method (threshold={threshold}): Removed {n_removed} outliers")
            logger.info(f"  x range: [{lower_x:.4f}, {upper_x:.4f}]")
            logger.info(f"  y range: [{lower_y:.4f}, {upper_y:.4f}]")

    else:
        raise ValueError(
            f"Unknown method: {method}. Choose from: 'iqr', 'zscore', 'mad'"
        )

    return df_clean[mask].reset_index(drop=True)


def compute_hysteresis_fit_metrics(
    x: np.ndarray,
    y: np.ndarray,
    thresholds: List[Tuple[float, float]],
    low_values: List[float],
    high_values: List[float],
) -> Dict[str, float]:
    """
    Compute fit quality metrics for hysteresis model.

    Provides statistical measures of how well the hysteresis model fits
    the observed data. Use these metrics to decide if parameter re-estimation
    or model refinement is needed.

    Parameters
    ----------
    x : np.ndarray
        Input values (e.g., power in MW)
    y : np.ndarray
        Measured output values (e.g., flow rate in m³/h)
    thresholds : List[Tuple[float, float]]
        Hysteresis threshold parameters (ascending, descending)
    low_values : List[float]
        Low state values for each level
    high_values : List[float]
        High state values for each level

    Returns
    -------
    Dict[str, float]
        Dictionary containing:
        - 'rmse': Root Mean Square Error
        - 'mae': Mean Absolute Error
        - 'max_error': Maximum absolute error
        - 'r_squared': Coefficient of determination (R²)
        - 'mape': Mean Absolute Percentage Error (%) [if y has no zeros]
        - 'n_points': Number of data points
        - 'match_rate': Percentage of points within 10% of true value

    Examples
    --------
    >>> import numpy as np
    >>> # Create test data with hysteresis
    >>> x = np.concatenate([np.linspace(0, 10, 50), np.linspace(10, 0, 50)])
    >>> thresholds = [(3, 2), (7, 5)]
    >>> low_values = [100, 200]
    >>> high_values = [200, 300]
    >>> y = multi_level_hysteresis(x, thresholds, low_values, high_values)
    >>> y += np.random.normal(0, 2, len(y))  # Add noise
    >>> 
    >>> metrics = compute_hysteresis_fit_metrics(x, y, thresholds, 
    ...                                          low_values, high_values)
    >>> print(f"RMSE: {metrics['rmse']:.2f}")
    >>> print(f"R²: {metrics['r_squared']:.4f}")
    >>> 
    >>> # Decision logic
    >>> if metrics['r_squared'] < 0.85:
    ...     print("Poor fit - consider refit with different n_levels")
    >>> elif metrics['rmse'] > 10:
    ...     print("High error - check for outliers or data quality")
    >>> else:
    ...     print("Good fit quality")

    Notes
    -----
    **Interpretation Guidelines:**
    
    **R² (Coefficient of Determination):**
    - > 0.95: Excellent fit, very clear hysteresis
    - 0.85-0.95: Good fit, hysteresis well-captured
    - 0.70-0.85: Fair fit, consider refinement
    - < 0.70: Poor fit, refit recommended
    
    **RMSE (Root Mean Square Error):**
    - Compare to typical output range
    - < 5% of range: Excellent
    - < 10% of range: Good
    - > 15% of range: Poor, refit needed
    
    **Match Rate:**
    - Percentage of points within 10% of true value
    - > 90%: Excellent
    - > 80%: Good
    - < 70%: Consider refit
    
    **When to Refit:**
    - R² < 0.85
    - RMSE > 10% of output range
    - Match rate < 80%
    - Systematic patterns in residuals
    - New operating conditions not covered by original fit
    - Control system parameters changed
    
    **Troubleshooting Poor Fits:**
    1. Try different n_levels in estimate_hysteresis_parameters()
    2. Remove outliers before fitting (use remove_outliers())
    3. Check if data truly exhibits hysteresis or is just noisy
    4. Ensure sufficient data coverage of all operating regions
    5. Verify time-series ordering (hysteresis needs sequential data)
    """
    # Compute fitted values
    y_fit = multi_level_hysteresis(x, thresholds, low_values, high_values)

    # Basic metrics
    residuals = y - y_fit
    n_points = len(y)

    rmse = float(np.sqrt(np.mean(residuals**2)))
    mae = float(np.mean(np.abs(residuals)))
    max_error = float(np.max(np.abs(residuals)))

    # R²
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

    # MAPE (only if no zeros)
    if np.all(y != 0):
        mape = float(np.mean(np.abs(residuals / y)) * 100)
    else:
        mape = None

    # Match rate (percentage within 10% of true value)
    tolerance = 0.10
    if np.all(y != 0):
        within_tolerance = np.abs(residuals / y) <= tolerance
        match_rate = float(np.mean(within_tolerance) * 100)
    else:
        # Use absolute tolerance if y has zeros
        abs_tolerance = np.std(y) * 0.1
        within_tolerance = np.abs(residuals) <= abs_tolerance
        match_rate = float(np.mean(within_tolerance) * 100)

    metrics = {
        "rmse": rmse,
        "mae": mae,
        "max_error": max_error,
        "r_squared": r_squared,
        "n_points": int(n_points),
        "match_rate": match_rate,
    }

    if mape is not None:
        metrics["mape"] = mape

    return metrics


def plot_hysteresis_model(
    x: np.ndarray,
    thresholds: List[Tuple[float, float]],
    low_values: List[float],
    high_values: List[float],
    xlabel: str = "Input",
    ylabel: str = "Output",
    title: Optional[str] = None,
    show: bool = True,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 6),
) -> None:
    """
    Plot hysteresis model showing input signal and output with threshold levels.

    Parameters
    ----------
    x : np.ndarray
        Input signal
    thresholds : List[Tuple[float, float]]
        List of (ascending_threshold, descending_threshold) pairs
    low_values : List[float]
        Output values for low state at each level
    high_values : List[float]
        Output values for high state at each level
    xlabel : str, default="Input"
        Label for x-axis
    ylabel : str, default="Output"
        Label for y-axis
    title : str, optional
        Plot title
    show : bool, default=True
        Display plot interactively
    save_path : str, optional
        Path to save plot (if provided)
    figsize : Tuple[float, float], default=(10, 6)
        Figure size (width, height) in inches

    Examples
    --------
    >>> import numpy as np
    >>> x = np.concatenate([np.linspace(0, 10, 50), np.linspace(10, 0, 50)])
    >>> thresholds = [(3, 2), (7, 5)]
    >>> low_values = [100, 200]
    >>> high_values = [200, 300]
    >>> plot_hysteresis_model(x, thresholds, low_values, high_values,
    ...                       xlabel="Power (MW)", ylabel="Flow (m³/h)")
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available. Cannot create plot.")
        return

    # Compute output with hysteresis
    y = multi_level_hysteresis(x, thresholds, low_values, high_values)

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # Plot 1: Input signal
    ax1.plot(range(len(x)), x, 'b-', linewidth=1.5, label='Input signal')
    ax1.set_ylabel(xlabel, fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: Output with hysteresis
    ax2.plot(range(len(y)), y, 'r-', linewidth=1.5, label='Output (with hysteresis)')
    
    # Draw threshold levels as horizontal lines
    for i, (asc_thresh, desc_thresh) in enumerate(thresholds):
        # Mark high value for this level
        ax2.axhline(y=high_values[i], color='g', linestyle='--', alpha=0.5,
                   linewidth=1, label=f'Level {i} high' if i == 0 else '')
        # Mark low value for this level
        if i == 0:
            ax2.axhline(y=low_values[i], color='orange', linestyle='--', alpha=0.5,
                       linewidth=1, label='Level low')
    
    ax2.set_xlabel('Sample Index', fontsize=11)
    ax2.set_ylabel(ylabel, fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    if title:
        fig.suptitle(title, fontsize=13, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_hysteresis_fit(
    df: pd.DataFrame,
    thresholds: List[Tuple[float, float]],
    low_values: List[float],
    high_values: List[float],
    x_col: str = "x",
    y_col: str = "y",
    xlabel: str = "Input",
    ylabel: str = "Output",
    title: Optional[str] = None,
    show: bool = True,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 8),
) -> None:
    """
    Plot raw data and fitted hysteresis model for comparison.

    Parameters
    ----------
    df : pd.DataFrame
        Raw data with x and y columns
    thresholds : List[Tuple[float, float]]
        Fitted threshold parameters
    low_values : List[float]
        Fitted low values
    high_values : List[float]
        Fitted high values
    x_col : str, default="x"
        Name of input column in df
    y_col : str, default="y"
        Name of output column in df
    xlabel : str, default="Input"
        Label for x-axis
    ylabel : str, default="Output"
        Label for y-axis
    title : str, optional
        Plot title
    show : bool, default=True
        Display plot interactively
    save_path : str, optional
        Path to save plot
    figsize : Tuple[float, float], default=(12, 8)
        Figure size (width, height) in inches

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'power': power_data, 'flow': flow_data})
    >>> result = estimate_hysteresis_parameters(df, x_col='power', y_col='flow')
    >>> plot_hysteresis_fit(df, result['thresholds'], result['low_values'],
    ...                     result['high_values'], x_col='power', y_col='flow',
    ...                     xlabel='Power (MW)', ylabel='Flow (m³/h)')
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available. Cannot create plot.")
        return

    x = df[x_col].values
    y = df[y_col].values

    # Compute fitted model
    y_fit = multi_level_hysteresis(x, thresholds, low_values, high_values)

    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    ax1 = fig.add_subplot(gs[0, :])  # Top: time series
    ax2 = fig.add_subplot(gs[1, 0])  # Bottom left: x-y plot
    ax3 = fig.add_subplot(gs[1, 1])  # Bottom right: residuals

    # Plot 1: Time series comparison
    ax1.plot(y, 'b.', markersize=3, alpha=0.5, label='Raw data')
    ax1.plot(y_fit, 'r-', linewidth=2, label='Fitted model')
    ax1.set_xlabel('Sample Index', fontsize=11)
    ax1.set_ylabel(ylabel, fontsize=11)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Time Series: Raw Data vs Fitted Model', fontsize=12)

    # Plot 2: X-Y scatter with fitted curve
    ax2.plot(x, y, 'b.', markersize=3, alpha=0.3, label='Raw data')
    ax2.plot(x, y_fit, 'r-', linewidth=2, label='Fitted model')
    
    # Mark thresholds
    y_min, y_max = ax2.get_ylim()
    for i, (asc_thresh, desc_thresh) in enumerate(thresholds):
        ax2.axvline(x=asc_thresh, color='g', linestyle='--', alpha=0.6,
                   linewidth=1.5, label=f'Ascending thresholds' if i == 0 else '')
        ax2.axvline(x=desc_thresh, color='orange', linestyle='--', alpha=0.6,
                   linewidth=1.5, label=f'Descending thresholds' if i == 0 else '')
    
    ax2.set_xlabel(xlabel, fontsize=11)
    ax2.set_ylabel(ylabel, fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Input-Output Relationship', fontsize=12)

    # Plot 3: Residuals
    residuals = y - y_fit
    ax3.plot(x, residuals, 'k.', markersize=3, alpha=0.5)
    ax3.axhline(y=0, color='r', linestyle='-', linewidth=1)
    ax3.set_xlabel(xlabel, fontsize=11)
    ax3.set_ylabel('Residual', fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.set_title(f'Residuals (RMSE: {np.sqrt(np.mean(residuals**2)):.3f})', fontsize=12)

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)
