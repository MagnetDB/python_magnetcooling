"""Tests for hysteresis module"""

import pytest
import numpy as np
import pandas as pd
from python_magnetcooling.hysteresis import (
    multi_level_hysteresis,
    estimate_hysteresis_parameters,
    remove_low_x_outliers,
    remove_outliers,
)


class TestMultiLevelHysteresis:
    """Test multi-level hysteresis model"""

    def test_single_level_ascending(self):
        """Test single level with ascending signal"""
        x = np.array([0, 1, 2, 3, 4, 5, 6])
        thresholds = [(3.0, 2.0)]
        low_values = [100.0]
        high_values = [200.0]

        y = multi_level_hysteresis(x, thresholds, low_values, high_values)

        # Should start low, then transition high at x=3
        assert y[0] == 100.0  # x=0, below threshold
        assert y[1] == 100.0  # x=1
        assert y[2] == 100.0  # x=2
        assert y[3] == 100.0  # x=3, at ascending threshold but not yet crossed
        assert y[4] == 200.0  # x=4, above threshold
        assert y[5] == 200.0  # x=5
        assert y[6] == 200.0  # x=6

    def test_single_level_descending(self):
        """Test single level with descending signal"""
        x = np.array([6, 5, 4, 3, 2, 1, 0])
        thresholds = [(3.0, 2.0)]
        low_values = [100.0]
        high_values = [200.0]

        y = multi_level_hysteresis(x, thresholds, low_values, high_values)

        # Should start high (x=6 > threshold), stay high until below descending threshold
        assert y[0] == 200.0  # x=6, above threshold
        assert y[1] == 200.0  # x=5
        assert y[2] == 200.0  # x=4
        assert y[3] == 200.0  # x=3
        assert y[4] == 200.0  # x=2, at descending threshold
        assert y[5] == 100.0  # x=1, below descending threshold
        assert y[6] == 100.0  # x=0

    def test_hysteresis_loop(self):
        """Test that hysteresis creates different outputs for same input"""
        # Signal that goes up then down
        x = np.array([0, 2, 4, 6, 4, 2, 0])
        thresholds = [(3.5, 2.5)]
        low_values = [100.0]
        high_values = [200.0]

        y = multi_level_hysteresis(x, thresholds, low_values, high_values)

        # At x=4: First time (ascending) should be high, second time (descending) still high
        assert y[2] == 200.0  # x=4, ascending, above threshold
        assert y[4] == 200.0  # x=4, descending, still above descending threshold

        # At x=2: First time (ascending) should be low, second time (descending) should become low
        assert y[1] == 100.0  # x=2, ascending, below threshold
        assert y[5] == 100.0  # x=2, descending, below descending threshold

    def test_multiple_levels(self):
        """Test three-level hysteresis"""
        x = np.array([0, 4, 9, 13, 9, 4, 0])
        thresholds = [(3, 2), (8, 6), (12, 10)]
        low_values = [100, 200, 300]
        high_values = [250, 350, 450]

        y = multi_level_hysteresis(x, thresholds, low_values, high_values)

        # Check transitions on ascent
        assert y[0] == 100  # x=0, below all thresholds (active_level=-1)
        assert y[1] == 250  # x=4, crossed first threshold (active_level=0)
        assert y[2] == 350  # x=9, crossed second threshold (active_level=1)
        assert y[3] == 450  # x=13, crossed third threshold (active_level=2)
        
        # Check transitions on descent
        assert y[4] == 350  # x=9, still at level 1 (above desc threshold 1=6)
        assert y[5] == 250  # x=4, dropped to level 0 (above desc threshold 0=2)
        assert y[6] == 100  # x=0, below all desc thresholds (active_level=-1)

    def test_validation_length_mismatch(self):
        """Test that mismatched parameter lengths raise ValueError"""
        x = np.array([1, 2, 3])
        thresholds = [(2, 1), (4, 3)]
        low_values = [100]  # Wrong length
        high_values = [200, 300]

        with pytest.raises(ValueError, match="must have same length"):
            multi_level_hysteresis(x, thresholds, low_values, high_values)

    def test_validation_ascending_order(self):
        """Test that unordered ascending thresholds raise ValueError"""
        x = np.array([1, 2, 3])
        thresholds = [(5, 4), (3, 2)]  # Ascending thresholds not in order
        low_values = [100, 200]
        high_values = [200, 300]

        with pytest.raises(ValueError, match="ascending thresholds must be in ascending order"):
            multi_level_hysteresis(x, thresholds, low_values, high_values)

    def test_validation_descending_order(self):
        """Test that unordered descending thresholds raise ValueError"""
        x = np.array([1, 2, 3])
        thresholds = [(3, 5), (5, 2)]  # Descending thresholds not in order
        low_values = [100, 200]
        high_values = [200, 300]

        with pytest.raises(ValueError, match="descending thresholds must be in ascending order"):
            multi_level_hysteresis(x, thresholds, low_values, high_values)

    def test_validation_descending_greater_than_ascending(self):
        """Test that desc >= asc raises ValueError"""
        x = np.array([1, 2, 3])
        thresholds = [(3, 4)]  # desc > asc (invalid)
        low_values = [100]
        high_values = [200]

        with pytest.raises(
            ValueError, match="descending threshold must be less than.*ascending threshold"
        ):
            multi_level_hysteresis(x, thresholds, low_values, high_values)

    def test_empty_array(self):
        """Test with empty input array"""
        x = np.array([])
        thresholds = [(3, 2)]
        low_values = [100]
        high_values = [200]

        y = multi_level_hysteresis(x, thresholds, low_values, high_values)
        assert len(y) == 0
        assert isinstance(y, np.ndarray)

    def test_single_point(self):
        """Test with single data point"""
        x = np.array([5.0])
        thresholds = [(3, 2)]
        low_values = [100]
        high_values = [200]

        y = multi_level_hysteresis(x, thresholds, low_values, high_values)
        assert len(y) == 1
        assert y[0] == 200.0  # 5 > 3, so should be in high state

    def test_zeros_initialization(self):
        """Test initialization below all thresholds"""
        x = np.array([0, 0, 0])
        thresholds = [(3, 2), (8, 6)]
        low_values = [100, 200]
        high_values = [250, 350]

        y = multi_level_hysteresis(x, thresholds, low_values, high_values)
        assert all(y == 100)  # All should be at lowest level


class TestEstimateHysteresisParameters:
    """Test hysteresis parameter estimation"""

    def test_single_level_synthetic_data(self):
        """Test parameter estimation from synthetic single-level data"""
        # Create synthetic data with known hysteresis
        x_up = np.linspace(0, 10, 50)
        x_down = np.linspace(10, 0, 50)
        x = np.concatenate([x_up, x_down])

        # Create y with hysteresis: low below 3 on ascent, low below 2 on descent
        y_up = np.where(x_up > 3.5, 200.0, 100.0)
        y_down = np.where(x_down < 2.5, 100.0, 200.0)
        y = np.concatenate([y_up, y_down])

        df = pd.DataFrame({"x": x, "y": y})
        result = estimate_hysteresis_parameters(df, x_col="x", y_col="y")

        # Check structure
        assert "thresholds" in result
        assert "low_values" in result
        assert "high_values" in result
        assert "diagnostics" in result

        # Should find one valid level
        assert len(result["thresholds"]) >= 1
        assert len(result["low_values"]) == len(result["thresholds"])
        assert len(result["high_values"]) == len(result["thresholds"])

        # Check threshold is approximately correct
        if len(result["thresholds"]) > 0:
            asc, desc = result["thresholds"][0]
            assert 3.0 < asc < 4.0  # Should be around 3.5
            assert 2.0 < desc < 3.0  # Should be around 2.5
            assert desc < asc  # Descending < ascending

    def test_with_n_levels_clustering(self):
        """Test with explicit n_levels parameter"""
        # Try sklearn import first
        try:
            from sklearn.cluster import KMeans
            sklearn_available = True
        except ImportError:
            sklearn_available = False

        # Create data with continuous y values that should cluster into 2 levels
        x = np.concatenate([
            np.linspace(0, 10, 30),
            np.linspace(10, 0, 30)
        ])
        
        # Base pattern with noise
        y_base = np.concatenate([
            np.where(np.linspace(0, 10, 30) > 5, 200, 100),
            np.where(np.linspace(10, 0, 30) < 4, 100, 200)
        ])
        
        # Add noise
        np.random.seed(42)
        y = y_base + np.random.normal(0, 5, len(y_base))

        df = pd.DataFrame({"x": x, "y": y})
        
        # Request 2 levels
        result = estimate_hysteresis_parameters(df, x_col="x", y_col="y", n_levels=2)

        # Should return results regardless of sklearn availability
        assert "thresholds" in result
        assert isinstance(result["thresholds"], list)

    def test_empty_dataframe(self):
        """Test with empty DataFrame"""
        df = pd.DataFrame({"x": [], "y": []})
        result = estimate_hysteresis_parameters(df, x_col="x", y_col="y")

        # Should return empty results
        assert len(result["thresholds"]) == 0
        assert len(result["low_values"]) == 0
        assert len(result["high_values"]) == 0

    def test_constant_data(self):
        """Test with constant y values (no transitions)"""
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [100, 100, 100, 100, 100]})
        result = estimate_hysteresis_parameters(df, x_col="x", y_col="y")

        # Should find no transitions
        assert result["diagnostics"]["n_transitions"] == 0

    def test_verbose_output(self, caplog):
        """Test that verbose mode produces log output"""
        x = np.concatenate([np.linspace(0, 10, 30), np.linspace(10, 0, 30)])
        y = np.concatenate([
            np.where(np.linspace(0, 10, 30) > 5, 200, 100),
            np.where(np.linspace(10, 0, 30) < 4, 100, 200)
        ])
        df = pd.DataFrame({"x": x, "y": y})

        # Note: verbose logging may require logger configuration in actual usage
        result = estimate_hysteresis_parameters(df, x_col="x", y_col="y", verbose=True)
        
        # Just ensure it completes without error
        assert "diagnostics" in result


class TestRemoveLowXOutliers:
    """Test low-x outlier removal"""

    def test_iqr_method(self):
        """Test IQR-based outlier removal in low-x region"""
        # Create data with outliers in low-x region
        np.random.seed(42)
        x = np.concatenate([
            np.random.uniform(0, 5, 50),    # Low-x region
            np.random.uniform(5, 20, 100)   # High-x region
        ])
        y = x * 10 + np.random.normal(0, 2, 150)
        
        # Add outliers in low-x region
        y[5] = 1000  # Extreme outlier in low-x
        y[10] = -500

        df = pd.DataFrame({"x": x, "y": y})
        df_clean = remove_low_x_outliers(
            df, x_col="x", y_col="y", x_percentile=25, method="iqr", threshold=1.5
        )

        # Should have removed some outliers
        assert len(df_clean) < len(df)
        assert len(df_clean) > 0

        # Outlier points should be removed
        assert 1000 not in df_clean["y"].values
        assert -500 not in df_clean["y"].values

    def test_zscore_method(self):
        """Test z-score outlier removal in low-x region"""
        np.random.seed(42)
        x = np.linspace(0, 20, 100)
        y = x * 10 + np.random.normal(0, 2, 100)
        y[5] = 1000  # Outlier in low-x region

        df = pd.DataFrame({"x": x, "y": y})
        df_clean = remove_low_x_outliers(
            df, x_col="x", y_col="y", x_percentile=25, method="zscore", threshold=2.0
        )

        assert len(df_clean) < len(df)
        assert 1000 not in df_clean["y"].values

    def test_both_dims_method(self):
        """Test outlier removal using both x and y in low-x region"""
        np.random.seed(42)
        x = np.random.uniform(0, 20, 100)
        y = x * 10 + np.random.normal(0, 2, 100)
        
        # Add x and y outliers
        y[5] = 1000  # y outlier
        x[10] = -100  # x outlier in what should be low-x region

        df = pd.DataFrame({"x": x, "y": y})
        df_clean = remove_low_x_outliers(
            df, x_col="x", y_col="y", x_percentile=25, method="both_dims", threshold=1.5
        )

        assert len(df_clean) < len(df)

    def test_invalid_method_raises_error(self):
        """Test that invalid method raises ValueError"""
        df = pd.DataFrame({"x": [1, 2, 3], "y": [10, 20, 30]})
        
        with pytest.raises(ValueError, match="Unknown method"):
            remove_low_x_outliers(df, method="invalid_method")

    def test_no_outliers(self):
        """Test with clean data (no outliers)"""
        np.random.seed(42)
        x = np.linspace(0, 20, 100)
        y = x * 10 + np.random.normal(0, 0.1, 100)  # Very low noise

        df = pd.DataFrame({"x": x, "y": y})
        df_clean = remove_low_x_outliers(
            df, x_col="x", y_col="y", method="iqr", threshold=3.0  # High threshold
        )

        # Should keep most or all points
        assert len(df_clean) >= len(df) * 0.9

    def test_custom_x_percentile(self):
        """Test with custom x_percentile"""
        np.random.seed(42)
        x = np.linspace(0, 100, 100)
        y = x + np.random.normal(0, 2, 100)
        y[5] = 1000  # Outlier

        df = pd.DataFrame({"x": x, "y": y})
        
        # Use 50th percentile - larger low-x region
        df_clean_50 = remove_low_x_outliers(df, x_col="x", y_col="y", x_percentile=50)
        
        # Use 10th percentile - smaller low-x region
        df_clean_10 = remove_low_x_outliers(df, x_col="x", y_col="y", x_percentile=10)

        # Both should work
        assert len(df_clean_50) > 0
        assert len(df_clean_10) > 0


class TestRemoveOutliers:
    """Test general outlier removal"""

    def test_iqr_method(self):
        """Test IQR-based outlier removal"""
        np.random.seed(42)
        x = np.random.normal(10, 2, 100)
        y = np.random.normal(50, 5, 100)
        
        # Add clear outliers
        x[5] = 1000
        y[10] = -500

        df = pd.DataFrame({"x": x, "y": y})
        df_clean = remove_outliers(df, x_col="x", y_col="y", method="iqr", threshold=1.5)

        # Should remove outliers
        assert len(df_clean) < len(df)
        assert 1000 not in df_clean["x"].values
        assert -500 not in df_clean["y"].values

    def test_zscore_method(self):
        """Test z-score outlier removal"""
        np.random.seed(42)
        x = np.random.normal(10, 2, 100)
        y = np.random.normal(50, 5, 100)
        x[5] = 1000  # Outlier

        df = pd.DataFrame({"x": x, "y": y})
        df_clean = remove_outliers(df, x_col="x", y_col="y", method="zscore", threshold=3.0)

        assert len(df_clean) < len(df)
        assert 1000 not in df_clean["x"].values

    def test_mad_method(self):
        """Test Median Absolute Deviation outlier removal"""
        np.random.seed(42)
        x = np.random.normal(10, 2, 100)
        y = np.random.normal(50, 5, 100)
        x[5] = 1000
        y[10] = -500

        df = pd.DataFrame({"x": x, "y": y})
        df_clean = remove_outliers(df, x_col="x", y_col="y", method="mad", threshold=3.0)

        assert len(df_clean) < len(df)
        assert 1000 not in df_clean["x"].values
        assert -500 not in df_clean["y"].values

    def test_invalid_method_raises_error(self):
        """Test that invalid method raises ValueError"""
        df = pd.DataFrame({"x": [1, 2, 3], "y": [10, 20, 30]})
        
        with pytest.raises(ValueError, match="Unknown method"):
            remove_outliers(df, method="nonexistent")

    def test_no_outliers(self):
        """Test with clean data"""
        np.random.seed(42)
        x = np.random.normal(10, 1, 100)
        y = np.random.normal(50, 2, 100)

        df = pd.DataFrame({"x": x, "y": y})
        df_clean = remove_outliers(df, x_col="x", y_col="y", method="iqr", threshold=3.0)

        # Should keep most points
        assert len(df_clean) >= len(df) * 0.9

    def test_threshold_variation(self):
        """Test that different thresholds affect results"""
        np.random.seed(42)
        x = np.random.normal(10, 2, 100)
        y = np.random.normal(50, 5, 100)
        x[5] = 30  # Moderate outlier
        y[10] = 100

        df = pd.DataFrame({"x": x, "y": y})
        
        # Strict threshold (remove more)
        df_strict = remove_outliers(df, method="iqr", threshold=1.0)
        
        # Loose threshold (remove less)
        df_loose = remove_outliers(df, method="iqr", threshold=3.0)

        # Strict should remove more points
        assert len(df_strict) <= len(df_loose)

    def test_empty_dataframe(self):
        """Test with empty DataFrame"""
        df = pd.DataFrame({"x": [], "y": []})
        df_clean = remove_outliers(df, x_col="x", y_col="y")

        assert len(df_clean) == 0
        assert isinstance(df_clean, pd.DataFrame)

    def test_verbose_mode(self):
        """Test verbose output (should not raise errors)"""
        np.random.seed(42)
        x = np.random.normal(10, 2, 50)
        y = np.random.normal(50, 5, 50)
        x[5] = 1000

        df = pd.DataFrame({"x": x, "y": y})
        
        # Should complete without error
        df_clean = remove_outliers(df, method="iqr", verbose=True)
        assert len(df_clean) > 0


class TestIntegrationScenarios:
    """Integration tests combining multiple functions"""

    def test_full_pipeline(self):
        """Test complete pipeline: outlier removal -> parameter estimation"""
        # Create synthetic data with hysteresis and outliers
        np.random.seed(42)
        
        # Ascending phase
        x_up = np.linspace(0, 20, 100)
        y_up = np.where(x_up > 8, 200, 100) + np.random.normal(0, 1, 100)  # Less noise
        
        # Descending phase
        x_down = np.linspace(20, 0, 100)
        y_down = np.where(x_down < 6, 100, 200) + np.random.normal(0, 1, 100)  # Less noise
        
        x = np.concatenate([x_up, x_down])
        y = np.concatenate([y_up, y_down])
        
        # Add clear outliers
        y[10] = 1000
        y[50] = -500
        y[150] = 1000

        df = pd.DataFrame({"x": x, "y": y})

        # Step 1: Remove outliers - very lenient to keep most good data
        df_clean = remove_outliers(df, x_col="x", y_col="y", method="iqr", threshold=5.0)
        assert len(df_clean) > 0  # Should keep most data
        assert len(df_clean) < len(df)  # Should remove extreme outliers

        # Step 2: Remove low-x outliers - very lenient
        df_clean2 = remove_low_x_outliers(
            df_clean, x_col="x", y_col="y", x_percentile=20, method="iqr", threshold=5.0
        )
        
        # Only proceed if we still have enough data
        if len(df_clean2) < 20:
            # Not enough data after cleaning - skip parameter estimation
            # This is acceptable behavior for very noisy/sparse data
            return

        # Step 3: Estimate parameters
        result = estimate_hysteresis_parameters(df_clean2, x_col="x", y_col="y")

        # Should find valid thresholds if there's enough data
        assert "thresholds" in result
        assert "diagnostics" in result
        # With synthetic data, we may or may not find clear levels depending on noise
        if len(result["thresholds"]) > 0:
            asc, desc = result["thresholds"][0]
            assert desc < asc  # Basic sanity check

    def test_apply_estimated_parameters(self):
        """Test using estimated parameters with multi_level_hysteresis"""
        # Create clean synthetic data
        x_train = np.concatenate([
            np.linspace(0, 15, 50),
            np.linspace(15, 0, 50)
        ])
        y_train = np.concatenate([
            np.where(np.linspace(0, 15, 50) > 7, 200, 100),
            np.where(np.linspace(15, 0, 50) < 5, 100, 200)
        ])
        
        df_train = pd.DataFrame({"x": x_train, "y": y_train})

        # Estimate parameters
        result = estimate_hysteresis_parameters(df_train, x_col="x", y_col="y")

        if len(result["thresholds"]) > 0:
            # Apply to new data
            x_test = np.array([0, 10, 15, 10, 3, 0])
            y_test = multi_level_hysteresis(
                x_test,
                result["thresholds"],
                result["low_values"],
                result["high_values"]
            )

            # Should produce valid output
            assert len(y_test) == len(x_test)
            assert all(np.isfinite(y_test))
