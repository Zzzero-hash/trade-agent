"""
Comprehensive script to check for data leakage in the feature engineering output.
"""

import numpy as np
import pandas as pd


def comprehensive_leakage_check():
    """Comprehensive check for data leakage."""
    # Load the feature data
    features_df = pd.read_parquet('data/large_fe.parquet')
    print(f"Loaded feature data with {len(features_df)} rows and {len(features_df.columns)} columns")

    # Load the original data to compute actual returns
    original_df = pd.read_parquet('data/large_sample_data.parquet')

    # Compute actual returns for comparison
    actual_returns = pd.Series(
        np.log(original_df['close'] / original_df['close'].shift(1)),
        index=original_df.index
    )

    print(f"Computed actual returns with {len(actual_returns)} values")

    # Align the data
    # We need to make sure we're comparing the right timestamps
    # The features are computed with shifts, so we need to be careful

    # Let's check a few specific cases
    print("\nChecking for data leakage:")
    print("Verifying that features at time t don't depend on returns at time t or future times")

    leakage_detected = False

    # Take a sample of timestamps
    sample_timestamps = features_df.index[:10]  # First 10 timestamps

    for timestamp in sample_timestamps:
        print(f"\nAnalyzing timestamp: {timestamp}")

        # Get the feature values at this timestamp
        if timestamp not in features_df.index:
            continue

        feature_values = features_df.loc[timestamp]

        # Get the actual return at this timestamp
        if timestamp not in actual_returns.index:
            continue

        actual_return_t = actual_returns.loc[timestamp]
        print(f"  Actual return at t: {actual_return_t:.6f}")

        # Get the actual return at t-1
        timestamp_loc = actual_returns.index.get_loc(timestamp)
        if timestamp_loc > 0:
            prev_timestamp = actual_returns.index[timestamp_loc - 1]
            actual_return_t_minus_1 = actual_returns.loc[prev_timestamp]
            print(f"  Actual return at t-1: {actual_return_t_minus_1:.6f}")
        else:
            actual_return_t_minus_1 = np.nan

        # Check log_returns feature - should equal return at t-1
        if not pd.isna(feature_values['log_returns']) and not pd.isna(actual_return_t_minus_1):
            if abs(feature_values['log_returns'] - actual_return_t_minus_1) < 1e-10:
                print("  ✓ log_returns feature correctly uses return at t-1")
            else:
                print(f"  ✗ log_returns feature mismatch: feature={feature_values['log_returns']:.6f}, actual t-1 return={actual_return_t_minus_1:.6f}")
                leakage_detected = True

        # For other features, check if they're suspiciously close to the return at t
        # This would indicate potential leakage
        for feature_name in features_df.columns:
            if feature_name in ['log_returns', 'mu_hat', 'sigma_hat']:
                continue  # Skip log_returns (already checked) and targets

            feature_value = feature_values[feature_name]
            if pd.isna(feature_value) or pd.isna(actual_return_t):
                continue

            # If feature is very close to the actual return at t, it might indicate leakage
            if abs(feature_value - actual_return_t) < 1e-10:
                print(f"  ⚠️  Potential leakage: {feature_name} = {feature_value:.6f} ≈ return at t = {actual_return_t:.6f}")
                leakage_detected = True

    # Check for constant columns
    print("\nChecking for constant columns:")
    constant_columns = []
    for col in features_df.columns:
        variance = features_df[col].var()
        if pd.isna(variance) or variance == 0:
            print(f"  ⚠️  Constant column: {col} (variance: {variance})")
            constant_columns.append(col)

    # Summary
    print(f"\n{'='*60}")
    if leakage_detected:
        print("❌ DATA LEAKAGE DETECTED!")
        print("Please review the warnings above.")
        return False
    else:
        print("✅ No data leakage detected in this sample.")
        print("All features appear to be correctly computed using only past/present data.")
        return True


if __name__ == "__main__":
    comprehensive_leakage_check()
