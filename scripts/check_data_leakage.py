"""
Script to check for data leakage in the feature engineering output.
"""

import numpy as np
import pandas as pd


def check_data_leakage():
    """Check for data leakage in the feature engineering output."""
    # Load the feature data
    features_df = pd.read_parquet('data/large_fe.parquet')
    print(f"Loaded feature data with {len(features_df)} rows and {len(features_df.columns)} columns")
    print(f"Columns: {list(features_df.columns)}")

    # Load the original data to compute actual returns
    original_df = pd.read_parquet('data/large_sample_data.parquet')
    print(f"Loaded original data with {len(original_df)} rows")

    # Compute actual returns for comparison
    actual_returns = np.log(original_df['close'] / original_df['close'].shift(1))
    print(f"Computed actual returns with {len(actual_returns)} values")

    # Align the indices
    # The features are already aligned with the original data (after NaN handling)
    # Let's take a sample of 10 random rows to check
    sample_indices = np.random.choice(features_df.index, size=min(10, len(features_df)), replace=False)
    print(f"\nChecking {len(sample_indices)} random rows for data leakage:")

    leakage_detected = False

    for idx in sample_indices:
        print(f"\nRow with timestamp {idx}:")

        # Get the feature values for this timestamp
        feature_row = features_df.loc[idx]

        # Get the actual return for this timestamp
        if idx in actual_returns.index:
            actual_return = actual_returns.loc[idx]
            print(f"  Actual return at t: {actual_return:.6f}")
        else:
            print("  Actual return at t: Not available")
            continue

        # Check if any feature is highly correlated with the actual return at t
        # This would indicate potential leakage
        feature_names = [col for col in features_df.columns if col not in ['mu_hat', 'sigma_hat']]

        for feature_name in feature_names:
            feature_value = feature_row[feature_name]
            if pd.isna(feature_value) or pd.isna(actual_return):
                continue

            # For log_returns feature, it should be the return at t-1, not t
            if feature_name == 'log_returns':
                # This should be the return at t-1, not t
                # Let's check if we can find the previous timestamp
                prev_timestamp = actual_returns.index[actual_returns.index.get_loc(idx) - 1] if actual_returns.index.get_loc(idx) > 0 else None
                if prev_timestamp is not None:
                    prev_return = actual_returns.loc[prev_timestamp]
                    if abs(feature_value - prev_return) < 1e-10:
                        print(f"  ✓ log_returns feature correctly uses return at t-1: {feature_value:.6f}")
                    else:
                        print(f"  ✗ log_returns feature mismatch: feature={feature_value:.6f}, actual t-1 return={prev_return:.6f}")
                        leakage_detected = True
                continue

            # For other features, check if they're correlated with the return at t
            # This would indicate leakage
            correlation = abs(feature_value - actual_return)  # Simple check for now

            # If the feature value is very close to the actual return at t, it might indicate leakage
            # But we need to be careful as some features might naturally be close to returns
            if correlation < 1e-10 and feature_name not in ['log_returns']:
                print(f"  ⚠️  Potential leakage detected in {feature_name}: feature={feature_value:.6f}, return at t={actual_return:.6f}")
                leakage_detected = True

    # Check for constant columns (variance)
    print("\nChecking for constant columns (variance):")
    constant_columns = []
    for col in features_df.columns:
        variance = features_df[col].var()
        if pd.isna(variance) or variance == 0:
            print(f"  ⚠️  Constant column detected: {col} (variance: {variance})")
            constant_columns.append(col)
        else:
            print(f"  ✓ {col}: variance = {variance:.6f}")

    if constant_columns:
        leakage_detected = True

    # Summary
    print(f"\n{'='*50}")
    if leakage_detected:
        print("❌ DATA LEAKAGE DETECTED!")
        print("Please review the warnings above.")
    else:
        print("✅ No data leakage detected.")
        print("All features appear to be correctly computed using only past/present data.")

    return not leakage_detected


if __name__ == "__main__":
    check_data_leakage()
