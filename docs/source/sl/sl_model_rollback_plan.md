# SL Model Implementation - Rollback Plan

## 1. Overview

This document outlines the rollback procedures for the supervised learning model implementation. These procedures ensure that in case of failures or issues during or after deployment, the system can be reverted to a stable state with minimal disruption.

## 2. Rollback Scenarios

### 2.1 Model Performance Issues

- **Trigger**: SL model performance degrades below acceptable thresholds
- **Impact**: Trading strategy performance may be negatively affected
- **Rollback Action**: Revert to previous model version

### 2.2 Model Training Failures

- **Trigger**: Model training fails or produces errors
- **Impact**: No new model can be deployed
- **Rollback Action**: Remove newly created model files and restore original training approach

### 2.3 Integration Problems

- **Trigger**: Issues with feature engineering pipeline integration
- **Impact**: Data processing pipeline may be disrupted
- **Rollback Action**: Revert to previous integration approach

### 2.4 System Stability Issues

- **Trigger**: Memory leaks, excessive CPU usage, or other stability problems
- **Impact**: System performance degradation
- **Rollback Action**: Remove problematic components and restore previous version

## 3. Rollback Procedures

### 3.1 Revert to Previous Model Version

#### 3.1.1 Identify Previous Version

```bash
# Check model registry for previous production version
cat models/registry.json | jq '.[] | select(.is_production==true and .version!="CURRENT_VERSION")'
```

#### 3.1.2 Load Previous Model

```python
def rollback_to_previous_model():
    """
    Rollback to the previous production model version.
    """
    registry = ModelRegistry()

    # Get current production model
    current_model = registry.get_production_model("sl_forecaster")

    # Get previous production model
    previous_model = registry.get_previous_production_model("sl_forecaster")

    if previous_model:
        # Load previous model
        model = load_model(previous_model["model_path"])

        # Update registry to mark previous as production
        registry.set_production_model("sl_forecaster", previous_model["version"])

        # Archive current model
        registry.archive_model("sl_forecaster", current_model["version"])

        return model
    else:
        raise ValueError("No previous production model found")
```

#### 3.1.3 Verify Rollback

```python
def verify_rollback():
    """
    Verify that rollback was successful.
    """
    # Check that previous model is now production
    registry = ModelRegistry()
    prod_model = registry.get_production_model("sl_forecaster")

    # Load and test model
    model = load_model(prod_model["model_path"])

    # Run quick validation
    X_test, y_test = load_validation_data()
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)

    assert mse < 0.1, "Rollback model performance is unacceptable"

    return True
```

### 3.2 Remove Newly Created Model Files

#### 3.2.1 Identify New Files

```bash
# List recently created model files
find models/ -name "*.pkl" -newer backup_timestamp.txt
find models/ -name "*metadata.json" -newer backup_timestamp.txt
```

#### 3.2.2 Remove Files

```python
def remove_new_model_files(current_version):
    """
    Remove newly created model files.
    """
    import glob
    import os

    # Remove model files for current version
    model_files = glob.glob(f"models/sl_model_{current_version}*")
    for file_path in model_files:
        try:
            os.remove(file_path)
            print(f"Removed {file_path}")
        except OSError as e:
            print(f"Error removing {file_path}: {e}")

    # Update registry
    registry = ModelRegistry()
    registry.remove_model("sl_forecaster", current_version)
```

#### 3.2.3 Verify Cleanup

```bash
# Verify files are removed
find models/ -name "*CURRENT_VERSION*" 2>&1 | grep "No such file"
```

### 3.3 Restore Original Training Approach

#### 3.3.1 Revert Configuration

```python
def restore_original_training_config():
    """
    Restore original training configuration.
    """
    import shutil

    # Restore config from backup
    if os.path.exists("configs/sl/training_configs.yaml.backup"):
        shutil.copy("configs/sl/training_configs.yaml.backup", "configs/sl/training_configs.yaml")
        print("Restored original training configuration")
    else:
        # Create default config
        create_default_training_config()
        print("Created default training configuration")
```

#### 3.3.2 Revert Code Changes

```bash
# If using git, revert to previous commit
git checkout HEAD~1 -- src/sl/

# Or revert specific files
git checkout HEAD~1 -- src/sl/models/traditional.py
git checkout HEAD~1 -- src/sl/models/tree_based.py
```

## 4. Backup and Recovery

### 4.1 Automated Backups

Before any major changes, automated backups should be created:

```python
def create_backup():
    """
    Create backup of current SL model components.
    """
    import shutil
    import datetime

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Backup model files
    if os.path.exists("models/"):
        shutil.copytree("models/", f"backups/models_{timestamp}/")

    # Backup source code
    if os.path.exists("src/sl/"):
        shutil.copytree("src/sl/", f"backups/src_sl_{timestamp}/")

    # Backup configs
    if os.path.exists("configs/sl/"):
        shutil.copytree("configs/sl/", f"backups/configs_sl_{timestamp}/")

    # Backup registry
    if os.path.exists("models/registry.json"):
        shutil.copy("models/registry.json", f"backups/registry_{timestamp}.json")

    # Record backup timestamp
    with open("backup_timestamp.txt", "w") as f:
        f.write(timestamp)

    return timestamp
```

### 4.2 Version Control

All changes should be tracked using git:

```bash
# Before making changes
git add -A
git commit -m "Backup before SL model implementation"

# After changes
git add -A
git commit -m "Implemented SL model components"

# Tag releases
git tag -a sl-v1.0 -m "SL Model Version 1.0"
```

### 4.3 Automated Rollback Scripts

#### 4.3.1 Full Rollback Script

```bash
#!/bin/bash
# rollback_sl_model.sh

echo "Starting SL model rollback procedure..."

# 1. Stop any running model training
pkill -f "sl_model_training"

# 2. Identify current version
CURRENT_VERSION=$(python -c "from src.sl.persistence.registry import ModelRegistry; r = ModelRegistry(); m = r.get_production_model('sl_forecaster'); print(m['version'] if m else 'unknown')")

echo "Current model version: $CURRENT_VERSION"

# 3. Rollback to previous version
python -c "
from src.sl.persistence.registry import ModelRegistry
registry = ModelRegistry()
previous_model = registry.get_previous_production_model('sl_forecaster')
if previous_model:
    registry.set_production_model('sl_forecaster', previous_model['version'])
    print(f'Rolled back to version: {previous_model[\"version\"]}')
else:
    print('No previous version found')
"

# 4. Remove current version files
if [ "$CURRENT_VERSION" != "unknown" ]; then
    echo "Removing files for version: $CURRENT_VERSION"
    rm -f models/sl_model_${CURRENT_VERSION}*
fi

# 5. Verify rollback
python -c "
from src.sl.persistence.registry import ModelRegistry
registry = ModelRegistry()
prod_model = registry.get_production_model('sl_forecaster')
if prod_model:
    print(f'Rollback successful. Production model is now: {prod_model[\"version\"]}')
else:
    print('Rollback failed. No production model found.')
"

echo "SL model rollback procedure completed."
```

#### 4.3.2 Quick Verification Script

```python
# verify_rollback.py
def quick_verification():
    """
    Quick verification that rollback was successful.
    """
    try:
        # Import SL components
        from src.sl.pipelines.forecasting_pipeline import ForecastingPipeline
        from src.sl.persistence.registry import ModelRegistry

        # Check registry
        registry = ModelRegistry()
        prod_model = registry.get_production_model("sl_forecaster")

        if not prod_model:
            raise Exception("No production model found")

        # Test model loading
        from src.sl.models.factory import SLModelFactory
        model = SLModelFactory.create_model("xgboost", {})

        # Test pipeline
        pipeline = ForecastingPipeline()

        print("Quick verification passed")
        return True

    except Exception as e:
        print(f"Quick verification failed: {e}")
        return False

if __name__ == "__main__":
    quick_verification()
```

## 5. Rollback Verification Steps

### 5.1 System Stability Check

- Verify that no processes are consuming excessive resources
- Check system logs for errors
- Confirm that all services are running normally

### 5.2 Model Performance Check

- Run validation on the rolled-back model
- Compare performance with expected baseline
- Ensure predictions are within acceptable ranges

### 5.3 Integration Check

- Verify that feature pipeline integration works correctly
- Check that data flows properly through the system
- Confirm that no data leakage occurs

### 5.4 User Impact Assessment

- Assess impact on trading strategy performance
- Check for any disruptions to downstream systems
- Document any issues that may affect users

## 6. Rollback Communication Plan

### 6.1 Internal Communication

- Notify development team of rollback
- Document reasons for rollback in team communication channels
- Update project management tools with rollback status

### 6.2 External Communication

- If rollback affects production systems, notify stakeholders
- Provide timeline for resolution
- Communicate expected impact and recovery plan

## 7. Post-Rollback Actions

### 7.1 Root Cause Analysis

- Investigate what caused the need for rollback
- Document findings and lessons learned
- Implement preventive measures to avoid similar issues

### 7.2 Testing and Validation

- Thoroughly test rolled-back system
- Validate that all functionality works as expected
- Run acceptance tests to ensure quality

### 7.3 Future Planning

- Plan for re-implementation of rolled-back features
- Schedule additional testing for problematic components
- Update documentation with lessons learned

## 8. Rollback Checklist

### 8.1 Pre-Rollback

- [ ] Identify rollback trigger and impact
- [ ] Create backup of current state
- [ ] Notify relevant stakeholders
- [ ] Prepare rollback scripts and tools

### 8.2 During Rollback

- [ ] Execute rollback procedures in order
- [ ] Monitor system during rollback
- [ ] Document any issues encountered
- [ ] Verify each step as completed

### 8.3 Post-Rollback

- [ ] Verify system stability and performance
- [ ] Test critical functionality
- [ ] Communicate rollback completion
- [ ] Document rollback process and results
- [ ] Plan next steps for issue resolution

## 9. Dependencies

- Git version control system
- Model registry and versioning system
- Backup storage with sufficient space
- Monitoring and alerting systems
- Communication channels for stakeholder notification

## 10. Estimated Rollback Time

- **Simple rollback (version revert)**: 15-30 minutes
- **Complex rollback (code revert)**: 1-2 hours
- **Full system rollback**: 2-4 hours

## 11. Rollback Success Criteria

- System returns to stable operating state
- Previous model version performs within acceptable parameters
- No data loss or corruption occurs
- All integrations function correctly
- Stakeholders are notified of successful rollback
