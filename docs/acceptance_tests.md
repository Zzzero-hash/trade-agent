# Acceptance Tests for trade-agent Project Initialization

## Overview

This document defines the acceptance tests that must pass to verify successful completion of Step 1: Project skeleton & dependencies (Chunk0).

## Test Categories

### 1. Dependency Installation Tests

Verify that all required dependencies can be installed without conflicts.

#### Test 1.1: Dependency Resolution

- **Description**: All dependencies specified in pyproject.toml can be resolved
- **Command**: `pip install -e .`
- **Expected Result**: Successful installation with no dependency conflicts
- **Acceptance Criteria**: Exit code 0, no error messages

#### Test 1.2: Dependency Versions

- **Description**: All dependencies are installed with correct versions
- **Command**: `pip list | grep -E "(torch|pandas|numpy|gymnasium|stable-baselines3|scikit-learn|optuna|ta|pyarrow|fastapi)"`
- **Expected Result**: All required packages listed with versions meeting minimum requirements
- **Acceptance Criteria**: All packages present with versions >= minimum requirements

### 2. Smoke Test Verification

Verify that the smoke test runs successfully and imports all dependencies.

#### Test 2.1: Smoke Test Execution

- **Description**: Main module runs smoke test successfully
- **Command**: `python src/main.py --smoke-test`
- **Expected Result**: All dependencies imported successfully, "OK" printed
- **Acceptance Criteria**: Exit code 0, output contains "OK"

#### Test 2.2: Individual Dependency Import

- **Description**: Each required dependency can be imported individually
- **Command**: `python -c "import torch; import pandas; import numpy; import gymnasium; import stable_baselines3; import sklearn; import optuna; import ta; import pyarrow; import fastapi; print('All imports successful')"`
- **Expected Result**: All imports succeed without errors
- **Acceptance Criteria**: Exit code 0, "All imports successful" printed

### 3. Directory Structure Verification

Verify that the directory structure matches requirements.

#### Test 3.1: Required Directories Exist

- **Description**: All required directories are present
- **Command**: `ls -la data/ models/ reports/ src/data/ src/features/ src/sl/ src/envs/ src/rl/ src/ensemble/ src/eval/ src/serve/`
- **Expected Result**: All directories exist and are accessible
- **Acceptance Criteria**: Exit code 0, all directories listed

#### Test 3.2: Directory Permissions

- **Description**: All directories have appropriate read/write permissions
- **Command**: `test -r data && test -w data && test -r models && test -w models && test -r reports && test -w reports`
- **Expected Result**: Directories are readable and writable
- **Acceptance Criteria**: Exit code 0

### 4. File Creation Verification

Verify that required files are properly created.

#### Test 4.1: Main Module Exists

- **Description**: Main entry point exists
- **Command**: `test -f src/main.py`
- **Expected Result**: File exists
- **Acceptance Criteria**: Exit code 0

#### Test 4.2: Makefile Exists

- **Description**: Build automation file exists
- **Command**: `test -f Makefile`
- **Expected Result**: File exists
- **Acceptance Criteria**: Exit code 0

### 5. Makefile Command Tests

Verify that Makefile commands work correctly.

#### Test 5.1: Makefile Help

- **Description**: Makefile help command works
- **Command**: `make help`
- **Expected Result**: Help text displayed with all commands
- **Acceptance Criteria**: Exit code 0, help text contains all expected commands

#### Test 5.2: Makefile Smoke Test

- **Description**: Makefile smoke test command works
- **Command**: `make smoke-test`
- **Expected Result**: Smoke test runs successfully
- **Acceptance Criteria**: Exit code 0, output contains "OK"

## Automated Test Script

Create a test script to automate verification:

```bash
#!/bin/bash
# acceptance_tests.sh

echo "Running acceptance tests for trade-agent initialization..."

# Test 1.1: Dependency Resolution
echo "Test 1.1: Dependency Resolution"
pip install -e . > /tmp/install.log 2>&1
if [ $? -eq 0 ]; then
    echo "✓ PASS: Dependencies installed successfully"
else
    echo "✗ FAIL: Dependency installation failed"
    cat /tmp/install.log
    exit 1
fi

# Test 1.2: Dependency Versions
echo "Test 1.2: Dependency Versions"
pip list | grep -E "(torch|pandas|numpy|gymnasium|stable-baselines3|scikit-learn|optuna|ta|pyarrow|fastapi)" > /tmp/versions.log
if [ $? -eq 0 ]; then
    echo "✓ PASS: All required dependencies present"
    cat /tmp/versions.log
else
    echo "✗ FAIL: Some dependencies missing"
    exit 1
fi

# Test 2.1: Smoke Test Execution
echo "Test 2.1: Smoke Test Execution"
python src/main.py --smoke-test > /tmp/smoke.log 2>&1
if [ $? -eq 0 ] && grep -q "OK" /tmp/smoke.log; then
    echo "✓ PASS: Smoke test successful"
else
    echo "✗ FAIL: Smoke test failed"
    cat /tmp/smoke.log
    exit 1
fi

# Test 3.1: Required Directories Exist
echo "Test 3.1: Required Directories Exist"
REQUIRED_DIRS="data models reports src/data src/features src/sl src/envs src/rl src/ensemble src/eval src/serve"
for dir in $REQUIRED_DIRS; do
    if [ ! -d "$dir" ]; then
        echo "✗ FAIL: Directory $dir missing"
        exit 1
    fi
done
echo "✓ PASS: All required directories exist"

# Test 4.1: Main Module Exists
echo "Test 4.1: Main Module Exists"
if [ -f "src/main.py" ]; then
    echo "✓ PASS: Main module exists"
else
    echo "✗ FAIL: Main module missing"
    exit 1
fi

# Test 4.2: Makefile Exists
echo "Test 4.2: Makefile Exists"
if [ -f "Makefile" ]; then
    echo "✓ PASS: Makefile exists"
else
    echo "✗ FAIL: Makefile missing"
    exit 1
fi

echo "All acceptance tests passed!"
```

## Manual Verification Steps

### Step 1: Verify Installation

```bash
pip install -e .
```

### Step 2: Run Smoke Test

```bash
python src/main.py --smoke-test
```

### Step 3: Check Directory Structure

```bash
ls -la data/ models/ reports/ src/
```

### Step 4: Test Makefile Commands

```bash
make help
make smoke-test
```

## Success Criteria

The initialization step is considered successful when:

1. All dependencies install without conflicts (Test 1.1, 1.2)
2. Smoke test runs successfully (Test 2.1, 2.2)
3. Directory structure matches requirements (Test 3.1, 3.2)
4. Required files are created (Test 4.1, 4.2)
5. Makefile commands work correctly (Test 5.1, 5.2)

## Failure Handling

If any acceptance test fails:

1. Identify the specific test that failed
2. Check error messages and logs
3. Refer to rollback plan for recovery steps
4. Document the failure for future reference
