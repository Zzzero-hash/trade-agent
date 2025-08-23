# Plan for src/main.py Implementation

## Purpose

Create a main entry point for the trading RL project with a smoke test functionality that imports all dependencies and prints 'OK'.

## File Location

`src/main.py`

## Required Functionality

### 1. Command Line Interface

- Support `--smoke-test` argument
- When run with `--smoke-test`, execute dependency import verification
- Default behavior can be defined later

### 2. Smoke Test Implementation

The smoke test should:

- Import all required dependencies:
  - PyTorch
  - pandas
  - numpy
  - gymnasium
  - stable-baselines3
  - scikit-learn
  - optuna
  - ta (or talib)
  - pyarrow
  - fastapi
- Print "OK" if all imports succeed
- Print error message if any import fails

### 3. Implementation Plan

```python
#!/usr/bin/env python3
"""
Main entry point for the trading RL project.
"""

import argparse
import sys


def smoke_test():
    """
    Run smoke test to verify all dependencies can be imported.
    """
    dependencies = [
        ("torch", "PyTorch"),
        ("pandas", "pandas"),
        ("numpy", "NumPy"),
        ("gymnasium", "Gymnasium"),
        ("stable_baselines3", "Stable Baselines3"),
        ("sklearn", "scikit-learn"),
        ("optuna", "Optuna"),
        ("ta", "Technical Analysis"),
        ("pyarrow", "PyArrow"),
        ("fastapi", "FastAPI"),
    ]

    print("Running smoke test...")

    failed_imports = []

    for module, name in dependencies:
        try:
            __import__(module)
            print(f"✓ {name} imported successfully")
        except ImportError as e:
            failed_imports.append((name, str(e)))
            print(f"✗ {name} import failed: {e}")

    if failed_imports:
        print("\nFailed imports:")
        for name, error in failed_imports:
            print(f"  - {name}: {error}")
        return False
    else:
        print("\nOK")
        return True


def main():
    """
    Main entry point.
    """
    parser = argparse.ArgumentParser(description="Trading RL Project")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run smoke test to verify dependencies"
    )

    args = parser.parse_args()

    if args.smoke_test:
        success = smoke_test()
        sys.exit(0 if success else 1)
    else:
        print("Trading RL Project")
        print("Use --smoke-test to verify dependencies")


if __name__ == "__main__":
    main()
```

## Usage Examples

1. Run smoke test:

   ```bash
   python src/main.py --smoke-test
   ```

2. Run default:
   ```bash
   python src/main.py
   ```

## Expected Output

Successful smoke test:

```
Running smoke test...
✓ PyTorch imported successfully
✓ pandas imported successfully
✓ NumPy imported successfully
✓ Gymnasium imported successfully
✓ Stable Baselines3 imported successfully
✓ scikit-learn imported successfully
✓ Optuna imported successfully
✓ Technical Analysis imported successfully
✓ PyArrow imported successfully
✓ FastAPI imported successfully

OK
```

Failed smoke test:

```
Running smoke test...
✓ PyTorch imported successfully
✓ pandas imported successfully
✗ Technical Analysis import failed: No module named 'ta'
...

Failed imports:
  - Technical Analysis: No module named 'ta'
```
