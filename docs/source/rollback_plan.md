# Rollback Plan for trade-agent Project Initialization

## Overview

This document defines the rollback procedures to recover from failures during Step 1: Project skeleton & dependencies (Chunk0).

## Rollback Scenarios

### Scenario 1: Dependency Installation Failure

When dependencies cannot be installed due to conflicts or missing packages.

#### Recovery Steps:

1. **Revert pyproject.toml changes**
   - Restore from version control: `git checkout HEAD -- pyproject.toml`
   - Or manually remove added dependencies

2. **Clean Python environment**

   ```bash
   pip uninstall -y torch pandas numpy gymnasium stable-baselines3 scikit-learn optuna ta pyarrow fastapi
   ```

3. **Reinstall original dependencies**

   ```bash
   pip install -e .
   ```

4. **Verify recovery**
   ```bash
   python -c "import sys; print('Python OK')"
   ```

### Scenario 2: Directory Creation Failure

When required directories cannot be created or have incorrect permissions.

#### Recovery Steps:

1. **Remove newly created directories**

   ```bash
   rm -rf models/ reports/ src/serve/
   ```

2. **Restore directory structure from backup**
   - If using version control: `git checkout HEAD -- .`
   - Recreate missing directories if needed

3. **Verify directory permissions**
   ```bash
   ls -la data/ models/ reports/ src/
   ```

### Scenario 3: File Creation Failure

When required files (src/main.py, Makefile) cannot be created or are corrupted.

#### Recovery Steps:

1. **Remove corrupted files**

   ```bash
   rm -f src/main.py Makefile
   ```

2. **Restore from version control**

   ```bash
   git checkout HEAD -- src/main.py Makefile
   ```

3. **If no version control history, recreate files**
   - Follow implementation plans in documentation
   - Create minimal working versions

### Scenario 4: Smoke Test Failure

When smoke test fails after installation.

#### Recovery Steps:

1. **Identify failed dependencies**

   ```bash
   python src/main.py --smoke-test
   ```

2. **Check specific imports**

   ```bash
   python -c "import torch"  # Test each failed import
   ```

3. **Reinstall problematic packages**

   ```bash
   pip uninstall package_name
   pip install package_name
   ```

4. **Check for version conflicts**
   ```bash
   pip check
   ```

## Backup Procedures

### Before Making Changes

1. **Commit current state to version control**

   ```bash
   git add .
   git commit -m "Backup before trade-agent initialization"
   ```

2. **Export current environment**

   ```bash
   pip freeze > backup_requirements.txt
   ```

3. **Backup pyproject.toml**
   ```bash
   cp pyproject.toml pyproject.toml.backup
   ```

### During Implementation

1. **Stage changes incrementally**

   ```bash
   git add pyproject.toml
   git commit -m "Update dependencies"

   git add src/main.py Makefile
   git commit -m "Add main module and Makefile"

   git add data/ models/ reports/ src/serve/
   git commit -m "Create required directories"
   ```

## Recovery Commands

### Complete Rollback

```bash
# Revert to previous state
git reset --hard HEAD~1

# Or if multiple commits were made
git reset --hard <commit-hash-before-changes>

# Clean up any untracked files
git clean -fd
```

### Selective Rollback

```bash
# Revert only pyproject.toml
git checkout HEAD~1 -- pyproject.toml
pip install -e .

# Revert only directory structure
git checkout HEAD~1 -- data/ models/ reports/ src/

# Revert only source files
git checkout HEAD~1 -- src/main.py Makefile
```

## Verification After Rollback

### Check 1: Environment State

```bash
pip list | grep -E "(torch|pandas|numpy|gymnasium|stable-baselines3|scikit-learn|optuna|ta|pyarrow|fastapi)"
```

Expected: Only originally installed packages

### Check 2: Directory Structure

```bash
ls -la data/ models/ reports/ src/
```

Expected: Original directory structure restored

### Check 3: File Integrity

```bash
test -f src/main.py && echo "main.py exists" || echo "main.py missing"
test -f Makefile && echo "Makefile exists" || echo "Makefile missing"
```

Expected: Files in expected state

## Common Issues and Solutions

### Issue 1: Permission Denied

**Symptom**: `Permission denied` errors when creating files/directories
**Solution**:

```bash
sudo chown -R $USER:$USER .
chmod -R u+rwx .
```

### Issue 2: Disk Space Full

**Symptom**: `No space left on device` errors
**Solution**:

```bash
# Check disk usage
df -h

# Clean up temporary files
sudo apt-get clean  # On Ubuntu/Debian
# Or
brew cleanup  # On macOS

# Remove Python cache files
find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -exec rm -rf {} +
```

### Issue 3: Network Issues During Installation

**Symptom**: `Connection timeout` or `Failed to fetch` during pip install
**Solution**:

```bash
# Retry with timeout
pip install -e . --timeout 1000

# Use alternative index
pip install -e . -i https://pypi.org/simple/

# Install from cache if available
pip install -e . --no-index --find-links /path/to/cache
```

## Emergency Procedures

### If Version Control is Unavailable

1. **Manual backup of current state**

   ```bash
   cp -r . ../trade-agent-backup-$(date +%Y%m%d-%H%M%S)
   ```

2. **Document current configuration**
   ```bash
   pip freeze > current_state_requirements.txt
   ls -la > current_state_files.txt
   ```

### If Complete System Failure

1. **Restore from external backup**
   - Cloud backup (if available)
   - Local backup on external drive
   - Re-clone from remote repository

2. **Reinstall system dependencies**
   ```bash
   # Reinstall Python and pip
   sudo apt-get install python3 python3-pip  # Ubuntu/Debian
   # Or
   brew install python  # macOS
   ```

## Post-Rollback Actions

1. **Verify system stability**

   ```bash
   python -c "print('Python OK')"
   ```

2. **Test basic functionality**

   ```bash
   # Run any existing tests
   python -m pytest tests/ 2>/dev/null || echo "No tests to run"
   ```

3. **Document the rollback**
   - Record what failed
   - Note recovery steps taken
   - Update issue tracking system if applicable

4. **Plan next steps**
   - Identify root cause of failure
   - Determine if approach needs modification
   - Schedule retry with fixes
