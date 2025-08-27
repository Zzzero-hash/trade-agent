import re
import subprocess
from pathlib import Path


SCRIPT = Path("scripts/bump_version.py")
VERSION_FILE = Path("version.py")


def read_version() -> str:
    text = VERSION_FILE.read_text(encoding="utf-8")
    m = re.search(r'__version__\s*=\s*"([^"]+)"', text)
    assert m, f"Failed to find version in {VERSION_FILE}"
    return m.group(1)


def test_bump_version_patch_dry_run_and_real() -> None:
    original_content = VERSION_FILE.read_text(encoding="utf-8")
    original_version = read_version()
    try:
        # Dry run
        r = subprocess.run(
            ["python", str(SCRIPT), "--part", "patch", "--dry-run"],
            check=True,
            capture_output=True,
            text=True,
        )
        assert "[dry-run]" in r.stdout
        assert read_version() == original_version

        # Real bump
        r2 = subprocess.run(
            ["python", str(SCRIPT), "--part", "patch"],
            check=True,
            capture_output=True,
            text=True,
        )
        new_version = read_version()
        assert new_version != original_version
        assert r2.returncode == 0

        # Idempotency: setting same version shouyld exit code 1
        r3 = subprocess.run(
            ["python", str(SCRIPT), "--set", new_version],
            capture_output=True,
            text=True,
        )
        assert r3.returncode == 1
    finally:
        VERSION_FILE.write_text(original_content, encoding="utf-8")
