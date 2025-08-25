# Feature Flags (Phase 0)

Purpose: Gate premium or experimental features without breaking core OSS.

Current flags:
- premium.large_models
- premium.cloud_execution

Usage example:
```python
from shared import flags
if flags.is_enabled("premium.large_models"):
    # enable larger model downloads
    ...
```

Planned providers:
- Local JSON file
- Env var prefix
- Remote service (SaaS) signed
