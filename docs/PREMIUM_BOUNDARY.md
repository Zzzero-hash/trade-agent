# Premium Boundary

Directory `premium/` reserved for proprietary or licensed extensions not shipped in the core open-source distribution.

Core rules:
1. No hard runtime dependency from OSS core into `premium`.
2. Premium modules must register via plugin or dynamic discovery.
3. Clear separation for licensing audits.

Examples of future premium modules:
- High-frequency data connectors
- GPU/TPU remote training orchestrators
- Advanced risk dashboards
