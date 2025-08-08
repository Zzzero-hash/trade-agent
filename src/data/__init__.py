"""
Data processing package for the Trade Agent.

Modules:
- ingestion: Data fetching interfaces (APIs, files, etc.)
- processing: Transformations and feature engineering
- cleaning: Data cleaning utilities
- validation: Symbol validation logic
- symbol_corrector: Automated correction suggestions
- unified_orchestrator: Orchestrates the full pipeline
"""

__all__ = [
    "ingestion",
    "processing",
    "cleaning",
    "validation",
    "symbol_corrector",
    "unified_orchestrator",
]
