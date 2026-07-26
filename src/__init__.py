"""APR heart-sound classification: importable library code.

Nothing in `src/` has side effects at import time. All orchestration lives in
`scripts/`. This separation is what makes the modules unit-testable and lets a
phase be re-run without re-running its neighbours.
"""

__version__ = "1.0.0"
