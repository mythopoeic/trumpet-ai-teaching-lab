"""Pytest configuration for the teaching-engine project.

Adds ``services/`` to ``sys.path`` so tests can import bare-package modules
like ``audio.dp_tier_a``, mirroring the runtime convention established by
``app/main.py`` (see project ``CLAUDE.md``). Without this, ``python -m pytest``
launched from ``teaching-engine/`` would not see the modules under
``services/`` because ``services/`` is not a package and is not auto-added
to ``sys.path``.
"""

import sys
from pathlib import Path

_TE_ROOT = Path(__file__).resolve().parent.parent
_SERVICES_DIR = _TE_ROOT / "services"
if str(_SERVICES_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVICES_DIR))
