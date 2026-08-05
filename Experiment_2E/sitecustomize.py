"""Process-local compatibility hooks for Experiment 2E.

This module is imported automatically by Python when the project root is on
``PYTHONPATH``. The hook is opt-in so ordinary analysis commands are untouched.
"""

from __future__ import annotations

import os


if os.environ.get("EXPERIMENT_2E_FORCE_SHM") == "1":
    import verl.utils.device as _verl_device

    # torch 2.8 changed the CUDA IPC reduction handle layout expected by the
    # bundled veRL 0.9 receiver. veRL already supports a shared-memory fallback;
    # force that supported path for this run instead of editing the shared repo.
    _verl_device.is_support_ipc = lambda: False
