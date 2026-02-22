# Server & NAS Environment Setup Guide

This document outlines the specific challenges and solutions for deploying the NotebookLM skill in enterprise or research environments (Servers, NAS, VNC, etc.).

## 1. NAS File Locking Issues

### The Problem
Chromium (and Chrome) uses a file called `SingletonLock` inside the user profile directory to ensure only one instance of the browser is running per profile. Many network file systems (NFS, SMB, NAS) do not support the atomic file locking required by Chromium, leading to an `Input/output error (5)` when creating the lock.

### The Solution: Local Cache Redirection
We modified `config.py` to redirect the browser state to a **local** directory on the server's SSD/HDD, bypassing the NAS.

**Change made in `scripts/config.py`:**
```python
import os
# Redefine DATA_DIR to use a local path
BASE_DATA_DIR = Path(os.path.expanduser("~/.cache/antigravity/notebooklm"))
DATA_DIR = BASE_DATA_DIR / "data"
# ... other paths derived from DATA_DIR ...
```

**Benefit:** Chromium can lock files locally at high speed, while the skill scripts remain on the shared NAS.

---

## 2. Shared Environment Integration

### The Problem
The default skill setup creates a local `.venv` inside the skill folder. In large projects, this leads to:
1. Redundant library installations.
2. Complicated dependency management when using multiple servers.

### The Solution: Flexible `run.py`
We updated `scripts/run.py` to be "environment aware". It now:
1. First checks if `patchright` is available in the **current active environment** (e.g., your project's Conda environment).
2. Only attempts to create/use a local `.venv` if the current environment is missing dependencies.

**Integration Tip:** Add the skill dependencies to your project's `pyproject.toml` or `requirements.txt`:
```toml
# pyproject.toml example
dependencies = [
    "patchright==1.55.2",
    "python-dotenv==1.0.0",
]
```

---

## 3. Remote Authentication (VNC/X11)

### The Problem
Setup (`auth_manager.py setup`) requires a **visible browser** for Google Sign-in and 2FA. In a remote SSH session, the browser cannot find a display to render to.

### The Solution: DISPLAY Variable
When using VNC or X-Forwarding, you must explicitly pass the `$DISPLAY` variable to the command.

**How to find your VNC Display:**
```bash
# Check running processes for Xvnc
ps aux | grep Xvnc
# Usually it is :1 or :10.0
```

**Running Authentication:**
```bash
export DISPLAY=:1  # Replace with your actual display
python scripts/run.py auth_manager.py setup
```

---

## 4. Browser Binary Management

On servers with restricted internet access, we recommend pre-installing the browser binaries into a shared cache:

```bash
# Install into the project's conda environment
conda activate grid_world_pain
python -m patchright install chromium
```

The binaries will be stored in `~/.cache/ms-playwright/`, which is shared across all environments for that user.
