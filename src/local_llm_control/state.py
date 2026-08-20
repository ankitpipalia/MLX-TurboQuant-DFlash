"""Process-wide controller state shared by the API routers."""

from .config import load_settings
from .manager import RuntimeManager


settings = load_settings()
manager = RuntimeManager(settings)


def refresh_manager_settings():
    """Reload profiles.toml and atomically replace the manager snapshot."""
    fresh = load_settings()
    manager.update_settings(fresh)
    return fresh
