"""Compatibility shim for isaaclab.utils.configclass.

When isaaclab is not installed, this provides a lightweight fallback.
The config classes are used as plain objects whose __dict__ is read by agents.
The MISSING sentinel is preserved for documentation purposes.
"""
from dataclasses import MISSING as _MISSING

MISSING = _MISSING

try:
    from isaaclab.utils import configclass
except ImportError:
    def configclass(cls=None, **kwargs):
        """Drop-in replacement for isaaclab's configclass.

        Returns the class unchanged. Agents read cfg.__dict__ to get config values,
        so no dataclass machinery is needed. This avoids issues with MISSING fields
        and dataclass inheritance ordering.
        """
        if cls is None:
            return lambda c: c
        return cls
