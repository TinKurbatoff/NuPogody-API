"""Make BindsNET friends with modern PyTorch.

BindsNET 0.2.7 (the latest on PyPI) still imports ``torch._six`` — a module
PyTorch removed back in 1.9. That single reference is the *only* thing that
breaks the import on torch 2.x; with it shimmed, ``MSTDPET`` reward-modulated
learning runs unchanged (verified on torch 2.12). So instead of pinning an
ancient torch (which would drag in an ancient Python), we install a tiny shim
providing the three names BindsNET's ``datasets/collate.py`` looks for.

Import this module *before* importing anything from ``bindsnet``. It is
idempotent and a no-op on any torch that still ships ``torch._six``.
"""

from __future__ import annotations

import collections.abc
import sys
import types


def install() -> None:
    """Register a stand-in ``torch._six`` if PyTorch no longer ships one."""
    import torch

    if getattr(torch, "_six", None) is not None and "torch._six" in sys.modules:
        return
    try:  # a torch old enough to still have it — nothing to do.
        import torch._six  # noqa: F401

        return
    except ImportError:
        pass

    six = types.ModuleType("torch._six")
    six.container_abcs = collections.abc
    six.string_classes = (str, bytes)
    six.int_classes = (int,)
    sys.modules["torch._six"] = six
    torch._six = six  # type: ignore[attr-defined]


install()
