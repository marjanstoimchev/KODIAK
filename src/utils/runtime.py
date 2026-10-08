"""Runtime helpers shared by the entry-point scripts."""

from typing import Any, List, Tuple, Union

import torch


def resolve_accelerator_and_devices(devices: Union[str, int, List[int], None]) -> Tuple[str, Any]:
    """Translate a config/CLI ``devices`` value into Lightning ``(accelerator, devices)``.

    Configs list GPU indices (e.g. ``[0, 1]``). When CUDA is not available the
    same list would make Lightning's CPU accelerator fail ("devices ... should be
    an int > 0"), so on CPU we fall back to a single process.

    Args:
        devices: ``"auto"``, an int, a list of GPU indices, or ``None``.

    Returns:
        ``(accelerator, devices)`` ready to pass to ``pl.Trainer``.
    """
    if torch.cuda.is_available():
        if devices is None:
            return "gpu", "auto"
        return "gpu", devices

    # CPU: a list of indices is meaningless; use a single process.
    if devices in (None, "auto") or isinstance(devices, (list, tuple)):
        return "cpu", 1
    return "cpu", max(1, int(devices))


def count_devices(devices: Union[str, int, List[int], None]) -> int:
    """Number of training processes implied by ``devices`` (used to split the global batch size)."""
    if isinstance(devices, (list, tuple)):
        return max(1, len(devices))
    if devices in (None, "auto"):
        return torch.cuda.device_count() if torch.cuda.is_available() else 1
    return max(1, int(devices))
