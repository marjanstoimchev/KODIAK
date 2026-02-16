# callbacks/__init__.py
"""Lightning callbacks for KODIAK training."""

from .progress_bar import KodiakProgressBar
from .timer import TrainingTimer

# Backward compatibility alias
MPPProgressBar = KodiakProgressBar

__all__ = ['KodiakProgressBar', 'MPPProgressBar', 'TrainingTimer']
