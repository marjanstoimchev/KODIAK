# =============================================================================
# Data Samplers
# =============================================================================
# DINOv3-style samplers for distributed and infinite training.
#
# Sampler types:
#   - EpochSampler: Finite sampler for epoch-based training
#   - InfiniteSampler: Infinite iterator with optional shuffling
#   - ShardedInfiniteSampler: Sharded infinite sampler for distributed training
#   - DistributedSampler: PyTorch's distributed sampler (wrapper)
#
# Usage:
#   from data.samplers import SamplerType, make_sampler
#   sampler = make_sampler(
#       dataset=dataset,
#       sampler_type=SamplerType.SHARDED_INFINITE,
#       shuffle=True,
#       seed=42,
#   )
# =============================================================================

import itertools
import warnings
from enum import Enum
from typing import Any, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, Sampler


# -----------------------------------------------------------------------------
# Distributed utilities
# -----------------------------------------------------------------------------

def get_rank() -> int:
    """Get current process rank in distributed setting."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """Get total number of processes in distributed setting."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


# -----------------------------------------------------------------------------
# Sampler type enumeration
# -----------------------------------------------------------------------------

class SamplerType(Enum):
    """Supported sampler types for training."""
    DISTRIBUTED = 0      # PyTorch DistributedSampler
    EPOCH = 1            # Finite epoch-based sampler
    INFINITE = 2         # Infinite sampler (non-sharded)
    SHARDED_INFINITE = 3 # Sharded infinite sampler
    SHARDED_INFINITE_NEW = 4  # Sharded infinite with new shuffle


# Checkpoints written before Oct 2026 contain SamplerType in the datamodule
# hyper-parameters. Allow-list it so they load under torch.load(weights_only=True).
if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals([SamplerType])


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def _get_numpy_dtype(size: int) -> Any:
    """Get appropriate numpy dtype based on dataset size."""
    return np.int32 if size <= 2**31 else np.int64


def _get_torch_dtype(size: int) -> Any:
    """Get appropriate torch dtype based on dataset size."""
    return torch.int32 if size <= 2**31 else torch.int64


def _generate_randperm_indices(*, size: int, generator: torch.Generator):
    """
    Generate indices of a random permutation.

    This matches PyTorch's CPU implementation for reproducibility.
    See: https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/TensorFactories.cpp
    """
    dtype = _get_torch_dtype(size)
    perm = torch.arange(size, dtype=dtype)
    for i in range(size):
        j = torch.randint(i, size, size=(1,), generator=generator).item()
        # Always swap even if no-op
        value = perm[j].item()
        perm[j] = perm[i].item()
        perm[i] = value
        yield value


def _shuffle_tensor_slice(
    *, tensor: torch.Tensor, start: int = 0, step: int = 1, generator: torch.Generator
) -> np.ndarray:
    """Shuffle a slice of tensor without full permutation."""
    stop = len(tensor)
    count = stop // step
    drop_count = stop - step * count
    if drop_count:
        warnings.warn(f"# of dropped samples: {drop_count}", stacklevel=1)

    dtype = _get_numpy_dtype(stop)
    result = np.empty(count, dtype=dtype)

    for i in range(count):
        j = torch.randint(0, i + 1, size=(1,), generator=generator).item() if i > 0 else 0
        result[i] = result[j]
        result[j] = tensor[start + i * step].item()

    return result


def _new_shuffle_tensor_slice(
    *, tensor: torch.Tensor, start: int = 0, step: int = 1, generator: torch.Generator
) -> np.ndarray:
    """Shuffle tensor slice using randperm (newer, faster implementation)."""
    stop = len(tensor)
    count = stop // step
    dtype = torch.int64
    drop_count = stop - step * count
    if drop_count:
        warnings.warn(f"# of dropped samples: {drop_count}", stacklevel=1)
    indices = torch.randperm(count, dtype=dtype, generator=generator)
    return tensor[start::step][indices].numpy()


def _make_seed(seed: int, start: int, iter_count: int) -> int:
    """Create deterministic seed for reproducibility."""
    return seed + start + (iter_count << 24)


# -----------------------------------------------------------------------------
# Sampler implementations
# -----------------------------------------------------------------------------

class EpochSampler(Sampler):
    """
    Epoch-based sampler for finite training.

    Samples a fixed number of indices per epoch with optional shuffling.
    Supports distributed training by slicing indices based on rank.

    Args:
        size: Total number of samples to draw per epoch.
        sample_count: Size of the underlying dataset.
        shuffle: Whether to shuffle samples each epoch.
        seed: Random seed for reproducibility.
        start: Starting index for distributed slicing (default: rank).
        step: Step size for distributed slicing (default: world_size).
    """

    def __init__(
        self,
        *,
        size: int,
        sample_count: int,
        shuffle: bool = False,
        seed: int = 0,
        start: Optional[int] = None,
        step: Optional[int] = None,
    ):
        self._size = size
        self._sample_count = sample_count
        self._shuffle = shuffle
        self._seed = seed
        self._start = get_rank() if start is None else start
        self._step = get_world_size() if step is None else step
        self._epoch = 0

    def __iter__(self):
        count = (self._size + self._sample_count - 1) // self._sample_count
        tiled_indices = np.tile(np.arange(self._sample_count), count)
        if self._shuffle:
            seed = self._seed * self._epoch if self._seed != 0 else self._epoch
            rng = np.random.default_rng(seed)
            iterable = rng.choice(tiled_indices, self._size, replace=False)
        else:
            iterable = tiled_indices[: self._size]

        yield from itertools.islice(iterable, self._start, None, self._step)

    def __len__(self):
        return (self._size - self._start + self._step - 1) // self._step

    def set_epoch(self, epoch: int):
        """Set the epoch for shuffling reproducibility."""
        self._epoch = epoch


class InfiniteSampler(Sampler):
    """
    Infinite sampler for streaming training.

    Generates an infinite stream of sample indices with optional shuffling.
    Useful for training without explicit epochs.

    Args:
        sample_count: Size of the underlying dataset.
        shuffle: Whether to shuffle samples.
        seed: Random seed for reproducibility.
        start: Starting index for distributed slicing (default: rank).
        step: Step size for distributed slicing (default: world_size).
        advance: Number of samples to skip at the start (for resuming).
    """

    def __init__(
        self,
        *,
        sample_count: int,
        shuffle: bool = False,
        seed: int = 0,
        start: Optional[int] = None,
        step: Optional[int] = None,
        advance: int = 0,
    ):
        self._sample_count = sample_count
        self._seed = seed
        self._shuffle = shuffle
        self._start = get_rank() if start is None else start
        self._step = get_world_size() if step is None else step
        self._advance = advance

    def __iter__(self):
        if self._shuffle:
            iterator = self._shuffled_iterator()
        else:
            iterator = self._iterator()

        yield from itertools.islice(iterator, self._advance, None)

    def _iterator(self):
        assert not self._shuffle
        while True:
            iterable = range(self._sample_count)
            yield from itertools.islice(iterable, self._start, None, self._step)

    def _shuffled_iterator(self):
        assert self._shuffle
        # Instantiate generator here (not in __init__) to keep class picklable
        generator = torch.Generator().manual_seed(self._seed)
        while True:
            iterable = _generate_randperm_indices(
                size=self._sample_count, generator=generator
            )
            yield from itertools.islice(iterable, self._start, None, self._step)


class ShardedInfiniteSampler(Sampler):
    """
    Sharded infinite sampler for efficient distributed training.

    Each worker sees a different shard of the shuffled data, with the
    global shuffle computed once and then distributed. More efficient
    than InfiniteSampler for large-scale distributed training.

    Args:
        sample_count: Size of the underlying dataset.
        shuffle: Whether to shuffle samples.
        seed: Random seed for reproducibility.
        start: Starting index for distributed slicing (default: rank).
        step: Step size for distributed slicing (default: world_size).
        advance: Number of samples to skip at the start (for resuming).
        use_new_shuffle_tensor_slice: Use newer, faster shuffle implementation.
    """

    def __init__(
        self,
        *,
        sample_count: int,
        shuffle: bool = False,
        seed: int = 0,
        start: Optional[int] = None,
        step: Optional[int] = None,
        advance: int = 0,
        use_new_shuffle_tensor_slice: bool = False,
    ):
        self._sample_count = sample_count
        self._seed = seed
        self._shuffle = shuffle
        self._start = get_rank() if start is None else start
        self._step = get_world_size() if step is None else step
        self._advance = advance
        self._iter_count = 0
        self._shuffle_tensor_slice_fn = (
            _new_shuffle_tensor_slice if use_new_shuffle_tensor_slice else _shuffle_tensor_slice
        )

    def __iter__(self):
        iter_count = self._advance // self._sample_count
        if iter_count > 0:
            self._advance -= iter_count * self._sample_count
            self._iter_count += iter_count

        if self._shuffle:
            iterator = self._shuffled_iterator()
        else:
            iterator = self._iterator()

        yield from itertools.islice(iterator, self._advance, None)

    def _iterator(self):
        assert not self._shuffle
        while True:
            iterable = range(self._sample_count)
            yield from itertools.islice(iterable, self._start, None, self._step)

    def _shuffled_iterator(self):
        assert self._shuffle
        # Instantiate generator here (not in __init__) to keep class picklable
        generator = torch.Generator()

        # Always shuffle everything first
        generator.manual_seed(self._seed)
        dtype = _get_torch_dtype(self._sample_count)
        perm = torch.randperm(self._sample_count, dtype=dtype, generator=generator)

        while True:
            # Re-seed on each iteration to allow skipping whole permutations
            seed = _make_seed(self._seed, self._start, self._iter_count)
            generator.manual_seed(seed)

            iterable = self._shuffle_tensor_slice_fn(
                tensor=perm, start=self._start, step=self._step, generator=generator
            )
            yield from iterable
            self._iter_count += 1


# -----------------------------------------------------------------------------
# Sampler factory function
# -----------------------------------------------------------------------------

def make_sampler(
    *,
    dataset: Dataset,
    sampler_type: Optional[SamplerType] = None,
    shuffle: bool = False,
    seed: int = 0,
    size: int = -1,
    advance: int = 0,
) -> Optional[Sampler]:
    """
    Create a sampler of the specified type.

    Factory function for creating training samplers with consistent API.

    Args:
        dataset: The dataset to sample from.
        sampler_type: Type of sampler (DISTRIBUTED, EPOCH, INFINITE, etc.).
        shuffle: Whether to shuffle samples.
        seed: Random seed for reproducibility.
        size: Number of samples per epoch (EPOCH sampler only, -1 for full dataset).
        advance: Number of samples to skip (for resuming, INFINITE samplers only).

    Returns:
        Configured sampler instance, or None if sampler_type is None.

    Example:
        >>> sampler = make_sampler(
        ...     dataset=train_dataset,
        ...     sampler_type=SamplerType.SHARDED_INFINITE,
        ...     shuffle=True,
        ...     seed=42,
        ... )
        >>> dataloader = DataLoader(dataset, sampler=sampler, batch_size=32)
    """
    sample_count = len(dataset)

    if sampler_type == SamplerType.INFINITE:
        if size > 0:
            raise ValueError("sampler size > 0 is invalid for INFINITE sampler")
        return InfiniteSampler(
            sample_count=sample_count,
            shuffle=shuffle,
            seed=seed,
            advance=advance,
        )

    elif sampler_type in (SamplerType.SHARDED_INFINITE, SamplerType.SHARDED_INFINITE_NEW):
        if size > 0:
            raise ValueError("sampler size > 0 is invalid for SHARDED_INFINITE sampler")
        use_new_shuffle = sampler_type == SamplerType.SHARDED_INFINITE_NEW
        return ShardedInfiniteSampler(
            sample_count=sample_count,
            shuffle=shuffle,
            seed=seed,
            advance=advance,
            use_new_shuffle_tensor_slice=use_new_shuffle,
        )

    elif sampler_type == SamplerType.EPOCH:
        if advance > 0:
            raise NotImplementedError("sampler advance > 0 is not supported for EPOCH sampler")
        size = size if size > 0 else sample_count
        return EpochSampler(
            size=size,
            sample_count=sample_count,
            shuffle=shuffle,
            seed=seed,
        )

    elif sampler_type == SamplerType.DISTRIBUTED:
        if size > 0:
            raise ValueError("sampler size > 0 is invalid for DISTRIBUTED sampler")
        if advance > 0:
            raise ValueError("sampler advance > 0 is invalid for DISTRIBUTED sampler")
        # Check if distributed training is actually initialized
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return torch.utils.data.DistributedSampler(
                dataset=dataset,
                shuffle=shuffle,
                seed=seed,
                drop_last=False,
            )
        else:
            # Fallback to EPOCH sampler when distributed not initialized
            print("Warning: Distributed sampler requested but distributed not initialized. Using EPOCH sampler.")
            return EpochSampler(
                size=sample_count,
                sample_count=sample_count,
                shuffle=shuffle,
                seed=seed,
            )

    return None
