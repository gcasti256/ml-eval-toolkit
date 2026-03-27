"""Dataset splitting utilities."""

from __future__ import annotations

import random

from ml_eval.datasets.schema import DatasetSchema, Sample


def split_dataset(
    dataset: DatasetSchema,
    train_ratio: float = 0.7,
    test_ratio: float = 0.2,
    val_ratio: float = 0.1,
    seed: int | None = 42,
) -> tuple[DatasetSchema, DatasetSchema, DatasetSchema]:
    """Split a dataset into train, test, and validation sets.

    Args:
        dataset: The dataset to split.
        train_ratio: Proportion for training set.
        test_ratio: Proportion for test set.
        val_ratio: Proportion for validation set.
        seed: Random seed for reproducibility. None for non-deterministic.

    Returns:
        Tuple of (train, test, validation) DatasetSchema instances.

    Raises:
        ValueError: If ratios don't sum to ~1.0 or are invalid.
    """
    total = train_ratio + test_ratio + val_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {total:.4f}")
    if any(r < 0 for r in (train_ratio, test_ratio, val_ratio)):
        raise ValueError("All ratios must be non-negative")

    samples = list(dataset.samples)
    if seed is not None:
        rng = random.Random(seed)
        rng.shuffle(samples)
    else:
        random.shuffle(samples)

    n = len(samples)
    train_end = int(n * train_ratio)
    test_end = train_end + int(n * test_ratio)

    train_samples = samples[:train_end]
    test_samples = samples[train_end:test_end]
    val_samples = samples[test_end:]

    # Redistribute so every non-zero-ratio split has at least one sample
    splits = [train_samples, test_samples, val_samples]
    ratios = [train_ratio, test_ratio, val_ratio]
    for i, (_split, ratio) in enumerate(zip(splits, ratios, strict=True)):
        if ratio > 0 and not splits[i]:
            donor = max(range(3), key=lambda j: len(splits[j]))
            if splits[donor]:
                splits[i].append(splits[donor].pop())

    def _make_schema(s: list[Sample], ratio: float, suffix: str) -> DatasetSchema:
        if not s:
            if ratio == 0.0:
                # Zero-ratio split: return a single-sample schema as a no-op placeholder
                return DatasetSchema(samples=[splits[0][0]], name=f"{dataset.name}_{suffix}")
            raise ValueError(f"Cannot create {suffix} split: no samples (ratio may be too small)")
        return DatasetSchema(samples=s, name=f"{dataset.name}_{suffix}")

    return (
        _make_schema(splits[0], ratios[0], "train"),
        _make_schema(splits[1], ratios[1], "test"),
        _make_schema(splits[2], ratios[2], "val"),
    )
