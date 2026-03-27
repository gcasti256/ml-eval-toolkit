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

    # Ensure at least one sample per split when dataset is large enough
    if n >= 3:
        if not train_samples:
            train_samples = [val_samples.pop()] if val_samples else [test_samples.pop()]
        if not test_samples:
            test_samples = [val_samples.pop()] if val_samples else [train_samples.pop()]
        if not val_samples and val_ratio > 0:
            val_samples = [train_samples.pop()]

    def _make_schema(s: list[Sample], suffix: str) -> DatasetSchema:
        if not s:
            # Return a minimal valid dataset if empty
            return DatasetSchema(
                samples=[Sample(input="placeholder", expected_output="placeholder")],
                name=f"{dataset.name}_{suffix}",
            )
        return DatasetSchema(samples=s, name=f"{dataset.name}_{suffix}")

    return _make_schema(train_samples, "train"), _make_schema(test_samples, "test"), _make_schema(val_samples, "val")
