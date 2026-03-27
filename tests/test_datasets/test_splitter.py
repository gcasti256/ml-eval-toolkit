"""Tests for dataset splitting."""

import pytest

from ml_eval.datasets.schema import DatasetSchema, Sample
from ml_eval.datasets.splitter import split_dataset


def _make_dataset(n: int) -> DatasetSchema:
    return DatasetSchema(
        samples=[Sample(input=f"Q{i}", expected_output=f"A{i}") for i in range(n)],
        name="split_test",
    )


class TestSplitDataset:
    def test_standard_split(self) -> None:
        ds = _make_dataset(100)
        train, test, val = split_dataset(ds)
        assert len(train) + len(test) + len(val) == 100
        assert len(train) == 70
        assert len(test) == 20
        assert len(val) == 10

    def test_custom_ratios(self) -> None:
        ds = _make_dataset(100)
        train, test, val = split_dataset(ds, train_ratio=0.8, test_ratio=0.1, val_ratio=0.1)
        assert len(train) == 80
        assert len(test) == 10
        assert len(val) == 10

    def test_no_validation_split(self) -> None:
        ds = _make_dataset(20)
        train, test, _val = split_dataset(ds, train_ratio=0.8, test_ratio=0.2, val_ratio=0.0)
        # val_ratio=0 returns a placeholder; all real samples go to train+test
        assert len(train) + len(test) == 20

    def test_deterministic_with_seed(self) -> None:
        ds = _make_dataset(50)
        train1, _, _ = split_dataset(ds, seed=42)
        train2, _, _ = split_dataset(ds, seed=42)
        assert [s.input for s in train1.samples] == [s.input for s in train2.samples]

    def test_different_seeds_differ(self) -> None:
        ds = _make_dataset(50)
        train1, _, _ = split_dataset(ds, seed=42)
        train2, _, _ = split_dataset(ds, seed=99)
        assert [s.input for s in train1.samples] != [s.input for s in train2.samples]

    def test_small_dataset(self) -> None:
        ds = _make_dataset(3)
        train, test, val = split_dataset(ds)
        total = len(train) + len(test) + len(val)
        assert total == 3

    def test_ratios_must_sum_to_one(self) -> None:
        ds = _make_dataset(10)
        with pytest.raises(ValueError, match=r"sum to 1\.0"):
            split_dataset(ds, train_ratio=0.5, test_ratio=0.5, val_ratio=0.5)

    def test_negative_ratio_rejected(self) -> None:
        ds = _make_dataset(10)
        with pytest.raises(ValueError, match="non-negative"):
            split_dataset(ds, train_ratio=-0.1, test_ratio=0.6, val_ratio=0.5)

    def test_preserves_sample_content(self) -> None:
        ds = _make_dataset(10)
        train, test, val = split_dataset(ds)
        all_inputs = set()
        for split in (train, test, val):
            for s in split.samples:
                all_inputs.add(s.input)
        expected = {f"Q{i}" for i in range(10)}
        assert all_inputs == expected

    def test_naming(self) -> None:
        ds = _make_dataset(10)
        train, test, val = split_dataset(ds)
        assert train.name == "split_test_train"
        assert test.name == "split_test_test"
        assert val.name == "split_test_val"
