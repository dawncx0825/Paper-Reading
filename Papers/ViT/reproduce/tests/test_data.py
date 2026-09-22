from collections import Counter

from vit_repro.data import stratified_dev_indices


def test_stratified_split_contains_ten_per_class():
    targets = [class_id for class_id in range(100) for _ in range(50)]
    indices = stratified_dev_indices(targets, seed=42, per_class=10)
    assert len(indices) == 1000
    assert len(set(indices)) == 1000
    counts = Counter(targets[index] for index in indices)
    assert set(counts.values()) == {10}


def test_stratified_split_is_deterministic():
    targets = [class_id for class_id in range(4) for _ in range(12)]
    assert stratified_dev_indices(targets, seed=7, per_class=3) == stratified_dev_indices(
        targets, seed=7, per_class=3
    )

