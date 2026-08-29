import numpy as np
import json
from sklearn.model_selection import train_test_split

from src.config import SplitConfig, split_dir
from src.utils import ensure_dir


def create_split_indices(y: np.ndarray, config: SplitConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_indices = np.arange(len(y))

    development_indices, test_indices = train_test_split(all_indices,
                                                         test_size=config.test_fraction,
                                                         random_state=config.split_seed[0],
                                                         stratify=y)

    relative_valid_fraction = config.validation_fraction/(1-config.test_fraction)

    train_indices, validation_indices = train_test_split(development_indices,
                                                         test_size=relative_valid_fraction,
                                                         random_state=config.split_seed[1],
                                                         stratify=y[development_indices])

    return train_indices, validation_indices, test_indices


def create_eval_split_indices(y: np.ndarray, config: SplitConfig) -> tuple[np.ndarray, np.ndarray]:
    all_indices = np.arange(len(y))

    train_indices, test_indices = train_test_split(all_indices, test_size=config.test_fraction,
                                                   random_state=config.split_seed[0], stratify=y)

    return train_indices, test_indices


def save_split(y: np.ndarray, train_indices: np.ndarray, validation_indices: np.ndarray,
               test_indices: np.ndarray, config: SplitConfig) -> None:

    validate_split(y, train_indices, validation_indices, test_indices)

    print("Begin saving split data...")
    ensure_dir(split_dir)

    indices_path = split_dir/f"{config.split_id}.npz"
    metadata_path = split_dir/f"{config.split_id}_metadata.json"

    if indices_path.exists() or metadata_path.exists():
        print(f"The '{config.split_id}' split already exists and will not be overwritten.")
        return

    np.savez_compressed(str(indices_path),
                        train_indices=train_indices,
                        validation_indices=validation_indices,
                        test_indices=test_indices)

    def class_counts(indices):
        labels, counts = np.unique(y[indices], return_counts=True)
        return {
            str(int(label)): int(count)
            for label, count in zip(labels, counts)
        }

    metadata = {
        "split_id": config.split_id,
        "strategy": "stratified_random",
        "split_seed": config.split_seed,
        "test_fraction": config.test_fraction,
        "validation_fraction": config.validation_fraction,
        "sample_counts": {
            "train": len(train_indices),
            "validation": len(validation_indices),
            "test": len(test_indices)
        },
        "class_counts": {
            "train": class_counts(train_indices),
            "validation": class_counts(validation_indices),
            "test": class_counts(test_indices)
        }
    }

    print("Split was saved successfully.")

    with metadata_path.open("w", encoding="utf-8") as json_file:
        json.dump(metadata, json_file, indent=2)


def save_eval_split(y: np.ndarray, train_indices: np.ndarray, test_indices: np.ndarray, config: SplitConfig) -> None:

    validate_eval_split(y, train_indices, test_indices)

    print("Begin saving the final evaluation split data...")
    ensure_dir(split_dir)

    indices_path = split_dir/f"{config.split_id}.npz"
    metadata_path = split_dir/f"{config.split_id}_metadata.json"

    if indices_path.exists() or metadata_path.exists():
        print(f"The '{config.split_id}' split already exists and will not be overwritten.")
        return

    np.savez_compressed(str(indices_path),
                        train_indices=train_indices,
                        test_indices=test_indices)

    def class_counts(indices):
        labels, counts = np.unique(y[indices], return_counts=True)
        return {
            str(int(label)): int(count)
            for label, count in zip(labels, counts)
        }

    metadata = {
        "split_id": config.split_id,
        "strategy": "stratified_random",
        "split_seed": config.split_seed,
        "test_fraction": config.test_fraction,
        "validation_fraction": config.validation_fraction,
        "sample_counts": {
            "train": len(train_indices),
            "test": len(test_indices)
        },
        "class_counts": {
            "train": class_counts(train_indices),
            "test": class_counts(test_indices)
        }
    }

    print("Final evaluation split was saved successfully.")

    with metadata_path.open("w", encoding="utf-8") as json_file:
        json.dump(metadata, json_file, indent=2)


def load_split(split_id: str, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    print("Loading saved split.")
    indices_path = split_dir/f"{split_id}.npz"

    if not indices_path.is_file():
        raise FileNotFoundError(
            f"The file at '{indices_path}' was not found."
        )

    expected_arrays = {
        "train_indices",
        "validation_indices",
        "test_indices"
    }

    with np.load(str(indices_path)) as split:
        missing_arrays = expected_arrays.difference(split.files)  # What's missing in the loaded data?

        if missing_arrays:
            raise ValueError(
                f"Split is missing the following arrays: {missing_arrays}"
            )

        train_indices = split["train_indices"]
        validation_indices = split["validation_indices"]
        test_indices = split["test_indices"]

    print("Split loaded successfully.")

    validate_split(y, train_indices, validation_indices, test_indices)

    return train_indices, validation_indices, test_indices


def load_eval_split(split_id: str, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    print("Loading saved final evaluation split.")
    indices_path = split_dir/f"{split_id}.npz"

    if not indices_path.is_file():
        raise FileNotFoundError(
            f"The file at '{indices_path}' was not found."
        )

    expected_arrays = {
        "train_indices",
        "test_indices"
    }

    with np.load(str(indices_path)) as split:
        missing_arrays = expected_arrays.difference(split.files)  # What's missing in the loaded data?

        if missing_arrays:
            raise ValueError(
                f"Split is missing the following arrays: {missing_arrays}"
            )

        train_indices = split["train_indices"]
        test_indices = split["test_indices"]

    print("Final evaluation split loaded successfully.")

    validate_eval_split(y, train_indices, test_indices)

    return train_indices, test_indices


def validate_split(y: np.ndarray,
                   train_indices: np.ndarray,
                   validation_indices: np.ndarray,
                   test_indices: np.ndarray) -> None:
    print("Begin validation of the split...")
    if y.ndim != 1:
        raise ValueError(
            f"'y' must be a one-dimensional array, but its shape is {y.shape}."
        )

    sample_count = y.shape[0]

    splits = {
        "train": train_indices,
        "validation": validation_indices,
        "test": test_indices
    }

    for split_name, indices in splits.items():
        if not isinstance(indices, np.ndarray):
            raise TypeError(
                f"{split_name}_indices must be a Numpy array."
            )

        if indices.ndim != 1:
            raise ValueError(
                f"{split_name}_indices must be a one-dimensional Numpy array."
            )

        if indices.size == 0:
            raise ValueError(
                f"{split_name}_indices cannot be empty."
            )

        if not np.issubdtype(indices.dtype, np.integer):
            raise ValueError(
                f"{split_name}_indices must be integer values."
            )

        if np.any(indices < 0) or np.any(indices >= sample_count):
            raise ValueError(
                f"{split_name}_indices must be within the [0 to {sample_count-1}] range."
            )

        if np.unique(indices).size != indices.size:
            raise ValueError(
                f"{split_name}_indices must not contain duplicate values."
            )

    split_pairs = [
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test")
    ]

    # Verify that none of the sets overlap.
    for first_set, second_set in split_pairs:
        overlap = np.intersect1d(splits[first_set], splits[second_set], assume_unique=True)  # AU=True -> faster exec
        if overlap.size > 0:
            raise ValueError(
                f"The {first_set} and {second_set} contain overlapping indices: {overlap}."
            )

    all_indices = np.concatenate(list(splits.values()))
    expected_indices = np.arange(sample_count)

    # Verify that the loaded indices are the same as the expected indices.
    if not np.array_equal(np.sort(all_indices), expected_indices):
        missing_indices = np.setdiff1d(expected_indices, all_indices)  # returns items that are in the
        raise ValueError(                                              # first array but not in the second one
            "Unexpected index values."
            f"Missing index values found: {missing_indices}."
        )

    # Verify that every split contains all the expected classes.
    expected_classes = np.unique(y)

    for split_name, indices in splits.items():
        split_classes = np.unique(y[indices])
        missing_classes = np.setdiff1d(expected_classes, split_classes)

        if missing_classes.size > 0:
            raise ValueError(
                f"The {split_name} split is missing classes: {missing_classes.tolist()}."
            )

    print("Validation of the split is completed successfully.")


def validate_eval_split(y: np.ndarray,
                        train_indices: np.ndarray,
                        test_indices: np.ndarray) -> None:
    print("Begin validation of the final evaluation split...")
    if y.ndim != 1:
        raise ValueError(
            f"'y' must be a one-dimensional array, but its shape is {y.shape}."
        )

    sample_count = y.shape[0]

    splits = {
        "train": train_indices,
        "test": test_indices
    }

    for split_name, indices in splits.items():
        if not isinstance(indices, np.ndarray):
            raise TypeError(
                f"{split_name}_indices must be a Numpy array."
            )

        if indices.ndim != 1:
            raise ValueError(
                f"{split_name}_indices must be a one-dimensional Numpy array."
            )

        if indices.size == 0:
            raise ValueError(
                f"{split_name}_indices cannot be empty."
            )

        if not np.issubdtype(indices.dtype, np.integer):
            raise ValueError(
                f"{split_name}_indices must be integer values."
            )

        if np.any(indices < 0) or np.any(indices >= sample_count):
            raise ValueError(
                f"{split_name}_indices must be within the [0 to {sample_count-1}] range."
            )

        if np.unique(indices).size != indices.size:
            raise ValueError(
                f"{split_name}_indices must not contain duplicate values."
            )

    # Verify that none of the sets overlap.

    overlap = np.intersect1d(splits["train"], splits["test"], assume_unique=True)  # AU=True -> faster exec
    if overlap.size > 0:
        raise ValueError(
            f"The train and test sets contain overlapping indices: {overlap}."
        )

    all_indices = np.concatenate(list(splits.values()))
    expected_indices = np.arange(sample_count)

    # Verify that the loaded indices are the same as the expected indices.
    if not np.array_equal(np.sort(all_indices), expected_indices):
        missing_indices = np.setdiff1d(expected_indices, all_indices)  # returns items that are in the
        raise ValueError(                                              # first array but not in the second one
            "Unexpected index values."
            f"Missing index values found: {missing_indices}."
        )

    # Verify that every split contains all the expected classes.
    expected_classes = np.unique(y)

    for split_name, indices in splits.items():
        split_classes = np.unique(y[indices])
        missing_classes = np.setdiff1d(expected_classes, split_classes)

        if missing_classes.size > 0:
            raise ValueError(
                f"The {split_name} split is missing classes: {missing_classes.tolist()}."
            )

    print("Validation of the final evaluation split is completed successfully.")