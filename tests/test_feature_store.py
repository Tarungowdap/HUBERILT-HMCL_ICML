import json

import h5py
import numpy as np

from data_utils.features import FeatureStore


def _write_h5(path, features, identifiers, curv=None, id_key="labels"):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("features", data=np.asarray(features, dtype=np.float32))
        handle.create_dataset(id_key, data=np.asarray(identifiers, dtype=np.int64))
        if curv is not None:
            handle.attrs["curv"] = curv


def test_load_classification(tmp_path):
    dataset_dir = tmp_path / "toy"
    _write_h5(dataset_dir / "toy_train_image.h5", [[1, 2], [3, 4]], [0, 1])
    _write_h5(dataset_dir / "toy_train_text.h5", [[5, 6], [7, 8]], [0, 1])

    split = FeatureStore(tmp_path).load_classification("toy", "train")
    assert split["features"].shape == (2, 2)
    assert split["text_features"].shape == (2, 2)
    assert split["labels"].tolist() == [0, 1]


def test_load_retrieval(tmp_path):
    dataset_dir = tmp_path / "coco"
    _write_h5(
        dataset_dir / "coco_test_image.h5", [[1, 2]], [10], curv=0.2, id_key="ids"
    )
    _write_h5(
        dataset_dir / "coco_test_text.h5",
        [[3, 4], [5, 6]],
        [20, 21],
        curv=0.2,
        id_key="ids",
    )
    ground_truth = {
        "image_to_text_gt": {"10": [20, 21]},
        "text_to_image_gt": {"20": [10], "21": [10]},
    }
    (dataset_dir / "coco_test_retrieval_gt.json").write_text(json.dumps(ground_truth))

    split = FeatureStore(tmp_path).load_retrieval("coco", "test")
    assert split["curv"] == 0.2
    assert split["image_to_text_gt"][10] == {20, 21}
    assert split["text_to_image_gt"][20] == {10}
