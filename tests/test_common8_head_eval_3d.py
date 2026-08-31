import numpy as np

from tools.eval_common8_orgslot_heads_3d import (
    accumulate_slab,
    overlap_metrics,
    postprocess_volume,
    surface_metrics,
)


def test_overlapping_slabs_are_averaged_by_original_slice_index():
    probability_sum = np.zeros((1, 3, 2, 2), dtype=np.float32)
    counts = np.zeros(3, dtype=np.uint16)
    targets = [None, None, None]
    slice_to_position = {10: 0, 11: 1, 12: 2}
    labels_a = np.zeros((2, 2, 2), dtype=np.uint8)
    labels_b = np.zeros((2, 2, 2), dtype=np.uint8)
    labels_b[1] = 1
    probabilities_a = np.full((1, 2, 2, 2), 0.2, dtype=np.float32)
    probabilities_a[:, 1] = 0.4
    probabilities_b = np.full((1, 2, 2, 2), 0.8, dtype=np.float32)
    probabilities_b[:, 1] = 0.6

    accumulate_slab(
        probability_sum, counts, targets, slice_to_position,
        [10, 11], probabilities_a, labels_a,
    )
    accumulate_slab(
        probability_sum, counts, targets, slice_to_position,
        [11, 12], probabilities_b, labels_b,
    )

    averaged = probability_sum / counts[None, :, None, None]
    assert counts.tolist() == [1, 2, 1]
    assert np.allclose(averaged[0, :, 0, 0], [0.2, 0.6, 0.6])
    assert np.array_equal(targets[2], np.ones((2, 2), dtype=np.uint8))


def test_overlap_metrics_identical_masks_are_one():
    target = np.zeros((3, 4, 5), dtype=bool)
    target[1, 1:3, 2:4] = True
    metrics = overlap_metrics(target, target)
    assert metrics["dice"] == 1.0
    assert metrics["iou"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0


def test_surface_metrics_use_physical_spacing():
    target = np.zeros((5, 7, 7), dtype=bool)
    prediction = np.zeros_like(target)
    target[2, 3, 3] = True
    prediction[3, 3, 3] = True
    metrics = surface_metrics(
        prediction, target, spacing_zyx=(2.0, 0.7, 0.7), tolerance_mm=1.0,
    )
    assert metrics["nsd"] == 0.0
    assert np.isclose(metrics["hd95_mm"], 2.0)


def test_3d_component_filter_is_not_slice_wise():
    prediction = np.zeros((3, 5, 5), dtype=bool)
    prediction[:, 2, 2] = True
    prediction[0, 0, 0] = True
    cleaned = postprocess_volume(
        prediction,
        min_component_voxels=3,
        opening_radius_mm=0.0,
        spacing_zyx=(2.0, 0.7, 0.7),
    )
    assert cleaned[:, 2, 2].all()
    assert not cleaned[0, 0, 0]


if __name__ == "__main__":
    test_overlapping_slabs_are_averaged_by_original_slice_index()
    test_overlap_metrics_identical_masks_are_one()
    test_surface_metrics_use_physical_spacing()
    test_3d_component_filter_is_not_slice_wise()
    print("4 passed")
