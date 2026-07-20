import pytest

from AAAICOTB.geometry import bbox_iou, cluster_targets


def test_single_link_clustering_preserves_shared_target_chain():
    labels, centers = cluster_targets([(0.10, 0.10), (0.14, 0.10), (0.18, 0.10), (0.80, 0.80)], 0.05)
    assert labels == [0, 0, 0, 1]
    assert centers[0] == pytest.approx((0.14, 0.10))
    assert len(centers) == 2


def test_bbox_iou_detects_duplicate_boxes():
    assert bbox_iou([0.1, 0.1, 0.3, 0.3], [0.1, 0.1, 0.3, 0.3]) == 1.0
    assert bbox_iou([0.1, 0.1, 0.2, 0.2], [0.8, 0.8, 0.9, 0.9]) == 0.0


def test_empty_target_list_is_supported():
    labels, centers = cluster_targets([], 0.06)
    assert labels == []
    assert centers == []
