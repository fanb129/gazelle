from scripts.audit_gazefollow_query_groups import query_statistics


def _heads(count, *, inframe=True):
    return [
        {"inout": 1 if inframe else 0}
        for _ in range(count)
    ]


def test_query_statistics_reports_multi_head_distribution():
    records = [
        {"path": "a.jpg", "heads": _heads(1)},
        {"path": "b.jpg", "heads": _heads(2)},
        {
            "path": "c.jpg",
            "heads": _heads(3) + _heads(1, inframe=False),
        },
    ]

    stats = query_statistics(records, (0, 1, 2))

    assert stats["record_count"] == 3
    assert stats["total_head_count"] == 7
    assert stats["total_inframe_head_count"] == 6
    assert stats["single_head_record_count"] == 1
    assert stats["multi_head_record_count"] == 2
    assert stats["multi_head_person_count"] == 5
    assert stats["max_inframe_heads_per_image"] == 3
    assert stats["inframe_head_count_histogram"] == {"1": 1, "2": 1, "3": 1}
