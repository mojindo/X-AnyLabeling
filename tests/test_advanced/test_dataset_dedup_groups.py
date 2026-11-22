from anylabeling.views.labeling.advanced.dedup.cluster import build_dedup_groups


def test_near_duplicate_chain_stays_split():
    groups = build_dedup_groups(3, [], [(0, 1), (1, 2)])
    assert sorted(map(tuple, groups)) == [(0, 1), (1, 2)]


def test_exact_copy_inherits_near_duplicate_edge():
    groups = build_dedup_groups(3, [[0, 1]], [(0, 2)])
    assert groups == [[0, 1, 2]]


def test_exact_group_without_neighbors_is_kept():
    groups = build_dedup_groups(4, [[0, 1]], [])
    assert groups == [[0, 1]]