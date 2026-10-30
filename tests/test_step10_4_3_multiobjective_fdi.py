import numpy as np

from aeris.fdi.multiobjective import (
    fuse_scores,
    pareto_frontier,
    reject_unknown,
    route_domain,
    threshold_summary,
)


def test_pareto_frontier_keeps_non_dominated_rows():
    rows = [
        {"f1": 0.8, "fpr": 0.05},
        {"f1": 0.7, "fpr": 0.10},
        {"f1": 0.9, "fpr": 0.20},
    ]
    assert pareto_frontier(rows, {"f1"}, {"fpr"}) == [0, 2]


def test_threshold_routing_fusion_and_rejection_are_deterministic():
    summary = threshold_summary(
        np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.7, 0.9]), np.array([0.5])
    )
    assert summary[0]["recall"] == 1.0
    assert route_domain(0.01) == "KNOWN_OPERATING_ENVELOPE"
    assert route_domain(0.10) == "SHIFTED_OPERATING_ENVELOPE"
    assert route_domain(0.30) == "UNKNOWN_AMBIGUOUS_ENVELOPE"
    np.testing.assert_allclose(fuse_scores(np.array([0.2, 0.4]), np.array([0.4, 0.6])), [0.3, 0.5])
    np.testing.assert_array_equal(reject_unknown(np.array([0.2, 0.8]), 0.5), [True, False])
