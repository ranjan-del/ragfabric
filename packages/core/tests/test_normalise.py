import math

from ragfabric_core.embeddings.normalise import normalise, normalise_all


def test_normalise_returns_a_unit_vector():
    out = normalise([3.0, 4.0])
    assert math.isclose(math.sqrt(sum(v * v for v in out)), 1.0, rel_tol=1e-9)
    assert math.isclose(out[0], 0.6, rel_tol=1e-9)
    assert math.isclose(out[1], 0.8, rel_tol=1e-9)


def test_normalise_leaves_a_zero_vector_alone_rather_than_dividing_by_zero():
    assert normalise([0.0, 0.0, 0.0]) == [0.0, 0.0, 0.0]


def test_normalise_all_maps_every_row():
    out = normalise_all([[3.0, 4.0], [0.0, 5.0]])
    assert math.isclose(out[1][1], 1.0, rel_tol=1e-9)
    assert len(out) == 2
