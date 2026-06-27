import numpy as np
import pytest

import specomp.steps  # noqa: F401
from specomp.abstract.steps import Step, LossyStep


def _round_trip_cases():
    for cls in Step.subclass_registry.values():
        for input_type in cls.domain:
            yield cls, input_type


@pytest.mark.parametrize(
    "cls,input_type",
    _round_trip_cases(),
    ids=lambda v: v.__name__,
)
def test_round_trip_on_example_data(cls, input_type):
    s = cls()
    arr = input_type.generate_example()
    assert input_type.validate(arr)

    x, side = s.forward(arr)
    out = s.inverse(x, side)

    assert input_type.validate(out)
    if issubclass(cls, LossyStep):
        assert np.max(np.abs(out.astype(float) - arr)) <= s.error_bound
    else:
        np.testing.assert_array_equal(out, arr)