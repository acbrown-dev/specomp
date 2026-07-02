import json

import numpy as np

_DTYPE_SHAPE_TAG = "dtype_shape"


def _encode_side(side):
    if side is None:
        return None
    if (
        isinstance(side, tuple)
        and len(side) == 2
        and isinstance(side[0], np.dtype)
        and isinstance(side[1], tuple)
    ):
        return {
            _DTYPE_SHAPE_TAG: True,
            "dtype": np.dtype(side[0]).name,
            "shape": list(side[1]),
        }
    raise TypeError(f"Unsupported side type for serialization: {type(side)!r}")


def _decode_side(value):
    if value is None:
        return None
    if isinstance(value, dict) and value.get(_DTYPE_SHAPE_TAG):
        return np.dtype(value["dtype"]), tuple(value["shape"])
    raise TypeError(f"Unsupported serialized side value: {value!r}")


def serialize_sides(sides: list) -> str:
    return json.dumps([_encode_side(side) for side in sides])


def deserialize_sides(serialized: str) -> list:
    return [_decode_side(value) for value in json.loads(serialized)]
