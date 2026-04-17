import numpy as np

from src.utils.geometry import invert_transform, make_transform


def test_invert_transform_round_trip():
    rotation = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    translation = np.array([1.5, -2.0, 0.75], dtype=np.float32)
    transform = make_transform(rotation=rotation, translation=translation)

    inverse = invert_transform(transform)
    identity = transform @ inverse

    np.testing.assert_allclose(identity, np.eye(4, dtype=np.float32), atol=1e-6)
