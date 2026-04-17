import logging

from src.data.sync import find_best_lidar_for_camera


class FakeNuScenes:
    def __init__(self, sample_data_by_token):
        self.sample_data_by_token = sample_data_by_token

    def get(self, table_name, token):
        assert table_name == "sample_data"
        return self.sample_data_by_token[token]


def test_find_best_lidar_for_camera_accepts_pair_within_threshold():
    nusc = FakeNuScenes(
        {
            "cam-token": {"timestamp": 1_000_000},
            "lidar-token": {"timestamp": 1_040_000},
        }
    )
    sample = {"token": "sample-a", "data": {"CAM_FRONT": "cam-token", "LIDAR_TOP": "lidar-token"}}

    pair = find_best_lidar_for_camera(
        nusc=nusc,
        sample=sample,
        camera_name="CAM_FRONT",
        lidar_name="LIDAR_TOP",
        max_time_diff_ms=50.0,
    )

    assert pair is not None
    assert pair.time_diff_ms == 40.0


def test_find_best_lidar_for_camera_rejects_pair_and_logs_skip(caplog):
    nusc = FakeNuScenes(
        {
            "cam-token": {"timestamp": 1_000_000},
            "lidar-token": {"timestamp": 1_090_000},
        }
    )
    sample = {"token": "sample-b", "data": {"CAM_FRONT": "cam-token", "LIDAR_TOP": "lidar-token"}}

    with caplog.at_level(logging.WARNING):
        pair = find_best_lidar_for_camera(
            nusc=nusc,
            sample=sample,
            camera_name="CAM_FRONT",
            lidar_name="LIDAR_TOP",
            max_time_diff_ms=50.0,
            scene_name="scene-0001",
        )

    assert pair is None
    assert "Skipping camera/lidar pair" in caplog.text
    assert "scene-0001" in caplog.text
