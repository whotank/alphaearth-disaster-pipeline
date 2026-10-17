from __future__ import annotations

import verify_release


def test_camera_ready_release_artifacts() -> None:
    assert verify_release.main() == 0
