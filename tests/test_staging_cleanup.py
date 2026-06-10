import os
import time

from apps.streamlit_translate import sweep_stale_staging


def test_sweep_removes_only_stale_dirs(tmp_path):
    old_dir = tmp_path / "old_session"
    old_dir.mkdir()
    (old_dir / "recording.wav").write_bytes(b"x" * 64)
    stale_time = time.time() - 48 * 3600
    os.utime(old_dir, (stale_time, stale_time))

    fresh_dir = tmp_path / "fresh_session"
    fresh_dir.mkdir()
    (fresh_dir / "recording.wav").write_bytes(b"x" * 64)

    removed = sweep_stale_staging(root=tmp_path, max_age_seconds=24 * 3600)

    assert removed == 1
    assert not old_dir.exists()
    assert fresh_dir.exists()


def test_sweep_handles_missing_root(tmp_path):
    assert sweep_stale_staging(root=tmp_path / "does_not_exist") == 0
