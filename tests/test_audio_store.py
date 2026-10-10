import numpy as np
import pytest

from kata.audio.store import ARRAYS, CURVE_COLUMNS, FEATURE_VERSION, SILENCE, FeatureStore


def arrays(T, v=1.0):
    return {"mel": np.full((T, 128), v), "stem_mel": np.full((T, 4, 64), v),
            "chroma": np.full((T, 12), v), "curves": np.full((T, 12), v)}


def test_write_load_round_trip(tmp_path):
    st = FeatureStore(tmp_path)
    st.open_for_write({"a": 1})
    st.write("abcdef", arrays(300, 0.5), [0.5, 1.0], [0.5])
    assert st.has("abcdef") and not st.has("other")
    assert st.song_dir("abcdef") == tmp_path / FEATURE_VERSION / "ab" / "abcdef"
    mel = st.load("abcdef", "mel")
    assert isinstance(mel, np.memmap) and mel.dtype == np.float16 and mel.shape == (300, 128)
    b, d = st.events("abcdef")
    np.testing.assert_array_equal(b, [0.5, 1.0])
    np.testing.assert_array_equal(d, [0.5])
    assert not list(st.song_dir("abcdef").glob("*.tmp"))


def test_frame_count_mismatch_is_rejected(tmp_path):
    a = arrays(300)
    a["chroma"] = np.zeros((299, 12))
    with pytest.raises(ValueError, match="frame count"):
        FeatureStore(tmp_path).write("abcdef", a, [], [])


def test_config_is_pinned_per_version(tmp_path):
    FeatureStore(tmp_path).open_for_write({"fps": 100})
    FeatureStore(tmp_path).open_for_write({"fps": 100})  # same config: fine
    with pytest.raises(ValueError, match="bump FEATURE_VERSION"):
        FeatureStore(tmp_path).open_for_write({"fps": 50})


def test_window_pads_with_silence_rows(tmp_path):
    st = FeatureStore(tmp_path)
    st.write("abcdef", arrays(300, 1.0), [], [])  # 3 s
    w = st.window("abcdef", 2.5, 3.5)  # 100 frames, 50 inside the song
    assert {k: v.shape[0] for k, v in w.items()} == {k: 100 for k in ARRAYS}
    assert (w["mel"][:50] == 1).all() and (w["mel"][50:] == -100).all()
    np.testing.assert_array_equal(w["curves"][-1], SILENCE["curves"])
    assert (st.window("abcdef", -0.5, 0.5)["chroma"][:50] == 0).all()


def test_curve_silence_row_matches_the_columns():
    assert len(SILENCE["curves"]) == len(CURVE_COLUMNS)
    assert all((v == -100) == c.startswith("rms_") for v, c in zip(SILENCE["curves"], CURVE_COLUMNS))


def test_window_length_does_not_depend_on_the_start_time(tmp_path):
    st = FeatureStore(tmp_path)
    st.write("abcdef", arrays(3000), [], [])
    lengths = {st.window("abcdef", s, s + 12.0)["mel"].shape[0] for s in np.arange(0.0, 10.0, 0.0037)}
    assert lengths == {1200}
