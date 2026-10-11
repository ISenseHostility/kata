import json

import polars as pl
import pytest

from kata.validate.reference import (Reference, ValidatorDir, _anchors, merge_bands, stratum_key,
                                     validator_config)
from kata.validate.rules import ERROR_RULES, LOAD_RULES


def test_merge_bands_merges_the_smallest_cell_into_its_smaller_neighbour():
    assert merge_bands({0: 50, 1: 200, 2: 30, 3: 150}) == [[0, 1, 250], [2, 3, 180]]
    assert merge_bands({1: 40, 3: 30}) == [[1, 3, 70]]           # gaps count as empty bands
    assert merge_bands({2: 500}) == [[2, 2, 500]]


def test_stratum_key_clamps_to_the_outer_cells():
    cells = [[0, 1, 250], [2, 3, 180]]
    assert stratum_key("Hard", cells, 3.9) == "Hard|0-4"
    assert stratum_key("Hard", cells, 4.0) == "Hard|4-8"
    assert stratum_key("Hard", cells, 30.0) == "Hard|4-8"
    assert stratum_key("Hard", [[0, 5, 80]], 3.0) == "Hard|all"


def test_error_anchors_keep_a_minimum_spacing():
    assert _anchors([0.0] * 50, error=True) == {"q90": 0.2, "q99": 1.2, "zero": 3.2}
    a = _anchors(list(range(101)), error=False)
    assert a["q90"] == pytest.approx(90) and a["q99"] == pytest.approx(99) and a["zero"] == pytest.approx(117)


def _metrics(n=400):
    rows = []
    for i in range(n):
        label = "Expert" if i % 2 else "Hard"
        rows.append({"difficulty": label, "nps": 3.0 + (i % 8), "njs": 16.0, "reaction_time_s": 0.6,
                     "dt_p5": 0.15, **{r: float(i % 10) / 10 for r in ERROR_RULES + LOAD_RULES}})
    return pl.DataFrame(rows)


def test_reference_build_has_strata_anchors_and_baseline_targets():
    ref = Reference.build(_metrics())
    assert set(ref.strata) == {"Easy", "Normal", "Hard", "Expert", "ExpertPlus"}
    assert all(sum(c[2] for c in cells) >= 100 or len(cells) == 1 for cells in ref.strata.values())
    s = ref.stratum("Expert", 6.0)
    assert set(ref.anchors("L1", s)) == {"q90", "q99", "zero"}
    assert ref.anchors("E1", "anything") == ref.data["rules"]["E1"]["pooled"]
    assert ref.baseline["njs"]["Expert"] == 16.0 and ref.baseline["min_gap_s"][s] == 0.15
    assert Reference(json.loads(json.dumps(ref.data))).anchors("L1", s) == ref.anchors("L1", s)


def test_validator_dir_pins_its_config(tmp_path):
    vd = ValidatorDir(tmp_path)
    vd.open_for_write()
    vd.open_for_write()                                   # same config: fine
    changed = {**validator_config(), "overlap_m": 0.4}
    with pytest.raises(ValueError, match="bump VALIDATOR_VERSION"):
        vd.open_for_write(changed)


def test_config_pins_every_score_changing_parameter():
    c = validator_config()
    assert c["rules"] == {"hard": ["H1", "H2", "H3"], "error": ["E1", "E2", "E3"], "load": ["L1", "L2", "L3"]}
    assert c["spread_deg"] == {"H3": 90.0, "E3": 45.0}
    assert c["score_floor"] == 0.01 and c["load_anchor_ratio"] == 1.05
    assert c["swing"]["backhand_prior"] == 0.1


def test_loading_the_reference_refuses_a_changed_config(tmp_path, toy_ref):
    vd = ValidatorDir(tmp_path)
    vd.open_for_write()
    vd.write_reference(toy_ref)
    assert vd.load_reference().data == toy_ref.data
    vd.config_path.write_text('{"version": "v1", "something": "else"}')
    with pytest.raises(ValueError, match="bump VALIDATOR_VERSION"):
        vd.load_reference()
