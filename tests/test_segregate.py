import numpy as np

from src.config import load_config
from src.detector import Detection
from src.segregate import (ACCUMULATION, DUMP, OVERFLOW, analyse_frame, annotate_frame,
                           average_composition, dominant, parse_composition, shares, summary_text)

SHAPE = (120, 160, 3)           # h, w, c  -> frame area 19200 px
CFG = load_config()
CFG["pipeline"].update(flag_conf=0.4, min_items=3, min_area=0.5)   # area rule effectively off


def det(label, conf=0.9, box=(0, 0, 20, 20)):
    return Detection(label, conf, box)


def test_below_min_items_is_not_flagged():
    info = analyse_frame([det("plastic"), det("metal")], CFG, SHAPE)
    assert not info.flagged and info.count == 2 and info.area == 0.0


def test_enough_items_flags_and_reports_materials():
    dets = [det("plastic"), det("plastic"), det("metal")]
    info = analyse_frame(dets, CFG, SHAPE)
    assert info.flagged and info.label == ACCUMULATION
    assert info.counts == {"plastic": 2, "metal": 1}
    assert abs(info.comp["plastic"] - 2 * info.comp["metal"]) < 1e-9


def test_big_area_flags_even_with_one_item():
    cfg = load_config()
    cfg["pipeline"].update(flag_conf=0.4, min_items=10, min_area=0.1)
    info = analyse_frame([det("glass", box=(0, 0, 100, 100))], cfg, SHAPE)
    assert info.flagged and info.area > 0.1


def test_direct_class_flags_alone_and_sets_label():
    info = analyse_frame([det("bin_overflowing", box=(0, 0, 80, 60))], CFG, SHAPE)
    assert info.flagged and info.label == "bin_overflowing"


def test_low_confidence_and_unknown_classes_ignored():
    dets = [det("plastic", conf=0.2), det("plastic", conf=0.2), det("plastic", conf=0.2),
            det("person"), det("bin_ok")]
    info = analyse_frame(dets, CFG, SHAPE)
    assert not info.flagged and info.count == 0 and info.kept == []


def test_roi_excludes_items_outside_the_zone():
    dets = [det("plastic", box=(0, 0, 10, 10)), det("metal", box=(100, 60, 110, 70)),
            det("paper", box=(100, 70, 110, 80)), det("glass", box=(100, 80, 110, 90))]
    roi = [0.5, 0.4, 1.0, 1.0]            # right-bottom part of the frame
    info = analyse_frame(dets, CFG, SHAPE, roi)
    assert "plastic" not in info.counts and info.count == 3 and info.flagged


def test_shares_sorted_and_normalised():
    s = shares({"metal": 1.0, "plastic": 3.0, "glass": 0.0})
    assert list(s) == ["plastic", "metal"]
    assert abs(sum(s.values()) - 1.0) < 0.01
    assert shares({}) == {}


def test_dominant_and_summary():
    assert dominant({"a": 0.2, "b": 0.8}) == "b" and dominant({}) is None
    info = analyse_frame([det("plastic"), det("plastic"), det("metal")], CFG, SHAPE)
    assert summary_text(info) == "plastic x2, metal x1"
    assert summary_text(analyse_frame([], CFG, SHAPE)) == "no waste detected"


def test_parse_and_average_composition():
    assert parse_composition('{"plastic": 0.5}') == {"plastic": 0.5}
    assert parse_composition(None) == {} and parse_composition("not json") == {}
    assert parse_composition(float("nan")) == {}
    avg = average_composition(['{"plastic": 1.0}', '{"metal": 1.0}', None, ""])
    assert avg == {"plastic": 0.5, "metal": 0.5}


# ----------------------------------------------------------------- bins
# Bin box (60,40,100,100) is 40 wide x 60 high; with bin_pad 0.35 the "overflow zone"
# around it is x 46..114, y 19..121.
BIN = (60, 40, 100, 100)
OVER_A = (65, 20, 80, 34)      # centre above the rim, inside the zone
OVER_B = (80, 22, 95, 36)
INSIDE = (70, 60, 80, 70)      # centre inside the bin
FAR = [(0, 0, 10, 10), (0, 20, 10, 30), (0, 40, 10, 50)]   # nowhere near the bin


def test_waste_inside_a_bin_is_normal_even_if_many_items():
    dets = [det("bin", box=BIN)] + [det("plastic", box=INSIDE) for _ in range(5)]
    info = analyse_frame(dets, CFG, SHAPE)
    assert not info.flagged and info.count == 5
    assert len(info.bins) == 1 and not info.bins[0]["overflowing"]


def test_waste_just_outside_a_bin_means_overflow_with_its_own_mix():
    dets = [det("bin", box=BIN), det("plastic", box=OVER_A), det("metal", box=OVER_B),
            det("glass", box=INSIDE)]
    info = analyse_frame(dets, CFG, SHAPE)
    assert info.flagged and info.label == OVERFLOW
    assert info.bins[0]["overflowing"] and info.bins[0]["overflow_items"] == 2
    assert set(info.problem_comp) == {"plastic", "metal"}          # the glass inside is not blamed
    assert set(info.comp) == {"plastic", "metal", "glass"}         # but it is still visible/counted
    assert info.area > 0


def test_single_stray_item_near_a_bin_is_not_an_overflow():
    info = analyse_frame([det("bin", box=BIN), det("plastic", box=OVER_A)], CFG, SHAPE)
    assert not info.flagged and not info.bins[0]["overflowing"]


def test_waste_far_from_bins_is_a_dump_not_an_overflow():
    dets = [det("bin", box=BIN)] + [det("paper", box=b) for b in FAR]
    info = analyse_frame(dets, CFG, SHAPE)
    assert info.flagged and info.label == DUMP and not info.bins[0]["overflowing"]
    assert info.problem_comp.keys() == {"paper"}


def test_no_bin_in_frame_behaves_like_loose_waste():
    info = analyse_frame([det("paper", box=b) for b in FAR], CFG, SHAPE)
    assert info.flagged and info.label == DUMP and info.bins == []


def test_two_bins_are_judged_independently():
    bin2 = (120, 40, 158, 100)
    dets = [det("bin", box=BIN), det("bin", box=bin2),
            det("plastic", box=OVER_A), det("plastic", box=OVER_B)]      # only around bin 1
    info = analyse_frame(dets, CFG, SHAPE)
    assert [b["overflowing"] for b in info.bins] == [True, False]


def test_summary_and_annotation_mention_bins():
    dets = [det("bin", box=BIN), det("plastic", box=OVER_A), det("plastic", box=OVER_B)]
    info = analyse_frame(dets, CFG, SHAPE)
    assert summary_text(info) == "plastic x2 | bins 1 (1 overflowing)"
    frame = np.zeros(SHAPE, np.uint8)
    out = annotate_frame(frame, info, CFG, "x")
    assert out.shape == frame.shape and out.any() and not frame.any()
