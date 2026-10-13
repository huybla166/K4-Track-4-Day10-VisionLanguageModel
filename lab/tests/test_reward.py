"""Test hàm reward IoU (viết TRƯỚC khi chạy GRPO). Chạy: python -m pytest lab/tests -q"""
import json

import pytest

from lab.reward import compute_reward, completion_text, gt_from_objects, iou_reward

GT = [{"label": "pothole", "bbox": (100.0, 100.0, 200.0, 200.0)},
      {"label": "crack", "bbox": (500.0, 500.0, 700.0, 600.0)}]


def _js(boxes):
    return json.dumps([{"label": lab, "bbox_2d": list(bb)} for lab, bb in boxes])


def test_perfect_prediction_gets_one():
    assert compute_reward(_js([("pothole", (100, 100, 200, 200)), ("crack", (500, 500, 700, 600))]),
                          GT) == pytest.approx(1.0)


def test_broken_json_gets_zero():
    assert compute_reward('[{"label": "pothole", "bbox_2d": [1, 2', GT) == 0.0
    assert compute_reward("I see a pothole.", GT) == 0.0
    assert compute_reward('[{"label": "pothole", "bbox_2d": [9, 9, 1, 1]}]', GT) == 0.0  # x2<=x1


def test_wrong_class_only_gets_format_bonus():
    pred = _js([("crack", (100, 100, 200, 200)), ("pothole", (500, 500, 700, 600))])
    assert compute_reward(pred, GT) == pytest.approx(0.2)


def test_extra_boxes_are_penalised_by_denominator():
    one_gt = GT[:1]
    pred = _js([("pothole", (100, 100, 200, 200)), ("pothole", (800, 800, 900, 900))])
    assert compute_reward(pred, one_gt) == pytest.approx(0.2 + 0.8 * 1.0 / 2)  # 0.6


def test_both_empty_gets_one_and_hallucination_on_empty_gets_format_only():
    assert compute_reward("[]", []) == 1.0
    assert compute_reward(_js([("pothole", (0, 0, 10, 10))]), []) == pytest.approx(0.2)


def test_empty_prediction_on_nonempty_image_is_not_rewarded_like_perfect():
    assert compute_reward("[]", GT) == pytest.approx(0.2)


def test_giant_box_does_not_hack_reward():
    pred = _js([("pothole", (0, 0, 1000, 1000)), ("crack", (0, 0, 1000, 1000))])
    assert compute_reward(pred, GT) == pytest.approx(0.2)  # IoU << 0.5 → không cặp nào khớp


def test_partial_iou_is_continuous():
    pred = _js([("pothole", (100, 100, 200, 190))])  # IoU 0.9 với box pothole
    assert compute_reward(pred, GT[:1]) == pytest.approx(0.2 + 0.8 * 0.9)


def test_fenced_json_is_accepted():
    raw = "```json\n" + _js([("pothole", (100, 100, 200, 200))]) + "\n```"
    assert compute_reward(raw, GT[:1]) == pytest.approx(1.0)


def test_one_to_one_matching():
    pred = _js([("pothole", (100, 100, 200, 200)), ("pothole", (100, 100, 200, 200))])
    assert compute_reward(pred, GT[:1]) == pytest.approx(0.6)  # box trùng chỉ khớp 1 lần


def test_gt_from_objects_matches_score_convention():
    gt = gt_from_objects([{"label": "manhole", "yolo": [0.5, 0.5, 0.2, 0.1]}])
    assert gt[0]["label"] == "manhole"
    assert gt[0]["bbox"] == pytest.approx((400.0, 450.0, 600.0, 550.0))


def test_trl_interface_with_conversational_completions():
    comps = [[{"role": "assistant", "content": _js([("pothole", (100, 100, 200, 200))])}], "oops"]
    gts = [[{"label": "pothole", "bbox": [100, 100, 200, 200]}], []]
    assert iou_reward(comps, gt=gts) == pytest.approx([1.0, 0.0])
    assert completion_text([{"role": "assistant", "content": [{"type": "text", "text": "[]"}]}]) == "[]"
