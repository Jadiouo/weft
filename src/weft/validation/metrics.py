"""SDD §5.2 的指標計算。

只負責「算出數字」；「數字要多少才算通過」在 thresholds.py。
兩者分開，是為了讓調門檻這件事顯眼到無法混進一般 commit。
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from .thresholds import BOUNDARY_TOLERANCE_SEC


@dataclass(frozen=True)
class PRF:
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int
    #: 每個配對成功的 (gt, pred) 時間差，供誤差分布分析
    matched_deltas: tuple[float, ...] = ()

    def __str__(self) -> str:
        return (
            f"P={self.precision:.3f} R={self.recall:.3f} F1={self.f1:.3f} "
            f"(TP={self.tp} FP={self.fp} FN={self.fn})"
        )


def boundary_prf(
    predicted: list[float],
    ground_truth: list[float],
    tolerance_sec: float = BOUNDARY_TOLERANCE_SEC,
) -> PRF:
    """換頁邊界的 precision / recall / F1。

    配對方式：依「時間差最小」的順序做全域貪婪一對一配對。這比「對每個 gt
    取最近 pred」穩健——後者在兩個 gt 相鄰時會讓同一個 pred 被搶走，虛報 FN。

    一個 pred 只能配一個 gt，反之亦然；超出容忍窗的不配。
    """
    pairs = sorted(
        (
            (abs(p - g), pi, gi)
            for pi, p in enumerate(predicted)
            for gi, g in enumerate(ground_truth)
            if abs(p - g) <= tolerance_sec
        )
    )
    used_pred: set[int] = set()
    used_gt: set[int] = set()
    deltas: list[float] = []
    for delta, pi, gi in pairs:
        if pi in used_pred or gi in used_gt:
            continue
        used_pred.add(pi)
        used_gt.add(gi)
        deltas.append(predicted[pi] - ground_truth[gi])

    tp = len(used_gt)
    fp = len(predicted) - tp
    fn = len(ground_truth) - tp
    precision = tp / len(predicted) if predicted else (1.0 if not ground_truth else 0.0)
    recall = tp / len(ground_truth) if ground_truth else (1.0 if not predicted else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return PRF(precision, recall, f1, tp, fp, fn, tuple(deltas))


def classification_accuracy(predicted: list[str], ground_truth: list[str]) -> float:
    """speaker/slide 二分類 accuracy。長度必須一致——對不齊代表抽幀本身有問題，
    那是比分類錯誤更嚴重的 bug，不該被平均掉。"""
    if len(predicted) != len(ground_truth):
        raise ValueError(f"長度不符：predicted={len(predicted)} ground_truth={len(ground_truth)}")
    if not ground_truth:
        raise ValueError("ground_truth 為空，無法計算 accuracy")
    hits = sum(1 for p, g in zip(predicted, ground_truth) if p == g)
    return hits / len(ground_truth)


def median_absolute_error(predicted: list[float], ground_truth: list[float]) -> float:
    """對齊邊界誤差中位數。配對後才有意義，故要求已一一對應。"""
    if len(predicted) != len(ground_truth):
        raise ValueError(f"長度不符：predicted={len(predicted)} ground_truth={len(ground_truth)}")
    if not ground_truth:
        raise ValueError("ground_truth 為空")
    return statistics.median(abs(p - g) for p, g in zip(predicted, ground_truth))


def correction_prf(
    applied: list[tuple[int, str, str]],
    expected: list[tuple[int, str, str]],
) -> PRF:
    """術語校正的 precision / recall。

    每筆為 `(cue_index, from_text, to_text)`。判定為完全相符——改對位置但
    改錯字、或改對字但改錯位置，都算錯。
    """
    applied_set = set(applied)
    expected_set = set(expected)
    tp = len(applied_set & expected_set)
    fp = len(applied_set - expected_set)
    fn = len(expected_set - applied_set)
    precision = tp / len(applied_set) if applied_set else (1.0 if not expected_set else 0.0)
    recall = tp / len(expected_set) if expected_set else (1.0 if not applied_set else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return PRF(precision, recall, f1, tp, fp, fn)


def merge_accuracy(predicted_counts: list[int], expected_counts: list[int]) -> float:
    """逐條動畫合併正確率（對抗樣本 A2）。

    每個元素是「某個邏輯頁面被偵測成幾張投影片」。正確 = 恰好 1 張。
    門檻為 1.00，所以這裡不做部分給分。
    """
    if len(predicted_counts) != len(expected_counts):
        raise ValueError(f"長度不符：predicted={len(predicted_counts)} expected={len(expected_counts)}")
    if not expected_counts:
        raise ValueError("expected_counts 為空")
    hits = sum(1 for p, e in zip(predicted_counts, expected_counts) if p == e)
    return hits / len(expected_counts)


def correction_outcome_prf(
    applied: list[tuple[int, str, str]],
    expected: list[tuple[int, str, str]],
    ideal_texts: dict[int, str],
    corrected_texts: dict[int, str],
) -> PRF:
    """以**結果**判定的術語校正 precision／recall。

    為什麼不用 `correction_prf` 的完全相符：校正器可能匹配到比 ground truth
    更長的詞庫條目。實測案例——注入的錯誤是「形照 → 形兆」，校正器匹配到
    詞庫中的「形兆胚也」，改成「形照胚也 → 形兆胚也」。**結果文字完全正確**，
    但 span 不同，完全相符會把它記成 FP 兼 FN，同時低估 precision 與 recall。

    這不是放寬標準：改錯的編輯仍然是 FP。判定改成——

      TP：`to` 出現在理想文字中，且 `from` 不出現在理想文字中（真的修好了）
      FP：其餘（改到了本來就對的地方，或改成了錯的東西）
      recall：每個注入的錯誤，最終文字是否已含正解且不含錯字
    """
    tp = fp = 0
    for index, from_text, to_text in applied:
        ideal = ideal_texts.get(index, "")
        if to_text in ideal and from_text not in ideal:
            tp += 1
        else:
            fp += 1

    fixed = 0
    for index, wrong, right in expected:
        final = corrected_texts.get(index, "")
        if right in final and wrong not in final:
            fixed += 1
    fn = len(expected) - fixed

    precision = tp / (tp + fp) if (tp + fp) else (1.0 if not expected else 0.0)
    recall = fixed / len(expected) if expected else (1.0 if not applied else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return PRF(precision, recall, f1, tp, fp, fn)
