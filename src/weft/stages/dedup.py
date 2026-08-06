"""S1c — 投影片去重。SDD §4.3b。純本地 CV，不花額度。

S1b 找的是「畫面靜止的區段」，而**同一張投影片會反覆出現**。
實測 42 分鐘影片有 49 個候選區段，相異投影片只有 11 張——
《太上老君內觀經》那一張出現 10 次。不去重的話：

1. 同一張圖被送去理解 10 次
2. **更糟的是 `slide_text` 每次都不完全一樣**（實測本地模型變異 5.2%），
   而它是 §5.4 的比對基準——基準本身在抖

**只做一件事：ink Jaccard 單連結分群。** 門檻 0.20–0.40 結果完全相同。

> **實作時修正了 SDD §4.3b 的設計（D26）。** 原本規劃兩道，第 1 道用
> 「與全片中位幀的距離」**剔除**攝影棚定鏡。實作後量測發現那一道是多餘的，
> 而且有害：
>
> | | 呼叫數 | 誰在做分類 |
> |---|---|---|
> | 兩道（剔除） | 21 | **CV**——被剔除的畫面 VLM 根本看不到 |
> | 只分群 | 22 | **VLM**——攝影棚那群送一張代表幀去判 |
>
> 攝影棚定鏡**本來就會自己聚成一群**（實測 17 張聚成 1 群），
> 所以分群已經達到同樣的呼叫數縮減，差別只有 1 次。
> 而「剔除」等於讓 CV 做了「這不是投影片」的判斷——
> **正是 v0.3 以 D16 刻意移除的東西**。
>
> 中位幀距離改為**只診斷不剔除**：算出來記進 S-1 的 profile
> （§4.0 要用它判斷「本支的背景基準是否與前一支不同」），但不影響分群。
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from ..ir import CandidateSet, Slide
from .detect import ink_jaccard
from .frames import _ink_mask

log = logging.getLogger(__name__)


def _dedup_mask(image_path, short_side: int) -> np.ndarray:
    """去重用的 ink 遮罩。

    **與 §4.3 用同一套演算法與解析度，但不做高斯模糊。**

    模糊在 S1b 存在的理由是壓制**相鄰幀之間**的雷射筆紅點與壓縮雜訊。
    但去重比較的是**不同的投影片**，模糊抹掉的正是鑑別力——實測
    `slide_013`（紅底橫幅「01生命之初」）與 `slide_047`（黃底橫幅
    「問題：為何小產…」）在模糊後 ink 遮罩幾乎相同，會被誤併成同一張。
    那是 recall 違規，而 §5.2 把去重 recall 設在 1.00。

    | 短邊 | 模糊 | 群數 | 真組完整 | 013/047 誤併 |
    |---|---|---|---|---|
    | 180–720 | 0.0 | 22 | ✓ | 否 |
    | 180/240/320 | 2.0 | 21 | ✓ | **是** |
    | 720 | 2.0 | 23 | **✗** | 否 |

    解析度在 180–720 之間對結果沒有影響，因此沿用 §4.3 的
    `downscale_short_side`，只關掉模糊。
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"讀不到投影片圖：{image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    scale = short_side / min(h, w)
    resized = cv2.resize(gray, (max(1, int(w * scale)), max(1, int(h * scale))),
                         interpolation=cv2.INTER_AREA)
    return _ink_mask(resized)


def median_frame(frame_paths, short_side: int, stride: int) -> np.ndarray | None:
    """全片的中位幀——「攝影棚常態」的代表。

    取樣而非全讀：實測 2519 幀取 1/4（630 幀）與全讀的結果一致，
    而記憶體與時間都降到四分之一。
    """
    sampled = frame_paths[::max(1, stride)]
    if not sampled:
        return None
    stack = []
    for path in sampled:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        h, w = img.shape
        scale = short_side / min(h, w)
        stack.append(cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                                interpolation=cv2.INTER_AREA))
    return np.median(np.stack(stack), axis=0) if stack else None


def studio_distance(image_path, median: np.ndarray, short_side: int) -> float:
    """與中位幀的平均絕對差，正規化到 0–1。"""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"讀不到投影片圖：{image_path}")
    h, w = img.shape
    scale = short_side / min(h, w)
    resized = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                         interpolation=cv2.INTER_AREA)
    if resized.shape != median.shape:
        resized = cv2.resize(resized, (median.shape[1], median.shape[0]),
                             interpolation=cv2.INTER_AREA)
    return float(np.abs(resized.astype(int) - median).mean()) / 255.0


def group_by_ink(slide_ids: list[str], masks: dict[str, np.ndarray],
                 threshold: float) -> list[list[str]]:
    """單連結分群。距離低於 `threshold` 即視為同一張。

    單連結（而非完全連結）是刻意的：逐條動畫的中間幀與最終幀之間
    是**鏈狀**相似，完全連結會把它們拆開。
    """
    parent = {s: s for s in slide_ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, a in enumerate(slide_ids):
        for b in slide_ids[i + 1:]:
            if ink_jaccard(masks[a], masks[b]) < threshold:
                parent[find(a)] = find(b)

    groups: dict[str, list[str]] = {}
    for s in slide_ids:
        groups.setdefault(find(s), []).append(s)
    # 依群內第一張的原始順序排，讓輸出可重現
    return sorted(groups.values(), key=lambda g: slide_ids.index(g[0]))


def s1c_dedup(cfg, work, slides: list[Slide], candidates: CandidateSet) -> dict:
    """對整支影片跑去重。**就地更新** `slides` 的 `duplicate_of` 與
    `occurrences`，回傳供 S-1 profile 使用的統計。

    失敗行為（§4.3b）：分群把絕大多數候選幀併成一群時（門檻過鬆），
    **記錄並跳過去重**，退化為逐候選幀處理。**不自動調門檻**——
    調參數把結果拉到「看起來合理」正是 §5.5 #4 禁止的事。
    """
    p = cfg.s1c
    stats = {"candidates": len(slides), "distinct": len(slides), "skipped": None}

    # 無論是否去重，每張都先有自己的 occurrence——下游可以無條件依賴它
    for slide in slides:
        slide.occurrences = [(slide.t_first_seen, slide.t_last_seen)]
        slide.duplicate_of = None

    if not p.enabled or len(slides) < 2:
        stats["skipped"] = "未啟用" if not p.enabled else "候選幀不足 2 張"
        return stats

    by_id = {s.slide_id: s for s in slides}
    paths = {s.slide_id: work.dir / s.image_path for s in slides}
    sids = [s.slide_id for s in slides]

    # ---- ink Jaccard 分群 ----------------------------------------
    masks = {sid: _dedup_mask(paths[sid], cfg.s1b.downscale_short_side) for sid in sids}
    groups = group_by_ink(sids, masks, p.jaccard_threshold)

    largest = max(len(g) for g in groups)
    if largest / len(sids) >= p.max_group_ratio:
        log.error("S1c %s：最大群佔 %d/%d，超過上限 %.0f%%——門檻可能過鬆，"
                  "跳過去重並退化為逐候選幀（§4.3b）。**不自動調門檻**",
                  work.video_id, largest, len(sids), p.max_group_ratio * 100)
        stats["skipped"] = f"最大群佔比 {largest / len(sids):.0%}"
        return stats

    # ---- 中位幀距離：只診斷不剔除（D26），供 S-1 的 profile 用 ----
    frame_paths = sorted(work.frames_dir.glob("*")) if work.frames_dir.exists() else []
    median = median_frame(frame_paths, cfg.s1b.downscale_short_side, p.median_stride)
    if median is not None:
        distances = [studio_distance(paths[sid], median, cfg.s1b.downscale_short_side)
                     for sid in sids]
        stats["studio_distance_min"] = round(min(distances), 4)
        stats["studio_distance_median"] = round(float(np.median(distances)), 4)

    # ---- 就地更新 IR ---------------------------------------------
    for group in groups:
        if len(group) == 1:
            continue
        # 代表幀取群內 ink 量最大者——最完整的一幀，同 §4.3 步驟 5
        rep = max(group, key=lambda sid: int(masks[sid].sum()))
        spans = sorted((by_id[sid].t_first_seen, by_id[sid].t_last_seen) for sid in group)
        by_id[rep].occurrences = spans
        by_id[rep].t_first_seen = spans[0][0]
        by_id[rep].t_last_seen = spans[-1][1]
        for sid in group:
            if sid != rep:
                # **不刪除**被合併的候選幀——§5.6 要能複核「真的是同一張嗎」
                by_id[sid].duplicate_of = rep

    stats["distinct"] = sum(1 for s in slides if s.duplicate_of is None)
    stats["groups"] = len(groups)
    stats["largest_group"] = largest
    #: S-1（§4.0）要把這個比例納入 profile：它在同系列各集之間若大幅跳動，
    #: 代表門檻不能通用。
    stats["distinct_ratio"] = round(stats["distinct"] / len(slides), 3)
    log.info("S1c %s：%d 個候選幀 → %d 張相異投影片（比例 %.2f）",
             work.video_id, len(slides), stats["distinct"], stats["distinct_ratio"])
    return stats


def representatives(slides: list[Slide]) -> list[Slide]:
    """只回傳代表幀——S4a 的處理單位（§4.7a）。"""
    return [s for s in slides if s.duplicate_of is None]
