"""SDD §5.2 的量化門檻。

┌──────────────────────────────────────────────────────────────────────┐
│ 這些數值是**設計目標**，不是可調參數。                                  │
│                                                                      │
│ SDD §5.5 #7：不得為了讓測試通過而調低本檔案的門檻。                      │
│ SDD §5.2  ：門檻寫死在測試程式碼中，不得在測試失敗時調整。                │
│                                                                      │
│ 若你（人類或 AI coding agent）認為某個門檻不合理，正確做法是**停下來提出   │
│ 討論並修改 SDD**，而不是編輯這個檔案讓紅燈變綠燈。                        │
│ 本檔案的任何變更都必須在 commit message 中引用對應的 SDD 修訂。            │
└──────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

from typing import Final

#: 換頁偵測 boundary F1 的時間容忍窗（秒）。§5.2 表格括號內的「±2 秒」。
BOUNDARY_TOLERANCE_SEC: Final[float] = 2.0

#: 換頁偵測 boundary F1 —— 合成影片。環境完全可控，應接近滿分。
BOUNDARY_F1_SYNTHETIC: Final[float] = 0.95

#: 換頁偵測 boundary F1 —— 真實影片。
#: 參考：文獻中 naive frame diff 約 0.50–0.60，好方法約 0.81+。
BOUNDARY_F1_REAL: Final[float] = 0.75

#: 「這張候選幀是不是投影片」的分類 accuracy。
#:
#: **v0.4 改綁對象（D30）。** 原名 `FRAME_CLASS_ACCURACY`，量的是
#: `FrameLabel.frame_class`——但 v0.3 的 D16 把 CV 的 speaker/slide 分類
#: 整個移除後，那個欄位**恆為 `slide`**（實測 2519/2519），
#: 門檻因此測不到任何東西。
#:
#: 分類這件事沒有消失，只是搬到 **S4a 的 `is_slide`**（§4.7a）。
#: 門檻改綁到那裡，值不變。
SLIDE_CLASSIFICATION_ACCURACY: Final[float] = 0.95

#: 逐條動畫合併正確率（對抗樣本 A2）。這是設計目標，不容失敗。
PROGRESSIVE_MERGE_ACCURACY: Final[float] = 1.00

#: 術語校正 precision（黃金集）。寧可漏改，不可亂改。
TERM_CORRECTION_PRECISION: Final[float] = 0.90

#: 術語校正 recall —— **記錄但不設硬門檻**（首版求穩）。
#: 值為 None 表示「必須量測並寫入報告，但不 assert」。
TERM_CORRECTION_RECALL: Final[None] = None

#: 對齊邊界誤差中位數（黃金集），秒。
ALIGNMENT_MEDIAN_ERROR_SEC: Final[float] = 5.0

#: **單支影片**的溯源通過率閘門。低於此值 → 整支 `needs_review`，
#: 不進入 `chunks.jsonl`。這是 §5.4 從一開始就定義的東西。
#:
#: v0.5 更名（原 `PROVENANCE_PASS_RATE`）。原名沒有說它的對象是誰，
#: 而 §5.2 的表格把它寫成「對象：全部」的全域驗收門檻——**同一個數字
#: 承載了兩種語意**，於是 R27 算出來的「四支合計 0.838」被當成驗收依據，
#: 而那個數字混了四種成因（歸屬錯、分類誤報、靠校正、真的溯不到）。
#:
#: 全域的比率**沒有門檻**，只記錄（`provenance_rate_overall`）。
#: 這不是調低門檻（§5.5 #7）——per-video 閘門的值一個字都沒動，
#: 是把一個從來不該存在的全域門檻拿掉。見 docs/FROZEN.md F2。
PROVENANCE_PER_VIDEO_GATE: Final[float] = 0.95

#: §5.4：每支影片 unverified 比例超過此值 → 整支標記 needs_review，
#: 不進入 chunks.jsonl。與 `PROVENANCE_PER_VIDEO_GATE` 是同一件事的兩面
#: （1 - 0.95 = 0.05），保留兩個名字是因為程式碼兩種問法都會用到。
MAX_UNVERIFIED_RATIO: Final[float] = 0.05

# --------------------------------------------------------------------------
# 以下**不是** §5.2 的驗收門檻，而是 §4.9／§5.3 的處理參數。
# 它們可以經設定檔調整；上面那些不行。
# --------------------------------------------------------------------------

#: §4.9 chunk 切分規則：單一 content_block 超過此字數則按句切分。
#: 這是處理參數而非驗收門檻——`S6Config.max_chunk_chars` 以它為預設值。
MAX_CHUNK_CHARS: Final[int] = 800

#: §5.3 不變量 2：segments 聯集等於影片全長的容忍值（秒）。
COVERAGE_TOLERANCE_SEC: Final[float] = 1.0


#: **只記錄、不設門檻**的量測項。它們必須被量出來並寫進報告，
#: 但沒有一個數字可以說「到這裡就算過了」。
#:
#: `provenance_rate_overall`：跨影片的溯源通過率合計。R27 證明這個數字
#: 混了四種成因，本來就不該當驗收依據——真正的閘門是 per-video 的。
#: 它的用途是**趨勢**：改了東西之後整體是往上還是往下。
OBSERVED_ONLY: Final[frozenset[str]] = frozenset({
    "provenance_rate_overall",
})


#: §5.2 的驗收門檻名稱。**這些不得出現在設定檔中**（§5.5 #7）。
#: 機械式護欄見 `test_unit_conventions.py`。
ACCEPTANCE_THRESHOLDS: Final[frozenset[str]] = frozenset({
    "BOUNDARY_F1_SYNTHETIC",
    "BOUNDARY_F1_REAL",
    "SLIDE_CLASSIFICATION_ACCURACY",
    "PROGRESSIVE_MERGE_ACCURACY",
    "TERM_CORRECTION_PRECISION",
    "TERM_CORRECTION_RECALL",
    "ALIGNMENT_MEDIAN_ERROR_SEC",
    "PROVENANCE_PER_VIDEO_GATE",
    "MAX_UNVERIFIED_RATIO",
    "BOUNDARY_TOLERANCE_SEC",
    # 這一項是**門檻的門檻**（見下方定義）。登記在這裡的理由與其他項相同：
    # §5.5 #7 禁止把它放進設定檔——「測試紅了就調高覆蓋率上限」和
    # 「測試紅了就調低 F1 門檻」是同一件事，而前者更難被看見。
    "MAX_TOLERANCE_COVERAGE",
})


#: 容忍窗覆蓋率的上限——**這是「門檻有沒有效」的門檻**。
#:
#: `boundary_prf` 的分數有多少來自「切得準」、有多少來自「切得多」，
#: 取決於 `±tolerance` 涵蓋了多少比例的時間軸。覆蓋率接近 1 時
#: 任何刀都會落在某個窗裡，**F1 主要由刀數決定**。
#:
#: 實測（R37）覆蓋率與「不看內容等距切」對照組拿到的 F1：
#:
#:   46% → 0.200   80% → 0.444   83% → 0.500   95% → 0.459   100% → 全部滿分
#:
#: 0.80 是從這組觀察取的**下限**，不是理論值：到 80% 時對照組已經
#: 拿得到 0.44，方法與「不看內容」的差距被壓到量不出來。
#:
#: 換頁偵測目前是 **5%**（10 刀 / 14 分鐘、±2s）——離飽和很遠。
#: 換成 ±20s 會變 46%，仍在上限內。
#:
#: 出事的是 R30／R37 的**分段實驗**：那裡刀數是換頁的 3–6 倍（30–57 刀），
#: 腳本又報在 ±20s，覆蓋率 80–95%，數字不可用（known-risks R30）。
#: 同一個容忍窗在兩處一好一壞，因為**飽和是容忍窗 × 刀數密度的性質**——
#: 這也是為什麼不能靠「把 BOUNDARY_TOLERANCE_SEC 全域改小」來修。
MAX_TOLERANCE_COVERAGE = 0.80
