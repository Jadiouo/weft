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

#: speaker/slide 分類 accuracy。合成與真實皆適用。硬切素材，這題應該很簡單。
FRAME_CLASS_ACCURACY: Final[float] = 0.95

#: 逐條動畫合併正確率（對抗樣本 A2）。這是設計目標，不容失敗。
PROGRESSIVE_MERGE_ACCURACY: Final[float] = 1.00

#: 術語校正 precision（黃金集）。寧可漏改，不可亂改。
TERM_CORRECTION_PRECISION: Final[float] = 0.90

#: 術語校正 recall —— **記錄但不設硬門檻**（首版求穩）。
#: 值為 None 表示「必須量測並寫入報告，但不 assert」。
TERM_CORRECTION_RECALL: Final[None] = None

#: 對齊邊界誤差中位數（黃金集），秒。
ALIGNMENT_MEDIAN_ERROR_SEC: Final[float] = 5.0

#: 溯源通過率（全部）。見 §5.4。
PROVENANCE_PASS_RATE: Final[float] = 0.95

#: §5.4：每支影片 unverified 比例超過此值 → 整支標記 needs_review，
#: 不進入 chunks.jsonl。
MAX_UNVERIFIED_RATIO: Final[float] = 0.05

#: §4.9 chunk 切分規則：單一 content_block 超過此字數則按句切分。
MAX_CHUNK_CHARS: Final[int] = 800

#: §5.3 不變量 2：segments 聯集等於影片全長的容忍值（秒）。
COVERAGE_TOLERANCE_SEC: Final[float] = 1.0
