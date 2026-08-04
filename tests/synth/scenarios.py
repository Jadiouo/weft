"""SDD §5.1 表格的 A1–A7 對抗樣本定義。

§5.5 #8：**不得縮小測試集或移除對抗樣本。A1–A7 為必選。**
新增場景可以；刪除或弱化既有場景不行。每個場景的 `expectation` 欄位是
SDD 表格中「期望行為」一欄的原文，用來讓測試報告能直接對回規格。
"""

from __future__ import annotations

from .truth import LogicalPage, SynthTruth

# --------------------------------------------------------------------------
# 投影片內容（文言文 + 直排 + 雙欄 + 箭頭 + 色彩編碼，對應 SDD §1.3）
# --------------------------------------------------------------------------

C_OPENING = {
    "title": "太上老君內觀經．其一",
    "items": [
        "天地媾精，陰陽布化，萬物以生。",
        "承其宿業，分靈道一，父母和合。",
        "人受其生，一月為胞，精血凝也。",
        "二月為胎，形兆胚也。",
        "三月陽神為三魂，動而生也。",
        "四月陰靈為七魄，靜鎮形也。",
    ],
}
C_VERTICAL = {
    "title": "經文原文（直排）",
    "items": [
        "一月為胞精血凝也",
        "二月為胎形兆胚也",
        "三月陽神為三魂",
        "四月陰靈為七魄",
        "五月五行分藏以安神",
        "六月六律定腑用滋靈",
    ],
}
C_TWO_COL = {
    "title": "經文與白話對照",
    "items": [
        "一月為胞，精血凝也。｜受精卵一月成胞，為父精母血之凝聚。",
        "二月為胎，形兆胚也。｜二月成胎，胚體初具，形兆已見。",
        "三月陽神為三魂。｜三月之時，陽神化為三魂，主動。",
        "四月陰靈為七魄。｜四月陰靈化為七魄，主靜以鎮形。",
        "五月五行分藏。｜五行之氣分入五藏，以安其神。",
        "六月六律定腑。｜六律之數定於六腑，用以滋靈。",
    ],
}
C_ARROW = {
    "title": "識蘊入胎次第",
    "items": ["父母和合", "精血凝聚為胞", "形兆為胚胎", "陽神為三魂", "陰靈為七魄"],
}
C_COLORED = {
    "title": "四月至九月",
    "items": [
        "四月陰靈為七魄，靜鎮形也",
        "五月五行分藏，以安其神",
        "六月六律定腑，用滋其靈",
        "七月七精開竅，通光明也",
        "八月八景神具，降真靈也",
        "九月宮室羅布，以定精也",
    ],
}
C_BUILD = {
    "title": "十月懷胎，逐月化生",
    "items": [
        "一月為胞，精血凝也",
        "二月為胎，形兆胚也",
        "三月陽神為三魂",
        "四月陰靈為七魄",
        "五月五行分藏以安神",
        "六月六律定腑用滋靈",
    ],
}
C_EMBED = {
    "title": "胚胎發育影片（內嵌播放）",
    "items": ["下方為顯微攝影短片", "示第一週至第四週之變化"],
}


#: 講者說的話。每個元素為 `(文字, 注入的 ASR 錯誤 → 正解)`。
#: 錯誤全部是**真實中文 ASR 會犯的同音錯**，並且對應的正解一定出現在
#: 同時段的投影片上——這正是 §4.5 術語校正要處理的情形。
S_OPENING = (
    ("今天我們接著看太上老君內觀經的第一段", None),
    ("經文說天地媾精陰陽布化萬物以生", None),
    ("這裡講的經血凝聚是父精母血的結合", ("經血", "精血")),
)
S_VERTICAL = (
    ("這一頁是直排的經文原文請大家看右邊", None),
    ("一月為胞講的是受精卵凝聚成形的階段", None),
    ("羊神為三魂說的是第三個月的變化", ("羊神", "陽神")),
)
S_TWO_COL = (
    ("左邊是經文右邊是白話解說可以對照著看", None),
    ("形照胚也的意思是胚體已經初具形狀", ("形照", "形兆")),
)
S_ARROW = (
    ("這張圖用箭頭表示入胎的先後次第", None),
    ("從父母和合一直到陽神為三魂", None),
    ("時運進入的時機就在這個階段", ("時運", "識蘊")),
)
S_COLORED = (
    ("這一頁用顏色把四月到九月分開", None),
    ("每一個月都有它對應的變化", None),
)
S_BUILD = (
    ("我們一條一條看十月懷胎的過程", None),
    ("先看第一個月然後是第二個月", None),
    ("接下來三月四月五月六月依序出現", None),
)
S_SPEAKER = (
    ("這邊我先離題講一個題外話", None),
    ("這跟我們今天的主題其實是相關的", None),
)
S_EMBED = (
    ("下面這段影片是胚胎發育的顯微攝影", None),
    ("大家可以看到第一週到第四週的變化", None),
)


def _slide(label: str, duration: float, layout: str, content: dict,
           speech: tuple = (), **render) -> LogicalPage:
    return LogicalPage(
        label=label,
        kind="slide",
        duration=duration,
        expected_slides=1,
        render={"layout": layout, "content": content, **render},
        speech=speech,
    )


def _speaker(label: str, duration: float, seed: int = 0, speech: tuple = ()) -> LogicalPage:
    return LogicalPage(
        label=label,
        kind="speaker",
        duration=duration,
        expected_slides=0,
        render={"kind": "speaker", "seed": seed},
        speech=speech or S_SPEAKER,
    )


# --------------------------------------------------------------------------
# A1–A7
# --------------------------------------------------------------------------

A1 = SynthTruth(
    name="A1_standard",
    description="標準整頁換頁，每頁停留 30–120 秒",
    expectation="全部正確偵測",
    pages=(
        _slide("p1", 30, "plain", C_OPENING, S_OPENING),
        _slide("p2", 60, "vertical", C_VERTICAL, S_VERTICAL),
        _slide("p3", 45, "two_column", C_TWO_COL, S_TWO_COL),
        _slide("p4", 120, "arrow", C_ARROW, S_ARROW),
        _slide("p5", 40, "colored", C_COLORED, S_COLORED),
    ),
)

A2 = SynthTruth(
    name="A2_progressive",
    description="逐條動畫，一頁分 6 次疊加出現",
    expectation="偵測為 1 頁，取最後一幀",
    pages=(
        _slide("before", 30, "plain", C_OPENING, S_OPENING),
        LogicalPage(
            label="build_page",
            kind="slide",
            duration=48,
            expected_slides=1,  # ← 這是 A2 的全部重點
            build_offsets=(0, 8, 16, 24, 32, 40),
            keyframe_offset_window=(40, 48),  # 內容最完整的那一段
            render={"layout": "plain", "content": C_BUILD, "progressive": True, "steps": 6},
            speech=S_BUILD,
        ),
        _slide("after", 30, "colored", C_COLORED, S_COLORED),
    ),
)

A3 = SynthTruth(
    name="A3_speaker_only",
    description="純講者無投影片（靜態人像 + 輕微晃動）",
    expectation="偵測為 0 頁",
    pages=(_speaker("speaker", 120, seed=11),),
)

A4 = SynthTruth(
    name="A4_laser_pointer",
    description="移動紅點疊加在靜態投影片上",
    expectation="偵測為 1 頁（紅點不觸發換頁）",
    pages=(_slide("p1", 90, "two_column", C_TWO_COL, S_TWO_COL, overlay="laser"),),
)

A5 = SynthTruth(
    name="A5_embedded_video",
    description="投影片內嵌播放的短影片",
    expectation="不切成數十頁",
    pages=(
        _slide("p1", 20, "plain", C_OPENING, S_OPENING),
        _slide("p2", 60, "plain", C_EMBED, S_EMBED, overlay="embedded_video"),
        _slide("p3", 20, "colored", C_COLORED, S_COLORED),
    ),
)

A6 = SynthTruth(
    name="A6_interleaved",
    description="講者頁與投影片頁交錯出現",
    expectation="分類正確，投影片頁數正確",
    pages=(
        _slide("s1", 25, "plain", C_OPENING, S_OPENING),
        _speaker("k1", 20, seed=3),
        _slide("s2", 30, "vertical", C_VERTICAL, S_VERTICAL),
        _speaker("k2", 15, seed=4),
        _slide("s3", 25, "arrow", C_ARROW, S_ARROW),
    ),
)

A7 = SynthTruth(
    name="A7_backtrack",
    description="回放：講者退回前一頁再前進",
    expectation="偵測為 3 次切換（非 2 次）",
    pages=(
        _slide("p1", 30, "two_column", C_TWO_COL, S_TWO_COL),
        _slide("p2", 30, "arrow", C_ARROW, S_ARROW),
        # 內容與 p1 相同，仍是獨立段落
        _slide("p1_again", 25, "two_column", C_TWO_COL, S_TWO_COL),
        _slide("p2_again", 30, "arrow", C_ARROW, S_ARROW),
    ),
)

# --------------------------------------------------------------------------
# A8、A9：2026-08-04 依真實素材實測結果新增（SDD v0.2 §5.1）
#
# A1–A7 的合成素材**無鏡頭運動、無交叉淡化**（見 docs/decisions.md D5），
# 而這兩者正是真實素材上失敗的直接原因——合成全綠、真實素材產出垃圾。
# --------------------------------------------------------------------------

S_ZOOM = (
    ("這一段我想先把整體的架構講清楚", None),
    ("因為後面的細節都建立在這個基礎上", None),
    ("大家如果聽不懂可以隨時打斷我", None),
)

A8 = SynthTruth(
    name="A8_camera_zoom",
    description="攝影機緩慢推近（純講者背景，全程無投影片）",
    expectation="偵測為 0 頁（鏡頭運動不觸發換頁）",
    pages=(
        LogicalPage(
            label="zoom_in",
            kind="speaker",
            duration=90,
            expected_slides=0,
            render={"kind": "speaker", "seed": 21},
            speech=S_ZOOM,
            # 90 秒內推近 30%，與真實素材的推近幅度同量級
            zoom=(1.0, 1.30),
        ),
    ),
)

A9 = SynthTruth(
    name="A9_crossfade",
    description="1 秒交叉淡化轉場進出全螢幕投影片",
    expectation="段落邊界正確（±2 秒內），且 keyframe 不得取到轉場幀",
    pages=(
        LogicalPage(
            label="p1", kind="slide", duration=40, expected_slides=1,
            render={"layout": "plain", "content": C_OPENING},
            speech=S_OPENING, crossfade_out_sec=1.0,
        ),
        LogicalPage(
            label="p2", kind="slide", duration=45, expected_slides=1,
            render={"layout": "vertical", "content": C_VERTICAL},
            speech=S_VERTICAL, crossfade_out_sec=1.0,
        ),
        LogicalPage(
            label="p3", kind="slide", duration=40, expected_slides=1,
            render={"layout": "arrow", "content": C_ARROW},
            speech=S_ARROW, crossfade_out_sec=1.0,
        ),
        LogicalPage(
            label="p4", kind="slide", duration=35, expected_slides=1,
            render={"layout": "colored", "content": C_COLORED},
            speech=S_COLORED,
        ),
    ),
)

ALL_SCENARIOS: tuple[SynthTruth, ...] = (A1, A2, A3, A4, A5, A6, A7, A8, A9)
BY_NAME: dict[str, SynthTruth] = {s.name: s for s in ALL_SCENARIOS}
