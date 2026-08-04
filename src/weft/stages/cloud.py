"""S4–S6：理解、統整、渲染。SDD §4.7–§4.9，實作屬 Phase 2–3（§7.3–§7.4）。

S4 是**唯一花額度的階段**。額度來源固定為 Gemini API key（AI Studio free
tier）；§5.5 #12 明文禁止把 AI Pro/Ultra 的訂閱額度接入程式化呼叫。
"""

from __future__ import annotations

from ..config import Config
from ..ir import Chunk, Lexicon, Segment, Understanding, VideoIR
from ..paths import OutPaths, WorkPaths
from . import pending


def s4_understand(
    cfg: Config,
    work: WorkPaths,
    segments: list[Segment],
    lexicon: Lexicon,
) -> list[Understanding]:
    """S4 聯合理解。SDD §4.7。

    冪等鍵：segment_id + prompt_version + model
    失敗行為：單一 segment 失敗 → 重試 2 次（指數退避）→ understanding=null 並繼續
    """
    pending(
        "S4 聯合理解",
        "§4.7",
        "Phase 2",
        [
            "per segment 送出：投影片圖 + transcript_corrected + 前段 summary（≤200 字）+ 相關詞庫",
            "Gemini structured output，schema 對應 IR 的 Understanding",
            "prompt 必須要求：結構化版面描述、每 block 標 type 與 provenance、"
            "展開指涉性語句、不得推測畫面與逐字稿都沒有的資訊",
            "可將 2–3 個相鄰 segment 併為一次呼叫，但輸出仍逐 segment 分開",
            "降級：speaker_only 不送圖；transcript_only 退化為逐字稿結構化",
            "額度耗盡 → 停止並記錄進度。本地 fallback 須為明確設定開關且標記 "
            "model_used（§5.5 #6，預設 allow_local_fallback=False）",
            "呼叫前用 quota ledger 主動估算節流，不靠撞 429（§5.5 #13）",
        ],
    )


def s5_synthesize(cfg: Config, work: WorkPaths, ir: VideoIR) -> VideoIR:
    """S5 全片統整。SDD §4.8。只讀 S4 輸出，**不再讀圖**。成本 1–2 次呼叫。"""
    pending(
        "S5 全片統整",
        "§4.8",
        "Phase 2",
        [
            "讀 07_understanding/*.json，產出 TL;DR、術語總表、章節結構",
            "寫入 08_video.json 的頂層欄位",
        ],
    )


def s6_render(cfg: Config, ir: VideoIR, work: WorkPaths, out: OutPaths) -> list[Chunk]:
    """S6 渲染。SDD §4.9。

    chunk 切分規則：一個 content_block = 一個 chunk；超過 800 字則按句切分，
    但每個切片都要複製完整 metadata。
    """
    pending(
        "S6 渲染",
        "§4.9",
        "Phase 3",
        [
            "產出 out/chunks.jsonl",
            "產出 out/debug/{video_id}.md，含內嵌圖片與可點的時間戳連結",
            "needs_review 的影片不得進入 chunks.jsonl（§5.4）",
            "chunk 自足性：text 不得含「如上圖」「前面提到的」等指涉語句與 markdown 裝飾（§3.5）",
        ],
    )
