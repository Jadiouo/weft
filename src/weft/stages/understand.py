"""聯合理解。SDD §4.7。**唯一花額度的階段。**

「拆時間軸，不拆模態軸。」（§2.1 原則一）——投影片圖與逐字稿必須在同一次
呼叫中被聯合理解，分別理解再合併會在壓縮階段丟掉對齊所需的資訊。

額度來源固定為 **Gemini API key**（AI Studio free tier）。
§5.5 #12：不得以任何方式把 AI Pro/Ultra 的訂閱額度接入程式化呼叫。
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

PROMPT_VERSION = "v1"


class ApiKeyMissing(RuntimeError):
    """沒有 Gemini API key。

    §2.3：「額度來源明確為 Gemini API key（AI Studio 取得，free tier）。
    訂閱額度（AI Pro/Ultra）在架構上無法接入程式化呼叫，不得嘗試繞道。」
    """


#: Gemini structured output 的 schema。對應 IR 的 Understanding（§3.4）。
#: 這裡刻意手寫而非從 pydantic 產生——Gemini 的 schema 方言只支援 OpenAPI
#: 的子集（不吃 $defs／anyOf），自動轉換出來的東西會被靜默忽略，然後模型
#: 回傳自由格式的 JSON，錯誤要到反序列化才浮現。
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "segments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "segment_id": {"type": "string"},
                    "summary": {"type": "string"},
                    "layout_description": {"type": "string"},
                    "content_blocks": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "type": {
                                    "type": "string",
                                    "enum": ["經文原文", "白話解說", "圖表描述", "口頭延伸"],
                                },
                                "text": {"type": "string"},
                                "provenance_kind": {
                                    "type": "string",
                                    "enum": ["slide_ocr", "transcript"],
                                },
                                "provenance_ref": {"type": "string"},
                            },
                            "required": ["type", "text", "provenance_kind", "provenance_ref"],
                        },
                    },
                    "terms": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["segment_id", "summary", "content_blocks", "terms"],
            },
        }
    },
    "required": ["segments"],
}


SYSTEM_PROMPT = """你在為一個「講經影片 → 可檢索知識庫」的系統做內容理解。
你會拿到一段影片的投影片畫面與該時段的逐字稿，要輸出結構化的理解結果。

## 絕對規則

1. **不得推測畫面上與逐字稿中都沒有的資訊。** 人名、書名、數字、年代尤其
   如此。你寫下的每一個具體事實都必須能在來源中找到。系統會做自動溯源
   檢查，編造的內容會被標記並退回。

2. **每個 content_block 都必須標註來源**：
   - `provenance_kind: "slide_ocr"` + `provenance_ref: "<slide_id>"`
     ——內容取自投影片畫面
   - `provenance_kind: "transcript"` + `provenance_ref: "<起秒>-<迄秒>"`
     ——內容取自講者口述

3. **展開所有指涉性語句。** 「這個式子」要寫成它實際指的東西，「前面提到
   的」要寫出前面提到的是什麼。輸出的每一段文字都會被單獨取出當作檢索
   結果，讀者看不到上下文。

4. **不要只做複製貼上。** `白話解說` 與 `口頭延伸` 必須是理解與整合後的
   表述；系統會檢查逐字複製率，整段照抄會被判為失敗。
   `經文原文` 是例外——那本來就該逐字引用。

5. 不要使用 markdown 裝飾（`**`、`#`、`-`）。輸出會直接進向量庫。

## content_blocks 的型別（封閉列舉，不得自創）

- `經文原文`：投影片上的經典原文引用，逐字照錄
- `白話解說`：對經文的白話翻譯或解釋
- `圖表描述`：圖片、表格、流程圖的**文字化**描述。RAG 讀不到圖，所以
  版面所隱含的結構關係（箭頭指向什麼、雙欄如何對應、色彩編碼代表什麼）
  必須寫成文字
- `口頭延伸`：講者在投影片之外補充的說明、比喻、離題

## layout_description

用一段文字描述這一頁的版面結構與其語意，不只是列出文字。例如：
「上方為四張胚胎顯微照片（第一至第四周），下方紫底區塊為經文引文，
右側以箭頭指向對應的白話解說。」
"""


@dataclass
class BatchResult:
    """一次呼叫的結果。"""

    per_segment: dict[str, dict]
    input_tokens: int
    output_tokens: int
    model_used: str


def _client():
    """建立 Gemini client。

    只從 `GEMINI_API_KEY` / `GOOGLE_API_KEY` 取值。**不讀瀏覽器 cookie、
    不走 OAuth**——那是 §5.5 #12 禁止的訂閱額度繞道。
    """
    from google import genai

    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise ApiKeyMissing(
            "找不到 GEMINI_API_KEY。請至 https://aistudio.google.com/apikey 取得"
            "（free tier 即可），然後 export GEMINI_API_KEY=...\n"
            "SDD §2.3：額度來源固定為 API key；訂閱額度（AI Pro/Ultra）"
            "在架構上無法接入程式化呼叫，不得嘗試繞道。"
        )
    return genai.Client(api_key=key)


def build_prompt(segments, slides_by_id, prev_summary: str | None, lexicon_terms: list[str]) -> str:
    """組出一次呼叫的使用者訊息。SDD §4.7 的輸入清單。"""
    parts: list[str] = []
    if prev_summary:
        parts.append(f"## 前一段的摘要（僅供銜接，不要重複其內容）\n{prev_summary}\n")
    if lexicon_terms:
        parts.append(
            "## 本系列的術語（逐字稿已用它校正過，請沿用這些寫法）\n"
            + "、".join(lexicon_terms)
            + "\n"
        )

    for seg in segments:
        header = f"## segment {seg.segment_id}（{seg.t_start:.1f}s – {seg.t_end:.1f}s）"
        if seg.mode.value == "slide" and seg.slide_ref:
            header += f"\n投影片：{seg.slide_ref}（圖見下方附件）"
            ocr = (slides_by_id.get(seg.slide_ref) or "").strip()
            if ocr:
                header += f"\n投影片 OCR 文字（供對照，畫面為準）：\n{ocr}"
        elif seg.mode.value == "speaker_only":
            header += "\n此段畫面為講者，無投影片。"
        else:
            header += "\n此段無投影片畫面。"

        transcript = (seg.transcript_corrected or seg.transcript_raw).strip()
        header += f"\n\n逐字稿：\n{transcript or '（此段無逐字稿）'}"
        parts.append(header)

    parts.append(
        "\n請為上述每一個 segment 各輸出一組結果，`segment_id` 必須與上面完全一致。"
    )
    return "\n\n".join(parts)


def call_gemini(
    segments,
    slides_by_id: dict[str, str],
    image_paths: dict[str, Path],
    prev_summary: str | None,
    lexicon_terms: list[str],
    cfg,
) -> BatchResult:
    """送出一次批次呼叫。SDD §4.7。

    §4.7 批次策略：可將 2–3 個相鄰 segment 合併為一次呼叫以節省額度，
    但**輸出仍須逐 segment 分開**——所以 schema 是 `{"segments": [...]}`
    而不是單一物件。
    """
    from google.genai import types

    client = _client()
    contents: list = [build_prompt(segments, slides_by_id, prev_summary, lexicon_terms)]

    for seg in segments:
        path = image_paths.get(seg.slide_ref or "")
        if path and path.exists():
            contents.append(
                types.Part.from_bytes(data=path.read_bytes(), mime_type="image/png")
            )

    response = client.models.generate_content(
        model=cfg.model,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=RESPONSE_SCHEMA,
            temperature=0.2,
        ),
    )

    payload = json.loads(response.text)
    usage = getattr(response, "usage_metadata", None)
    return BatchResult(
        per_segment={s["segment_id"]: s for s in payload.get("segments", [])},
        input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
        output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
        model_used=cfg.model,
    )


def to_understanding(raw: dict, segment, cfg):
    """把模型回傳的 dict 轉成 IR 的 Understanding。

    `provenance` 在 IR 中是必填且不得為 null（§3.4、§5.3 不變量 7）。
    模型漏填時**不補假值**——那會讓溯源檢查失去意義。改為丟掉該 block
    並記錄，讓 unverified 比例如實反映品質。
    """
    from ..ir import ContentBlock, ContentType, Provenance, ProvenanceKind, Understanding

    blocks = []
    for i, item in enumerate(raw.get("content_blocks", [])):
        kind = item.get("provenance_kind")
        ref = (item.get("provenance_ref") or "").strip()
        if not kind or not ref:
            log.warning("%s 的 block#%d 缺 provenance，已丟棄（§5.3 不變量 7）",
                        segment.segment_id, i)
            continue
        try:
            blocks.append(
                ContentBlock(
                    type=ContentType(item["type"]),
                    text=item["text"].strip(),
                    provenance=Provenance(kind=ProvenanceKind(kind), ref=ref),
                )
            )
        except (KeyError, ValueError) as exc:
            log.warning("%s 的 block#%d 格式不符（%s），已丟棄", segment.segment_id, i, exc)

    return Understanding(
        summary=(raw.get("summary") or "").strip(),
        layout_description=(raw.get("layout_description") or "").strip() or None,
        content_blocks=blocks,
        terms=[t for t in raw.get("terms", []) if t],
        model_used=cfg.model,
        prompt_version=cfg.prompt_version,
    )


def with_retries(fn, max_retries: int, backoff_sec: float):
    """指數退避重試。SDD §4.7 失敗行為：重試 2 次後標記 understanding=null。

    **額度耗盡不重試**——重試只會再撞一次牆，而 §6.1 說 429 會讓做到一半
    的 segment 白費。
    """
    from ..quota import QuotaExhausted

    last: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except (QuotaExhausted, ApiKeyMissing):
            raise
        except Exception as exc:  # noqa: BLE001
            last = exc
            if attempt == max_retries:
                break
            wait = backoff_sec * (2**attempt)
            log.warning("呼叫失敗（%s），%.0fs 後重試（%d/%d）",
                        exc, wait, attempt + 1, max_retries)
            time.sleep(wait)
    raise RuntimeError(f"重試 {max_retries} 次後仍失敗：{last}")
