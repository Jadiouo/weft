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
#:
#: **欄位順序有意義。** structured output 是逐欄生成的，先產生的欄位不能
#: 因後面的內容而回頭修改。所以順序是：
#:   is_slide → slide_text（逐字轉錄）→ content_blocks（詮釋）
#: 讓模型**先把畫面上的字抄下來，再據以詮釋**。slide_text 同時是 §5.4
#: 溯源檢查的比對來源，這個順序是它僅存的獨立性（見 known-risks R9）。
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "segments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "segment_id": {"type": "string"},
                    "is_slide": {"type": "boolean"},
                    "reject_reason": {"type": "string"},
                    "slide_text": {"type": "string"},
                    "layout_description": {"type": "string"},
                    "corrections": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "from": {"type": "string"},
                                "to": {"type": "string"},
                                "reason": {"type": "string"},
                            },
                            "required": ["from", "to", "reason"],
                        },
                    },
                    "summary": {"type": "string"},
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
                "required": [
                    "segment_id", "is_slide", "slide_text",
                    "corrections", "summary", "content_blocks", "terms",
                ],
            },
        }
    },
    "required": ["segments"],
}


SYSTEM_PROMPT = """你在為一個「講經影片 → 可檢索知識庫」的系統做內容理解。
你會拿到一段影片的**代表畫面**與該時段的逐字稿。

代表畫面是自動抽出的——系統只知道「這段時間畫面是靜止的」，**不知道那是
投影片還是講者鏡頭**。判斷它是什麼，是你的第一項工作。

## 你的四項工作，按順序

### 1. 判斷這張圖是不是投影片（`is_slide`）

**是投影片**：畫面主體為文字、圖表、經文、流程圖等準備好的教材內容。
**不是投影片**：講者的攝影棚鏡頭、片頭片尾動畫、純裝飾畫面。

注意：講者所在的攝影棚背景**經常有大量裝飾文字**（標語、書法、招牌）。
那些是**佈景**，不是投影片內容。判斷依據是「這是為了講解而製作的教材」，
不是「畫面上有沒有字」。

`is_slide: false` 時，填 `reject_reason`（一句話），`slide_text` 留空字串，
`content_blocks` 只能用 `provenance_kind: "transcript"`。

**若這一段既不是投影片、逐字稿又是空的**（例如片頭動畫、還沒開始講話的
畫面），`content_blocks` 請回**空陣列**。這種段落沒有任何可溯源的材料，
描述畫面上看到的字會變成無法驗證的內容，系統會整批退回。

### 2. 逐字轉錄投影片上的文字（`slide_text`）

**先抄，再詮釋。** 把畫面上的文字**原樣**打出來，保留換行與排列順序
（直排請由右至左、由上而下）。這一欄是後續溯源檢查的比對基準，
**不要在這裡改寫、摘要或補充**。

`is_slide: false` 時填空字串。

### 3. 對照投影片修正逐字稿的錯字（`corrections`）

逐字稿來自語音辨識或字幕，專有術語常出現**同音錯字**。你同時看得到
投影片上的正確寫法與逐字稿，請找出這類錯誤。

每一筆填 `{"from": 逐字稿中的錯字, "to": 正確寫法, "reason": 理由}`。

規則：
- `from` 必須是**逐字稿中實際出現的字串**，一字不差
- 只改**有把握**的：正確寫法出現在這張投影片上，且與錯字同音或近音
- **寧可漏改，不可亂改。** 講者本來就講對的詞不要動；一般用語不要動
- 沒有要改的就回空陣列

### 4. 理解與結構化（`summary`、`content_blocks`、`terms`）

## 絕對規則

1. **不得推測畫面上與逐字稿中都沒有的資訊。** 人名、書名、數字、年代
   尤其如此。系統會做自動溯源檢查，編造的內容會被標記並退回。

2. **每個 content_block 都必須標註來源**：
   - `provenance_kind: "slide_ocr"` + `provenance_ref: "<slide_id>"`
     ——內容取自投影片畫面
   - `provenance_kind: "transcript"` + `provenance_ref: "<起秒>-<迄秒>"`
     ——內容取自講者口述

3. **展開所有指涉性語句。** 「這個式子」要寫成它實際指的東西。輸出的每
   一段文字都會被單獨取出當作檢索結果，讀者看不到上下文。

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

用一段文字描述這一頁的版面結構與其語意，不只是列出文字。
`is_slide: false` 時可留空。
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


def build_prompt(segments, prev_summary: str | None) -> str:
    """組出一次呼叫的使用者訊息。SDD §4.7 的輸入清單。"""
    parts: list[str] = []
    if prev_summary:
        parts.append(f"## 前一段的摘要（僅供銜接，不要重複其內容）\n{prev_summary}\n")

    for seg in segments:
        header = f"## segment {seg.segment_id}（{seg.t_start:.1f}s – {seg.t_end:.1f}s）"
        if seg.candidate_ref:
            header += f"\n代表畫面：{seg.candidate_ref}（圖見附件，依序對應）"
        else:
            header += "\n此段沒有代表畫面（純逐字稿）。"

        transcript = (seg.transcript_corrected or seg.transcript_raw).strip()
        header += f"\n\n逐字稿：\n{transcript or '（此段無逐字稿）'}"
        parts.append(header)

    parts.append(
        "\n請為上述每一個 segment 各輸出一組結果，`segment_id` 必須與上面完全一致。"
    )
    return "\n\n".join(parts)

def call_gemini(
    segments,
    image_paths: dict[str, Path],
    prev_summary: str | None,
    cfg,
) -> BatchResult:
    """送出一次批次呼叫。SDD §4.7。

    §4.7 批次策略：可將 2–3 個相鄰 segment 合併為一次呼叫以節省額度，
    但**輸出仍須逐 segment 分開**——所以 schema 是 `{"segments": [...]}`
    而不是單一物件。
    """
    from google.genai import types

    client = _client()
    contents: list = [build_prompt(segments, prev_summary)]

    for seg in segments:
        path = image_paths.get(seg.candidate_ref or "")
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


def validate_corrections(raw: dict, segment) -> list:
    """把模型回傳的校正轉成 IR 的 Correction，並**丟掉對不上原文的**。

    §5.3 不變量 10 要求每一筆的 `from` 字串實際出現在 `transcript_raw` 中。
    模型偶爾會回傳改寫過的片段（例如加了標點、或抄成了正確版本），那種
    校正無法套用也無法稽核，**丟掉並記錄**——不做模糊比對硬套上去。
    """
    from ..ir import Correction, CorrectionMethod

    out: list = []
    seen: set[tuple[str, str]] = set()
    for item in raw.get("corrections", []):
        from_text = (item.get("from") or "").strip()
        to_text = (item.get("to") or "").strip()
        if not from_text or not to_text or from_text == to_text:
            continue
        if from_text not in segment.transcript_raw:
            log.warning("%s：校正 %r→%r 的原字串不在逐字稿中，已丟棄（§5.3 不變量 10）",
                        segment.segment_id, from_text, to_text)
            continue
        if (from_text, to_text) in seen:
            continue
        seen.add((from_text, to_text))
        out.append(
            Correction(
                **{
                    "from": from_text,
                    "to": to_text,
                    "source": segment.candidate_ref or segment.segment_id,
                    "method": CorrectionMethod.VLM,
                    "reason": (item.get("reason") or "").strip(),
                }
            )
        )
    return out


def apply_corrections(transcript, segment, corrections) -> None:
    """把校正套用到該 segment 涵蓋的逐字稿句子上。

    **只改 `text_corrected`，`text_raw` 永不覆寫**（§4.5 約束 3、§5.3 不變量 9）。
    套用後重建 `segment.transcript_corrected`，讓 segment 與 cue 兩層一致——
    不一致的話 §5.4 的溯源檢查會拿到與 debug markdown 不同的文字。
    """
    by_index = {c.index: c for c in transcript.cues}
    for index in segment.cue_indices:
        cue = by_index.get(index)
        if cue is None:
            continue
        text = cue.text_corrected or cue.text_raw
        applied = []
        for correction in corrections:
            if correction.from_text in cue.text_raw:
                text = text.replace(correction.from_text, correction.to_text)
                applied.append(correction)
        cue.text_corrected = text
        cue.corrections = applied

    picked = [by_index[i] for i in segment.cue_indices if i in by_index]
    segment.transcript_corrected = "".join(c.text_corrected or c.text_raw for c in picked)
    segment.corrections = list(corrections)


def to_understanding(raw: dict, segment, cfg):
    """把模型回傳的 dict 轉成 IR 的 Understanding。

    `provenance` 在 IR 中是必填且不得為 null（§3.4、§5.3 不變量 7）。
    模型漏填時**不補假值**——那會讓溯源檢查失去意義。改為丟掉該 block
    並記錄，讓 unverified 比例如實反映品質。
    """
    from ..ir import ContentBlock, ContentType, Provenance, ProvenanceKind, Understanding

    is_slide = bool(raw.get("is_slide", True))
    slide_text = (raw.get("slide_text") or "").strip()

    blocks = []
    for i, item in enumerate(raw.get("content_blocks", [])):
        kind = item.get("provenance_kind")
        ref = (item.get("provenance_ref") or "").strip()
        if not kind or not ref:
            log.warning("%s 的 block#%d 缺 provenance，已丟棄（§5.3 不變量 7）",
                        segment.segment_id, i)
            continue
        # 判定不是投影片時，不得有 slide_ocr 來源的 block——沒有投影片可溯源。
        # 這是 prompt 已明說的規則，但模型不一定照做，所以這裡也擋一次。
        if not is_slide and kind == "slide_ocr":
            log.warning("%s 的 block#%d 宣稱來自投影片，但該段判定不是投影片，已丟棄",
                        segment.segment_id, i)
            continue
        # 逐字稿為空時，transcript 來源同樣無法溯源。
        #
        # 實測（v0.3 首跑）：片頭 0–50 秒有 12 個 segment 沒有任何字幕，
        # VLM 判定 is_slide=false（正確），卻仍描述畫面上的字並標成
        # transcript 來源——因為 prompt 只留了這一個選項給它。9 個這樣的
        # block 全部溯源失敗，佔未通過總數的四分之一。
        if kind == "transcript" and not (
            segment.transcript_corrected or segment.transcript_raw
        ).strip():
            log.warning("%s 的 block#%d 宣稱來自逐字稿，但該段逐字稿為空，已丟棄",
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
        is_slide=is_slide,
        reject_reason=(raw.get("reject_reason") or "").strip() or None if not is_slide else None,
        slide_text=slide_text or None,
        corrections=validate_corrections(raw, segment),
        summary=(raw.get("summary") or "").strip(),
        layout_description=(raw.get("layout_description") or "").strip() or None,
        content_blocks=blocks,
        terms=[t for t in raw.get("terms", []) if t],
        model_used=cfg.model,
        prompt_version=cfg.prompt_version,
    )


class PermanentApiError(RuntimeError):
    """重試不會改變結果的錯誤（模型不存在、請求格式錯、認證失敗）。"""


class QuotaHit(RuntimeError):
    """API 回了 429。帶著它從回應中解析出的**真實配額上限**。"""

    def __init__(self, message: str, limit: int | None = None) -> None:
        super().__init__(message)
        self.limit = limit


#: HTTP 狀態碼 → 是否值得重試。
#: 404／400／401／403 重試**沒有意義**——模型不存在不會因為再問一次就存在，
#: 而每次重試都消耗一次配額。實測 v0.3 首跑時 `gemini-2.5-flash-lite` 回 404
#: （對新使用者已停用），15 個批次各重試 2 次＝45 次呼叫，把 20 RPD 的
#: 配額燒光，一個 segment 都沒完成。
_PERMANENT_STATUS = (400, 401, 403, 404, 405)
_QUOTA_STATUS = (429,)


def _status_of(exc: Exception) -> int | None:
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if isinstance(code, int):
        return code
    text = str(exc)
    for status in (*_PERMANENT_STATUS, *_QUOTA_STATUS, 500, 503):
        if text.startswith(f"{status} ") or f"'code': {status}" in text:
            return status
    return None


def _parse_quota_limit(exc: Exception) -> int | None:
    """從 429 的回應中取出真實的配額上限。

    SDD §9 對「Google 曾大幅調降免費額度」的緩解是「**Ledger 讀取實際配額
    而非寫死**」。429 的回應正好帶著它：
        {'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier',
         'quotaValue': '20'}
    """
    import re

    text = str(exc)
    if "PerDay" not in text:
        return None
    match = re.search(r"['\"]quotaValue['\"]:\s*['\"](\d+)['\"]", text)
    if match:
        return int(match.group(1))
    match = re.search(r"limit:\s*(\d+)", text)
    return int(match.group(1)) if match else None


def with_retries(fn, max_retries: int, backoff_sec: float, on_attempt=None):
    """指數退避重試。SDD §4.7 失敗行為：重試 2 次後標記 understanding=null。

    **只重試暫時性錯誤。** 永久性錯誤（404 模型不存在、400 請求格式錯）
    重試不會成功，只會多燒配額；額度耗盡（429）更是重試只會再撞一次牆，
    §6.1 明說「429 會讓做到一半的 segment 白費」。

    `on_attempt` 在**每次實際 API 呼叫後**被呼叫（無論成敗），讓呼叫端能把
    每一次呼叫都記進帳本——記「批次」而不記「呼叫」會讓帳本嚴重低估用量。
    """
    from ..quota import QuotaExhausted

    last: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            result = fn()
            if on_attempt:
                on_attempt(True, None)
            return result
        except (QuotaExhausted, ApiKeyMissing):
            raise
        except Exception as exc:  # noqa: BLE001
            last = exc
            status = _status_of(exc)
            if on_attempt:
                on_attempt(False, exc)

            if status in _QUOTA_STATUS:
                raise QuotaHit(str(exc), _parse_quota_limit(exc)) from exc
            if status in _PERMANENT_STATUS:
                raise PermanentApiError(
                    f"HTTP {status}，重試不會改變結果：{exc}"
                ) from exc
            if attempt == max_retries:
                break
            wait = backoff_sec * (2**attempt)
            log.warning("呼叫失敗（%s），%.0fs 後重試（%d/%d）",
                        exc, wait, attempt + 1, max_retries)
            time.sleep(wait)
    raise RuntimeError(f"重試 {max_retries} 次後仍失敗：{last}")
