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
from functools import lru_cache
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
#: **v0.4：`is_slide` 與 `slide_text` 移到 S4a**（§4.7a）。這裡只剩
#: 「校正 → 理解」。投影片的文字由 S4a 讀出後以**文字**掛進 prompt，
#: 所以本階段不需要、也不應該重讀圖——重讀等於讓同一個模型既產生來源
#: 又產生待驗證的內容，那正是 R9 記的獨立性問題。
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "segments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "segment_id": {"type": "string"},
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
                            },
                            # `provenance_ref` **不向模型要**：一段只會拿到一張
                            # 投影片的文字，合法來源只有一個，逐字稿的區間就是
                            # 這一段的起訖。兩者都由管線填（`_canonicalize_refs`）。
                            # 曾經要過，模型把 `slide_015` 寫成 `015`，查不到
                            # 投影片 → 來源為空 → 未通過（R27）。
                            "required": ["type", "text", "provenance_kind"],
                        },
                    },
                    "terms": {"type": "array", "items": {"type": "string"}},
                },
                "required": [
                    "segment_id", "corrections", "summary", "content_blocks", "terms",
                ],
            },
        }
    },
    "required": ["segments"],
}


SYSTEM_PROMPT = """你在為一個「講經影片 → 可檢索知識庫」的系統做內容理解。
你會拿到一段影片的逐字稿，以及該時段螢幕上投影片的**文字內容與版面描述**。

投影片的文字**已經由前一個階段逐字讀出**，你不需要也不應該重讀。
你的工作是理解「講者在這段時間講了什麼」，投影片是參考材料。

## 你的兩項工作

### 1. 修正逐字稿的錯字（`corrections`）

逐字稿來自語音辨識或字幕，專有術語常出現**同音錯字**。請找出這類錯誤。

每一筆填 `{"from": 逐字稿中的錯字, "to": 正確寫法, "reason": 理由}`。

**這一欄只處理一件事：把聽錯的字改成聽對的字。**
判準是「講者嘴巴發出的音沒變，只是字寫錯了」。

可以改：

- 同音或近音的專名誤寫：`時運`→`識蘊`、`憍梵钵提`→`憍梵波提`、`涅盤`→`涅槃`
- 音近漏字：`意地論`→`瑜伽師地論`、`學古學的`→`學古文學的`
- 語音辨識造成的疊字：`家家當`→`家當`、`這這個`→`這個`
- 被截斷的詞：`未`→`未來`（限後文已出現完整詞）

**絕對不可以改**（以下每一種都會被系統自動退回）：

- **補上講者沒說的字。** `陽神為三魂`→`陽神為三魂，動而生也` ✗
  講者只唸了半句就是只唸了半句。投影片上有下半句**不是**補上去的理由。
  逐字稿要忠實記錄講者說了什麼，不是記錄投影片寫了什麼。
- **改事實。** `啟示經`→`創世記` ✗、`唐朝`→`宋朝` ✗、`三百年`→`五百年` ✗
  講者講錯典故或年代，那是講者說的話。你的工作是聽寫正確，不是校訂內容。
  講者的口誤若重要，寫進 `summary` 說明，不要動逐字稿。
- **換成意思相近的別的詞。** `投胎轉世`→`十個月懷胎` ✗、`買房子`→`受精卵著床` ✗
  發音完全不同就不是聽錯。把比喻換成本體、把代詞展開成詮釋，
  都是 `content_blocks` 的工作，不是這一欄的。
- **潤稿與格式化。** 刪贅詞 ✗、改大小寫 ✗、翻譯外來語 ✗
  `沒sense`→`沒Sense` ✗、`好像很像`→`很像` ✗

其他規則：

- `from` 必須是**逐字稿中實際出現的字串**，一字不差
- `to` 與 `from` 讀音要接近。系統會自動比對拼音，差太多的會被丟棄
- **寧可漏改，不可亂改。** 講者本來就講對的詞不要動；一般用語不要動
- 沒有要改的就回空陣列（多數段落都該是空的）

### 2. 理解與結構化（`summary`、`content_blocks`、`terms`）

**若這一段沒有投影片、逐字稿又是空的**（例如片頭動畫、還沒開始講話的
畫面），`content_blocks` 請回**空陣列**。這種段落沒有任何可溯源的材料，
描述畫面上看到的字會變成無法驗證的內容，系統會整批退回。

## 絕對規則

1. **不得推測畫面上與逐字稿中都沒有的資訊。** 人名、書名、數字、年代
   尤其如此。系統會做自動溯源檢查，編造的內容會被標記並退回。

2. **每個 content_block 都要標對來源**：`provenance_kind` 只有兩個值。

   - `slide_ocr`——這段話的**字**主要來自投影片
   - `transcript`——這段話的**字**主要來自講者口述

   **摘要投影片上的內容要標 `slide_ocr`**，即使講者同時也在講它。
   標錯的話系統會拿另一份材料去驗證，必然對不上，整段被判無法溯源。

3. **展開所有指涉性語句。** 「這個式子」要寫成它實際指的東西。輸出的每
   一段文字都會被單獨取出當作檢索結果，讀者看不到上下文。

4. **不要用「講者提到…」「講者解釋…」這種轉述框架。** 直接寫那件事本身。

   ✗ 「講者將胎生過程比作搬家，需要把累世人的種子搬過來。」
   ✓ 「胎生過程如同搬家，要把累世的種子逐步搬過來。」

   來源與時間戳已經記在 metadata 裡，「講者說」三個字對檢索沒有價值，
   只是稀釋內容。而且轉述框架會加入來源中沒有的字，
   系統的溯源檢查會因此判定找不到依據——實測用這種框架的段落
   **6/10 未通過**，不用的只有 3/39。

5. **不要只做複製貼上。** `白話解說` 與 `口頭延伸` 必須是理解與整合後的
   表述；系統會檢查逐字複製率，整段照抄會被判為失敗。
   `經文原文` 是例外——那本來就該逐字引用。

6. 不要使用 markdown 裝飾（`**`、`#`、`-`）。輸出會直接進向量庫。

## content_blocks 的型別（封閉列舉，不得自創）

- `經文原文`：投影片上的經典原文引用，逐字照錄
- `白話解說`：對經文的白話翻譯或解釋
- `圖表描述`：圖片、表格、流程圖的**文字化**描述。RAG 讀不到圖，所以
  版面所隱含的結構關係（箭頭指向什麼、雙欄如何對應、色彩編碼代表什麼）
  必須寫成文字
- `口頭延伸`：講者在投影片之外補充的說明、比喻、離題

**`slide_ocr` 型的來源是給你參考的投影片文字**，不是你讀出來的——
所以引用它時要如實引用，不要「修正」它。
"""


@dataclass
class BatchResult:
    """一次呼叫的結果。"""

    per_segment: dict[str, dict]
    input_tokens: int
    output_tokens: int
    model_used: str


@lru_cache(maxsize=1)
def _cached_client(key: str):
    """實際建立 client。**整個行程共用一個**——見 `_client` 的說明。"""
    from google import genai

    return genai.Client(api_key=key)


def _client():
    """取得 Gemini client。

    只從 `GEMINI_API_KEY` / `GOOGLE_API_KEY` 取值。**不讀瀏覽器 cookie、
    不走 OAuth**——那是 §5.5 #12 禁止的訂閱額度繞道。

    **共用單一實例**（D21）。原本每次呼叫都 `genai.Client(...)`，S4 跑完
    17 個批次後那些 client 一起被 GC，關掉了底層共用的 httpx 傳輸層，
    接著 S5 就拿到 `Cannot send a request, as the client has been closed`。
    S4 的結果有保住（S5 失敗會降級繼續），但 `tldr` 與 `chapters` 是空的。
    """
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise ApiKeyMissing(
            "找不到 GEMINI_API_KEY。請至 https://aistudio.google.com/apikey 取得"
            "（free tier 即可），然後 export GEMINI_API_KEY=...\n"
            "SDD §2.3：額度來源固定為 API key；訂閱額度（AI Pro/Ultra）"
            "在架構上無法接入程式化呼叫，不得嘗試繞道。"
        )
    return _cached_client(key)


def seg_id_of(segments, image_key: str) -> str:
    """找出用這張圖的區段 id，純粹為了讓日誌指得出是誰。"""
    return next((s.segment_id for s in segments if s.candidate_ref == image_key), "?")


def build_parts(segments, prev_summary: str | None,
                slide_context: dict | None = None,
                send_images: bool = True) -> list[tuple[str, str | None]]:
    """組出一次呼叫的內容序列。SDD §4.7c 的輸入清單。

    `slide_context` 是 `{slide_id: {"slide_text": ..., "description": ...}}`，
    由 S4a 產出。有它時**改掛文字、不送圖**——投影片的內容已經文字化，
    再送一次圖只是重複，而且會把 D20 的錯位風險帶回來。

    回傳 `[(文字, 圖的 slide_id 或 None), ...]`。每個 tuple 是**一段文字，
    後面緊接它自己的那張圖**——`call_gemini` 照這個順序疊 parts。

    **不要改成「文字全部在前、圖片全部在後」。** v0.3 首跑就是那樣寫的
    （一段文字加一句「圖見附件，依序對應」，然後三張裸圖），實測 49 個區段
    中 **15 個（30.6%）** 拿到隔壁那張圖的分析結果，而且**每一個錯都落在
    批次內、沒有一個跨批次邊界**——證明是綁定問題不是模型能力問題。
    見 `experiments/r14_image_binding/REPORT.md` 與 docs/decisions.md D20。
    """
    parts: list[tuple[str, str | None]] = []
    if prev_summary:
        parts.append((f"## 前一段的摘要（僅供銜接，不要重複其內容）\n{prev_summary}", None))

    for seg in segments:
        header = f"## segment {seg.segment_id}（{seg.t_start:.1f}s – {seg.t_end:.1f}s）"

        # v0.4：投影片的文字與版面已由 S4a 讀出，這裡用**文字**掛進來，
        # 預設不再送圖（§4.7c）。送圖只在設定明確要求時發生。
        slide = slide_context.get(seg.slide_ref or seg.candidate_ref or "") if slide_context else None
        image_key = (seg.candidate_ref or None) if send_images else None
        if slide and (slide.get("slide_text") or slide.get("description")):
            header += f"\n\n這一段螢幕上的投影片（{seg.slide_ref or seg.candidate_ref}）："
            if slide.get("slide_text"):
                header += f"\n逐字內容：\n{slide['slide_text']}"
            if slide.get("description"):
                header += f"\n版面：{slide['description']}"
        elif image_key:
            header += f"\n**下面緊接的那一張圖**就是這一段的代表畫面（{image_key}）。"
        else:
            header += "\n此段沒有投影片（純逐字稿）。"

        transcript = (seg.transcript_corrected or seg.transcript_raw).strip()
        header += f"\n\n逐字稿：\n{transcript or '（此段無逐字稿）'}"
        parts.append((header, image_key))

    parts.append(
        ("\n請為上述每一個 segment 各輸出一組結果，`segment_id` 必須與上面完全一致。"
         "每一段只能依據**緊接在它標頭後面**的那張圖作答，不要參考其他段的圖。", None)
    )
    return parts

def call_model(
    segments,
    image_paths: dict[str, Path],
    prev_summary: str | None,
    cfg,
    slide_context: dict | None = None,
) -> BatchResult:
    """送出一次 S4c 的批次呼叫。SDD §4.7c。

    §4.7c 批次策略：可將 2–3 個相鄰 segment 合併為一次呼叫以節省額度，
    但**輸出仍須逐 segment 分開**——所以 schema 是 `{"segments": [...]}`
    而不是單一物件。

    **v0.4：預設不送圖。** 投影片的文字已由 S4a 讀出、以文字掛進 prompt；
    再送一次圖等於讓同一個模型既產生來源又產生待驗證的內容（R9），
    而且會把 D20 的錯位風險帶回來。`cfg.send_images` 要明確打開才送。
    """
    from .providers import Part, generate

    send_images = bool(getattr(cfg, "send_images", False))

    # 文字與圖**交錯**：每張圖緊接在它自己的區段標頭之後。
    # 首跑用「文字全在前、圖全在後」，30.6% 的區段拿到隔壁的圖（D20）。
    parts: list[Part] = []
    for text, image_key in build_parts(segments, prev_summary, slide_context, send_images):
        parts.append(Part(text=text))
        if image_key is None:
            continue
        path = image_paths.get(image_key)
        if path and path.exists():
            parts.append(Part(image=path.read_bytes()))
        else:
            # 圖不見了就明說，**不要靜靜跳過**——靜靜跳過會讓後面的圖遞補
            # 上來，整批的對應全部位移。
            log.warning("%s：代表畫面 %s 不存在，改以純逐字稿處理",
                        seg_id_of(segments, image_key), image_key)
            parts.append(Part(text="（這一段的代表畫面檔案遺失，請當作沒有畫面處理。）"))

    result = generate(cfg.model, SYSTEM_PROMPT, parts, RESPONSE_SCHEMA)
    return BatchResult(
        per_segment={s["segment_id"]: s for s in result.payload.get("segments", [])},
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        model_used=result.model_used,
    )


#: 舊名保留，避免既有呼叫端與測試一次全斷。
call_gemini = call_model


def validate_corrections(raw: dict, segment) -> list:
    """把模型回傳的校正轉成 IR 的 Correction，並**丟掉對不上原文的**。

    §5.3 不變量 10 要求每一筆的 `from` 字串實際出現在 `transcript_raw` 中。
    模型偶爾會回傳改寫過的片段（例如加了標點、或抄成了正確版本），那種
    校正無法套用也無法稽核，**丟掉並記錄**——不做模糊比對硬套上去。

    R13 再加一道：不變量 10 只驗 `from`，驗不到 `to` 是否加了原文沒有的
    內容。`unauthorized_reason` 擋掉插入與語意改寫，見
    `weft.validation.corrections`。
    """
    from ..ir import Correction, CorrectionMethod
    from ..validation.corrections import unauthorized_reason

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
        reason = unauthorized_reason(from_text, to_text)
        if reason is not None:
            log.warning("%s：校正 %r→%r 超出授權，已丟棄（R13）——%s",
                        segment.segment_id, from_text, to_text, reason)
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


def _replace_outside_existing(text: str, from_text: str, to_text: str) -> str:
    """把 `from_text` 換成 `to_text`，但**跳過已經是 `to_text` 一部分的位置**。

    D22 的第二個 bug：`未 → 未來` 用 `str.replace` 會把**已經正確的**
    「未來」裡的「未」也換掉——「現在、過去、未來」變成「未來來」。
    這類「截斷補全」的校正（R13 判定為合法）只要 `from` 是 `to` 的子字串，
    全域取代就必然過度套用。

    作法：先標出文字中所有既有的 `to_text` 位置當保護區，
    只取代保護區**之外**的 `from_text`。

    這同時保證冪等——套用結果本身不含保護區外的 `from_text`，再跑一次不會變。
    """
    if from_text not in to_text:
        return text.replace(from_text, to_text)

    protected: list[tuple[int, int]] = []
    start = 0
    while (i := text.find(to_text, start)) != -1:
        protected.append((i, i + len(to_text)))
        start = i + len(to_text)

    out: list[str] = []
    pos = 0
    while pos < len(text):
        i = text.find(from_text, pos)
        if i == -1:
            out.append(text[pos:])
            break
        end = i + len(from_text)
        inside = any(a <= i and end <= b for a, b in protected)
        out.append(text[pos:i])
        out.append(from_text if inside else to_text)
        pos = end
    return "".join(out)


def apply_corrections(transcript, segment, corrections) -> None:
    """把校正套用到該 segment 涵蓋的逐字稿句子上。

    **只改 `text_corrected`，`text_raw` 永不覆寫**（§4.5 約束 3、§5.3 不變量 9）。
    套用後重建 `segment.transcript_corrected`，讓 segment 與 cue 兩層一致——
    不一致的話 §5.4 的溯源檢查會拿到與 debug markdown 不同的文字。

    **一律從 `text_raw` 重新推導，不在 `text_corrected` 上疊加**（D22）。
    原本是 `text = cue.text_corrected or cue.text_raw`，續跑時等於在已校正的
    文字上再套一次：`未→未來` 跑第二次會變成 `未來來`、第三次 `未來來來`。
    實測第二次跑同一支影片，溯源通過率從 98.6% 掉到 47.9%。
    `text_corrected` 必須是 `text_raw + corrections` 的**純函數**——
    這樣重跑幾次都一樣，才符合 §6.3 的續跑要求。
    """
    by_index = {c.index: c for c in transcript.cues}
    for index in segment.cue_indices:
        cue = by_index.get(index)
        if cue is None:
            continue
        text = cue.text_raw
        applied = []
        for correction in corrections:
            if correction.from_text in cue.text_raw:
                text = _replace_outside_existing(
                    text, correction.from_text, correction.to_text)
                applied.append(correction)
        cue.text_corrected = text
        cue.corrections = applied

    picked = [by_index[i] for i in segment.cue_indices if i in by_index]
    segment.transcript_corrected = "".join(c.text_corrected or c.text_raw for c in picked)
    segment.corrections = list(corrections)


def to_understanding(raw: dict, segment, cfg, slide_obj=None):
    """把模型回傳的 dict 轉成 IR 的 Understanding。

    `provenance` 在 IR 中是必填且不得為 null（§3.4、§5.3 不變量 7）。
    模型漏填時**不補假值**——那會讓溯源檢查失去意義。改為丟掉該 block
    並記錄，讓 unverified 比例如實反映品質。
    """
    from ..ir import ContentBlock, ContentType, Provenance, ProvenanceKind, Understanding

    # v0.4：`is_slide` 與 `slide_text` 由 **S4a** 決定（§4.7a）。
    # 這裡從 slide 物件讀回來，而不是從模型回傳讀——S4c 已經不產生它們。
    slide = slide_obj if slide_obj is not None else None
    is_slide = bool(slide is not None and slide.slide_text)
    slide_text = (slide.slide_text if slide is not None else None) or ""

    blocks = []
    for i, item in enumerate(raw.get("content_blocks", [])):
        kind = item.get("provenance_kind")
        if not kind:
            log.warning("%s 的 block#%d 缺 provenance_kind，已丟棄（§5.3 不變量 7）",
                        segment.segment_id, i)
            continue
        # 佔位值，之後由 `_canonicalize_refs` 換成管線的權威值。
        # **不能留空**——`Provenance.ref` 在 IR 是必填（§5.3 不變量 7）。
        ref = "?"
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
        reject_reason=(slide.reject_reason if slide is not None else None),
        slide_text=slide_text or None,
        corrections=validate_corrections(raw, segment),
        summary=(raw.get("summary") or "").strip(),
        layout_description=(slide.layout_description if slide is not None else None),
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
