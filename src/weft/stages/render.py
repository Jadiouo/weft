"""chunks.jsonl 與 debug markdown。SDD §4.9、§3.5。

chunk 是**產品**——最終進向量庫的就是這些。§3.5 的自足性要求因此是硬約束：
「`text` 欄位不得包含『如上圖』『前面提到的』等指涉性語句，也不得包含
markdown 裝飾。」
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

log = logging.getLogger(__name__)

#: §3.5 的指涉性語句。S4 的 prompt 已要求展開，這裡是最後一道檢查——
#: 漏網的會被記錄，不會靜默進入產品。
_REFERENTIAL = re.compile(
    r"(如上圖|如下圖|上圖|下圖|前面提到|前述|上述|如前所述|這個式子|這張圖|"
    r"剛才說的|以下|如右|如左)"
)
#: markdown 裝飾。向量庫存的是純文字，`**` 會變成檢索雜訊。
_MARKDOWN = re.compile(r"(\*\*|__|^#{1,6}\s|^\s*[-*+]\s|`{1,3})", re.MULTILINE)

#: 中文句末標點。超長 block 按句切分時用（§4.9）。
_SENTENCE_END = re.compile(r"(?<=[。！？；])")


def strip_markdown(text: str) -> str:
    """移除 markdown 裝飾，保留文字本身。"""
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"__(.+?)__", r"\1", text)
    text = re.sub(r"`{1,3}(.+?)`{1,3}", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
    return re.sub(r"\s+", " ", text).strip()


def referential_phrases(text: str) -> list[str]:
    """找出違反 §3.5 自足性的指涉語句。"""
    return sorted(set(_REFERENTIAL.findall(text)))


def split_long_text(text: str, max_chars: int) -> list[str]:
    """超過上限則按句切分。SDD §4.9。

    切在句末標點，不切在字數上限——切在句中會產生讀不通的 chunk，而
    chunk 是要單獨被檢索出來給人看的。
    """
    if len(text) <= max_chars:
        return [text]

    pieces: list[str] = []
    current = ""
    for sentence in _SENTENCE_END.split(text):
        if not sentence:
            continue
        if current and len(current) + len(sentence) > max_chars:
            pieces.append(current)
            current = sentence
        else:
            current += sentence
    if current:
        pieces.append(current)

    # 單一句子就超長（沒有標點的長段落）→ 只能硬切，但這值得記錄
    final: list[str] = []
    for piece in pieces:
        if len(piece) <= max_chars:
            final.append(piece)
            continue
        log.warning("單一句子超過 %d 字且無標點可切，改為硬切", max_chars)
        final += [piece[i : i + max_chars] for i in range(0, len(piece), max_chars)]
    return final


def build_chunks(ir, cfg) -> tuple[list, list[str]]:
    """把 VideoIR 展成 chunks。回傳 `(chunks, 品質警告)`。

    SDD §4.9：一個 `content_block` = 一個 chunk。
    """
    from ..ir import Chunk, ChunkMetadata, VerificationStatus

    chunks: list[Chunk] = []
    warnings: list[str] = []

    for seg in ir.segments:
        if seg.understanding is None:
            continue
        for b, block in enumerate(seg.understanding.content_blocks):
            # §5.4：溯源未通過的 block 不進產品
            if block.verification is not None and block.verification is not VerificationStatus.VERIFIED:
                warnings.append(f"{seg.segment_id} block#{b} 溯源為 {block.verification.value}，已排除")
                continue

            text = strip_markdown(block.text)
            referential = referential_phrases(text)
            if referential:
                warnings.append(
                    f"{seg.segment_id} block#{b} 含指涉語句 {referential}，違反 §3.5 自足性"
                )

            for k, piece in enumerate(split_long_text(text, cfg.max_chunk_chars)):
                if not piece.strip():
                    continue
                chunks.append(
                    Chunk(
                        id=f"{seg.segment_id}#b{b:02d}" + (f"-{k}" if k else ""),
                        text=piece,
                        metadata=ChunkMetadata(
                            video_id=ir.meta.video_id,
                            series_id=ir.meta.series_id,
                            video_title=ir.meta.title,
                            episode_index=ir.meta.episode_index,
                            t_start=seg.t_start,
                            t_end=seg.t_end,
                            url=f"{ir.meta.url}&t={int(seg.t_start)}s",
                            content_type=block.type,
                            slide_ref=seg.slide_ref,
                            terms=seg.understanding.terms,
                            provenance_kind=block.provenance.kind,
                        ),
                    )
                )
    return chunks, warnings


def write_chunks(chunks: list, path: Path, append: bool = True) -> int:
    """寫出 chunks.jsonl。

    `append=True` 是為了批次跑數十支影片時每支跑完就落地，中途失敗不會
    前功盡棄。但**同一支影片重跑時必須取代自己的舊紀錄，不是再寫一遍**。

    實測（2026-08-08）：`out/chunks.jsonl` 有 428 行，相異 `id` 只有 91 個，
    最高重複 **5 次**——重跑幾次就複製幾份。這是要進向量庫的產品輸出，
    重複的 chunk 會讓同一段內容在檢索結果裡佔掉數個名額。
    `Chunk.id` 是穩定的（segment + block index），所以重複是可辨識的，
    但沒有任何一步在辨識它。

    與 `write_unverified` 同一個修法：先移除這支影片的舊行，再寫新的。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if not append:
        with path.open("w", encoding="utf-8") as fh:
            for chunk in chunks:
                fh.write(chunk.model_dump_json() + "\n")
        return len(chunks)

    video_ids = {c.metadata.video_id for c in chunks}
    kept = _lines_excluding_videos(path, video_ids)
    with path.open("w", encoding="utf-8") as fh:
        for line in kept:
            fh.write(line + "\n")
        for chunk in chunks:
            fh.write(chunk.model_dump_json() + "\n")
    return len(chunks)


def _lines_excluding_videos(path: Path, video_ids: set[str]) -> list[str]:
    """既有檔案中**不屬於**這幾支影片的行。

    `json.loads` 對 `[]`、`null`、`3` 都會成功，所以要檢查型別而不是只接
    `JSONDecodeError`。認不出來的行原樣留著——靜靜丟掉一行資料，
    比留著一行壞資料更糟。
    """
    import json

    if not path.exists():
        return []
    kept: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            vid = parsed.get("video_id") or (parsed.get("metadata") or {}).get("video_id")
            if vid in video_ids:
                continue
        kept.append(line)
    return kept


def write_debug_markdown(ir, work, path: Path) -> Path:
    """人工抽檢用（§5.6）。含內嵌圖片與可點的時間戳連結。

    §5.6 的檢查項目：投影片圖是否為該頁最完整版本、逐字稿術語是否正確、
    content_block 分類是否合理、chunk 是否自足。版面依此排列。

    **圖片連結必須從這份 markdown 的位置算相對路徑**（D23）。
    `slide.image_path` 是相對 `work/<video_id>/` 的，而這份文件寫到
    `out/debug/`——直接用會指向不存在的 `out/debug/03_slides/...`，
    整份文件一張圖都顯示不出來。而 §5.6 的人工複核**就是靠看圖**：
    D20 那個 30.6% 的圖片錯位，唯一的發現途徑就是逐張比對畫面與判定。
    """
    import os

    from ..ir import VerificationStatus

    def hhmmss(t: float) -> str:
        return f"{int(t) // 3600:02d}:{int(t) % 3600 // 60:02d}:{int(t) % 60:02d}"

    #: `work` 可能是 `WorkPaths` 也可能直接是目錄（測試就這樣傳）
    work_dir = Path(getattr(work, "dir", work))

    def image_link(image_path: str) -> str:
        """從 markdown 所在目錄指到 work 目錄下的實際圖檔。"""
        target = (work_dir / image_path).resolve()
        return os.path.relpath(target, path.parent.resolve()).replace(os.sep, "/")

    lines = [
        f"# {ir.meta.title}",
        "",
        f"- 影片：<{ir.meta.url}>",
        f"- 長度：{hhmmss(ir.meta.duration)}｜投影片 {len(ir.slides)} 張"
        f"｜segment {len(ir.segments)} 段",
    ]
    if ir.unverified_ratio is not None:
        flag = "⚠️ **needs_review**" if ir.needs_review else "✅"
        lines.append(f"- 溯源未通過比例：{ir.unverified_ratio:.1%} {flag}")
    if ir.tldr:
        lines += ["", "## TL;DR", "", ir.tldr]

    lines += ["", "---", "", "## 逐段檢視", ""]

    for seg in ir.segments:
        lines.append(
            f"### {seg.segment_id}　`{hhmmss(seg.t_start)}–{hhmmss(seg.t_end)}`"
            f"　[▶︎ 跳至此處]({ir.meta.url}&t={int(seg.t_start)}s)"
        )
        lines.append("")
        lines.append(f"模式：`{seg.mode.value}`" + (f"　投影片：`{seg.slide_ref}`" if seg.slide_ref else ""))
        if seg.boundary_shift_sec:
            lines.append(f"邊界吸附位移：{seg.boundary_shift_sec:+.1f}s（`{seg.boundary_method.value}`）")
        lines.append("")

        understanding = seg.understanding
        rejected = understanding is not None and not understanding.is_slide
        shown_ref = seg.slide_ref or seg.candidate_ref
        slide = ir.slide_by_id(shown_ref) if shown_ref else None
        if slide:
            lines.append(f"![{slide.slide_id}]({image_link(slide.image_path)})")
            if rejected:
                lines.append("")
                lines.append(
                    f"> ⚠️ VLM 判定**這不是投影片**："
                    f"{understanding.reject_reason or '未說明'}"
                    "　（此段已降級為 speaker_only，上圖僅供人工複核）"
                )
            if slide.is_progressive_final:
                lines.append("")
                lines.append(
                    f"> 此頁為逐條動畫，已合併 {len(slide.build_frames)} 個 build，"
                    "上圖為內容最完整的最後一幀。"
                )
            lines.append("")

        if seg.corrections:
            # 理由必須顯示——§5.6 的抽檢要判斷「改得對不對」，只看 from→to
            # 是判斷不了的
            lines.append("**術語校正**（VLM 對照投影片）：")
            for c in seg.corrections:
                lines.append(f"- `{c.from_text}` → `{c.to_text}`"
                             + (f"　<sub>{c.reason}</sub>" if c.reason else ""))
            lines.append("")

        if understanding is None:
            lines += ["_（此段無理解結果）_", ""]
        else:
            if understanding.layout_description:
                lines += [f"**版面**：{understanding.layout_description}", ""]
            lines += [f"**摘要**：{understanding.summary}", ""]
            for b, block in enumerate(understanding.content_blocks):
                mark = ""
                if block.verification is not None and block.verification is not VerificationStatus.VERIFIED:
                    mark = f"　⚠️ `{block.verification.value}`"
                    # 成因要跟著出現。人工複核看到一串 `unverified` 而不知道
                    # 是哪一種，只能一筆一筆重查——那正是 R27 之前的狀況。
                    if block.wrong_source:
                        mark += "　`來源型別可能標錯`"
                    if block.depends_on_correction:
                        mark += "　`靠術語校正才對得上`"
                sim = f"（相似度 {block.similarity:.2f}）" if block.similarity is not None else ""
                lines.append(
                    f"- **{block.type.value}**{mark}　"
                    f"<sub>來源：`{block.provenance.kind.value}:{block.provenance.ref}`{sim}</sub>"
                )
                lines.append(f"  {block.text}")
            lines.append("")

        # **人工複核看的是 raw**——溯源基準就是它（票 01）。校正後的版本
        # 另外附上，並且只在真的不同時才附：這份文件要回答的問題之一正是
        # 「這段內容是不是靠校正才成立的」，把校正後的當成唯一的逐字稿看，
        # 那個問題就問不出來了。
        lines += ["<details><summary>逐字稿（原始，溯源基準）</summary>", "",
                  seg.transcript_raw or "（無）", "", "</details>", ""]
        if seg.transcript_corrected and seg.transcript_corrected != seg.transcript_raw:
            # **不寫筆數。** `segment.corrections` 存的是**提案**，
            # `apply_corrections` 只套用 `from_text` 真的出現在 cue 裡的那些
            # （understand.py），兩者不相等。在 §5.6 的複核文件裡寫一個
            # 比實際多的數字，等於憑空捏造一個讓人據以判斷的事實。
            lines += ["<details><summary>逐字稿（S4b 校正後）</summary>",
                      "", seg.transcript_corrected, "", "</details>", ""]
        lines += ["---", ""]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_unverified(verdict, path: Path) -> int:
    """§5.4：所有 unverified 條目寫入 out/debug/unverified.jsonl 供人工檢視。

    **同一支影片重跑時取代舊紀錄，不附加。** 原本是純附加模式，重跑一次
    就把同一批未通過再寫一遍——實測跑兩次後檔案裡有 36 筆，實際只有 18 筆。
    人工複核從這個檔案數「有幾筆要看」會數成兩倍，而且新舊程式碼產生的
    紀錄混在一起看不出來。這與 §5.6 的用途直接衝突。
    """
    import json

    from ..ir import VerificationStatus

    rows = [v for v in verdict.verdicts if v.status is not VerificationStatus.VERIFIED]
    path.parent.mkdir(parents=True, exist_ok=True)

    kept = _lines_excluding_videos(path, {verdict.video_id})
    with path.open("w", encoding="utf-8") as fh:
        for line in kept:
            fh.write(line + "\n")
        for v in rows:
            fh.write(json.dumps({
                "video_id": verdict.video_id,
                "segment_id": v.segment_id,
                "block_index": v.block_index,
                "content_type": v.content_type,
                "status": v.status.value,
                "similarity": round(v.similarity, 4),
                "copy_ratio": round(v.copy_ratio, 4),
                "missing_entities": v.missing_entities,
                # 成因分類。**混在 reason 字串裡等於沒有**——
                # 人工複核要能一眼分出「歸屬標錯」「靠校正才成立」
                # 「兩個來源都對不上」，那是三種不同的修法（R27、票 01）。
                "wrong_source": v.wrong_source,
                "depends_on_correction": v.depends_on_correction,
                "reason": v.reason,
            }, ensure_ascii=False) + "\n")
    return len(rows)
