"""投影片 OCR。SDD §4.4。本地執行，無額度限制。

SDD §2.3 指定 PaddleOCR-VL。API 在 PaddleOCR 3.x 與 2.x 之間換過（`predict`
vs `ocr`），且 `PaddleOCRVL` 這個類別只在較新版本存在——所以此處**偵測**
可用的介面，而不是綁死某一個。裝到哪個版本在使用者機器上不受我們控制，
而 OCR 失敗會讓整個詞庫是空的，S2c 跟著空轉。
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)


class OcrUnavailable(RuntimeError):
    """PaddleOCR 裝不起來或載入失敗。"""


def _build_engine(cfg):
    """回傳 `(engine, kind)`，kind 決定怎麼呼叫它。"""
    import paddleocr

    # PaddleOCR-VL：SDD §2.3 指定的模型，中文／直排強項
    if hasattr(paddleocr, "PaddleOCRVL"):
        try:
            return paddleocr.PaddleOCRVL(), "vl"
        except Exception as exc:  # noqa: BLE001 —— 模型下載失敗、權重不符等
            log.warning("PaddleOCR-VL 載入失敗（%s），退回一般 PaddleOCR", exc)

    # enable_mkldnn=False 是必要的，不是效能取捨：PaddleOCR 3.7 + paddlepaddle
    # 3.3 的 oneDNN 後端在 PIR executor 下會拋
    #   (Unimplemented) ConvertPirAttribute2RuntimeAttribute not support
    #     [pir::ArrayAttribute<pir::DoubleAttribute>]
    # 導致**每一張投影片**都 OCR 失敗。關掉後直排中文可完整讀出。
    # 見 docs/decisions.md D10。
    try:
        return (
            paddleocr.PaddleOCR(
                lang=cfg.lang, use_textline_orientation=True, enable_mkldnn=False
            ),
            "v3",
        )
    except TypeError:
        # 2.x 的參數名不同
        return paddleocr.PaddleOCR(lang=cfg.lang, use_angle_cls=True, show_log=False), "v2"


def _extract_vl(result) -> tuple[str, float]:
    """PaddleOCR-VL 回傳的是版面結構化結果。"""
    lines: list[str] = []
    scores: list[float] = []
    for page in result if isinstance(result, list) else [result]:
        data = getattr(page, "json", None) or page
        if isinstance(data, dict):
            for block in data.get("parsing_res_list") or data.get("layout_parsing_result") or []:
                text = (block.get("block_content") or block.get("text") or "").strip()
                if text:
                    lines.append(text)
            if not lines and data.get("markdown"):
                lines.append(str(data["markdown"]))
    return "\n".join(lines), (sum(scores) / len(scores) if scores else 0.0)


def _extract_classic(result) -> tuple[str, float]:
    """一般 PaddleOCR：`[[box, (text, score)], ...]`，3.x 則是 dict。"""
    lines: list[str] = []
    scores: list[float] = []

    def walk(node) -> None:
        if isinstance(node, dict):
            texts = node.get("rec_texts") or []
            confs = node.get("rec_scores") or []
            lines.extend(t for t in texts if t)
            scores.extend(float(c) for c in confs)
            return
        if isinstance(node, (list, tuple)):
            # 葉節點形如 (text, score)
            if len(node) == 2 and isinstance(node[0], str) and isinstance(node[1], (int, float)):
                lines.append(node[0])
                scores.append(float(node[1]))
                return
            for child in node:
                walk(child)

    walk(result)
    return "\n".join(lines), (sum(scores) / len(scores) if scores else 0.0)


def to_traditional(text: str) -> str:
    """統一轉繁體。

    PaddleOCR 的 `ch` 模型輸出**簡體**，但本專案的素材是繁體（zh-Hant 字幕、
    繁體投影片）。不轉的話詞庫會混入簡體詞條，而 §4.5 的術語校正是拿詞庫去
    比對繁體逐字稿——簡體詞條永遠匹配不到，等於白建。

    實測：真實素材的 slide_002 OCR 出現「万批交城」「冷顺型辉」等簡體字。
    """
    if not text:
        return text
    from zhconv import convert

    return convert(text, "zh-hant")


def run_ocr(images: list[Path], cfg) -> list[tuple[str, float]]:
    """對每張圖做 OCR，回傳 `(text, confidence)`。

    單張失敗不中斷——一張讀不出來的投影片不該讓整支影片重跑。
    """
    try:
        engine, kind = _build_engine(cfg)
    except ImportError as exc:
        raise OcrUnavailable(
            "找不到 paddleocr。SDD §8：S2 屬 pipe-gpu 環境，"
            "請在該環境安裝（conda run -n pipe-gpu pip install paddleocr）"
        ) from exc

    log.info("S2：OCR 引擎 %s，共 %d 張", kind, len(images))
    out: list[tuple[str, float]] = []
    for image in images:
        try:
            if kind == "vl":
                text, conf = _extract_vl(engine.predict(str(image)))
            elif kind == "v3":
                text, conf = _extract_classic(engine.predict(str(image)))
            else:
                text, conf = _extract_classic(engine.ocr(str(image), cls=True))
        except Exception as exc:  # noqa: BLE001
            log.warning("%s 的 OCR 失敗：%s", image.name, exc)
            text, conf = "", 0.0
        out.append((to_traditional(text) if cfg.normalise_to_traditional else text, conf))
    return out
