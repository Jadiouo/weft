"""S4a — 投影片理解。SDD §4.7a（v0.4）。

處理單位是 **S1c 去重後的代表幀**，不是 S1b 的候選幀。
實測 49 個候選幀去重後只有 22 張代表幀。

**這一步只讀圖，不含逐字稿。** §5.4 的溯源基準必須獨立於待驗證的內容——
把逐字稿一起餵進來，模型就可能把聽到的字寫進 `slide_text`，
而那正是後面要拿來驗證逐字稿的東西。

**一張圖一次呼叫，不批次。** D20 的錯位就是批次造成的
（一批 3 段的文字放前面、圖片裸接在後面，30.6% 的區段拿到隔壁的圖），
而 S1c 之後呼叫數已經大幅下降，批次省不到什麼。
"""

from __future__ import annotations

import json
import logging

from ..ir import Slide
from .providers import Part, generate

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """你在為一個「講經影片 → 可檢索知識庫」的系統讀投影片。
你會拿到一張從影片中抽出的靜止畫面。

### 1. 判斷這張圖是不是投影片（`is_slide`）

**問兩件事，兩個都要成立才是投影片。**

**（一）畫面上有沒有「成段的、讀得出來的講解內容」？**

有：整頁經文、圖表、解剖圖、多行條列說明、本集的章節標題橫幅。
沒有：只有一兩個大字的標語、一行字幕、純圖案、看不清楚的遠景。

**（二）那些內容是不是「這一集特有的」？**

換到同系列的另一集還會一模一樣出現的，都不算——那是節目包裝與拍攝現場：

- 片頭的關鍵詞（「醫學困境」「遠古啟示」「整體視角」「未來之路」）
  搭配講者與一行字幕
- 頻道品牌畫面：太極水墨、logo、節目名稱
- 系列名稱卡（「古典醫學之 人體設計系列」）
- 主講人的姓名、現任、學經歷
- 攝影棚背板或講堂佈景上的書法標語
  （「真普眾」「五常導師宣言」「追求法喜的身體健康…」）
- 片尾的訂閱／按讚畫面

**講者有沒有入鏡不影響判定**，但他入鏡時要特別檢查第（一）題：

- 他站在投影幕前，畫面另一半是清楚可讀的解剖圖 → **是投影片**
- 片頭裡他配上一個大字加一行字幕 → **不是**，那不是成段的講解內容
- 他的學經歷雖然是成段的文字，但第（二）題不過 → **不是**

`is_slide: false` 時填 `reject_reason`（一句話），其餘欄位留空字串。

### 2. 逐字轉錄（`slide_text`）

**先抄，再詮釋。** 把畫面上的文字**原樣**打出來，保留換行與排列順序。

- 直排請由右至左、由上而下
- **多欄並排時，同一列的左右欄要寫在同一行**——
  「一月為胞，　精血凝也。」是一整句，不可拆成兩欄分別抄完
- 標點照畫面上的實際字元
- 純裝飾的背景文字不要抄

這一欄是後續溯源檢查的比對基準，**不要在這裡改寫、摘要或補充**。

### 3. 版面描述（`description`）

用一段文字說明這一頁的**版面結構與它表達的關係**：箭頭指向什麼、
雙欄如何對應、色彩編碼代表什麼、圖片畫的是什麼。

RAG 讀不到圖，所以「看得懂這張圖的人才知道的事」必須寫成文字。

**這一欄可以詮釋，但不得陳述畫面上沒有的資訊。** 不確定顏色、位置、
數量時，寧可不寫也不要猜——這一欄沒有自動的溯源檢查擋得住編造。

只輸出 JSON。"""

#: 欄位順序有意義：structured output 逐欄生成，先產生的不能回頭改。
#: 這個順序強制模型**先抄再詮釋**（SDD §4.7a）。
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "is_slide": {"type": "boolean"},
        "reject_reason": {"type": "string"},
        "slide_text": {"type": "string"},
        "description": {"type": "string"},
    },
    "required": ["is_slide", "reject_reason", "slide_text", "description"],
}

USER_PROMPT = "這是不是投影片？若是，逐字轉錄並描述版面。"

#: 只做分類的 prompt 與 schema（§2.3 的 S4a-1）。
#: 分開的理由：**分類錯的代價比轉錄錯高**——把講者鏡頭判成投影片，
#: 攝影棚佈景的書法就成了 §5.4 的「合法來源」。實跑單模型時 7 個誤報，
#: 溯源未通過比例 24.3%。
CLASSIFY_SYSTEM = """你在判斷一張從講經影片抽出的靜止畫面**是不是投影片**。

**問兩件事，兩個都要成立才是投影片。**

**（一）畫面上有沒有「成段的、讀得出來的講解內容」？**

有：整頁經文、圖表、解剖圖、多行條列說明、本集的章節標題橫幅。
沒有：只有一兩個大字的標語、一行字幕、純圖案、看不清楚的遠景。

**（二）那些內容是不是「這一集特有的」？**

換到同系列的另一集還會一模一樣出現的，都不算——那是節目包裝與拍攝現場：

- 片頭的關鍵詞（「醫學困境」「遠古啟示」「整體視角」「未來之路」）
  搭配講者與一行字幕
- 頻道品牌畫面：太極水墨、logo、節目名稱
- 系列名稱卡（「古典醫學之 人體設計系列」）
- 主講人的姓名、現任、學經歷
- 攝影棚背板或講堂佈景上的書法標語
  （「真普眾」「五常導師宣言」「追求法喜的身體健康…」）
- 片尾的訂閱／按讚畫面

**講者有沒有入鏡不影響判定**，但他入鏡時要特別檢查第（一）題：

- 他站在投影幕前，畫面另一半是清楚可讀的解剖圖 → **是投影片**
- 片頭裡他配上一個大字加一行字幕 → **不是**，那不是成段的講解內容
- 他的學經歷雖然是成段的文字，但第（二）題不過 → **不是**

只輸出 JSON。"""

CLASSIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "reason": {"type": "string"},
        "is_slide": {"type": "boolean"},
    },
    "required": ["reason", "is_slide"],
}

#: 欄位順序：**先講理由再判定**。實測（R16 §3）先描述再分類把準確率從
#: 76.2% 拉到 81.0%——模型先用文字看過一遍，判斷時那些話已經在上下文裡。
#: 那次量測 n=21 不足以下結論，但**分類是短輸出**，這裡的代價只有幾個 token，
#: 而分類錯的代價很高，所以採用。


def classify_slide(spec: str, image: bytes) -> dict:
    """只判斷是不是投影片。輸出短，不會撞到 context 上限。"""
    result = generate(spec, CLASSIFY_SYSTEM,
                      [Part(text="這是不是投影片？"), Part(image=image)],
                      CLASSIFY_SCHEMA, temperature=CLASSIFY_TEMPERATURE)
    return {"is_slide": bool(result.payload.get("is_slide")),
            "reject_reason": (result.payload.get("reason") or "").strip(),
            "classifier_used": result.model_used}


#: 逐字轉錄用 **temperature 0**。這不是創作——同一張圖應該永遠讀出同一份
#: 文字。而且去重之後每張投影片**只轉錄一次**，那一次就決定了它所有出現
#: 時段的 §5.4 溯源基準；取樣造成的變異在這裡沒有任何好處。
#: （R14 量到 qwen2.5vl 對同一張圖重複轉錄的變異是 18.0%。）
TRANSCRIBE_TEMPERATURE = 0.0
CLASSIFY_TEMPERATURE = 0.0


def understand_slide(spec: str, image: bytes) -> dict:
    """單張投影片。回傳 `{is_slide, reject_reason, slide_text, description, model_used}`。"""
    result = generate(spec, SYSTEM_PROMPT, [Part(text=USER_PROMPT), Part(image=image)],
                      RESPONSE_SCHEMA, temperature=TRANSCRIBE_TEMPERATURE)
    payload = dict(result.payload)
    payload["model_used"] = result.model_used
    payload["_tokens"] = (result.input_tokens, result.output_tokens)
    return payload


#: 第二個模型的描述任務（R23）。**只描述、不轉錄**——要比對的是
#: 「這張圖上有什麼」，混進逐字轉錄會讓兩份文字因為 OCR 差異而失分，
#: 量到的就不是描述的可信度了。
DESCRIBE_SYSTEM = """你在描述一張講經投影片的**版面與圖像內容**。

只寫你**確實看得見**的東西：有幾欄、標題在哪、有沒有圖、圖畫的是什麼、
用了哪些顏色。**不要**推測講者想表達什麼，**不要**補充背景知識，
**不要**逐字抄錄投影片上的文字（那由另一個步驟負責）。

不確定的就不要寫。寫少不扣分，寫錯才扣分。

只輸出 JSON。"""

DESCRIBE_SCHEMA = {
    "type": "object",
    "properties": {"description": {"type": "string"}},
    "required": ["description"],
}


def describe_slide(spec: str, image: bytes) -> dict:
    result = generate(spec, DESCRIBE_SYSTEM,
                      [Part(text="這張圖上有什麼？"), Part(image=image)],
                      DESCRIBE_SCHEMA, temperature=TRANSCRIBE_TEMPERATURE)
    return {"description": (result.payload.get("description") or "").strip(),
            "model_used": result.model_used}


def description_agreement(first: str, second: str) -> float:
    """兩份描述的雙向 bigram containment 取小者。

    **取小不取平均**：一份詳細、一份簡略時，簡略那份會被詳細那份完全包含，
    單向分數接近 1.0 卻什麼也沒驗證到。取小者才會反映「兩邊都提到的比例」。
    """
    from ..validation.provenance import containment

    if not first or not second:
        return 0.0
    return min(containment(first, second), containment(second, first))


def _cache_path(work, slide_id: str):
    return work.slide_understanding_dir / f"{slide_id}.json"


def _load_cached(work, slide_id: str, spec: str, prompt_version: str) -> dict | None:
    path = _cache_path(work, slide_id)
    if not path.exists():
        return None
    try:
        cached = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 —— 壞掉的快取當作沒有
        return None
    if cached.get("model_used") != spec or cached.get("prompt_version") != prompt_version:
        return None
    return cached


def apply_to_slide(slide: Slide, payload: dict) -> None:
    """把結果套到 Slide 上。**快取命中與新呼叫都要走這裡**（D22）。"""
    if payload.get("is_slide"):
        slide.slide_text = (payload.get("slide_text") or "").strip() or None
        slide.layout_description = (payload.get("description") or "").strip() or None
        slide.reject_reason = None
    else:
        # 不是投影片就不留文字——留著會被 §5.4 當成合法的比對來源
        slide.slide_text = None
        slide.layout_description = None
        slide.reject_reason = (payload.get("reject_reason") or "").strip() or "未說明"
    agree = payload.get("description_agreement")
    slide.description_agreement = None if agree is None else float(agree)


def s4a_understand_slides(cfg, work, slides: list[Slide], on_call=None) -> dict:
    """對所有**代表幀**跑投影片理解。就地更新 `slides`。

    **分兩趟：先全部分類，再全部轉錄。** 不是為了好看——
    16 GB 放不下兩個 VLM，逐張交錯會讓 ollama 在每張圖之間換載模型。
    實測 22 張交錯跑約 6 分鐘（16s/張），而 qwen2.5vl 單獨跑是 1.7s/張；
    換載不只慢，還在記憶體壓力下產生過一次**不完整的轉錄**——
    而去重之後那一次擲骰決定了該張投影片**所有出現時段**的溯源基準。

    `on_call(spec, in_tokens, out_tokens, slide_id, status)` 供額度記帳。
    失敗行為（§4.7a）：仍失敗則該張留空並記錄，**不得以部分輸出充數**。
    """
    from .dedup import representatives

    p = cfg.s4a
    work.slide_understanding_dir.mkdir(parents=True, exist_ok=True)
    reps = representatives(slides)
    stats = {"representatives": len(reps), "done": 0, "failed": 0,
             "is_slide": 0, "cached": 0}

    todo: list[Slide] = []
    for slide in reps:
        cached = _load_cached(work, slide.slide_id, p.model, p.prompt_version)
        if cached is not None:
            apply_to_slide(slide, cached)
            stats["cached"] += 1
            stats["done"] += 1
            stats["is_slide"] += bool(cached.get("is_slide"))
        else:
            todo.append(slide)

    # ---- 第 0 趟：跨集重現前濾（R26）--------------------------------
    # 在別集也一模一樣出現過的畫面，不必問模型就知道是節目包裝。
    # **零誤殺**（實測四集 26 張投影片一張都沒被砍），所以放在最前面：
    # 被它剔掉的不必再花一次分類呼叫。
    if p.cross_episode_filter:
        todo = _cross_episode_filter(cfg, work, todo, stats)

    # ---- 第 1 趟：分類（只在分類另用模型時）--------------------------
    verdicts: dict[str, dict] = {}
    if p.classifier_model and p.classifier_model != p.model:
        for slide in list(todo):
            image = (work.dir / slide.image_path).read_bytes()
            try:
                verdict = _retry(lambda img=image: classify_slide(p.classifier_model, img),
                                 p, on_call, slide.slide_id)
            except Exception as exc:  # noqa: BLE001
                log.error("S4a %s：%s 分類失敗，留空並繼續——%s",
                          work.video_id, slide.slide_id, str(exc)[:160])
                stats["failed"] += 1
                todo.remove(slide)
                continue
            if on_call:
                on_call(p.classifier_model, 0, 0, slide.slide_id, "ok")
            verdicts[slide.slide_id] = verdict
            if not verdict["is_slide"]:
                # 非投影片**不必轉錄**——省一次呼叫，也不會產生假的溯源來源
                payload = {**verdict, "slide_text": "", "description": "",
                           "model_used": p.model, "prompt_version": p.prompt_version}
                _write_cache(work, slide.slide_id, payload)
                apply_to_slide(slide, payload)
                stats["done"] += 1
                todo.remove(slide)
                log.info("S4a %s：%s 判定不是投影片（%s）", work.video_id,
                         slide.slide_id, verdict["reject_reason"] or "未說明")
        log.info("S4a %s：第 1 趟分類完成，%d 張進入轉錄", work.video_id, len(todo))

    # ---- 第 2 趟：轉錄 ----------------------------------------------
    for slide in todo:
        image = (work.dir / slide.image_path).read_bytes()
        try:
            payload = _retry(lambda img=image: understand_slide(p.model, img),
                             p, on_call, slide.slide_id)
        except Exception as exc:  # noqa: BLE001
            log.error("S4a %s：%s 轉錄失敗，留空並繼續——%s",
                      work.video_id, slide.slide_id, str(exc)[:160])
            stats["failed"] += 1
            continue

        if slide.slide_id in verdicts:
            # 分類已由第 1 趟定案，**不讓轉錄模型推翻**——分工的意義就在
            # 各做各擅長的，讓不擅長分類的那個有否決權等於白拆。
            payload["is_slide"] = True
            payload["reject_reason"] = ""

        if on_call:
            in_tok, out_tok = payload.pop("_tokens", (0, 0))
            on_call(p.model, in_tok, out_tok, slide.slide_id, "ok")
        payload.pop("_tokens", None)
        payload["prompt_version"] = p.prompt_version
        _write_cache(work, slide.slide_id, payload)

        apply_to_slide(slide, payload)
        stats["done"] += 1
        stats["is_slide"] += bool(payload.get("is_slide"))
        if not payload.get("is_slide"):
            log.info("S4a %s：%s 判定不是投影片（%s）", work.video_id, slide.slide_id,
                     payload.get("reject_reason") or "未說明")

    # ---- 第 3 趟：描述一致性把關（R23，選配）-------------------------
    # **不阻擋產出。** §5.4 對 `圖表描述` 的分離度是 0.00x——跨語言時
    # 忠實的描述本身 containment 就是 0，那型別沒有任何自動閘門可用。
    # 這一趟做的是**把 §5.6 的人工抽檢引導到最可能出錯的幾張**：
    # 目前抽檢沒有優先序，等於全片平均分配注意力。
    # R23 量到多模型一致性對「編造的事實主張」分離度 13.59x。
    if p.description_checker_model:
        _check_descriptions(cfg, work, reps, on_call, stats)

    # **全軍覆沒時大聲失敗，不要靜靜產出空的。**
    # 實測：ollama 服務沒開時 19 張全部連線失敗，每一張都「留空並繼續」，
    # 管線照樣往下走並產出一份沒有任何投影片文字的知識庫——
    # 那是環境壞了，不是「這支影片沒有投影片」。與 D22 的 rehydrate 同一類。
    attempted = stats["representatives"] - stats["cached"]
    if attempted and stats["failed"] == attempted:
        raise RuntimeError(
            f"S4a {work.video_id}：{attempted} 張代表幀**全部失敗**。"
            f"這通常是環境問題（模型服務沒開、模型名打錯、VRAM 不足），"
            f"不是素材問題。已中止，不產出空的投影片文字。"
        )

    # 被合併的候選幀沿用代表幀的結果——同一張投影片本來就該有同一份文字
    _propagate(slides)

    log.info("S4a %s：%d 張代表幀（快取 %d、失敗 %d），其中 %d 張判定為投影片",
             work.video_id, stats["representatives"], stats["cached"],
             stats["failed"], stats["is_slide"])
    return stats


def _cross_episode_filter(cfg, work, todo: list[Slide], stats: dict) -> list[Slide]:
    """剔除在別集也一模一樣出現過的畫面。回傳還需要問模型的那些。

    **參考集不足時明說並跳過，不靜默降級**（§5.5 #6 的精神）——
    這一步依賴「同系列其他集也處理過」，而那個前提不見得成立。
    """
    from .recurring import (
        MIN_REFERENCE_VIDEOS,
        count_reference_videos,
        recurring_slide_ids,
        reference_frames,
    )

    p = cfg.s4a
    work_dir = work.dir.parent
    n_ref = count_reference_videos(work_dir, work.video_id)
    if n_ref < MIN_REFERENCE_VIDEOS:
        log.warning(
            "S4a %s：跨集前濾已開啟，但只找到 %d 支參考影片（至少要 %d 支），"
            "**本次跳過**。片頭片尾類的誤報會回到模型身上（R26）。",
            work.video_id, n_ref, MIN_REFERENCE_VIDEOS)
        stats["cross_episode"] = "skipped:參考集不足"
        return todo

    refs = reference_frames(work_dir, work.video_id)
    hits = recurring_slide_ids(work, todo, p.cross_episode_mae, refs)
    stats["cross_episode"] = len(hits)
    if not hits:
        log.info("S4a %s：跨集前濾（%d 支參考、%d 張代表幀）沒有剔除任何畫面",
                 work.video_id, n_ref, len(refs))
        return todo

    remaining = []
    for slide in todo:
        mae = hits.get(slide.slide_id)
        if mae is None:
            remaining.append(slide)
            continue
        payload = {
            "is_slide": False,
            "reject_reason": (f"跨集重現：與其他 {n_ref} 集的代表幀最小灰階差異僅 "
                              f"{mae:.2f}（門檻 {p.cross_episode_mae}），"
                              f"是每集都一樣的節目包裝，不是本集的講解內容"),
            "slide_text": "", "description": "",
            "model_used": p.model, "prompt_version": p.prompt_version,
        }
        _write_cache(work, slide.slide_id, payload)
        apply_to_slide(slide, payload)
        stats["done"] += 1
    log.info("S4a %s：跨集前濾（%d 支參考、%d 張代表幀）剔除 %d 張，%d 張進入分類",
             work.video_id, n_ref, len(refs), len(hits), len(remaining))
    return remaining


def _check_descriptions(cfg, work, reps: list[Slide], on_call, stats: dict) -> None:
    """用第二個模型重描述一次，記下一致度。**只標記，不改寫也不擋。**

    只跑判定為投影片、且第一份描述非空的代表幀——非投影片沒有描述可比，
    第一份是空的表示轉錄那趟就失敗了，那是另一個問題。
    """
    p = cfg.s4a
    flagged = 0
    for slide in reps:
        if not slide.layout_description:
            continue
        cached = _load_cached(work, slide.slide_id, p.model, p.prompt_version) or {}
        if cached.get("description_agreement") is not None:
            slide.description_agreement = float(cached["description_agreement"])
            continue
        image = (work.dir / slide.image_path).read_bytes()
        try:
            second = _retry(lambda img=image: describe_slide(p.description_checker_model, img),
                            p, on_call, slide.slide_id)
        except Exception as exc:  # noqa: BLE001
            # 把關失敗**不影響主產出**——它本來就只是抽檢的排序訊號
            log.warning("S4a %s：%s 描述複驗失敗，略過——%s",
                        work.video_id, slide.slide_id, str(exc)[:120])
            continue
        if on_call:
            on_call(p.description_checker_model, 0, 0, slide.slide_id, "ok")
        score = description_agreement(slide.layout_description, second["description"])
        slide.description_agreement = score
        cached["description_agreement"] = score
        cached["description_checker"] = second["model_used"]
        cached["description_second"] = second["description"]  # 留著供人工比對
        _write_cache(work, slide.slide_id, cached)
        if score < p.description_agreement_min:
            flagged += 1
            log.info("S4a %s：%s 描述一致度 %.3f < %.2f，建議人工複核",
                     work.video_id, slide.slide_id, score, p.description_agreement_min)
    stats["description_flagged"] = flagged


def _retry(fn, p, on_call, slide_id):
    from .understand import with_retries

    def _attempt(ok: bool, exc, _sid=slide_id):
        if on_call and not ok:
            on_call(p.model, 0, 0, _sid, "error")

    return with_retries(fn, p.max_retries, p.retry_backoff_sec, on_attempt=_attempt)


def _write_cache(work, slide_id: str, payload: dict) -> None:
    _cache_path(work, slide_id).write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")


def _propagate(slides: list[Slide]) -> None:
    by_id = {s.slide_id: s for s in slides}
    for slide in slides:
        if slide.duplicate_of and slide.duplicate_of in by_id:
            rep = by_id[slide.duplicate_of]
            slide.slide_text = rep.slide_text
            slide.layout_description = rep.layout_description
            slide.reject_reason = rep.reject_reason
            # 衍生狀態要**每一條路都重建**（D22）——漏掉這行，被合併的
            # 候選幀在人工抽檢清單上永遠是「沒量過」，而它們共用同一份描述
            slide.description_agreement = rep.description_agreement


def rehydrate(cfg, work, slides: list[Slide]) -> int:
    """續跑時從快取重建投影片文字。回傳套用的張數。

    **衍生狀態不能只在「新計算」那條路上做**（D22）——不重建的話，
    續跑時 S4c 拿到的 `slide_context` 是空的，等於白拆。
    """
    p = cfg.s4a
    applied = 0
    for slide in slides:
        if slide.duplicate_of:
            continue
        cached = _load_cached(work, slide.slide_id, p.model, p.prompt_version)
        if cached is None:
            continue
        apply_to_slide(slide, cached)
        applied += 1

    _propagate(slides)
    return applied
