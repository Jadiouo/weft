"""R34（票 12）：檢索導向的可用性評測。**不用 LLM 當裁判。**

RAGAS 的三個指標（Answer Relevance / Context Relevance / Faithfulness）
**全部要 LLM 當裁判**。在這個專案裡那會是：用本地 LLM 去評判本地 LLM
的產出——正是票 01 剛修掉的「用來源 A 驗證來源 A」。
而且 R21／R26 已經量過，本地模型在這批中文素材上的表現對措辭極度敏感
（分類 0.636–1.000、`slide_text` CER 4.9–37.8%）。

改用**檢索**：把使用者真的會問的問題向量化，看正確的 chunk 排第幾。
它回答的是「我找不找得到我要的東西」——那正是「好不好用」的核心，
而且**完全機械化**，沒有裁判可以偏心。

    conda run -n pipe-gpu python -m experiments.r34_usability.retrieval
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
HERE = pathlib.Path(__file__).parent

#: 本機已快取，離線可用。**不是隨手挑的**：票 07 已經用它跑過分段，
#: 換一個模型結論會不會翻是已知的風險，所以兩個都跑。
MODELS = [
    "BAAI/bge-m3",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
]


def load_chunks() -> list[dict]:
    path = ROOT / "out" / "chunks.jsonl"
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def evaluate(model, chunks: list[dict], questions: list[dict]) -> dict:
    ids = [c["id"] for c in chunks]
    vecs = np.asarray(model.encode([c["text"] for c in chunks],
                                   normalize_embeddings=True, show_progress_bar=False))
    qvecs = np.asarray(model.encode([q["q"] for q in questions],
                                    normalize_embeddings=True, show_progress_bar=False))
    sims = qvecs @ vecs.T

    ranks: list[int | None] = []
    for row, q in zip(sims, questions, strict=True):
        order = np.argsort(-row)
        gold = set(q["gold"])
        hit = next((r for r, i in enumerate(order, start=1) if ids[i] in gold), None)
        ranks.append(hit)

    def recall_at(k: int) -> float:
        return sum(1 for r in ranks if r is not None and r <= k) / len(ranks)

    mrr = sum(1.0 / r for r in ranks if r is not None) / len(ranks)
    return {
        "recall@1": round(recall_at(1), 3),
        "recall@3": round(recall_at(3), 3),
        "recall@5": round(recall_at(5), 3),
        "recall@10": round(recall_at(10), 3),
        "mrr": round(mrr, 3),
        "ranks": ranks,
    }


#: 對照組的 gold 對映門檻：naive chunk **自身時長**有多少比例落在真 chunk 裡。
#:
#: 第一版用「任意 >0 秒重疊」，那讓對照組在一個**更容易的題目**上計分：
#: 實測 58 個 naive chunk 拿到 gold 資格，其中 12 個（21%）與真 chunk 的
#: 重疊不到自身時長的 25%——一個內容 75% 不相干的 200 字視窗被算成命中。
#: 加上候選池只有 40 個（weft 69），隨機命中率高 1.7 倍。
#: **於是「這個指標沒有區辨力」這個結論本身是建構假象。**
MIN_OVERLAP_RATIO = 0.5


def naive_chunks(video_ids: list[str], size_chars: int = 200) -> list[dict]:
    """對照組：直接把逐字稿切成固定長度，不做任何語意分段也不做理解。

    **沒有這一組，R@1=0.944 證明不了任何事。** 一個 69 筆的小語料上，
    任何合理的表示法都可能把正確答案排到第一——那時量到的是語料太小，
    不是 chunk 切得好。這與 §5.2 那四項「只 assert 常數值」的門檻同一類：
    綠燈不等於有在管。

    **`video_ids` 必須由呼叫端從真 chunk 推出來，不得寫死。** 寫死的話，
    語料一擴大（票 11 的 26 支）weft 側會讀到 800 個 chunk 而對照組還是
    那兩支，其餘題目的 gold 變成空集合、被算成未命中而不警告——
    naive 那一列會塌到真值的 2/26，報告讀起來就成了「weft 大勝 naive」，
    正好是這個實驗要抓的假象的反面。
    """
    out = []
    for vid in video_ids:
        cues = json.loads(
            (ROOT / "work" / vid / "05_transcript.json").read_text(encoding="utf-8"))["cues"]
        buf, start, idx = [], None, 0
        for c in cues:
            if start is None:
                start = c["t_start"]
            buf.append(c["text_raw"])
            if sum(len(x) for x in buf) >= size_chars:
                out.append({"id": f"{vid}#naive{idx:03d}", "text": "".join(buf),
                            "t_start": start, "t_end": c["t_end"], "video_id": vid})
                buf, start, idx = [], None, idx + 1
        if buf and start is not None:
            out.append({"id": f"{vid}#naive{idx:03d}", "text": "".join(buf),
                        "t_start": start, "t_end": cues[-1]["t_end"], "video_id": vid})
    return out


def remap_gold(questions: list[dict], real: list[dict], naive: list[dict],
               min_ratio: float = MIN_OVERLAP_RATIO) -> list[dict]:
    """把題目的 gold 從真 chunk 對映到 naive chunk。

    **門檻是 naive chunk 自身時長的比例，不是「有重疊就算」。**
    後者會讓對照組在更容易的題目上計分（見 `MIN_OVERLAP_RATIO`）——
    對照組出錯時不會長得像「對照組錯了」，會長得像「這個訊號沒那麼準」
    （R23 的教訓）。
    """
    by_id = {c["id"]: c["metadata"] for c in real}
    out = []
    for q in questions:
        gold = set()
        for g in q["gold"]:
            m = by_id.get(g)
            if not m:
                continue
            for n in naive:
                if n["video_id"] != m["video_id"]:
                    continue
                overlap = min(m["t_end"], n["t_end"]) - max(m["t_start"], n["t_start"])
                span = n["t_end"] - n["t_start"]
                if span > 0 and overlap / span >= min_ratio:
                    gold.add(n["id"])
        out.append({"q": q["q"], "gold": sorted(gold)})
    return out


def main() -> int:
    from sentence_transformers import SentenceTransformer

    chunks = load_chunks()
    questions = json.loads((HERE / "questions.json").read_text(encoding="utf-8"))["questions"]

    known = {c["id"] for c in chunks}
    missing = [g for q in questions for g in q["gold"] if g not in known]
    if missing:
        # **擋下來，不是印個警告就繼續。** 標註失效時 evaluate 會回傳
        # rank=None，recall 與 MRR 依失效比例下滑，而 results.json 照樣被
        # 覆寫成偏低的數字——讀它的人會當成「檢索變差了」而歸因到 chunk
        # 品質。那正是 §5.2 那四項假綠燈的鏡像：假紅燈一樣沒有資訊。
        raise SystemExit(
            f"標註指向 {len(missing)} 個不存在的 chunk：{missing[:5]}\n"
            f"chunks.jsonl 重新產生過（分段或 prompt 改了）。"
            f"**先重標 questions.json，不要拿失效的標註跑分數。**")

    print(f"{len(chunks)} 個 chunk，{len(questions)} 題\n")
    out = {}
    model_cache = {}
    for name in MODELS:
        # 快取：對照組要再用 MODELS[0]，重新載入 bge-m3（2.2 GB）
        # 等於同時在 16 GB 的卡上放兩個模型
        model_cache[name] = SentenceTransformer(name, device="cuda")
        r = evaluate(model_cache[name], chunks, questions)
        out[name] = r
        tag = name.split("/")[-1]
        print(f"{tag:<44} R@1={r['recall@1']:.3f} R@3={r['recall@3']:.3f} "
              f"R@5={r['recall@5']:.3f} R@10={r['recall@10']:.3f} MRR={r['mrr']:.3f}")

    # 哪幾題找不到——只給平均分看不出是哪一類問題會失敗。
    # **排序方向要對**：未命中最前，其餘按 rank 由大到小。
    # 第一版寫成 `reverse=True` 套在 `(is_none, -rank)` 上，
    # 結果把 rank 1 排到 rank 35 前面——印出來的是**最好的六題**，
    # 而那個區塊自己寫的目的是找出會失敗的那些。
    ref = out[MODELS[0]]["ranks"]
    print("\n排名較差的題（第一個模型）：")
    worst = sorted(zip(questions, ref, strict=True),
                   key=lambda x: (0 if x[1] is None else 1, -(x[1] or 0)))
    for q, rank in worst[:6]:
        print(f"  rank {rank if rank else '未命中'}：{q['q']}")

    # **模型結果先落地**，對照組失敗不得把它們一起丟掉。
    # 對照組讀 `work/*/05_transcript.json`，而 work/ 不進版控——
    # 在沒有 work/ 的機器上兩個模型的評測會白跑一遍然後拋例外。
    payload = {"chunks": len(chunks), "questions": len(questions),
               "embedding_model": MODELS[0], "models": out, "control": None}
    (HERE / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 對照組：naive 固定視窗 ----------------------------------
    video_ids = sorted({c["metadata"]["video_id"] for c in chunks})
    try:
        naive = naive_chunks(video_ids)
    except FileNotFoundError as exc:
        print(f"\n對照組跳過（缺 work/ 的逐字稿）：{exc}")
        return 0

    naive_q = remap_gold(questions, chunks, naive)
    empty = [q["q"] for q in naive_q if not q["gold"]]
    print(f"\n對照組（固定 200 字視窗，無語意分段、無理解）：{len(naive)} 個 chunk"
          f"，涵蓋 {len(video_ids)} 支")
    if empty:
        # **對映不到就明說。** 空 gold 會被 evaluate 當成未命中，
        # 那會讓對照組看起來比實際差——正好是這個實驗要抓的假象的反面。
        print(f"  ⚠️ {len(empty)} 題在對照組裡對映不到 gold，"
              f"**已排除**而非算成未命中：{empty[:3]}")
        keep = [q for q in naive_q if q["gold"]]
    else:
        keep = naive_q

    # **同一組題才可比。** weft 用 18 題、對照組用 16 題的話，
    # 分母不同，差距有多少來自題目、有多少來自 chunk 品質分不開。
    asked = {q["q"] for q in keep}
    same = [q for q in questions if q["q"] in asked]
    wr = evaluate(model_cache[MODELS[0]], chunks, same)
    nr = evaluate(model_cache[MODELS[0]], naive, keep)
    print(f"{'weft（同一組題）':<40} R@1={wr['recall@1']:.3f} R@3={wr['recall@3']:.3f} "
          f"R@5={wr['recall@5']:.3f} MRR={wr['mrr']:.3f}  （n={len(same)} 題）")
    print(f"{'naive-200':<44} R@1={nr['recall@1']:.3f} R@3={nr['recall@3']:.3f} "
          f"R@5={nr['recall@5']:.3f} MRR={nr['mrr']:.3f}  （n={len(keep)} 題）")
    print(f"{'差距':<44} R@1={wr['recall@1'] - nr['recall@1']:+.3f} "
          f"MRR={wr['mrr'] - nr['mrr']:+.3f}")

    payload["weft_on_same_questions"] = {k: v for k, v in wr.items() if k != "ranks"}
    payload["control"] = {
        "kind": "naive fixed-window 200 chars",
        "chunks": len(naive), "questions": len(keep),
        "embedding_model": MODELS[0],
        "min_overlap_ratio": MIN_OVERLAP_RATIO,
        **{k: v for k, v in nr.items() if k != "ranks"},
    }
    (HERE / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
