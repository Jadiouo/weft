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


def naive_chunks(size_chars: int = 200) -> list[dict]:
    """對照組：直接把逐字稿切成固定長度，不做任何語意分段也不做理解。

    **沒有這一組，R@1=0.944 證明不了任何事。** 一個 69 筆的小語料上，
    任何合理的表示法都可能把正確答案排到第一——那時量到的是語料太小，
    不是 chunk 切得好。這與 §5.2 那四項「只 assert 常數值」的門檻同一類：
    綠燈不等於有在管。

    對照組的 gold 對映用**時間重疊**：真 chunk 的時間範圍落在哪個 naive
    chunk 裡，那個就是對應的正確答案。
    """
    out = []
    for vid in ("cxrqHABhWOU", "2FjApOVIbUs"):
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


def remap_gold(questions: list[dict], real: list[dict], naive: list[dict]) -> list[dict]:
    """把題目的 gold 從真 chunk 對映到時間重疊的 naive chunk。"""
    by_id = {c["id"]: c["metadata"] for c in real}
    out = []
    for q in questions:
        gold = set()
        for g in q["gold"]:
            m = by_id.get(g)
            if not m:
                continue
            for n in naive:
                if (n["video_id"] == m["video_id"]
                        and min(m["t_end"], n["t_end"]) - max(m["t_start"], n["t_start"]) > 0):
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
        print(f"⚠️ 標註指向不存在的 chunk（產出變了？）：{missing}")

    print(f"{len(chunks)} 個 chunk，{len(questions)} 題\n")
    out = {}
    for name in MODELS:
        model = SentenceTransformer(name, device="cuda")
        r = evaluate(model, chunks, questions)
        out[name] = r
        tag = name.split("/")[-1]
        print(f"{tag:<44} R@1={r['recall@1']:.3f} R@3={r['recall@3']:.3f} "
              f"R@5={r['recall@5']:.3f} R@10={r['recall@10']:.3f} MRR={r['mrr']:.3f}")

    # 哪幾題找不到——只給平均分看不出是哪一類問題會失敗
    ref = out[MODELS[0]]["ranks"]
    print("\n排名較差的題（第一個模型）：")
    for q, rank in sorted(zip(questions, ref, strict=True),
                          key=lambda x: (x[1] is None, -(x[1] or 0)), reverse=True)[:6]:
        print(f"  rank {rank if rank else '未命中'}：{q['q']}")

    # ---- 對照組：naive 固定視窗 ----------------------------------
    naive = naive_chunks()
    naive_q = remap_gold(questions, chunks, naive)
    print(f"\n對照組（固定 200 字視窗，無語意分段、無理解）：{len(naive)} 個 chunk")
    model = SentenceTransformer(MODELS[0], device="cuda")
    nr = evaluate(model, naive, naive_q)
    print(f"{'naive-200':<44} R@1={nr['recall@1']:.3f} R@3={nr['recall@3']:.3f} "
          f"R@5={nr['recall@5']:.3f} MRR={nr['mrr']:.3f}")
    out["naive-200"] = nr

    (HERE / "results.json").write_text(
        json.dumps({"chunks": len(chunks), "questions": len(questions), "models": out},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
