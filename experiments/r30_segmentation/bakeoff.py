"""R30（票 07）：四種分段方案在中文逐字稿上的對打。

**先量再做。** 提案 §6 的三個方向到現在一個都沒量測過，直接實作等於
把猜測寫成規格——這個 repo 已經為此付過三次代價（CLAUDE.md 的表）。

四個候選：

1. `slide`     現行：投影片切換驅動（baseline，已知 35% 假邊界）
2. `sbert`     TextTiling + Sentence-BERT 的語意連貫度下降
3. `hybrid`    方向 2：投影片切換降級為候選，需語意佐證才採納
4. `ngram`     中文字元 n-gram 的 TextTiling（不需要模型）

**中文適用性不能假設。** TextTiling 原為英文設計，中文沒有詞邊界；
第 4 個方案存在的理由正是「不靠詞邊界也不靠模型，純字元共現」。

跑：`~/miniconda3/envs/pipe-gpu/bin/python experiments/r30_segmentation/bakeoff.py`
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

#: 本機已快取、離線可用。bge-m3 是多語系強模型，MiniLM 是小的對照組——
#: **兩個都跑**，因為「換一個模型結論就翻掉」本身就是要知道的事。
MODELS = [
    "BAAI/bge-m3",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
]

#: 把 cue 併成 TextTiling 的「偽句」。太短則雜訊大，太長則解析度不足。
#: 40 字約是這批素材的一到兩句話。**未經校準**——掃描幾個值一起報。
BLOCK_CHARS = 40
#: TextTiling 的比較視窗（左右各幾個 block）。
WINDOW = 3


def load_video(video_id: str) -> tuple[list[dict], dict]:
    work = ROOT / "work" / video_id
    cues = json.loads((work / "05_transcript.json").read_text(encoding="utf-8"))["cues"]
    golden = json.loads(
        (ROOT / "tests" / "golden" / f"{video_id}.golden.json").read_text(encoding="utf-8"))
    return cues, golden


def blocks_from_cues(cues: list[dict], lo: float, hi: float,
                     block_chars: int = BLOCK_CHARS) -> list[dict]:
    """把 body 範圍內的 cue 併成偽句 block。回傳 `{t_start, text}`。"""
    out, buf, start = [], [], None
    for c in cues:
        if c["t_start"] < lo or c["t_end"] > hi:
            continue
        if start is None:
            start = c["t_start"]
        buf.append(c["text_raw"])
        if sum(len(x) for x in buf) >= block_chars:
            out.append({"t_start": start, "text": "".join(buf)})
            buf, start = [], None
    if buf and start is not None:
        out.append({"t_start": start, "text": "".join(buf)})
    return out


def _depth_boundaries(scores: np.ndarray, blocks: list[dict]) -> list[float]:
    """TextTiling 的深度分數與斷點選取（Hearst 1997 的標準做法）。

    `scores[i]` 是 block i 與 i+1 之間的相似度。深度 = 左右兩側的局部高點
    各減去這個谷底。門檻用 `mean - std/2`——**這是文獻的預設值，不是我調的**。
    讓方法自己決定要切幾刀，而不是餵給它正確答案的數量。
    """
    if len(scores) < 3:
        return []
    depths = np.zeros_like(scores)
    for i in range(len(scores)):
        left = scores[i]
        j = i
        while j > 0 and scores[j - 1] >= scores[j]:
            j -= 1
            left = max(left, scores[j])
        right = scores[i]
        j = i
        while j < len(scores) - 1 and scores[j + 1] >= scores[j]:
            j += 1
            right = max(right, scores[j])
        depths[i] = (left - scores[i]) + (right - scores[i])

    cutoff = depths.mean() - depths.std() / 2
    picked = [i for i in range(1, len(depths) - 1)
              if depths[i] > cutoff and depths[i] >= depths[i - 1]
              and depths[i] >= depths[i + 1]]
    return [blocks[i + 1]["t_start"] for i in picked]


def _window_similarity(vectors: np.ndarray, window: int = WINDOW) -> np.ndarray:
    """每個 block 間隙的左右視窗餘弦相似度。"""
    n = len(vectors)
    out = np.zeros(n - 1)
    for i in range(n - 1):
        left = vectors[max(0, i - window + 1): i + 1].mean(axis=0)
        right = vectors[i + 1: min(n, i + 1 + window)].mean(axis=0)
        denom = np.linalg.norm(left) * np.linalg.norm(right)
        out[i] = float(left @ right / denom) if denom else 0.0
    return out


def char_ngram_vectors(texts: list[str], n: int = 2) -> np.ndarray:
    """字元 n-gram 的計數向量。**不需要斷詞也不需要模型。**

    中文沒有詞邊界，而 R18 已經量過：ASR 的錯誤幾乎全是等長同音替換，
    bigram 只是變稀疏不是歸零。用字元共現就繞開了斷詞這一層。
    """
    vocab: dict[str, int] = {}
    grams = []
    for t in texts:
        cleaned = "".join(t.split())
        g = [cleaned[i:i + n] for i in range(max(0, len(cleaned) - n + 1))]
        grams.append(g)
        for x in g:
            vocab.setdefault(x, len(vocab))
    mat = np.zeros((len(texts), len(vocab)), dtype=np.float32)
    for r, g in enumerate(grams):
        for x in g:
            mat[r, vocab[x]] += 1.0
    return mat


def _snap(primary: list[float], anchors: list[float], radius: float) -> list[float]:
    """把每個語意邊界吸到半徑內最近的視覺切點；沒有就留原位。

    語意邊界的時間解析度受限於 block 的大小，而投影片切換的時間戳很準。
    這是「投影片切換作為候選邊界」的另一種讀法——**它不否決語意邊界**，
    只調整位置，因此不會繼承 slide 的召回上限。
    """
    out = []
    for t in primary:
        near = [a for a in anchors if abs(a - t) <= radius]
        out.append(min(near, key=lambda a: abs(a - t)) if near else t)
    return sorted(set(out))


def slide_boundaries(video_id: str, lo: float, hi: float) -> list[float]:
    """現行方案：S3 的 segment 起點（由投影片切換決定）。"""
    segs = json.loads(
        (ROOT / "work" / video_id / "06_segments.json").read_text(encoding="utf-8"))
    return [s["t_start"] for s in segs if lo < s["t_start"] < hi]


def prf(pred: list[float], truth: list[float], tol: float) -> dict:
    from weft.validation.metrics import boundary_prf

    r = boundary_prf(sorted(pred), sorted(truth), tol)
    return {"P": round(r.precision, 3), "R": round(r.recall, 3),
            "F1": round(r.f1, 3), "n_pred": len(pred), "n_true": len(truth)}


def run_video(video_id: str, model_cache: dict, tolerances=(2.0, 10.0, 20.0)) -> dict:
    cues, golden = load_video(video_id)
    lo, hi = golden["body_start"], golden["body_end"]
    truth = [b["t"] for b in golden["segment_boundaries"] if b["status"] == "confirmed"]
    blocks = blocks_from_cues(cues, lo, hi)
    texts = [b["text"] for b in blocks]

    methods: dict[str, list[float]] = {}
    methods["slide"] = slide_boundaries(video_id, lo, hi)
    methods["ngram"] = _depth_boundaries(
        _window_similarity(char_ngram_vectors(texts)), blocks)

    for name in MODELS:
        model = model_cache[name]
        vecs = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        sem = _depth_boundaries(_window_similarity(np.asarray(vecs)), blocks)
        tag = name.split("/")[-1]
        methods[f"sbert[{tag}]"] = sem
        # 方向 2 的字面讀法：投影片切換**降級為候選**，只有語意也支持才採納。
        # 注意它結構上只能是 slide 的子集——**找不到 slide 沒找到的邊界**。
        methods[f"hybrid-filter[{tag}]"] = [
            t for t in methods["slide"] if any(abs(t - x) <= 20.0 for x in sem)
        ]
        # 方向 2 的另一種讀法，對它更公道：**語意為主**，投影片切換只用來
        # 把邊界吸附到附近的視覺切點（那裡的時間戳比語意的 block 邊界準）。
        methods[f"snap[{tag}]"] = _snap(sem, methods["slide"], 10.0)

    return {
        "video_id": video_id,
        "holdout": golden.get("holdout", False),
        "blocks": len(blocks),
        "truth": len(truth),
        "results": {
            m: {f"±{tol:g}s": prf(pred, truth, tol) for tol in tolerances}
            for m, pred in methods.items()
        },
    }


def main() -> None:
    from sentence_transformers import SentenceTransformer

    cache = {}
    for name in MODELS:
        print(f"載入 {name} ...", flush=True)
        cache[name] = SentenceTransformer(name, device="cuda")

    out = [run_video(v, cache) for v in ("cxrqHABhWOU", "2FjApOVIbUs")]
    path = pathlib.Path(__file__).with_name("results.json")
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    for r in out:
        tag = "保留集" if r["holdout"] else "調校集"
        print(f"\n=== {r['video_id']}［{tag}］ {r['blocks']} blocks，"
              f"{r['truth']} 個真實邊界 ===")
        for m, by_tol in r["results"].items():
            cells = "  ".join(
                f"{tol} F1={v['F1']:.3f}(P{v['P']:.2f}/R{v['R']:.2f})"
                for tol, v in by_tol.items())
            n = next(iter(by_tol.values()))["n_pred"]
            print(f"  {m:<42} 切 {n:>3} 刀  {cells}")
    print(f"\n寫入 {path}")


if __name__ == "__main__":
    main()
