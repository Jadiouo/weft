"""從 slide_text 建 Whisper 的 initial_prompt。

    conda run -n pipe-cpu python -m experiments.r20_lexicon.build_prompt

**詞庫來源必須是無字幕影片也拿得到的東西**——所以只用 `slide_text`
（VLM 從投影片讀出的文字），不用 `term_index`（那是人工字幕跑出來的）。

Whisper 的 initial_prompt 上限約 224 token，中文約 1–2 字/token，
所以要挑不能全塞。優先序：經文原文 > 專有名詞 > 一般詞。
"""
from __future__ import annotations
import json, re
from pathlib import Path

HERE = Path(__file__).parent
MAX_CHARS = 200


def main() -> int:
    ir = json.loads(Path("work/zIglvjoU9vo/08_video.json").read_text())
    lines: list[str] = []
    for s in ir["slides"]:
        for line in (s.get("slide_text") or "").splitlines():
            line = re.sub(r"\s+", "", line)
            # 只留看起來像經文／術語的短句：純中文、2–14 字
            if 2 <= len(line) <= 14 and re.fullmatch(r"[一-鿿，。、：；《》【】]+", line):
                lines.append(line)

    # **優先序很重要**：不排序的話會被講者簡歷（「行政院國家發展基金視訪
    # 委員」「中原大學工業系畢」）塞滿，經文只進得去一半。
    # 這裡用啟發式評分——**這個啟發式本身是可疑的**，換素材要重看。
    DOMAIN = "月魂魄神氣蘊胎胞精血藏腑竅靈識道陰陽經論"
    NOISE = "委員顧問畢講師負責人董事系班考試合格基金會大學中心"

    def score(line: str) -> int:
        return (sum(line.count(c) for c in DOMAIN) * 3
                - sum(line.count(c) for c in NOISE) * 4
                + (2 if line.endswith("也。") else 0))

    lines.sort(key=score, reverse=True)

    seen, picked, total = set(), [], 0
    for line in lines:
        key = re.sub(r"[，。、：；《》【】]", "", line)
        if key in seen or not key:
            continue
        seen.add(key)
        if total + len(line) > MAX_CHARS:
            continue
        picked.append(line)
        total += len(line)

    prompt = "".join(picked)
    (HERE / "initial_prompt.txt").write_text(prompt, encoding="utf-8")
    print(f"取 {len(picked)} 句 / {len(prompt)} 字：\n{prompt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
