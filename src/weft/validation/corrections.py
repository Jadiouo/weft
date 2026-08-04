"""術語校正的授權檢查（R13）。

S4 的 VLM 會回傳 `corrections`，把逐字稿裡的同音錯字改成正確寫法。
首跑實測發現它**超出授權**：9 筆中有 4 筆不是聽寫校正，而是事實修正、
語意改寫或**插入講者沒說的內容**。插入尤其危險——§5.3 不變量 10 只驗
`from` 出現在原文中，驗不到 `to` 是否加了原文沒有的東西。

這個模組是 prompt 之外的**程式端兜底**。它擋兩類（見
`experiments/r13_corrections/REPORT.md`）：

1. **插入**：`to` 完整包含 `from` 且多出超過 `MAX_INSERTED_CHARS` 個字
2. **語意改寫**：`to` 與 `from` 的拼音相似度低於 `MIN_PINYIN_SIMILARITY`
3. **大小寫**：兩者 casefold 後相同——大小寫聽不出來，定義上不是聽寫錯誤

**擋不到事實修正**（`唐朝`→`宋朝` 拼音 0.500、`六祖惠能`→`六祖慧能` 拼音
1.000，與授權樣本完全重疊）。那一類只能靠 prompt 的負面示例與 §5.6 人工抽檢。

門檻的餘裕只有 **1.60x**（授權最低 0.400 vs 語意改寫最高 0.250），
低於 D1／R12 沿用的 2x 判準。仍然採用的理由與代價寫在 REPORT §4。
"""

from __future__ import annotations

from pypinyin import Style, lazy_pinyin

#: `to` 包含 `from` 時，允許多出的字數上限。
#: 授權的補全（`未`→`未來`、`菩`→`菩薩`）最多 +1；
#: 實測的插入型違規是 +4～+6。取 2 留一格餘裕。
MAX_INSERTED_CHARS = 2

#: 拼音相似度下限。授權樣本最低 0.400（`意地論`→`瑜伽師地論`），
#: 語意改寫最高 0.250。取 0.35 置中——取 0.40 會貼齊授權最低值，零餘裕。
MIN_PINYIN_SIMILARITY = 0.35


def pinyin_key(text: str) -> list[str]:
    """轉成**不帶聲調**的拼音串；非漢字原樣保留並轉小寫。

    不帶聲調是刻意的：講者口誤造成的近音錯字聲調常常不同
    （`钵` bō / `波` bō 同調，但 `意` yì / `瑜` yú 不同調），
    帶聲調會把這類授權校正誤判為「讀音差太多」。
    """
    return [p.lower() for p in lazy_pinyin(text, style=Style.NORMAL)]


def pinyin_similarity(a: str, b: str) -> float:
    """1 - 拼音序列的正規化編輯距離。兩者皆空視為相同。"""
    x, y = pinyin_key(a), pinyin_key(b)
    if not x and not y:
        return 1.0
    if not x or not y:
        return 0.0
    prev = list(range(len(y) + 1))
    for i, cx in enumerate(x, 1):
        cur = [i] + [0] * len(y)
        for j, cy in enumerate(y, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (cx != cy))
        prev = cur
    return 1.0 - prev[-1] / max(len(x), len(y))


def unauthorized_reason(from_text: str, to_text: str) -> str | None:
    """回傳拒絕理由；`None` 表示這筆校正在授權範圍內。"""
    if from_text.casefold() == to_text.casefold():
        return "格式化：兩者只差英文大小寫。大小寫聽不出來，不可能是聽寫錯誤"

    inserted = len(to_text) - len(from_text)
    if from_text and from_text in to_text and inserted > MAX_INSERTED_CHARS:
        return (f"插入：`to` 在 `from` 之外多了 {inserted} 個字"
                f"（上限 {MAX_INSERTED_CHARS}）。校正不得補上講者沒說的內容")

    sim = pinyin_similarity(from_text, to_text)
    if sim < MIN_PINYIN_SIMILARITY:
        return (f"語意改寫：拼音相似度 {sim:.2f} 低於 {MIN_PINYIN_SIMILARITY}。"
                f"讀音差這麼多就不是聽寫錯誤")
    return None
