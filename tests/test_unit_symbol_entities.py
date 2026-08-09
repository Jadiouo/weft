"""數學符號的溯源診斷（R42，2026-08-09）。

**現行的具名實體檢查在 STEM 素材上是空轉的。**
實測 `extract_named_entities("四乘四矩陣用於表示坐標系B到坐標系A的完整轉換")`
回傳**空集合**——它抓的是書名、年代、中文數字，那是為中醫講經設計的。

而 R39 §3.1 手讀時判定為真編造的那一筆正是這類：`UiKi5#015#b03` 寫出
`R₁₁(a) • R₁₂(β)`，而講者說的是「對哪個軸轉幾度」。

**這是診斷不是閘門**（R42 量過：未通過組告警 25.0%、通過組 1.6%，
但被抓到的 block 本來就未通過；多抓的那些是正規化不是編造）。
所以這裡測的是「它抓得準不準」，不是「它擋不擋得住」。
"""

from __future__ import annotations

import pytest

from weft.validation.provenance import (
    extract_named_entities,
    symbol_entities,
    symbol_variants,
    unsupported_symbols,
)


class TestWhyThisExists:
    def test_the_existing_extractor_finds_nothing_in_stem_text(self):
        """**這一條是整個檔案的理由。** 前提沒了就該把這些程式碼刪掉。"""
        stem = "四乘四矩陣用於表示坐標系B到坐標系A的完整轉換"
        assert extract_named_entities(stem).all() == frozenset()
        assert symbol_entities(stem) == set()  # 中文寫法本身也不是符號

    def test_the_existing_extractor_still_works_on_its_own_material(self):
        """沒有回頭破壞中醫講經那邊。"""
        assert "黃帝內經" in extract_named_entities("《黃帝內經》說一月為胞").all()


class TestExtraction:
    @pytest.mark.parametrize(("text", "expected"), [
        ("R₁₁(a) 和 R₁₂(β) 的乘積", {"R₁₁", "R₁₂"}),
        ("可以用4x4矩陣來表示", {"4x4"}),
        ("旋轉矩陣是 3×3 的", {"3×3"}),
        ("Euler(φ, θ, ψ) = Rz(φ)Ry(θ)Rz(ψ)", {"Euler(φ, θ, ψ)", "Rz(φ)",
                                               "Ry(θ)", "Rz(φ)"}),
        ("φ=tan⁻¹(n₃/nₓ)", {"n₃", "nₓ"}),
    ])
    def test_extracts_structured_symbols(self, text, expected):
        assert expected <= symbol_entities(text)

    def test_does_not_split_nxn_into_a_phantom_symbol(self):
        """**R42 第一版的 bug**：`4x4` 被切出一個假的 `x4`。

        通過組的誤報有一半來自它。長樣式必須先吃掉字元。
        """
        found = symbol_entities("可以用4x4矩陣，也可以用3x3")
        assert "x4" not in found and "x3" not in found
        assert {"4x4", "3x3"} <= found

    def test_bare_capital_letters_are_not_entities(self):
        """「A 乘 B」裡的 A、B 太常見，抓了會把每個 block 都告警。"""
        assert symbol_entities("我會選 A 乘 B 還是 B 乘 A") == set()

    def test_plain_prose_yields_nothing(self):
        assert symbol_entities("胎生過程如同搬家，要把累世的種子逐步搬過來") == set()


class TestVariants:
    """**沒有變體正規化，書面化會被誤判成編造。**（R36 記過這個現象。）"""

    @pytest.mark.parametrize(("entity", "spoken"), [
        ("4x4", "四乘四"),
        ("3×3", "三乘三"),
        ("4x4", "4乘4"),
        ("R₁₁", "R11"),
        ("T2", "T二"),
    ])
    def test_spoken_forms_count_as_the_same_symbol(self, entity, spoken):
        assert spoken.replace(" ", "") in {
            v.replace(" ", "") for v in symbol_variants(entity)
        }

    def test_a_different_symbol_is_not_a_variant(self):
        assert "R₂₁" not in symbol_variants("R₁₁")
        assert "5x5" not in symbol_variants("4x4")


class TestUnsupportedSymbols:
    def test_the_case_r39_read_by_hand(self):
        """`UiKi5#015#b03`——講者從沒說過 R₁₁／R₁₂。"""
        block = "旋轉變換不具有交換性，即 R₁₁(a) • R₁₂(β) ≠ R₁₂(β) • R₁₁(a)"
        source = "旋轉本身是不可以交換　旋轉矩陣是個3乘3的　我對某個軸轉幾度"
        missing = unsupported_symbols(block, source)
        assert "R₁₁" in missing and "R₁₂" in missing

    def test_written_normalisation_is_not_flagged(self):
        """**這是誤報的主要來源，必須擋掉。**

        講者說「四乘四」，模型寫 `4x4`——那是書面化不是幻覺。
        """
        assert unsupported_symbols("用4x4矩陣表示", "我們用四乘四的矩陣") == []

    def test_a_symbol_present_in_the_source_is_not_flagged(self):
        assert unsupported_symbols("Rz(φ) 是繞 Z 軸", "投影片寫著 Rz(φ)Ry(θ)") == []

    def test_no_symbols_means_no_findings(self):
        """中醫講經那批沒有符號——這條保證它不會平白產生告警。"""
        assert unsupported_symbols("一月為胞，二月為膏", "講者說一月為胞") == []


class TestItIsRecordedAsADiagnosisNotAGate:
    def test_verdict_carries_the_symbols(self):
        from weft.ir import ContentBlock, ContentType, Provenance, ProvenanceKind
        from weft.validation.provenance import BlockVerdict, VerificationStatus

        v = BlockVerdict(segment_id="v#000", block_index=0,
                         content_type="白話解說",
                         status=VerificationStatus.UNVERIFIED,
                         similarity=0.0, copy_ratio=0.0)
        assert v.fabricated_symbols == []
        # 型別檢查：它是清單，不是 bool——**要看得見是哪些符號**，
        # 不然診斷跟「未通過」一樣不透明
        v.fabricated_symbols = ["R₁₁"]
        assert v.fabricated_symbols == ["R₁₁"]

        block = ContentBlock(
            type=ContentType.VERNACULAR, text="R₁₁(a) 的乘積",
            provenance=Provenance(kind=ProvenanceKind.TRANSCRIPT, ref="cue:0"))
        assert "R₁₁" in unsupported_symbols(block.text, "講者說對某個軸轉幾度")
