"""合成測試素材與其 ground truth 的自我驗證。SDD §5.1（A）。

ground truth 若是錯的，所有 §5.2 的指標都失去意義——而且會失敗得很安靜：
測試照跑、數字照出，只是全部不代表任何東西。所以 ground truth 本身必須被
測試。

§5.5 #8：不得縮小測試集或移除對抗樣本。A1–A7 為必選。
"""

from __future__ import annotations

import subprocess

import pytest

from tests.synth import scenarios as S
from tests.synth.build import DURATION_TOLERANCE_SEC, probe_duration
from tests.synth.truth import SynthTruth

REQUIRED = ("A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9")


# --------------------------------------------------------------------------
# 測試集完整性（§5.5 #8 的機械式護欄）
# --------------------------------------------------------------------------


def test_all_adversarial_samples_exist():
    """A1–A9 為必選。移除任何一個，這裡就紅燈。

    A8／A9 是 2026-08-04 依真實素材實測結果新增的——A1–A7 無鏡頭運動、
    無交叉淡化，那正是「合成全綠、真實素材產出垃圾」的直接原因。
    """
    prefixes = {s.name.split("_")[0] for s in S.ALL_SCENARIOS}
    missing = set(REQUIRED) - prefixes
    assert not missing, f"缺少 SDD §5.1 的對抗樣本：{sorted(missing)}"


def test_each_scenario_records_its_sdd_expectation():
    """每個場景都要帶 SDD 表格「期望行為」一欄的原文，讓報告能對回規格。"""
    for s in S.ALL_SCENARIOS:
        assert s.expectation.strip(), f"{s.name} 未記錄期望行為"
        assert s.description.strip(), f"{s.name} 未記錄描述"


def test_scenario_names_are_unique():
    names = [s.name for s in S.ALL_SCENARIOS]
    assert len(names) == len(set(names))


# --------------------------------------------------------------------------
# ground truth 的內部一致性
# --------------------------------------------------------------------------


@pytest.mark.parametrize("scenario", S.ALL_SCENARIOS, ids=lambda s: s.name)
def test_pages_tile_the_timeline_without_gaps(scenario: SynthTruth):
    placed = scenario.placed
    assert placed[0].t_start == 0.0
    for prev, cur in zip(placed, placed[1:]):
        assert cur.t_start == pytest.approx(prev.t_end), f"{scenario.name} 的頁面時間軸有斷裂"
    assert placed[-1].t_end == pytest.approx(scenario.duration)


@pytest.mark.parametrize("scenario", S.ALL_SCENARIOS, ids=lambda s: s.name)
def test_boundaries_are_sorted_and_within_duration(scenario: SynthTruth):
    b = scenario.slide_boundaries
    assert b == sorted(b)
    assert len(b) == len(set(b)), "邊界有重複"
    assert all(0 < x < scenario.duration for x in b)


@pytest.mark.parametrize("scenario", S.ALL_SCENARIOS, ids=lambda s: s.name)
def test_frame_classes_cover_whole_video(scenario: SynthTruth):
    classes = scenario.frame_classes(sample_fps=1.0)
    assert len(classes) == int(scenario.duration)
    assert set(classes) <= {"slide", "speaker"}


@pytest.mark.parametrize("scenario", S.ALL_SCENARIOS, ids=lambda s: s.name)
def test_build_frames_lie_inside_their_page(scenario: SynthTruth):
    for placed in scenario.placed:
        for bt in placed.build_times:
            assert placed.t_start <= bt < placed.t_end, f"{scenario.name} 的 build 時間戳越界"


# --------------------------------------------------------------------------
# 各對抗樣本的定義本身正確
# --------------------------------------------------------------------------


def test_a1_expects_five_slides_and_four_internal_boundaries():
    assert S.A1.expected_slide_count == 5
    assert len(S.A1.slide_boundaries) == 4  # 5 段相鄰投影片之間有 4 個切點


def test_a1_page_durations_are_within_sdd_range():
    """SDD 寫「每頁停留 30–120 秒」。"""
    for page in S.A1.pages:
        assert 30 <= page.duration <= 120


def test_a2_progressive_page_counts_as_one_slide():
    """A2 的全部重點：一頁分 6 次疊加，**必須**偵測為 1 頁。門檻 1.00。"""
    build_page = next(p for p in S.A2.pages if p.build_offsets)
    assert len(build_page.build_offsets) == 6
    assert build_page.expected_slides == 1


def test_a2_keyframe_window_is_after_last_build():
    """「取最後一幀」——代表幀必須落在最後一次疊加之後，內容才最完整。"""
    build_page = next(p for p in S.A2.pages if p.build_offsets)
    window = build_page.keyframe_offset_window
    assert window is not None
    assert window[0] >= max(build_page.build_offsets)
    assert window[1] <= build_page.duration


def test_a2_boundaries_do_not_include_build_steps():
    """關鍵：6 次疊加不得出現在 ground truth 邊界中，否則等於預設要切成 6 頁。"""
    build = next(p for p in S.A2.placed if p.page.build_offsets)
    for bt in build.build_times[1:]:
        assert bt not in S.A2.slide_boundaries


def test_a3_expects_zero_slides_and_zero_boundaries():
    """純講者：偵測為 0 頁。若演算法在此吐出任何邊界，F1 會是 0。"""
    assert S.A3.expected_slide_count == 0
    assert S.A3.slide_boundaries == []
    assert set(S.A3.frame_classes()) == {"speaker"}


def test_a4_laser_pointer_expects_single_page():
    """移動紅點不得觸發換頁。"""
    assert S.A4.expected_slide_count == 1
    assert S.A4.slide_boundaries == []
    assert S.A4.pages[0].render.get("overlay") == "laser"


def test_a5_embedded_video_expects_three_pages():
    """內嵌播放的影片不得被切成數十頁。"""
    assert S.A5.expected_slide_count == 3
    assert any(p.render.get("overlay") == "embedded_video" for p in S.A5.pages)


def test_a6_interleaves_speaker_and_slide():
    kinds = [p.kind for p in S.A6.pages]
    assert kinds == ["slide", "speaker", "slide", "speaker", "slide"]
    assert S.A6.expected_slide_count == 3


def test_a6_boundaries_include_speaker_transitions():
    """slide→speaker 與 speaker→slide 同樣是投影片段落的起訖，都算邊界。

    3 段投影片共 6 個端點，扣掉落在影片頭尾的 0 與 115 → 4 個內部邊界。
    """
    assert S.A6.slide_boundaries == [25.0, 45.0, 75.0, 90.0]


def test_a7_backtrack_yields_three_switches_not_two():
    """SDD 明寫「偵測為 3 次切換（非 2 次）」——內容相同的頁面重複出現時，
    不得因為內容一樣就被合併。"""
    assert len(S.A7.slide_boundaries) == 3
    assert S.A7.expected_slide_count == 4


def test_a7_repeated_pages_have_identical_content():
    """回放的前提：p1 與 p1_again 內容必須真的一樣，否則測的不是回放。"""
    pages = {p.label: p for p in S.A7.pages}
    assert pages["p1"].render["content"] is pages["p1_again"].render["content"]
    assert pages["p2"].render["content"] is pages["p2_again"].render["content"]


# --------------------------------------------------------------------------
# 素材特性（SDD §1.3）
# --------------------------------------------------------------------------


def test_a8_expects_no_page_turns():
    """A8：攝影機推近不是換頁。純講者背景，全程無投影片。"""
    assert S.A8.expected_slide_count == 0
    assert S.A8.slide_boundaries == []
    assert S.A8.pages[0].zoom is not None
    start, end = S.A8.pages[0].zoom
    assert end > start * 1.15, "推近幅度太小，測不出鏡頭運動的影響"


def test_a9_has_crossfade_at_every_internal_boundary():
    """A9：每個內部邊界都有 1 秒交叉淡化。"""
    windows = S.A9.crossfade_windows
    assert len(windows) == len(S.A9.slide_boundaries)
    for a, b in windows:
        assert b - a == pytest.approx(1.0), "轉場長度應為 1 秒（實測真實素材皆為 1 秒）"


def test_a9_crossfade_sits_at_the_page_boundary():
    """轉場屬於前一頁的尾巴，所以 slide_boundaries 的定義不必改。"""
    for (_, fade_end), boundary in zip(S.A9.crossfade_windows, S.A9.slide_boundaries):
        assert fade_end == pytest.approx(boundary)


def test_a9_keyframe_windows_exclude_the_crossfade():
    """A9 的核心：代表幀**不得**落在轉場區間內。

    §4.3 步驟 5 的舊作法「取段末幀」正好會取到它——`slide_017` 即為此例，
    還一度造成「素材有第三種疊加模式」的誤判（known-risks R8）。
    """
    for placed in S.A9.placed:
        window = placed.crossfade_window
        if window is None:
            continue
        assert window[0] > placed.t_start, "轉場佔滿整頁，代表幀無處可選"


def test_slides_include_vertical_text_and_semantic_layout():
    """SDD §1.3：投影片含直排文字、版面帶語意（箭頭、雙欄、色彩編碼）。
    只有橫排標題的合成素材，測出來的分數對真實講經影片沒有參考價值。"""
    layouts = {
        p.render.get("layout")
        for s in S.ALL_SCENARIOS
        for p in s.pages
        if p.kind == "slide"
    }
    for required in ("vertical", "two_column", "arrow", "colored"):
        assert required in layouts, f"合成素材缺少 {required} 版型"


def test_slide_text_contains_classical_chinese():
    """文言文是 §1.3 的關鍵特性，也是 §9 列為風險的項目。"""
    text = "".join(
        "".join(c["items"]) for c in (S.C_OPENING, S.C_VERTICAL, S.C_ARROW, S.C_COLORED)
    )
    for term in ("精血凝也", "形兆胚也", "陽神", "七魄"):
        assert term in text


# --------------------------------------------------------------------------
# 實際產物與 ground truth 吻合
# --------------------------------------------------------------------------


@pytest.mark.synth
@pytest.mark.parametrize("scenario", S.ALL_SCENARIOS, ids=lambda s: s.name)
def test_rendered_video_duration_matches_truth(synth_dir, scenario: SynthTruth):
    """產出時長與宣告的 ground truth 必須吻合，否則所有邊界時間都是偏的。"""
    video = synth_dir / f"{scenario.name}.mp4"
    assert video.exists()
    assert abs(probe_duration(video) - scenario.duration) <= DURATION_TOLERANCE_SEC


@pytest.mark.synth
@pytest.mark.parametrize("scenario", S.ALL_SCENARIOS, ids=lambda s: s.name)
def test_rendered_video_resolution_matches_truth(synth_dir, scenario: SynthTruth):
    video = synth_dir / f"{scenario.name}.mp4"
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height",
         "-of", "csv=p=0:s=x", str(video)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert out == f"{scenario.width}x{scenario.height}"


@pytest.mark.synth
def test_speaker_frames_are_detectable_as_faces(synth_dir):
    """合成講者幀若騙不過真實人臉偵測器，A3／A6 會產生**假失敗**，
    測試就從「檢驗演算法」變成「檢驗我的畫工」。這條把該假設釘住。"""
    import cv2

    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
    )

    def faces_at(name: str, t: float):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as td:
            frame = Path(td) / "f.png"
            subprocess.run(
                ["ffmpeg", "-v", "error", "-ss", str(t), "-i", str(synth_dir / name),
                 "-frames:v", "1", str(frame)],
                check=True,
            )
            img = cv2.imread(str(frame))
        h, w = img.shape[:2]
        found = cascade.detectMultiScale(
            cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 1.05, 4, minSize=(60, 60)
        )
        return [fw * fh / (w * h) for _, _, fw, fh in found]

    speaker = faces_at("A3_speaker_only.mp4", 60)
    assert len(speaker) == 1, "講者幀未被偵測為人臉——A3／A6 會產生假失敗"
    assert speaker[0] > 0.04, "人臉面積佔比低於 face_min_area_ratio 門檻"

    for t in (50, 110, 275):
        assert faces_at("A1_standard.mp4", t) == [], f"投影片幀在 t={t} 被誤判為講者"
