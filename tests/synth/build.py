"""用 ffmpeg 把場景定義合成為影片。SDD §5.1（A）。

「切換時間點由自己指定，ground truth 精確到毫秒、免費、可重現。」

作法：每個 LogicalPage 先編成一個獨立片段，再用 concat demuxer 串起來。
靜態頁面用 `-loop 1` 從單張 PNG 生成，不逐幀畫圖——943 秒的測試素材若
逐幀渲染，光產生就要好幾分鐘，會讓人不想跑測試，而不想跑的測試等於沒有。

動態干擾（雷射筆、講者晃動、內嵌影片）用 ffmpeg filter 的時間表達式做，
同樣不必逐幀出圖。
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from .render import render_slide, render_speaker, save
from .truth import PlacedPage, SynthTruth

FFMPEG = "ffmpeg"
FFPROBE = "ffprobe"

#: 產出時長與宣告 ground truth 的最大容許差距（秒）。
#: 超過就代表 ground truth 是錯的——那比偵測演算法出錯更嚴重，必須擋下。
DURATION_TOLERANCE_SEC = 0.5

#: crf 30 而非預設的 23：真實 YouTube 講經影片是重壓縮的，塊狀瑕疵與色帶
#: 都會抬高幀間距離的基線。用高畫質合成素材測出來的 F1 會虛高——實測未加
#: 雜訊前，多數相鄰幀是**像素完全相同**的，那對偵測器毫無挑戰。
_ENCODE = [
    "-c:v", "libx264",
    "-preset", "veryfast",
    "-crf", "30",
    "-pix_fmt", "yuv420p",
]

#: 疊加在靜態畫面上的干擾。ground truth 不變（換頁時刻由場景定義決定），
#: 變的只是「這題有多難」。
#:   noise    感測器／壓縮雜訊
#:   亮度浮動 攝影機自動曝光、投影機燈源、會場燈光變化
_REALISM = (
    "noise=alls=6:allf=t+u,"
    "eq=brightness='0.012*sin(2*PI*t/11)':contrast='1+0.015*sin(2*PI*t/7)'"
)


def _run(args: list[str]) -> None:
    proc = subprocess.run(args, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg 失敗：{' '.join(args[:6])}…\n{proc.stderr[-2000:]}")


def probe_duration(path: Path) -> float:
    proc = subprocess.run(
        [FFPROBE, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(proc.stdout.strip())


# --------------------------------------------------------------------------
# 片段生成
# --------------------------------------------------------------------------


def _laser_filter(w: int, h: int) -> str:
    """移動的雷射筆紅點（A4）。兩個週期互質，避免軌跡很快重複。"""
    dot = max(5, w // 100)
    x = f"'{w * 0.5:.0f}+{w * 0.28:.0f}*sin(2*PI*t/7)'"
    y = f"'{h * 0.55:.0f}+{h * 0.22:.0f}*cos(2*PI*t/5)'"
    return f"drawbox=x={x}:y={y}:w={dot}:h={dot}:color=red@0.95:t=fill"


def _encode_static(png: Path, duration: float, out: Path, truth: SynthTruth, vf: str | None = None) -> None:
    chain = f"{vf}," if vf else ""
    _run([
        FFMPEG, "-y", "-loglevel", "error",
        "-loop", "1", "-framerate", str(truth.fps), "-i", str(png),
        "-t", f"{duration:.3f}",
        "-vf", f"{chain}{_REALISM},fps={truth.fps},format=yuv420p",
        *_ENCODE, str(out),
    ])


def _encode_zoom(png: Path, duration: float, out: Path, truth: SynthTruth,
                 zoom: tuple[float, float]) -> None:
    """攝影機緩慢推近（對抗樣本 A8）。

    這是 A1–A7 完全沒有涵蓋的干擾：**整個畫面**相對於背景常態持續位移。
    真實素材 zIglvjoU9vo 實測有兩次推近（t=199→201、t=554→556），而
    R8 的實驗證明它是「以畫面變化為基礎的偵測」最根本的混淆源。
    """
    frames = max(1, int(round(duration * truth.fps)))
    start, end = zoom
    expr = f"{start}+{end - start}*on/{frames}"
    _run([
        FFMPEG, "-y", "-loglevel", "error",
        "-loop", "1", "-framerate", str(truth.fps), "-i", str(png),
        "-t", f"{duration:.3f}",
        "-vf", (f"zoompan=z='{expr}':d=1:s={truth.width}x{truth.height}:fps={truth.fps},"
                f"{_REALISM},format=yuv420p"),
        *_ENCODE, str(out),
    ])


def _encode_crossfade(before: Path, after: Path, duration: float, out: Path,
                      truth: SynthTruth) -> None:
    """交叉淡化轉場（對抗樣本 A9）。

    實測真實素材的轉場長度皆為 1 秒，且**兩側都有**（淡入與淡出）。
    這一秒的畫面是兩者的混合——OCR 讀不乾淨、VLM 看到的是疊影，所以
    §4.3 步驟 5 的代表幀選取必須避開它。
    """
    _run([
        FFMPEG, "-y", "-loglevel", "error",
        "-loop", "1", "-framerate", str(truth.fps), "-i", str(before),
        "-loop", "1", "-framerate", str(truth.fps), "-i", str(after),
        "-t", f"{duration:.3f}",
        "-filter_complex",
        # 混合比例隨時間由 0 走到 1，形成真正的漸變
        f"[0:v][1:v]blend=all_expr='A*(1-T/{duration})+B*(T/{duration})',"
        f"{_REALISM},fps={truth.fps},format=yuv420p",
        *_ENCODE, str(out),
    ])


def _encode_speaker(png: Path, duration: float, out: Path, truth: SynthTruth) -> None:
    """靜態人像 + 輕微晃動（A3）。以 crop 視窗隨時間平移模擬手持／呼吸晃動。"""
    w, h = truth.width, truth.height
    x = f"'{w * 0.06:.0f}+{w * 0.02:.0f}*sin(2*PI*t/9)'"
    y = f"'{h * 0.06:.0f}+{h * 0.02:.0f}*sin(2*PI*t/13)'"
    _run([
        FFMPEG, "-y", "-loglevel", "error",
        "-loop", "1", "-framerate", str(truth.fps), "-i", str(png),
        "-t", f"{duration:.3f}",
        "-vf", f"crop={w}:{h}:x={x}:y={y},{_REALISM},fps={truth.fps},format=yuv420p",
        *_ENCODE, str(out),
    ])


def _encode_embedded(png: Path, duration: float, out: Path, truth: SynthTruth) -> None:
    """投影片中嵌一塊持續變化的畫面（A5）。這是最容易被誤切成數十頁的樣本。"""
    w, h = truth.width, truth.height
    box_w, box_h = int(w * 0.46), int(h * 0.40)
    px, py = int(w * 0.27), int(h * 0.50)
    _run([
        FFMPEG, "-y", "-loglevel", "error",
        "-loop", "1", "-framerate", str(truth.fps), "-i", str(png),
        "-f", "lavfi", "-i", f"testsrc2=size={box_w}x{box_h}:rate={truth.fps}",
        "-t", f"{duration:.3f}",
        "-filter_complex",
        f"[0:v][1:v]overlay=x={px}:y={py}:shortest=0,{_REALISM},fps={truth.fps},format=yuv420p",
        *_ENCODE, str(out),
    ])


def _page_image(page, size, tmp: Path, tag: str) -> Path:
    """該頁的靜態畫面（逐條動畫取最完整的那一版）。"""
    render = page.render
    if render.get("kind") == "speaker":
        big = (int(size[0] * 1.12), int(size[1] * 1.12))
        return save(render_speaker(big, seed=render.get("seed", 0)), tmp / f"{tag}_spk.png")
    return save(render_slide(render["layout"], render["content"], size), tmp / f"{tag}_img.png")


def _build_page(placed: PlacedPage, truth: SynthTruth, tmp: Path, seq: int,
                next_page=None) -> list[Path]:
    """把一個 LogicalPage 編成一或多個片段（逐條動畫會是多個）。"""
    size = (truth.width, truth.height)
    render = placed.page.render
    parts: list[Path] = []

    # 尾端的交叉淡化自成一個片段，主體時間相應縮短——這樣總長不變，
    # ground truth 的邊界時刻也不必調整。
    fade = placed.page.crossfade_out_sec
    body_duration = placed.page.duration - fade

    if placed.page.zoom is not None:
        png = _page_image(placed.page, size, tmp, f"p{seq:02d}")
        out = tmp / f"p{seq:02d}.mp4"
        _encode_zoom(png, body_duration, out, truth, placed.page.zoom)
        parts.append(out)
    elif render.get("kind") == "speaker":
        # 渲染比輸出畫面大一圈，留給 crop 晃動的餘裕
        big = (int(truth.width * 1.12), int(truth.height * 1.12))
        png = save(render_speaker(big, seed=render.get("seed", 0)), tmp / f"p{seq:02d}_speaker.png")
        out = tmp / f"p{seq:02d}.mp4"
        _encode_speaker(png, body_duration, out, truth)
        parts.append(out)
    elif render.get("progressive"):
        layout, content = render["layout"], render["content"]
        steps = render["steps"]
        offsets = list(placed.page.build_offsets) + [body_duration]
        for i in range(steps):
            png = save(
                render_slide(layout, content, size, reveal=i + 1),
                tmp / f"p{seq:02d}_build{i}.png",
            )
            out = tmp / f"p{seq:02d}_b{i}.mp4"
            _encode_static(png, offsets[i + 1] - offsets[i], out, truth)
            parts.append(out)
    else:
        layout, content = render["layout"], render["content"]
        overlay = render.get("overlay")
        png = save(render_slide(layout, content, size), tmp / f"p{seq:02d}_slide.png")
        out = tmp / f"p{seq:02d}.mp4"
        if overlay == "laser":
            _encode_static(png, body_duration, out, truth,
                           vf=_laser_filter(truth.width, truth.height))
        elif overlay == "embedded_video":
            _encode_embedded(png, body_duration, out, truth)
        else:
            _encode_static(png, body_duration, out, truth)
        parts.append(out)

    if fade > 0 and next_page is not None:
        before = _page_image(placed.page, size, tmp, f"p{seq:02d}_fadea")
        after = _page_image(next_page, size, tmp, f"p{seq:02d}_fadeb")
        out = tmp / f"p{seq:02d}_fade.mp4"
        _encode_crossfade(before, after, fade, out, truth)
        parts.append(out)

    return parts


# --------------------------------------------------------------------------
# 對外入口
# --------------------------------------------------------------------------


def _vtt_time(t: float) -> str:
    h, rem = divmod(max(0.0, t), 3600)
    m, sec = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{sec:06.3f}"


def write_captions(truth: SynthTruth, path: Path) -> Path:
    """把場景的 speech 寫成 WebVTT。

    刻意寫成**手動字幕**的形式（一句一個 cue，不逐字滾動）——S1a 的字幕
    優先序會直接採用它，讓 S1a→S3 能在合成素材上端到端跑，不必動用
    Whisper（那需要 GPU，而且會讓測試變成分鐘級）。
    """
    lines = ["WEBVTT", ""]
    for t_start, t_end, text, _error in truth.all_cues:
        lines += [f"{_vtt_time(t_start)} --> {_vtt_time(t_end)}", text, ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def build_scenario(truth: SynthTruth, out_dir: Path, force: bool = False) -> tuple[Path, Path]:
    """產生 `{name}.mp4` 與 `{name}.truth.json`，回傳兩者路徑。

    已存在且 ground truth 未變更時直接沿用（比對 truth.json 內容，不只看
    檔案在不在——場景定義改了卻沿用舊影片，是最難察覺的測試失效方式）。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    video = out_dir / f"{truth.name}.mp4"
    truth_path = out_dir / f"{truth.name}.truth.json"
    expected = truth.to_dict()

    if not force and video.exists() and truth_path.exists():
        try:
            if json.loads(truth_path.read_text(encoding="utf-8")) == expected:
                captions = out_dir / f"{truth.name}.vtt"
                if not captions.exists():
                    write_captions(truth, captions)
                return video, truth_path
        except json.JSONDecodeError:
            pass

    if shutil.which(FFMPEG) is None:
        raise RuntimeError("找不到 ffmpeg。SDD §8：請用系統 apt 版本，不要用 conda 的。")

    with tempfile.TemporaryDirectory(prefix=f"synth_{truth.name}_") as td:
        tmp = Path(td)
        parts: list[Path] = []
        placed_pages = truth.placed
        for seq, placed in enumerate(placed_pages):
            following = (
                placed_pages[seq + 1].page if seq + 1 < len(placed_pages) else None
            )
            parts.extend(_build_page(placed, truth, tmp, seq, following))

        listing = tmp / "concat.txt"
        listing.write_text("".join(f"file '{p}'\n" for p in parts), encoding="utf-8")
        _run([
            FFMPEG, "-y", "-loglevel", "error",
            "-f", "concat", "-safe", "0", "-i", str(listing),
            "-c", "copy", str(video),
        ])

    actual = probe_duration(video)
    if abs(actual - truth.duration) > DURATION_TOLERANCE_SEC:
        video.unlink(missing_ok=True)
        raise RuntimeError(
            f"{truth.name}：產出時長 {actual:.3f}s 與 ground truth {truth.duration:.3f}s "
            f"相差 {abs(actual - truth.duration):.3f}s，超過容忍 {DURATION_TOLERANCE_SEC}s。"
            "ground truth 不可信，已刪除產物。"
        )

    truth.save(truth_path)
    write_captions(truth, out_dir / f"{truth.name}.vtt")
    return video, truth_path


def build_all(out_dir: Path, force: bool = False) -> dict[str, tuple[Path, Path]]:
    from .scenarios import ALL_SCENARIOS

    return {s.name: build_scenario(s, out_dir, force=force) for s in ALL_SCENARIOS}


if __name__ == "__main__":  # python -m tests.synth.build <out_dir>
    import sys

    target = Path(sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures/synth")
    for name, (mp4, tj) in build_all(target).items():
        print(f"{name}: {mp4} ({probe_duration(mp4):.2f}s)")
