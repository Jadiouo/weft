"""黃金集標註工具。SDD §5.1（B）。

用法：

    # 1. 產生候選與聯絡表（機器輔助，不是 ground truth）
    python -m tests.golden.annotate propose work/<video_id>

    # 2. 人工檢視 out/golden/<video_id>/ 下的聯絡表，編輯 *.golden.json
    #    每個候選標成 confirmed / rejected，並補上漏掉的切換點

    # 3. 檢查標註檔的一致性
    python -m tests.golden.annotate check tests/golden/<video_id>.golden.json

**為什麼候選不用 S1b 的輸出**：用受測演算法的輸出當作它自己的 ground
truth，測出來必然接近滿分，而且完全測不到它系統性漏掉的東西。這裡改用
一個**刻意過度切分**的簡單像素差偵測——它會產生大量假陽性（人工剔除很
快），但漏掉真實切換的機會低，因為它不做任何時間平滑。

即使如此，機器提案仍可能漏標。`golden.json` 的 `reviewed` 欄位必須由
標註者手動改成 true，未經人工確認的標註檔會被測試拒用。
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import cv2
import numpy as np

#: 提案階段刻意調低的門檻。寧可多給人工剔除，不可漏掉。
PROPOSAL_PERCENTILE = 92.0
PROPOSAL_MIN_GAP_SEC = 4.0


@dataclass
class Boundary:
    t: float
    status: str = "proposed"  # proposed | confirmed | rejected
    note: str = ""


@dataclass
class GoldenAnnotation:
    """一支真實影片的標註。"""

    video_id: str
    title: str
    duration: float
    url: str
    #: **必須由標註者手動改成 true。** 未經人工確認的標註檔測試會拒用。
    reviewed: bool = False
    annotator: str = ""
    annotated_on: str = ""
    boundaries: list[Boundary] = field(default_factory=list)
    #: 每個抽樣點的 speaker/slide 標註。
    #: **v0.4 已作廢**——v0.3 的 D16 移除 CV 分類後 `frame_class` 恆為
    #: `slide`，這份標註沒有東西可以對。欄位保留只為讀得動舊檔。
    frame_classes: dict[str, str] = field(default_factory=dict)
    #: `slide_id → 是不是投影片`。對應 **S4a 的 `is_slide`**（§4.7a）。
    #: 這是 `frame_classes` 的接替者（D30）。
    slide_classes: dict[str, bool] = field(default_factory=dict)
    #: `slide_id → 代表幀的 slide_id`。同一張投影片反覆出現時的分組，
    #: 對應 **S1c 去重**（§4.3b）。代表幀自己指向自己。
    slide_groups: dict[str, str] = field(default_factory=dict)
    notes: str = ""

    @property
    def confirmed(self) -> list[float]:
        return sorted(b.t for b in self.boundaries if b.status == "confirmed")

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(self)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> GoldenAnnotation:
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw["boundaries"] = [Boundary(**b) for b in raw.get("boundaries", [])]
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            # **不要靜靜忽略**——多出來的欄位通常代表標註格式改過而檔案沒跟上，
            # 忽略它會讓測試拿舊格式的資料驗新機制（D22 的同類問題）。
            raise ValueError(f"{path.name} 有不認得的欄位：{sorted(unknown)}")
        return cls(**raw)


def propose_boundaries(frames_dir: Path, fps: float) -> list[float]:
    """用簡單的像素差提出候選。**刻意過度切分。**

    不做 HMM、不做 ink 遮罩、不做時間平滑——與受測的 S1b 演算法完全不同，
    才不會把它的盲點一併帶進 ground truth。
    """
    paths = sorted(frames_dir.glob("f_*.png"))
    if len(paths) < 2:
        return []

    previous = None
    distances: list[float] = []
    for path in paths:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        small = cv2.resize(img, (160, 90), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        distances.append(0.0 if previous is None else float(np.mean(np.abs(small - previous))))
        previous = small

    values = np.array(distances)
    threshold = float(np.percentile(values[1:], PROPOSAL_PERCENTILE))
    min_gap = max(1, int(PROPOSAL_MIN_GAP_SEC * fps))

    proposed: list[float] = []
    last = -min_gap
    for i, value in enumerate(values):
        if i == 0 or value < threshold or i - last < min_gap:
            continue
        proposed.append(round(i / fps, 2))
        last = i
    return proposed


def contact_sheet(video: Path, times: list[float], out: Path, columns: int = 4) -> Path:
    """把候選時刻前後各一幀併成聯絡表，供人工一次看完。

    每組是「切換前 / 切換後」兩張並排——這是判斷「這裡到底有沒有換頁」
    最直接的形式，比看單張快得多。
    """
    out.mkdir(parents=True, exist_ok=True)
    made: list[Path] = []
    for i, t in enumerate(times):
        pair = out / f"pair_{i:03d}_{t:.0f}s.png"
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y",
             "-ss", f"{max(0, t - 1.5):.2f}", "-i", str(video), "-frames:v", "1",
             "-f", "image2pipe", "-vcodec", "png", "-",
             ], capture_output=True, check=False,
        )
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y",
             "-ss", f"{max(0, t - 1.5):.2f}", "-i", str(video),
             "-ss", f"{t + 1.5:.2f}", "-i", str(video),
             "-frames:v", "1",
             "-filter_complex",
             "[0:v]scale=480:-1,drawtext=text='before':x=8:y=8:fontsize=22:fontcolor=yellow[a];"
             "[1:v]scale=480:-1,drawtext=text='after':x=8:y=8:fontsize=22:fontcolor=yellow[b];"
             "[a][b]hstack",
             str(pair)],
            capture_output=True, check=False,
        )
        if pair.exists():
            made.append(pair)

    if not made:
        return out
    rows = [made[i : i + columns] for i in range(0, len(made), columns)]
    sheets: list[Path] = []
    for r, row in enumerate(rows):
        sheet = out / f"sheet_{r:02d}.png"
        args = ["ffmpeg", "-v", "error", "-y"]
        for p in row:
            args += ["-i", str(p)]
        if len(row) == 1:
            args += ["-c", "copy", str(sheet)]
        else:
            args += ["-filter_complex", "".join(f"[{i}:v]" for i in range(len(row)))
                     + f"vstack=inputs={len(row)},scale=1400:-1", str(sheet)]
        subprocess.run(args, capture_output=True, check=False)
        if sheet.exists():
            sheets.append(sheet)
    return out


def cmd_propose(work_dir: Path, out_root: Path) -> None:
    from weft.ir import CandidateSet, VideoMeta
    from weft.paths import WorkPaths

    video_id = work_dir.name
    work = WorkPaths(work_dir.parent, video_id)
    meta = VideoMeta.model_validate_json(work.meta.read_text(encoding="utf-8"))
    fps = 1.0
    if work.candidates.exists():
        fps = CandidateSet.model_validate_json(work.candidates.read_text(encoding="utf-8")).fps

    proposed = propose_boundaries(work.frames_dir, fps)
    annotation = GoldenAnnotation(
        video_id=video_id,
        title=meta.title,
        duration=meta.duration,
        url=meta.url,
        boundaries=[Boundary(t=t) for t in proposed],
        notes=(
            "候選由簡單像素差產生，刻意過度切分。請逐一檢視 out/golden/ 下的聯絡表，"
            "把每個候選標成 confirmed 或 rejected，並補上機器漏掉的切換點。"
            "全部確認後把 reviewed 改成 true。"
        ),
    )
    target = out_root / video_id
    annotation.save(target / f"{video_id}.golden.json")
    contact_sheet(work.video, proposed, target / "pairs")

    print(f"{video_id}：提出 {len(proposed)} 個候選")
    print(f"  聯絡表：{target / 'pairs'}")
    print(f"  標註檔：{target / f'{video_id}.golden.json'}")
    print("  請人工檢視後把 reviewed 改成 true，再移到 tests/golden/ 下。")


def cmd_check(path: Path) -> int:
    annotation = GoldenAnnotation.load(path)
    problems: list[str] = []

    if not annotation.reviewed:
        problems.append("reviewed 仍為 false——未經人工確認的標註不得用於驗收")
    if not annotation.annotator.strip():
        problems.append("annotator 為空")

    pending = [b.t for b in annotation.boundaries if b.status == "proposed"]
    if pending:
        problems.append(f"還有 {len(pending)} 個候選未裁決：{pending[:8]}")

    confirmed = annotation.confirmed
    if len(confirmed) != len(set(confirmed)):
        problems.append("confirmed 邊界有重複")
    for t in confirmed:
        if not 0 < t < annotation.duration:
            problems.append(f"邊界 {t} 超出影片長度 {annotation.duration}")

    print(f"{annotation.video_id}：{len(confirmed)} 個確認的邊界")
    for problem in problems:
        print(f"  ✗ {problem}")
    if not problems:
        print("  ✓ 標註檔可用")
    return 1 if problems else 0


if __name__ == "__main__":
    import sys

    command = sys.argv[1] if len(sys.argv) > 1 else "help"
    if command == "propose":
        cmd_propose(Path(sys.argv[2]), Path("out/golden"))
    elif command == "check":
        raise SystemExit(cmd_check(Path(sys.argv[2])))
    else:
        print(__doc__)
