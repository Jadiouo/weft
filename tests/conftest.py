from __future__ import annotations

from pathlib import Path

import pytest

from weft.config import Config

REPO_ROOT = Path(__file__).resolve().parents[1]
SYNTH_DIR = REPO_ROOT / "tests" / "fixtures" / "synth"


@pytest.fixture(scope="session")
def synth_dir() -> Path:
    """A1–A7 合成影片。build_scenario 會比對 ground truth 內容決定是否重建，
    所以場景定義改了會自動重新產生，改了卻沿用舊影片的情形不會發生。"""
    from tests.synth.build import build_all

    build_all(SYNTH_DIR)
    return SYNTH_DIR


@pytest.fixture(scope="session")
def synth_work(synth_dir: Path, tmp_path_factory) -> Path:
    """把合成影片擺成 SDD §3.1 的 work/ 佈局，讓階段函式能照正式路徑跑。

    抽幀結果在 session 範圍內共用——每個場景抽一次要數秒，七個場景乘上
    多條測試會讓跑一次測試變成分鐘級，而**沒人想跑的測試等於沒有測試**。
    """
    from tests.synth.scenarios import BY_NAME
    from weft.ir import VideoMeta
    from weft.paths import WorkPaths

    root = tmp_path_factory.mktemp("synth_work")
    for video in sorted(synth_dir.glob("*.mp4")):
        truth = BY_NAME[video.stem]
        work = WorkPaths(root, video.stem)
        work.ensure_dirs()
        work.video.symlink_to(video.resolve())

        captions = synth_dir / f"{video.stem}.vtt"
        if captions.exists():
            work.captions.symlink_to(captions.resolve())

        # 合成 S0 的產物。S0 本身需要網路，不該成為本地管線測試的前提。
        work.meta.write_text(
            VideoMeta(
                video_id=video.stem,
                title=f"合成測試素材：{truth.description}",
                duration=truth.duration,
                url=f"https://example.invalid/{video.stem}",
                series_id="PL_synthetic",
                episode_index=1,
                has_manual_caption=captions.exists(),
                caption_lang="zh-TW" if captions.exists() else None,
                video_path="01_video.mp4",
                caption_path="01_captions.vtt" if captions.exists() else None,
            ).model_dump_json(indent=2),
            encoding="utf-8",
        )
    return root


@pytest.fixture
def cfg() -> Config:
    return Config()


@pytest.fixture
def legal_ir(tmp_path: Path):
    """一份完全合法的 (VideoIR, Transcript, base_dir)。"""
    from tests.factories import make_ir, make_transcript

    return make_ir(tmp_path), make_transcript(), tmp_path
