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


@pytest.fixture
def cfg() -> Config:
    return Config()


@pytest.fixture
def legal_ir(tmp_path: Path):
    """一份完全合法的 (VideoIR, Transcript, base_dir)。"""
    from tests.factories import make_ir, make_transcript

    return make_ir(tmp_path), make_transcript(), tmp_path
