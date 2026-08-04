"""CLI。SDD §1.2：不做 UI，CLI + 設定檔。

兩個主要入口對應 §6.4 的 producer / consumer 分離：
  weft prepare <playlist|video>  S0–S3，不受額度限制，可一次跑完整個系列
  weft understand                S4–S6，消化 buffer 直到額度耗盡後自動停止
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import Config
from .logging_setup import setup_logging
from .paths import OutPaths, WorkPaths
from .state import PREPARE_STAGES, UNDERSTAND_STAGES


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="weft", description="講經影片 → 向量知識庫 pipeline")
    parser.add_argument("-c", "--config", type=Path, default=None, help="設定檔（YAML）")
    parser.add_argument("--log-level", default=None)
    sub = parser.add_subparsers(dest="command", required=True)

    survey = sub.add_parser(
        "survey", help="跑 S-1 素材勘查（新系列開跑前必跑，§4.0）")
    survey.add_argument("target", help="playlist URL / video URL / video_id")
    survey.add_argument("--sample", type=int, default=3,
                        help="抽樣幾支影片（預設 3）")

    prepare = sub.add_parser("prepare", help="跑 S0–S3（本地，不花額度）")
    prepare.add_argument("target", help="playlist URL / video URL / video_id")
    prepare.add_argument("--force", action="store_true", help="忽略既有 state，全部重跑")

    understand = sub.add_parser("understand", help="跑 S4–S6（消耗 Gemini 額度）")
    understand.add_argument("--video", default=None, help="只處理指定 video_id")
    understand.add_argument("--max-requests", type=int, default=None, help="本次執行的請求上限")

    sub.add_parser("status", help="掃描 work/，列出各影片的階段完成狀態")

    synth = sub.add_parser("synth", help="產生 §5.1 的 A1–A9 合成測試影片")
    synth.add_argument("--out", type=Path, default=Path("tests/fixtures/synth"))
    synth.add_argument("--force", action="store_true")

    return parser


def cmd_survey(args, cfg: Config) -> int:
    log = setup_logging(cfg.log_level, OutPaths(cfg.out_dir).root / "weft.log")
    log.info("survey：%s（抽樣 %d 支）", args.target, args.sample)
    from .pipeline import run_survey

    return run_survey(args.target, cfg, sample=args.sample)


def cmd_prepare(args, cfg: Config) -> int:
    from .stages import StageNotImplemented

    log = setup_logging(cfg.log_level, OutPaths(cfg.out_dir).root / "weft.log")
    log.info("prepare：%s（階段 %s）", args.target, " → ".join(s.value for s in PREPARE_STAGES))
    try:
        from .pipeline import run_prepare

        return run_prepare(args.target, cfg, force=args.force)
    except (ImportError, StageNotImplemented) as exc:
        log.error("尚未實作：%s", exc)
        return 2


def cmd_understand(args, cfg: Config) -> int:
    from .stages import StageNotImplemented

    log = setup_logging(cfg.log_level, OutPaths(cfg.out_dir).root / "weft.log")
    log.info("understand：階段 %s", " → ".join(s.value for s in UNDERSTAND_STAGES))
    try:
        from .pipeline import run_understand

        return run_understand(cfg, video_id=args.video, max_requests=args.max_requests)
    except (ImportError, StageNotImplemented) as exc:
        log.error("尚未實作：%s", exc)
        return 2


def cmd_status(args, cfg: Config) -> int:
    from .state import VideoState

    root = cfg.work_dir
    if not root.exists():
        print(f"{root} 不存在——尚未跑過 prepare。")
        return 0
    for video_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        work = WorkPaths(root, video_dir.name)
        if not work.state.exists():
            print(f"{video_dir.name}: 無 state.json")
            continue
        state = VideoState.load(work.state)
        done = [s.value for s, rec in state.stages.items() if rec.status == "done"]
        print(f"{video_dir.name}: {' '.join(done) or '（無完成階段）'}")
    return 0


def cmd_synth(args, cfg: Config) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tests.synth.build import build_all, probe_duration

    for name, (mp4, _) in build_all(args.out, force=args.force).items():
        print(f"{name}: {mp4} ({probe_duration(mp4):.2f}s)")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    cfg = Config.load(args.config)
    if args.log_level:
        cfg.log_level = args.log_level

    return {
        "survey": cmd_survey,
        "prepare": cmd_prepare,
        "understand": cmd_understand,
        "status": cmd_status,
        "synth": cmd_synth,
    }[args.command](args, cfg)


if __name__ == "__main__":
    raise SystemExit(main())
