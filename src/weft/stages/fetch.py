"""yt-dlp 封裝。SDD §4.1。

yt-dlp 以 Python 模組使用，不透過 subprocess——錯誤分類（影片私人／已刪除
／地區限制）需要例外型別，剖析 stderr 字串會在 yt-dlp 改版時默默失效。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

_VIDEO_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")
_URL_ID = re.compile(r"(?:v=|/shorts/|youtu\.be/|/embed/)([A-Za-z0-9_-]{11})")
_PLAYLIST_ID = re.compile(r"[?&]list=([A-Za-z0-9_-]+)")


class VideoUnavailable(Exception):
    """影片私人／已刪除／地區限制。SDD §4.1：記入 skip list，繼續下一支。"""


@dataclass(frozen=True)
class PlaylistItem:
    video_id: str
    series_id: str | None
    episode_index: int | None


def parse_target(target: str) -> tuple[str | None, str | None]:
    """把使用者給的字串解析成 `(video_id, playlist_id)`。

    兩者可能同時存在（playlist 中某一支的 URL）。此時以 playlist 為準——
    使用者貼這種 URL 通常是想跑整個系列。
    """
    playlist = _PLAYLIST_ID.search(target)
    playlist_id = playlist.group(1) if playlist else None

    if _VIDEO_ID.match(target.strip()):
        return target.strip(), playlist_id
    if target.startswith("PL") or target.startswith("UU") or target.startswith("OL"):
        return None, target.strip()

    found = _URL_ID.search(target)
    return (found.group(1) if found else None), playlist_id


def expand_playlist(playlist_id: str) -> list[PlaylistItem]:
    """展開 playlist 為影片清單，保留 episode_index（§3.5 的 metadata 欄位）。

    用 `extract_flat`：只要清單，不必逐支解析格式——一個 50 集的系列，
    完整解析要數分鐘，flat 只要幾秒。
    """
    import yt_dlp

    opts = {"quiet": True, "no_warnings": True, "extract_flat": "in_playlist"}
    url = f"https://www.youtube.com/playlist?list={playlist_id}"
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)

    items: list[PlaylistItem] = []
    for i, entry in enumerate(info.get("entries") or [], start=1):
        if not entry or not entry.get("id"):
            continue  # 私人／已刪除的項目在 flat 模式下會是 None
        items.append(PlaylistItem(entry["id"], playlist_id, i))
    return items


def _pick_caption(info: dict, langs: list[str]) -> tuple[str | None, bool]:
    """依 SDD §4.1 的優先序挑字幕：手動 > 自動 > 無。回傳 `(lang, is_manual)`。"""
    manual = info.get("subtitles") or {}
    auto = info.get("automatic_captions") or {}
    for lang in langs:
        if lang in manual:
            return lang, True
    for lang in langs:
        if lang in auto:
            return lang, False
    return None, False


def download(video_id: str, work, cfg) -> dict:
    """下載影片與字幕，回傳 yt-dlp 的 info dict。

    影片已存在則跳過下載但仍取 metadata——冪等鍵是 video_id（§4.1），
    重跑不該重新下載幾百 MB。
    """
    import yt_dlp
    from yt_dlp.utils import DownloadError, ExtractorError

    p = cfg.s0
    url = f"https://www.youtube.com/watch?v={video_id}"

    probe = {"quiet": True, "no_warnings": True, "skip_download": True}
    try:
        with yt_dlp.YoutubeDL(probe) as ydl:
            info = ydl.extract_info(url, download=False)
    except (DownloadError, ExtractorError) as exc:
        raise VideoUnavailable(f"{video_id}：{exc}") from exc
    if info is None:
        raise VideoUnavailable(f"{video_id}：yt-dlp 未回傳 metadata")

    lang, is_manual = _pick_caption(info, p.caption_langs)

    if not work.video.exists():
        opts = {
            "quiet": True,
            "no_warnings": True,
            "format": p.video_format,
            # yt-dlp 會依實際容器補副檔名，故 outtmpl 不含 .mp4
            "outtmpl": str(work.dir / "01_video.%(ext)s"),
            "merge_output_format": "mp4",
            "writesubtitles": bool(lang and is_manual),
            "writeautomaticsub": bool(lang and not is_manual and p.write_auto_caption),
            "subtitleslangs": [lang] if lang else [],
            "subtitlesformat": "vtt",
            "ratelimit": p.rate_limit,
        }
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([url])
        except (DownloadError, ExtractorError) as exc:
            raise VideoUnavailable(f"{video_id}：{exc}") from exc

        # 合流後的副檔名不一定是 mp4（例如純 webm 來源），統一成契約上的檔名
        for candidate in sorted(work.dir.glob("01_video.*")):
            if candidate != work.video and candidate.suffix != ".vtt":
                candidate.rename(work.video)
                break

    for vtt in sorted(work.dir.glob("01_video*.vtt")):
        vtt.rename(work.captions)
        break

    return info
