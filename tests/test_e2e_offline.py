"""離線端到端：不需要網路、不需要 ollama、不需要額度。

**這個檔案回答的問題是「擱置數週後回來，現在還是好的嗎」。**
它必須在一台沒有網路的機器上、在幾秒內跑完。

與 `test_e2e_pipeline.py` 的分工：

| | 依賴 | 涵蓋 |
|---|---|---|
| 這裡 | 無 | S6 渲染、§5.4 溯源、§5.3 不變量、chunks.jsonl 契約 |
| `@pytest.mark.synth` | ffmpeg | S1b–S3 本地管線 |
| `@pytest.mark.live` | 網路 + ollama | S0–S6 全程，實際呼叫模型 |

**這裡沒有 mock 模型**（§5.5 #10）——它根本不呼叫模型。
S4 的產物由 fixture 提供，測的是**拿到理解結果之後**那一段。
那正是擱置後最容易壞掉、又最不需要外部資源就能驗證的部分。

已知的涵蓋缺口，刻意留著並寫在這裡而不是假裝沒有：
fixture 是手寫的 happy path，不是真實影片的 IR。真實資料的形狀
（空 slide_text、批次少回、簡繁混雜）由 `@pytest.mark.live` 那層覆蓋。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from weft.config import Config
from weft.paths import OutPaths, WorkPaths
from weft.validation import invariants as inv

pytestmark = pytest.mark.e2e


@pytest.fixture
def offline_run(tmp_path: Path):
    """把一份合法 IR 擺成 work/ 佈局，回傳 (cfg, ir, transcript, work, out)。

    刻意用 `factories.make_ir` 而不是版控一份真實影片的 IR：
    真實素材是他人著作（SDD §9），而 §3.1 的 work/ 本來就不進版控。
    """
    from tests.factories import make_ir, make_transcript

    cfg = Config()
    cfg.work_dir = tmp_path / "work"
    cfg.out_dir = tmp_path / "out"

    ir = make_ir(tmp_path)
    transcript = make_transcript()

    work = WorkPaths(cfg.work_dir, ir.meta.video_id)
    work.ensure_dirs()
    # `make_ir` 把圖寫在 tmp_path 下，IR 的 image_path 是相對於 work.dir 的
    for src in (tmp_path / "03_slides").glob("*.png"):
        (work.dir / "03_slides" / src.name).write_bytes(src.read_bytes())

    out = OutPaths(cfg.out_dir)
    out.ensure_dirs()
    return cfg, ir, transcript, work, out


def test_render_produces_chunks_without_network_or_models(offline_run):
    """S6 從快取的 IR 產出 chunks.jsonl，全程不碰網路與模型。"""
    from weft.ir import Chunk
    from weft.stages.cloud import s6_render

    cfg, ir, _transcript, work, out = offline_run
    chunks = s6_render(cfg, ir, work, out)

    assert chunks, "S6 沒有產出任何 chunk"
    assert out.chunks.exists()
    written = [
        Chunk.model_validate_json(line)
        for line in out.chunks.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(written) == len(chunks)


def test_chunks_satisfy_all_invariants(offline_run):
    """§5.3 的不變量對真正落地的 chunk 全數成立。

    這是 `chunks.jsonl` 的契約：metadata 完整、時間軸單調、
    圖檔存在、`text_raw` 未被竄改。擱置後最可能悄悄壞掉的就是這一層。
    """
    from weft.stages.cloud import s6_render

    cfg, ir, transcript, work, out = offline_run
    chunks = s6_render(cfg, ir, work, out)

    violations = inv.check_all(ir, transcript, work.dir, chunks=chunks)
    assert violations == [], "\n".join(str(v) for v in violations)


def test_provenance_runs_and_fills_verdicts(offline_run):
    """§5.4 溯源檢查會就地填回每個 block 的判定與成因。

    只斷言「有跑、有填」，不斷言通過率——那個數字取決於 fixture 內容，
    寫死它會變成測 fixture 而不是測機制。
    """
    from weft.stages.cloud import s6_render
    from weft.validation.provenance import check_video

    cfg, ir, _transcript, work, out = offline_run
    s6_render(cfg, ir, work, out)

    blocks = [b for s in ir.segments if s.understanding for b in s.understanding.content_blocks]
    assert blocks
    assert all(b.verification is not None for b in blocks)
    assert all(b.similarity is not None for b in blocks)

    verdict = check_video(ir, cfg.provenance)
    assert verdict.total == len(blocks)


def test_debug_markdown_image_links_resolve(offline_run):
    """§5.6 的人工複核文件，圖片連結必須真的打得開。

    D23：這些連結曾經**一張都打不開**，而 §5.6 的人工抽檢正是靠看圖
    抓到 D20 的。連結壞掉不會有任何機械檢查報錯。
    """
    import re

    from weft.stages.cloud import s6_render

    cfg, ir, _transcript, work, out = offline_run
    s6_render(cfg, ir, work, out)

    md_path = out.debug_md(ir.meta.video_id)
    assert md_path.exists()
    links = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", md_path.read_text(encoding="utf-8"))
    assert links, "複核文件裡沒有任何圖片連結"
    for link in links:
        assert (md_path.parent / link).resolve().exists(), f"圖片連結打不開：{link}"


def test_rerun_is_idempotent(offline_run):
    """同一支影片跑兩次，產出不得變多。

    §6.3 的冪等性在這一層的意思是「重跑不會複製產出」。實測 `chunks.jsonl`
    曾累積到 428 行而相異 id 只有 91 個——重複的 chunk 會讓同一段內容
    在檢索結果裡佔掉數個名額。
    """
    from weft.stages.cloud import s6_render

    cfg, ir, _transcript, work, out = offline_run
    s6_render(cfg, ir, work, out)
    first = out.chunks.read_text(encoding="utf-8")
    s6_render(cfg, ir, work, out)
    assert out.chunks.read_text(encoding="utf-8") == first


def test_no_module_reaches_the_network_at_import_time():
    """匯入 weft 的任何模組都不得需要網路或 API key。

    沒有這一條，「離線可跑」會在某次有人把 client 建構搬到模組層級時
    悄悄失效，而失敗訊息會長得像別的問題。
    """
    import importlib
    import pkgutil

    import weft

    failed = []
    for mod in pkgutil.walk_packages(weft.__path__, prefix="weft."):
        try:
            importlib.import_module(mod.name)
        except Exception as exc:  # noqa: BLE001
            failed.append(f"{mod.name}: {exc}")
    assert not failed, "\n".join(failed)
