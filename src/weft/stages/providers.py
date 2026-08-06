"""模型供應者抽象層。SDD §2.3（v0.4）。

v0.4 起模型**逐子階段可設定**，因為實測同一批素材上不同子工作的最佳模型
不同（`experiments/r21_bakeoff/`）：

| | is_slide | slide_text CER（跨集） | 中位耗時 |
|---|---|---|---|
| gemma4:12b | **90.5%** | 37.8% | 20.5s |
| qwen2.5vl:7b | 66.7% | **4.9%** | **1.7s** |

模型規格字串的格式是 `供應者:模型名`：

    gemini:gemini-3.1-flash-lite
    ollama:qwen2.5vl:7b

**供應者必須明寫。** 不做「猜猜看這是本地還是雲端」——
§5.5 #6 要求本地 fallback 是明確的設定開關，猜測會讓那條規定失效。

**`model_used` 一定要記進輸出**：同一支影片的不同階段可能用不同模型，
§5.6 的人工抽檢要分得出「這段是誰產的」。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)

#: 送進模型的一段內容。`image` 為圖檔位元組，`text` 為文字，兩者擇一。
@dataclass(frozen=True)
class Part:
    text: str | None = None
    image: bytes | None = None

    def __post_init__(self) -> None:
        if (self.text is None) == (self.image is None):
            raise ValueError("Part 必須且只能有 text 或 image 其中一個")


@dataclass(frozen=True)
class Result:
    payload: dict
    input_tokens: int = 0
    output_tokens: int = 0
    model_used: str = ""


class UnknownProvider(RuntimeError):
    pass


#: 已知的供應者。**對照清單而不是「有冒號就算」**——`gemma4:12b` 會被
#: 切成 `("gemma4", "12b")`，兩邊都非空，若不查清單就會靜靜通過，
#: 到實際呼叫時才爆一個看不懂的錯。設定漏寫供應者是很容易犯的錯。
PROVIDERS = ("gemini", "ollama")


def split_spec(spec: str) -> tuple[str, str]:
    """`"ollama:qwen2.5vl:7b"` → `("ollama", "qwen2.5vl:7b")`。"""
    provider, _, model = spec.partition(":")
    if provider not in PROVIDERS or not model:
        raise UnknownProvider(
            f"模型規格要寫成 `供應者:模型名`（供應者為 {'／'.join(PROVIDERS)}），"
            f"例如 `ollama:gemma4:12b`；收到 {spec!r}"
        )
    return provider, model


def generate(spec: str, system: str, parts: list[Part], schema: dict,
             temperature: float = 0.2, num_ctx: int | None = None) -> Result:
    """依規格字串分派到對應的供應者。回傳解析後的 JSON。

    **JSON 解析失敗時拋例外**，由呼叫端的 `with_retries` 處理——
    這裡不吞錯，也不回傳半套結果。
    """
    provider, model = split_spec(spec)
    if provider == "gemini":
        return _gemini(model, system, parts, schema, temperature)
    return _ollama(model, system, parts, schema, temperature, num_ctx)


def _gemini(model: str, system: str, parts: list[Part], schema: dict,
            temperature: float) -> Result:
    from google.genai import types

    from .understand import _client

    contents: list = []
    for part in parts:
        if part.text is not None:
            contents.append(part.text)
        else:
            contents.append(types.Part.from_bytes(data=part.image, mime_type="image/png"))

    response = _client().models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_schema=schema,
            temperature=temperature,
        ),
    )
    usage = getattr(response, "usage_metadata", None)
    return Result(
        payload=json.loads(response.text),
        input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
        output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
        model_used=f"gemini:{model}",
    )


#: 每個 ollama 模型的 context 長度。**不是為了讓誰好看而個別調**——
#: gemma4:12b 在 8192 下對長文投影片回傳空內容（`done_reason=None`，
#: 即請求根本沒完成），16384 才正常。它的影像 token 比 gemma3 多。
#: 用不足的 ctx 去比等於在比「誰的圖比較小」。實測見 R21 §6。
OLLAMA_NUM_CTX: dict[str, int] = {"gemma4:12b": 16384}
DEFAULT_NUM_CTX = 8192

OLLAMA_URL = "http://localhost:11434/api/chat"


def _ollama(model: str, system: str, parts: list[Part], schema: dict,
            temperature: float, num_ctx: int | None) -> Result:
    import base64

    import requests

    text_parts = [p.text for p in parts if p.text is not None]
    images = [base64.b64encode(p.image).decode() for p in parts if p.image is not None]

    message: dict = {"role": "user", "content": "\n\n".join(text_parts)}
    if images:
        message["images"] = images

    response = requests.post(OLLAMA_URL, timeout=1800, json={
        "model": model,
        "messages": [{"role": "system", "content": system}, message],
        "format": schema,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_ctx": num_ctx or OLLAMA_NUM_CTX.get(model, DEFAULT_NUM_CTX),
        },
    })
    response.raise_for_status()
    body = response.json()
    content = body["message"].get("content") or ""
    if not content.strip():
        # gemma4 在 ctx 不足時就是這個症狀；qwen3-vl 的 thinking 變體也是。
        # **不要當成空結果吞掉**——那會讓一張投影片靜靜地沒有文字。
        raise RuntimeError(
            f"ollama {model} 回傳空內容（done_reason={body.get('done_reason')}，"
            f"eval_count={body.get('eval_count')}）。"
            f"常見原因：num_ctx 不足、或該 tag 是 thinking 變體"
        )
    return Result(
        payload=json.loads(content),
        input_tokens=body.get("prompt_eval_count", 0) or 0,
        output_tokens=body.get("eval_count", 0) or 0,
        model_used=f"ollama:{model}",
    )


def costs_quota(spec: str) -> bool:
    """這個規格會不會消耗雲端額度。§6 的估算依此計算。"""
    return split_spec(spec)[0] == "gemini"
