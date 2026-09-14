"""陪看猫娘 · 截屏辅助（可选，默认关闭）

两种"看懂画面"的方式，互为兜底：

1. 视觉模型（首选）：把当前帧压缩成 JPEG 交给视觉模型描述画面内容。
   接口对任意厂商通用（OpenAI 兼容 / Anthropic / 其它），靠 provider_type 适配；
   默认走宿主配置的 vision 通道（没配就退回 conversation），
   也可用插件级 vision_model / vision_base_url / vision_api_key 直接覆盖。
2. 本地 OCR（兜底）：rapidocr 本地推理，不联网、不上传。

帧来源支持两种：主屏实时抓帧（dxcam → ImageGrab → mss 多后端兜底）/
同步播放窗口抽帧（Playwright，见 _player_window）。

隐私红线：
- 默认关闭；只在配置开启后、且仅在 react_now/心跳/定时反应时取单帧
- 帧只存在于内存，随用随弃，不落盘
- 任何环节失败都静默降级（返回 not ok），绝不影响陪看主流程
"""

from __future__ import annotations

import asyncio
import re
from typing import Any, Optional

_JPEG_QUALITY = 80
_MAX_SIDE = 1280

VISION_PROMPT = (
    "这是一张B站视频的播放画面截图。请用不超过40字的中文客观描述画面里正在发生什么"
    "（人物/动作/场景/画面上的文字或字幕），不要加括号、不要评价、不要编造看不到的内容。"
)


# ── 抓帧后端链：dxcam → PIL.ImageGrab → mss ─────────────────────
# 借鉴官方 galgame_plugin 的做法：逐个后端试，失败记进 errors 再换下一个。
# 关键点：dxcam 依赖 cv2，而宿主自带运行时不一定有 cv2；
# 所以必须有"不依赖 cv2"的 Pillow(ImageGrab) 兜底，否则截屏会整体挂掉。


def _capture_dxcam() -> Any:
    """首选：dxcam 抓主屏（GPU 直读，最快，依赖 cv2）。"""
    import dxcam

    camera = dxcam.create(output_color="BGR")
    if camera is None:
        raise RuntimeError("dxcam 相机创建失败（可能被其它实例占用）")
    frame = camera.grab()
    if frame is None:
        # dxcam 偶发 None（无新帧），补一次带等待的抓取
        import time as _time

        _time.sleep(0.2)
        frame = camera.grab()
    if frame is None:
        raise RuntimeError("dxcam 抓帧为空")
    return frame


def _capture_imagegrab() -> Any:
    """兜底一：Pillow ImageGrab（宿主必带 Pillow，不依赖 cv2，只抓主屏）。"""
    from PIL import ImageGrab

    image = ImageGrab.grab()
    if image is None:
        raise RuntimeError("ImageGrab 返回空图")
    return image


def _capture_mss() -> Any:
    """兜底二：mss（跨平台，装了才可用）。"""
    import mss
    from PIL import Image

    with mss.mss() as sct:
        shot = sct.grab(sct.monitors[1])
    return Image.frombytes("RGB", shot.size, shot.rgb)


_CAPTURE_BACKENDS = (
    ("dxcam", _capture_dxcam),
    ("imagegrab", _capture_imagegrab),
    ("mss", _capture_mss),
)

_LAST_CAPTURE_BACKEND = ""


def last_capture_backend() -> str:
    """最近一次成功抓帧用的后端名（诊断用）。"""
    return _LAST_CAPTURE_BACKEND


def capture_frame(timeout: float = 3.0) -> tuple[bool, Any]:
    """抓一帧主屏，按后端链依次尝试，成功即返回。

    顺序：dxcam（最快）→ PIL.ImageGrab（宿主必带的 Pillow，不依赖 cv2）→ mss。
    返回 (成功, 帧或错误信息)；帧可能是 ndarray(BGR) 或 PIL.Image，
    下游 _to_pil / OCR 都能吃。
    """
    global _LAST_CAPTURE_BACKEND
    errors: list[str] = []
    for name, grab in _CAPTURE_BACKENDS:
        try:
            frame = grab()
        except Exception as exc:
            errors.append(f"{name}:{exc}")
            continue
        if frame is None:
            errors.append(f"{name}:空帧")
            continue
        _LAST_CAPTURE_BACKEND = name
        return True, frame
    return False, "截屏失败：" + ("；".join(errors) or "无可用后端")


def ocr_frame(frame: Any, keep_boxes: bool = False) -> tuple[bool, Any]:
    """对单帧做本地 OCR（rapidocr）。

    keep_boxes=False → 返回 (成功, 合并文字)
    keep_boxes=True  → 返回 (成功, [{text, x, y}] 列表，保留位置用于进度条识别)
    """
    try:
        from rapidocr_onnxruntime import RapidOCR

        ocr = RapidOCR()
        result, _ = ocr(frame)
        if not result:
            return True, ("" if not keep_boxes else [])
        items = []
        for item in result[:60]:
            if not (isinstance(item, (list, tuple)) and len(item) >= 2):
                continue
            text = str(item[1]).strip()
            if not text:
                continue
            if keep_boxes:
                box = item[0]
                xs = [pt[0] for pt in box] if box else [0]
                ys = [pt[1] for pt in box] if box else [0]
                items.append({"text": text, "x": min(xs), "y": min(ys)})
            else:
                items.append(text)
        merged = " ".join(items)[:600] if not keep_boxes else items
        return True, merged
    except Exception as exc:
        return False, f"OCR 失败：{exc}"


_TIME_RE = re.compile(r"(?<![0-9])([0-9]{1,2}):([0-5][0-9])(?::([0-5][0-9]))?(?![0-9])")


def extract_playback_time(ocr_items: list[dict[str, Any]], screen_h: int) -> Optional[dict[str, int]]:
    """从 OCR 结果里找播放器进度时间。

    只认"当前 / 总时长"成对格式（如 01:23 / 09:30）——
    单独出现的时间一律不采信：系统时钟（21:47）、视频时长标签都在屏幕底部，
    误认会把本地时间当成播放进度（真实事故）。
    返回 {position, total}，找不到 None。
    """
    pair_re = re.compile(
        r"([0-9]{1,2}:[0-5][0-9](?::[0-5][0-9])?)\s*/\s*([0-9]{1,2}:[0-5][0-9](?::[0-5][0-9])?)"
    )
    for it in ocr_items:
        text = it.get("text", "")
        m = pair_re.search(text)
        if not m:
            continue
        pos = _to_seconds(m.group(1))
        total = _to_seconds(m.group(2))
        if pos is None or total is None or total == 0:
            continue
        if pos > total:
            pos, total = total, pos  # 顺序颠倒自动纠正
        return {"position": pos, "total": total}
    return None


def _to_seconds(text: str) -> Optional[int]:
    parts = text.split(":")
    try:
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        return int(parts[0]) * 60 + int(parts[1])
    except (ValueError, IndexError):
        return None


def capture_screen_text(timeout: float = 3.0) -> dict[str, Any]:
    """截屏 + OCR 一站式：返回 {"ok", "text", "error"}。文字上限 600 字。"""
    ok, frame = capture_frame(timeout)
    if not ok:
        return {"ok": False, "text": "", "error": str(frame)}
    ok2, text = ocr_frame(frame)
    if not ok2:
        return {"ok": False, "text": "", "error": text}
    return {"ok": True, "text": text, "error": ""}


def build_screen_context(ocr_text: str, max_chars: int = 300) -> str:
    """把 OCR 文本拼成给猫娘看的上下文片段。"""
    text = (ocr_text or "").strip()
    if not text:
        return ""
    if len(text) > max_chars:
        text = text[:max_chars] + "…"
    return f"【画面上的文字】{text}"


# ── 视觉模型通道（厂商通用）───────────────────────────────────────


def _to_pil(frame: Any) -> Any:
    """把抓帧结果（ndarray/PIL）/ PNG·JPEG 字节 统一成 PIL.Image。"""
    from PIL import Image

    if frame is None:
        return None
    if isinstance(frame, Image.Image):
        return frame
    if isinstance(frame, (bytes, bytearray)):
        import io as _io

        try:
            return Image.open(_io.BytesIO(bytes(frame)))
        except Exception:
            return None
    if hasattr(frame, "shape"):
        try:
            import numpy as np

            arr = np.asarray(frame)
            if arr.ndim == 3 and arr.shape[2] >= 3:
                arr = arr[:, :, ::-1]  # dxcam 默认 BGR → RGB
        except Exception:
            arr = frame
        return Image.fromarray(arr)
    if hasattr(frame, "convert"):
        return frame
    return None


def frame_to_data_url(
    frame: Any,
    max_side: int = _MAX_SIDE,
    quality: int = _JPEG_QUALITY,
) -> tuple[bool, str]:
    """帧 → 压缩后的 JPEG data URL（供视觉模型消费）。

    压缩档位借鉴官方截屏工具：长边限制 + JPEG q80，先压再传，省流量也省额度。
    返回 (成功, data_url 或错误信息)。
    """
    try:
        import base64
        import io

        from PIL import Image
    except Exception as exc:
        return False, f"缺少图像库（Pillow）：{exc}"
    try:
        image = _to_pil(frame)
        if image is None:
            return False, "不支持的帧格式"
        image = image.convert("RGB")
        width, height = image.size
        longest = max(width, height)
        if max_side and longest > int(max_side):
            scale = float(max_side) / float(longest)
            image = image.resize(
                (max(1, int(width * scale)), max(1, int(height * scale))),
                Image.LANCZOS,
            )
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=max(30, min(95, int(quality))), optimize=True)
        raw = buffer.getvalue()
        if not raw:
            return False, "JPEG 编码为空"
        return True, "data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii")
    except Exception as exc:
        return False, f"帧编码失败：{exc}"


def resolve_vision_config(override: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """确定视觉模型参数：插件级覆盖优先，否则读宿主 vision → conversation 通道。

    返回 {model, base_url, api_key, provider_type, source}；不可用时含 "error"。
    provider_type 原样透传给宿主 LLM 客户端，所以任意厂商配置都能跑。
    """
    override = override or {}
    model = str(override.get("vision_model") or "").strip()
    base_url = str(override.get("vision_base_url") or "").strip().rstrip("/")
    api_key = str(override.get("vision_api_key") or "").strip()
    provider_type = str(override.get("vision_provider_type") or "").strip() or None
    if model and base_url:
        return {
            "model": model,
            "base_url": base_url,
            "api_key": api_key,
            "provider_type": provider_type,
            "source": "plugin",
        }
    try:
        from utils.config_manager import get_config_manager
    except Exception as exc:
        return {"error": f"宿主配置不可用：{exc}"}
    manager = get_config_manager()
    for group in ("vision", "conversation"):
        try:
            cfg = manager.get_model_api_config(group) or {}
        except Exception:
            cfg = {}
        m = str(cfg.get("model") or "").strip()
        b = str(cfg.get("base_url") or "").strip().rstrip("/")
        if not m or not b:
            continue
        return {
            "model": m,
            "base_url": b,
            "api_key": str(cfg.get("api_key") or "").strip(),
            "provider_type": str(cfg.get("provider_type") or "").strip() or None,
            "source": f"host:{group}",
        }
    return {"error": "宿主未配置可用的视觉/对话模型"}


def _content_to_text(response: Any) -> str:
    content = getattr(response, "content", None)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(str(block.get("text") or ""))
            else:
                parts.append(str(block))
        return "".join(parts).strip()
    return str(content or "").strip()


async def describe_frame(
    frame: Any,
    *,
    prompt: str = "",
    override: Optional[dict[str, Any]] = None,
    max_side: int = _MAX_SIDE,
    quality: int = _JPEG_QUALITY,
    timeout: float = 25.0,
    max_tokens: int = 1024,
) -> tuple[bool, str]:
    """把一帧交给视觉模型描述。返回 (成功, 描述或错误信息)。"""
    ok, data_url = frame_to_data_url(frame, max_side, quality)
    if not ok:
        return False, data_url
    cfg = resolve_vision_config(override)
    if cfg.get("error"):
        return False, str(cfg["error"])
    try:
        from utils.llm_client import create_chat_llm_async
    except Exception as exc:
        return False, f"视觉接口不可用：{exc}"
    try:
        llm = await create_chat_llm_async(
            model=cfg["model"],
            base_url=cfg["base_url"],
            api_key=cfg.get("api_key") or "",
            max_completion_tokens=int(max_tokens),
            timeout=float(timeout),
            provider_type=cfg.get("provider_type"),
        )
    except Exception as exc:
        return False, f"视觉模型初始化失败：{exc}"
    try:
        response = await asyncio.wait_for(
            llm.ainvoke(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": data_url}},
                            {"type": "text", "text": (prompt or VISION_PROMPT)[:800]},
                        ],
                    }
                ]
            ),
            timeout=float(timeout),
        )
        text = _content_to_text(response)
        if not text:
            return False, "视觉模型返回空内容"
        return True, text
    except asyncio.TimeoutError:
        return False, "视觉模型超时"
    except Exception as exc:
        return False, f"视觉模型调用失败：{exc}"
    finally:
        aclose = getattr(llm, "aclose", None)
        if callable(aclose):
            try:
                await aclose()
            except Exception:
                pass


async def describe_screen(
    *,
    prompt: str = "",
    override: Optional[dict[str, Any]] = None,
    max_side: int = _MAX_SIDE,
    quality: int = _JPEG_QUALITY,
    timeout: float = 25.0,
) -> dict[str, Any]:
    """抓主屏一帧 → 视觉模型描述。返回 {"ok","text","error","source"}。"""
    ok, frame = await asyncio.to_thread(capture_frame)
    if not ok:
        return {"ok": False, "text": "", "error": str(frame), "source": "screen"}
    ok2, text = await describe_frame(
        frame, prompt=prompt, override=override, max_side=max_side, quality=quality, timeout=timeout
    )
    return {
        "ok": bool(ok2),
        "text": text if ok2 else "",
        "error": "" if ok2 else text,
        "source": "screen",
    }
