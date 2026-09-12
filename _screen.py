"""陪看猫娘 · 截屏辅助（可选，默认关闭）

截取主屏画面 → 本地 OCR 提取文字 → 让猫娘"看见"画面内容（视频标题、画面文字、
弹幕框），弥补无字幕视频的感知空白。

隐私红线：
- 默认关闭；只在配置开启后、且仅在 react_now/心跳时截取单帧
- OCR 为本地推理（rapidocr），文字不落盘不上传
- 任何环节失败都静默降级（返回 not ok），绝不影响陪看主流程
"""

from __future__ import annotations

import re
from typing import Any

_OK_SENTINEL = "__SCREEN_OK__"


def capture_frame(timeout: float = 3.0) -> tuple[bool, Any]:
    """用 dxcam 抓一帧主屏。返回 (成功, 图像数组或错误信息)。"""
    try:
        import dxcam

        camera = dxcam.create(output_color="BGR")
        if camera is None:
            return False, "dxcam 相机创建失败（可能被其它实例占用）"
        frame = camera.grab()
        if frame is None:
            # dxcam 偶发 None（无新帧），补一次带等待的抓取
            import time as _time

            _time.sleep(0.2)
            frame = camera.grab()
        if frame is None:
            return False, "抓帧为空"
        return True, frame
    except Exception as exc:
        return False, f"截屏失败：{exc}"


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
        if pos is None or total is None or total == 0 or pos > total:
            continue
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
