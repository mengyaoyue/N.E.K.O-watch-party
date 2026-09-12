"""陪看猫娘 · 截屏辅助（可选，默认关闭）

截取主屏画面 → 本地 OCR 提取文字 → 让猫娘"看见"画面内容（视频标题、画面文字、
弹幕框），弥补无字幕视频的感知空白。

隐私红线：
- 默认关闭；只在配置开启后、且仅在 react_now/心跳时截取单帧
- OCR 为本地推理（rapidocr），文字不落盘不上传
- 任何环节失败都静默降级（返回 not ok），绝不影响陪看主流程
"""

from __future__ import annotations

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


def ocr_frame(frame: Any) -> tuple[bool, str]:
    """对单帧做本地 OCR（rapidocr），返回 (成功, 合并文字)。"""
    try:
        from rapidocr_onnxruntime import RapidOCR

        ocr = RapidOCR()
        result, _ = ocr(frame)
        if not result:
            return True, ""
        lines = []
        for item in result[:40]:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                text = str(item[1]).strip()
                if text:
                    lines.append(text)
        return True, " ".join(lines)[:600]
    except Exception as exc:
        return False, f"OCR 失败：{exc}"


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
