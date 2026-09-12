"""陪看猫娘 · 同步播放窗口（可选的不截屏同步方案）

插件用 Playwright 开一个**可见**的浏览器窗口播放B站视频，
随时直接读 video.currentTime —— 毫秒级真实进度，暂停/拖动/倍速全感知。

架构：独立线程 + 独立 asyncio 循环（Playwright 必须单线程使用），
公开方法线程安全。窗口被用户手动关闭时自动标记失效（is_alive()=False）。
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any, Optional

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
)


class PlayerWindow:
    """可见的B站播放窗口：线程安全的 open()/position()/close()。"""

    def __init__(self) -> None:
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._page: Any = None
        self._browser: Any = None
        self._pw: Any = None
        self._closed = threading.Event()
        self._lock = threading.Lock()

    # ── 循环管理 ───────────────────────────────────────────────
    def _ensure_loop(self) -> bool:
        if self._closed.is_set():
            return False
        if self._loop is None or not self._loop.is_running():
            self._loop = asyncio.new_event_loop()
            self._thread = threading.Thread(target=self._run_loop, daemon=True, name="neko-player-window")
            self._thread.start()
        return True

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _submit(self, coro, timeout: float) -> Any:
        if self._loop is None or self._loop.is_running() is False:
            raise RuntimeError("播放窗口循环未运行")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    # ── 异步实现 ───────────────────────────────────────────────
    async def _async_open(self, bvid: str) -> tuple[bool, str]:
        from playwright.async_api import async_playwright

        self._pw = await async_playwright().start()
        errors = []
        for channel in (None, "msedge", "chrome"):
            try:
                if channel:
                    self._browser = await self._pw.chromium.launch(headless=False, channel=channel)
                else:
                    self._browser = await self._pw.chromium.launch(headless=False)
                break
            except Exception as exc:
                errors.append(str(exc)[:60])
                self._browser = None
        if self._browser is None:
            return False, "浏览器起不来喵：" + "；".join(errors)
        self._page = await self._browser.new_page(user_agent=_UA, viewport=None)
        await self._page.goto(
            f"https://www.bilibili.com/video/{bvid}/",
            timeout=40000,
            wait_until="domcontentloaded",
        )
        await self._page.bring_to_front()
        return True, f"同步播放窗口已打开（{bvid}）喵，等预习完成后在这里看就行～"

    async def _async_position(self) -> Optional[float]:
        if self._page is None or self._page.is_closed():
            return None
        try:
            return await self._page.evaluate(
                "() => { const v = document.querySelector('video'); return v ? v.currentTime : null; }"
            )
        except Exception:
            return None

    async def _async_close(self) -> None:
        try:
            if self._browser:
                await self._browser.close()
        except Exception:
            pass
        try:
            if self._pw:
                await self._pw.stop()
        except Exception:
            pass
        self._page = None
        self._browser = None
        self._pw = None

    # ── 公开 API（线程安全）────────────────────────────────────
    def open(self, bvid: str, timeout: float = 45.0) -> tuple[bool, str]:
        if self.is_alive():
            return True, "同步播放窗口已经开着喵"
        if not self._ensure_loop():
            return False, "播放窗口已关闭"
        try:
            return self._submit(self._async_open(bvid), timeout)
        except Exception as exc:
            self._closed.set()
            return False, f"打开失败：{exc}"

    def position(self, timeout: float = 5.0) -> Optional[float]:
        """读真实播放进度（秒）。窗口关闭/无视频元素返回 None。"""
        if self._closed.is_set() or self._loop is None or not self._loop.is_running():
            return None
        try:
            return self._submit(self._async_position(), timeout)
        except Exception:
            return None

    def seek(self, seconds: float, timeout: float = 5.0) -> bool:
        """把窗口里的视频 seek 到指定秒数（跳到 X 分时连播放器一起同步）。"""
        if self._closed.is_set() or self._loop is None or not self._loop.is_running():
            return False
        try:
            return bool(self._submit(
                self._page.evaluate(
                    "() => { const v = document.querySelector('video'); if (v) { v.currentTime = arguments_placeholder; } return !!v; }"
                    .replace("arguments_placeholder", str(float(seconds)))
                ),
                timeout,
            ))
        except Exception:
            return False

    def is_alive(self) -> bool:
        if self._closed.is_set():
            return False
        if self._page is not None and self._page.is_closed():
            self._closed.set()
            return False
        return self._loop is not None and self._loop.is_running()

    def close(self, timeout: float = 10.0) -> None:
        if self._loop is not None and self._loop.is_running():
            try:
                self._submit(self._async_close(), timeout)
            except Exception:
                pass
            self._loop.call_soon_threadsafe(self._loop.stop)
        self._closed.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
