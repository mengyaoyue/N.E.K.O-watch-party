"""陪看猫娘 · 同步播放窗口（可选的不截屏同步方案）

插件用 Playwright 开一个**可见**的浏览器窗口播放B站视频，
随时直接读 video.currentTime —— 毫秒级真实进度，暂停/拖动/倍速全感知。

这个窗口同时承担三件事：
1. **登录B站**：用持久化用户目录（user_data_dir）保存登录态，
   窗口里登录一次，之后随时导出 SESSDATA 喂给字幕接口（免去 F12 手动复制）。
2. **真实进度**：读 video.currentTime，作为陪看时间轴的最高优先级来源。
3. **抽帧**：直接截取 <video> 元素画面，比抓主屏更干净，供视觉模型"看"。

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

_LOGIN_URL = "https://www.bilibili.com/"
_CHANNELS = (None, "msedge", "chrome")


class PlayerWindow:
    """可见的B站播放窗口：线程安全的 open()/position()/seek()/sessdata()/grab_frame()。"""

    def __init__(self, user_data_dir: Optional[str] = None) -> None:
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._page: Any = None
        self._context: Any = None
        self._browser: Any = None
        self._pw: Any = None
        self._persistent = False
        self._user_data_dir = str(user_data_dir) if user_data_dir else None
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

    # ── 浏览器启动（优先持久化上下文，保住登录态）─────────────
    async def _async_launch(self) -> tuple[bool, str]:
        from playwright.async_api import async_playwright

        if self._pw is not None:
            try:
                await self._pw.stop()
            except Exception:
                pass
            self._pw = None
        self._pw = await async_playwright().start()
        errors: list[str] = []

        if self._user_data_dir:
            try:
                import os

                os.makedirs(self._user_data_dir, exist_ok=True)
            except Exception as exc:
                errors.append(f"profile 目录不可用：{exc}"[:80])
                self._user_data_dir = None

        if self._user_data_dir:
            for channel in _CHANNELS:
                try:
                    self._context = await self._launch_persistent(channel)
                    self._persistent = True
                    return True, ""
                except Exception as exc:
                    errors.append(str(exc)[:80])
                    self._context = None
        for channel in _CHANNELS:
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
        self._context = await self._new_context()
        self._persistent = False
        return True, ""

    async def _launch_persistent(self, channel: Optional[str]) -> Any:
        kwargs: dict[str, Any] = {"headless": False, "user_agent": _UA, "no_viewport": True}
        if channel:
            kwargs["channel"] = channel
        try:
            return await self._pw.chromium.launch_persistent_context(self._user_data_dir, **kwargs)
        except TypeError:
            kwargs.pop("no_viewport", None)
            return await self._pw.chromium.launch_persistent_context(self._user_data_dir, **kwargs)

    async def _new_context(self) -> Any:
        try:
            return await self._browser.new_context(user_agent=_UA, no_viewport=True)
        except TypeError:
            return await self._browser.new_context(user_agent=_UA)

    # ── 页面操作 ───────────────────────────────────────────────
    def _pick_page(self):
        pages = list(getattr(self._context, "pages", []) or [])
        for page in pages:
            try:
                if not page.is_closed():
                    return page
            except Exception:
                continue
        return None

    async def _async_goto(self, url: str) -> tuple[bool, str]:
        if self._context is None:
            return False, "窗口没开喵"
        if self._page is None or self._page.is_closed():
            self._page = self._pick_page()
        if self._page is None:
            try:
                self._page = await self._context.new_page()
            except Exception as exc:
                return False, f"开新页面失败：{exc}"
        try:
            await self._page.goto(url, timeout=40000, wait_until="domcontentloaded")
            await self._page.bring_to_front()
        except Exception as exc:
            return False, f"打开页面失败：{exc}"
        return True, ""

    async def _async_open(self, bvid: str) -> tuple[bool, str]:
        if self._context is None:
            ok, err = await self._async_launch()
            if not ok:
                return False, err
        ok, err = await self._async_goto(f"https://www.bilibili.com/video/{bvid}/")
        if not ok:
            return False, err
        return True, f"同步播放窗口已打开（{bvid}）喵，等预习完成后在这里看就行～"

    async def _async_open_login(self) -> tuple[bool, str]:
        if self._context is None:
            ok, err = await self._async_launch()
            if not ok:
                return False, err
        ok, err = await self._async_goto(_LOGIN_URL)
        if not ok:
            return False, err
        if await self._async_sessdata():
            return True, "检测到已经登录过喵（登录态已保存），回面板点「读取登录态」就行～"
        return True, "登录窗口已打开喵，请在窗口里扫码或账号登录B站，登录完回面板点「读取登录态」～"

    async def _async_position(self) -> Optional[float]:
        if self._page is None or self._page.is_closed():
            return None
        try:
            return await self._page.evaluate(
                "() => { const v = document.querySelector('video'); return v ? v.currentTime : null; }"
            )
        except Exception:
            return None

    async def _async_sessdata(self) -> str:
        """从持久化上下文里读出 SESSDATA（httpOnly 也拿得到）。"""
        if self._context is None:
            return ""
        try:
            cookies = await self._context.cookies("https://www.bilibili.com")
        except Exception:
            return ""
        for cookie in cookies or []:
            if not isinstance(cookie, dict):
                continue
            if str(cookie.get("name")) == "SESSDATA":
                return str(cookie.get("value") or "").strip()
        return ""

    async def _async_grab_frame(self) -> Optional[bytes]:
        """截取 <video> 元素画面（拿不到就退整页截图）。返回 PNG 字节或 None。"""
        if self._page is None or self._page.is_closed():
            return None
        try:
            element = await self._page.query_selector("video")
            if element is not None:
                return await element.screenshot(type="png")
        except Exception:
            pass
        try:
            return await self._page.screenshot(type="png")
        except Exception:
            return None

    async def _async_close(self) -> None:
        try:
            if self._context is not None:
                await self._context.close()
        except Exception:
            pass
        try:
            if self._browser is not None:
                await self._browser.close()
        except Exception:
            pass
        try:
            if self._pw is not None:
                await self._pw.stop()
        except Exception:
            pass
        self._page = None
        self._context = None
        self._browser = None
        self._pw = None

    # ── 公开 API（线程安全）────────────────────────────────────
    def open(self, bvid: str, timeout: float = 45.0) -> tuple[bool, str]:
        if not self._ensure_loop():
            return False, "播放窗口已关闭"
        try:
            return self._submit(self._async_open(bvid), timeout)
        except Exception as exc:
            return False, f"打开失败：{exc}"

    def open_login(self, timeout: float = 45.0) -> tuple[bool, str]:
        """打开B站登录窗口（持久化上下文，登录一次长期有效）。"""
        if not self._ensure_loop():
            return False, "播放窗口已关闭"
        try:
            return self._submit(self._async_open_login(), timeout)
        except Exception as exc:
            return False, f"打开登录页失败：{exc}"

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

    def sessdata(self, timeout: float = 5.0) -> str:
        """导出当前窗口的 SESSDATA（已登录才有值）。"""
        if self._closed.is_set() or self._loop is None or not self._loop.is_running():
            return ""
        try:
            return str(self._submit(self._async_sessdata(), timeout) or "").strip()
        except Exception:
            return ""

    def grab_frame(self, timeout: float = 6.0) -> Optional[bytes]:
        """截取播放窗口里的视频画面（PNG 字节），供视觉模型消费。"""
        if self._closed.is_set() or self._loop is None or not self._loop.is_running():
            return None
        try:
            return self._submit(self._async_grab_frame(), timeout)
        except Exception:
            return None

    def is_alive(self) -> bool:
        if self._closed.is_set():
            return False
        if self._page is not None and self._page.is_closed():
            self._closed.set()
            return False
        return (
            self._loop is not None
            and self._loop.is_running()
            and self._context is not None
        )

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
