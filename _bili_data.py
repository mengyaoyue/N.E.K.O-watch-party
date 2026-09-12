"""陪看猫娘 · B站数据层 v2（WBI 签名 + 字幕时间轴）

学自 BiliGPT 类项目的标准架构：
  x/player/wbi/v2（WBI 签名）→ 字幕列表（含 AI 字幕）→ 下载字幕 JSON → 时间轴台词

- 优先宿主自带的 bilibili_api（自动处理 WBI），失败回退纯标准库手搓（含 WBI 签名实现）
- 字幕是带时间轴的台词：陪看反应可以挂在"正在说的那句话"上
- 全部匿名可用；纯标准库
"""

from __future__ import annotations

import functools
import hashlib
import json
import time
import urllib.parse
import urllib.request
from typing import Any, Callable, Optional

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
)
_HEADERS = {
    "User-Agent": _UA,
    "Referer": "https://www.bilibili.com/",
    "Origin": "https://www.bilibili.com",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "zh-CN,zh;q=0.9",
}

# WBI 混淆表（bilibili-API-collect 公开常量）
_MIXIN_KEY_ENC_TAB = [
    46, 47, 18, 2, 53, 8, 23, 32, 15, 50, 10, 31, 58, 3, 45, 35, 27, 43, 5, 49,
    33, 9, 42, 19, 29, 28, 14, 39, 12, 38, 41, 13, 37, 48, 7, 16, 24, 55, 40,
    61, 26, 17, 0, 1, 60, 51, 30, 4, 22, 25, 54, 21, 56, 59, 6, 63, 57, 62, 11,
    36, 20, 34, 44, 52,
]


def _get_bytes(url: str, cookie: str = "", timeout: float = 15.0) -> tuple[int, bytes]:
    headers = dict(_HEADERS)
    if cookie:
        headers["Cookie"] = cookie
    request = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            return resp.status, resp.read()
    except Exception:
        return 0, b""


def get_json(url: str, cookie: str = "", timeout: float = 15.0) -> Optional[dict[str, Any]]:
    status, body = _get_bytes(url, cookie, timeout)
    if status != 200 or not body:
        return None
    try:
        data = json.loads(body.decode("utf-8", "replace"))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def get_anonymous_cookie() -> str:
    """匿名 buvid3（风控最低要求）。失败返回空串。"""
    data = get_json("https://api.bilibili.com/x/frontend/finger/spi")
    if data and data.get("code") == 0:
        b3 = str((data.get("data") or {}).get("b_3") or "").strip()
        if b3:
            return f"buvid3={b3}"
    return ""


# ── WBI 签名（bilibili-API-collect 公开算法，纯标准库）────────────

def _mixin_key(img_key: str, sub_key: str) -> str:
    raw = img_key + sub_key
    return "".join(raw[i] for i in _MIXIN_KEY_ENC_TAB if i < len(raw))[:32]


def sign_wbi(params: dict[str, Any], img_key: str, sub_key: str, wts: Optional[int] = None) -> dict[str, Any]:
    """对参数做 WBI 签名：返回带 wts/w_rid 的新参数字典。"""
    mixin = _mixin_key(img_key, sub_key)
    wts = int(time.time()) if wts is None else wts
    cleaned = {k: str(v) for k, v in params.items() if v is not None}
    cleaned["wts"] = wts
    query = urllib.parse.urlencode(sorted(cleaned.items()))
    cleaned["w_rid"] = hashlib.md5((query + mixin).encode("utf-8")).hexdigest()
    return cleaned


def get_wbi_keys(cookie: str = "", timeout: float = 15.0) -> tuple[str, str]:
    """从 nav 接口拿当前 wbi img/sub key。失败返回空串。"""
    data = get_json("https://api.bilibili.com/x/web-interface/nav", cookie, timeout)
    wbi = ((data or {}).get("data") or {}).get("wbi_img") or {}
    return str(wbi.get("img_url") or ""), str(wbi.get("sub_url") or "")


def _key_from_url(url: str) -> str:
    name = url.rsplit("/", 1)[-1]
    return name.split(".")[0]


# ── 字幕获取（链：bilibili_api → 手搓 WBI player 接口）────────────

def _extract_subtitle_list(player_info: dict[str, Any]) -> list[dict[str, Any]]:
    """从 player_info 里抽出字幕列表（每项含字幕正文下载地址/语言）。"""
    subs = ((player_info.get("subtitle") or {}).get("subtitles")) or []
    out: list[dict[str, Any]] = []
    for s in subs:
        if isinstance(s, dict) and (s.get("subtitle_url") or s.get("url")):
            out.append(
                {
                    "lan": str(s.get("lan_doc") or s.get("lan") or ""),
                    "url": str(s.get("subtitle_url") or s.get("url")).replace("http://", "https://"),
                    "ai": bool(s.get("ai_type")),
                }
            )
    return out


def parse_subtitle_json(data: Any) -> list[dict[str, Any]]:
    """解析字幕正文：B站字幕 JSON（body[].from/to/content）→ [{t, dur, text}]。"""
    items = data.get("body") if isinstance(data, dict) else data
    out: list[dict[str, Any]] = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        try:
            t = float(item.get("from", 0))
            dur = float(item.get("to", item.get("from", 0))) - t
        except (TypeError, ValueError):
            continue
        text = str(item.get("content") or "").strip()
        if text:
            out.append({"t": max(0.0, t), "dur": max(0.0, dur), "text": text})
    out.sort(key=lambda x: x["t"])
    return out


def download_subtitle_body(url: str, cookie: str = "", timeout: float = 15.0) -> list[dict[str, Any]]:
    """下载字幕正文并解析。失败返回空列表。"""
    status, body = _get_bytes(url, cookie, timeout)
    if status != 200 or not body:
        return []
    try:
        data = json.loads(body.decode("utf-8", "replace"))
    except json.JSONDecodeError:
        return []
    return parse_subtitle_json(data)


def fetch_subtitles(bvid: str, cid: int, cookie: str = "", timeout: float = 15.0) -> list[dict[str, Any]]:
    """拉取视频字幕（时间轴台词）。无字幕/失败返回空列表。

    优先宿主自带 bilibili_api（自动 WBI）；失败回退手搓签名实现。
    """
    # 链 1：宿主自带 bilibili_api
    try:
        import asyncio as _asyncio

        from bilibili_api import video as bili_video

        async def _via_sdk() -> list[dict[str, Any]]:
            v = bili_video.Video(bvid=bvid)
            info = await v.get_player_info(cid=cid)
            subs = _extract_subtitle_list(info)
            for s in subs:
                body = download_subtitle_body(s["url"], cookie, timeout)
                if body:
                    return body
            return []

        try:
            loop = _asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is None:
            return _asyncio.run(_via_sdk())
    except Exception:
        pass

    # 链 2：手搓 WBI 签名 player 接口
    try:
        img_key, sub_key = get_wbi_keys(cookie, timeout)
        if img_key and sub_key:
            signed = sign_wbi({"bvid": bvid, "cid": cid}, img_key, sub_key)
            query = urllib.parse.urlencode(signed)
            data = get_json(f"https://api.bilibili.com/x/player/wbi/v2?{query}", cookie, timeout)
            subs = _extract_subtitle_list(data or {})
            for s in subs:
                body = download_subtitle_body(s["url"], cookie, timeout)
                if body:
                    return body
    except Exception:
        pass
    return []


# ── 时间轴工具 ────────────────────────────────────────────────────

def subtitle_window(subs: list[dict[str, Any]], position: float, window: float = 12.0) -> list[dict[str, Any]]:
    """取播放位置附近的台词（默认 ±12 秒）。"""
    return [s for s in subs if abs(s["t"] - position) <= window]


def subtitles_to_text(
    subs: list[dict[str, Any]],
    max_lines: int = 160,
    sample_keep: float = 0.35,
) -> str:
    """把时间轴台词压成 LLM 可读文本：长视频按比例抽样，短字幕全保留。"""
    if not subs:
        return ""
    if len(subs) > max_lines:
        keep = max(20, int(len(subs) * sample_keep))
        step = len(subs) / keep
        picked = [subs[int(i * step)] for i in range(keep)]
    else:
        picked = subs
    lines = []
    for s in picked:
        minutes, seconds = divmod(int(s["t"]), 60)
        lines.append(f"[{minutes:02d}:{seconds:02d}] {s['text']}")
    return "\n".join(lines)


# ── 页面通道（IP 风控时的备用入口）────────────────────────────────
# API 接口被 412 时，用真浏览器打开视频页：
#   1. window.__INITIAL_STATE__.videoData 提取 标题/aid/cid/时长/UP主
#   2. 页面上下文里带凭证 fetch 弹幕接口（浏览器自动处理压缩与 cookie）

_PAGE_STATE_JS = """() => {
    const s = window.__INITIAL_STATE__ || {};
    const v = s.videoData || {};
    return {title: v.title || '', bvid: v.bvid || '', aid: v.aid || 0,
            cid: v.cid || 0, duration: v.duration || 0,
            up: (v.owner || {}).name || '', ok: !!v.title};
}"""

_PAGE_DANMAKU_JS = """async (cid) => {
    const r = await fetch(`https://api.bilibili.com/x/v1/dm/list.so?oid=${cid}`, {credentials: 'include'});
    const buf = new Uint8Array(await r.arrayBuffer());
    return Array.from(buf);
}"""


def _browser_channel_sync(bvid: str, timeout: float = 30.0) -> tuple[Optional[dict[str, Any]], list[dict[str, Any]], str]:
    """真浏览器走视频页拿信息+弹幕。返回 (info, danmaku, error)。"""
    from playwright.sync_api import sync_playwright

    info: dict[str, Any] = {}
    danmaku: list[dict[str, Any]] = []
    error = ""
    with sync_playwright() as p:
        browser = None
        for channel in (None, "msedge", "chrome"):
            try:
                browser = (
                    p.chromium.launch(headless=True, channel=channel)
                    if channel
                    else p.chromium.launch(headless=True)
                )
                break
            except Exception:
                browser = None
        if browser is None:
            return {}, [], "浏览器内核不可用（chromium/Edge/chrome 全失败）"
        page = browser.new_page(user_agent=_UA, locale="zh-CN")
        try:
            page.goto(
                f"https://www.bilibili.com/video/{bvid}/",
                timeout=int(timeout * 1000),
                wait_until="load",
            )
            page.wait_for_timeout(3000)
            for _ in range(3):
                try:
                    state = page.evaluate(_PAGE_STATE_JS)
                    if state and state.get("ok"):
                        info = {
                            "bvid": str(state.get("bvid") or bvid),
                            "aid": int(state.get("aid") or 0),
                            "title": str(state.get("title") or "未知标题"),
                            "desc": "",
                            "up": str(state.get("up") or "未知UP主"),
                            "duration": int(state.get("duration") or 0),
                            "cid": int(state.get("cid") or 0),
                        }
                        break
                except Exception:
                    page.wait_for_timeout(1500)
            if info.get("cid"):
                raw = page.evaluate(_PAGE_DANMAKU_JS, info["cid"])
                xml_bytes = bytes(raw or [])
                try:
                    from . import _watch_logic as _wl
                except ImportError:  # 独立加载（测试）
                    import importlib.util as _ilu

                    _spec = _ilu.spec_from_file_location(
                        "neko_watch_party_logic",
                        str(__import__("pathlib").Path(__file__).parent / "_watch_logic.py"),
                    )
                    _wl = _ilu.module_from_spec(_spec)
                    _spec.loader.exec_module(_wl)
                danmaku = _wl.parse_danmaku_xml(xml_bytes)
        except Exception as exc:
            error = str(exc)
        finally:
            browser.close()
    if not info and not error:
        error = "视频页加载了但拿不到 INITIAL_STATE"
    return info, danmaku, error


def fetch_via_page(bvid: str, timeout: float = 30.0) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    """对外入口：页面通道（真浏览器）。见 _browser_channel_sync。"""
    return _browser_channel_sync(bvid, timeout)
