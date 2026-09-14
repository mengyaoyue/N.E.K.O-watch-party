from __future__ import annotations
"""陪看猫娘（neko_watch_party）v0.6 · 作者：MENGYAOYUE

陪主人看视频：开关打开后，猫娘**高频自主截屏**，把画面交给视觉模型理解，
看到内容就主动评论、抓笑点与梗；看不到画面就安静等着。

开关有两种打开方式（相当于一个开关）：
1. 主人说「陪我看」；
2. 猫娘自己截屏探测到主人正在看视频（低频探测，面板可关）。

**本插件没有任何"播放进度/时间点"概念**：猫娘只凭截屏画面说话，
不报时间、不猜进度、不读进度条——从根上杜绝进度类幻觉。

预学习（抓弹幕/字幕/热评存本地时间轴库）是**可选辅助**，默认不跑、主人想跑才跑；
素材只用于帮猫娘看懂画面里看不清的梗，绝不会变成"当前进度"喂给模型。

数据全部匿名读取公开接口（UA + buvid3），零第三方依赖；评论接口失败自动降级。
"""

from pathlib import Path

import asyncio
import json
import random
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Optional

from plugin.sdk.plugin import (
    Err,
    NekoPluginBase,
    Ok,
    SdkError,
    lifecycle,
    llm_tool,
    neko_plugin,
    plugin_entry,
)

from ._panel import PanelServer, find_open_port
from ._player_window import PlayerWindow
from ._bili_data import fetch_subtitles, fetch_via_page
from ._screen import (
    PLAYING_PROBE_PROMPT,
    capture_screen_text,
    describe_frame,
    describe_screen,
)
from ._timeline_db import TimelineDB
from ._watch_logic import (
    build_material_overview,
    build_screen_reaction_prompt,
    build_summary,
    format_reaction,
    format_video_intro,
    looks_like_video_ui,
    parse_danmaku_xml,
    parse_playing_answer,
    parse_screen_reaction,
    parse_video_id,
)

_PLUGIN_ID = "neko_watch_party"
_VERSION = "0.7.0"

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
)


def _safe_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        low = value.strip().lower()
        if low in {"1", "true", "yes", "on", "开"}:
            return True
        if low in {"0", "false", "no", "off", "关"}:
            return False
    return default

_LAST_BILI_CALL = [0.0]

def _bili_throttle(min_gap: float = 1.2) -> None:
    import time as _t
    gap = _t.time() - _LAST_BILI_CALL[0]
    if gap < min_gap:
        _t.sleep(min_gap - gap)
    _LAST_BILI_CALL[0] = _t.time()

def _safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _safe_int(value: Any, default: int) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value.strip()))
        except ValueError:
            return default
    return default


def _clamp_float(value: Any, low: float, high: float, default: float) -> float:
    """把配置值收敛到 [low, high]；非法值回退 default。"""
    if isinstance(value, bool) or value is None:
        return default
    try:
        num = float(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, num))


# 思考模式作用范围：none=都保留思考；chat=只关对话思考；vision=只关视觉思考；all=两条都关（最快）
_THINKING_MODES = ("none", "chat", "vision", "all")


def _safe_thinking_mode(value: Any, default: str = "all") -> str:
    """把配置值收敛到合法思考模式；非法值回退 default。"""
    text = _safe_str(value).lower()
    return text if text in _THINKING_MODES else default


def _http_get(url: str, cookie: str = "", timeout: float = 15.0) -> tuple[int, bytes]:
    """标准库 GET：返回 (状态码, 响应字节)；网络错误返回 (0, b"")。"""
    headers = {"User-Agent": _USER_AGENT, "Referer": "https://www.bilibili.com/"}
    if cookie:
        headers["Cookie"] = cookie
    request = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, b""
    except Exception:
        return 0, b""


def _http_get_json(url: str, cookie: str = "", timeout: float = 15.0) -> Optional[dict[str, Any]]:
    _bili_throttle()
    status, body = _http_get(url, cookie, timeout)
    if status != 200 or not body:
        return None
    try:
        data = json.loads(body.decode("utf-8", errors="replace"))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


async def _call_llm(system: str, user: str, timeout: float, disable_thinking: bool = False) -> str:
    """调用 N.E.K.O 配置的对话模型（与 neko_natural_command 同一套取数方式）。

    disable_thinking=True 时在请求体顶层加 `thinking:{type:disabled}`（DeepSeek 官方字段），
    跳过"先思考一大段再回答"，能明显降低出话延迟。
    """
    from utils.config_manager import get_config_manager

    cfg = get_config_manager().get_model_api_config("conversation")
    model = _safe_str(cfg.get("model"))
    base_url = _safe_str(cfg.get("base_url")).rstrip("/")
    api_key = _safe_str(cfg.get("api_key"))
    if not model or not base_url or not api_key:
        raise SdkError("N.E.K.O 尚未配置对话模型，猫娘没法预习视频喵。")

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.6,
    }
    if "chat/completions" not in base_url:
        base_url = base_url + "/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    def _make_request(data: dict[str, Any]) -> urllib.request.Request:
        return urllib.request.Request(
            base_url,
            data=json.dumps(data).encode("utf-8"),
            headers=headers,
            method="POST",
        )

    def _post(req: urllib.request.Request) -> tuple[int, str]:
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.status, resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            return exc.code, ""
        except Exception:
            return 0, ""

    if disable_thinking:
        fast_request = _make_request(dict(payload, thinking={"type": "disabled"}))
        status, text = await asyncio.to_thread(_post, fast_request)
        if status != 200:
            # 不是所有模型都认 thinking 字段：退回原始请求重试，避免"提速"反而发不出话
            status, text = await asyncio.to_thread(_post, _make_request(payload))
    else:
        status, text = await asyncio.to_thread(_post, _make_request(payload))
    if status != 200:
        raise SdkError(f"模型接口返回 {status}，预习失败了喵。")
    try:
        data = json.loads(text)
        return data["choices"][0]["message"]["content"]
    except (json.JSONDecodeError, KeyError, IndexError):
        raise SdkError("模型返回格式不对，预习失败了喵。")


@neko_plugin


class WatchPartyPlugin(NekoPluginBase):
    """陪看猫娘：一个开关 + 可配置的截屏节奏，看到画面才主动评论、抓笑点。

    - 开关打开（主人说「陪我看」或截屏探测到在看视频）→ 按 shot_interval_active 高频截屏评论；
    - 开关关闭且开着 auto_detect → 按 shot_interval_idle 低频探测要不要自动打开；
    - **全流程没有任何播放进度/时间点**，素材只做整片级背景参考。
    """

    def __init__(self, ctx):
        super().__init__(ctx)
        self.file_logger = self.enable_file_logging(log_level="INFO")
        self.logger = self.file_logger

        self.screen_assist: bool = True
        self.screen_vision: bool = True
        self.auto_detect: bool = True
        self.prelearn: bool = False
        self.shot_interval_active: float = 20.0
        self.shot_interval_idle: float = 120.0
        # 思考模式作用范围：none=都保留；chat=只关对话；vision=只关视觉；all=都关（出话最快）
        self.thinking_mode: str = "all"
        self.bili_sessdata: str = ""
        self.vision_timeout: float = 25.0
        self.vision_model: str = ""
        self.vision_base_url: str = ""
        self.vision_api_key: str = ""
        self.vision_provider_type: str = ""
        self.request_timeout: float = 15.0
        self.llm_timeout: float = 45.0
        self.poll_interval: float = 2.0
        self.auto_summary: bool = True
        self.catgirl_name: str = "猫娘"
        self.master_name: str = "主人"
        self._config_loaded = False

        # 陪看开关（内存态；重启即为关闭）
        self._lock = threading.Lock()
        self._watching: bool = False
        # 可选预学习素材（关闭预学习时为 None；只做整片级背景，不含任何时间点）
        self._session: Optional[dict[str, Any]] = None
        # 开关打开后本喵说过的话（用于去重 + 结束总结）
        self._said: list[dict[str, Any]] = []
        self._recent: list[str] = []
        self._last_comment_ts: float = 0.0
        # 待机探测：距上次"是否在看视频"探测的时间戳
        self._last_probe_ts: float = 0.0
        self._stop_event = threading.Event()
        self._wake_event = threading.Event()
        self._tick_thread: Optional[threading.Thread] = None
        self._panel_server = None
        self._panel_port: int = 15690
        self._panel_loop: Optional[asyncio.AbstractEventLoop] = None
        self._pwindow: Optional[PlayerWindow] = None
        self._panel_loop_thread: Optional[threading.Thread] = None
        # 本地时间轴库：仅在开着预学习时写入素材，供以后复用理解梗
        self._timeline: Optional[TimelineDB] = None

    # ── 配置 ───────────────────────────────────────────────────
    async def _load_config(self) -> None:
        try:
            cfg = await self.config.dump(timeout=5.0)
        except Exception as exc:
            self.logger.warning("[watch_party] 读取配置失败：{}", exc)
            cfg = {}
        section = cfg.get(_PLUGIN_ID) if isinstance(cfg, dict) else None
        section = section if isinstance(section, dict) else {}
        self.request_timeout = max(5.0, float(_safe_int(section.get("request_timeout"), 15)))
        self.llm_timeout = max(10.0, float(_safe_int(section.get("llm_timeout"), 45)))
        self.poll_interval = max(1.0, float(_safe_int(section.get("poll_interval"), 2)))
        self.auto_summary = bool(section.get("auto_summary", True))
        self.bili_sessdata = _safe_str(section.get("bili_sessdata"))
        self.screen_assist = _safe_bool(section.get("screen_assist"), True)
        # 截屏节奏：激活（在看视频）高频、待机低频探测，均可面板自定义
        self.shot_interval_active = _clamp_float(section.get("shot_interval_active"), 5.0, 600.0, 20.0)
        self.shot_interval_idle = _clamp_float(section.get("shot_interval_idle"), 10.0, 3600.0, 120.0)
        self.auto_detect = _safe_bool(section.get("auto_detect"), True)
        self.prelearn = _safe_bool(section.get("prelearn"), False)
        # 视觉看片：优先用多模态模型描述画面，失败再降级本地 OCR
        self.screen_vision = _safe_bool(section.get("screen_vision"), True)
        # 思考模式：关掉能省掉"先想一大段"的时间，明显降低出话延迟
        self.thinking_mode = _safe_thinking_mode(section.get("thinking_mode"), "all")
        self.vision_timeout = max(5.0, float(_safe_int(section.get("vision_timeout"), 25)))
        self.vision_model = _safe_str(section.get("vision_model"))
        self.vision_base_url = _safe_str(section.get("vision_base_url")).rstrip("/")
        self.vision_api_key = _safe_str(section.get("vision_api_key"))
        self.vision_provider_type = _safe_str(section.get("vision_provider_type"))
        self.catgirl_name = _safe_str(section.get("catgirl_name"), "猫娘") or "猫娘"
        self.master_name = _safe_str(section.get("master_name"), "主人") or "主人"
        self._config_loaded = True

    async def _ensure_config_loaded(self) -> None:
        if not self._config_loaded:
            await self._load_config()

    # ── 生命周期 ───────────────────────────────────────────────
    @lifecycle(id="startup")
    async def startup(self, **_):
        await self._load_config()
        self._stop_event.clear()
        self._wake_event.clear()
        self._tick_thread = threading.Thread(
            target=self._tick_loop, daemon=True, name="neko-watch-party-tick"
        )
        self._tick_thread.start()
        self._start_panel_loop()
        saved: dict[str, Any] = {}
        try:
            state_path = Path(self.data_path()) / "panel_state.json"
            saved = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
        except Exception:
            saved = {}
        if not self.bili_sessdata:
            self.bili_sessdata = str(saved.get("bili_sessdata") or "")
        self._apply_saved_panel_state(saved)
        self._start_panel()
        self.logger.info(
            "[watch_party] 启动：截屏辅助 {}｜开关频率 {:.0f}s｜待机探测 {}s（{}）｜预学习 {}",
            self.screen_assist,
            self.shot_interval_active,
            self.shot_interval_idle,
            "开" if self.auto_detect else "关",
            "开" if self.prelearn else "关",
        )
        return Ok({"status": "running", "version": _VERSION})

    def _apply_saved_panel_state(self, saved: dict[str, Any]) -> None:
        """面板改过的节奏/开关落盘后，重启时覆盖配置默认值。"""
        if not isinstance(saved, dict):
            return
        if saved.get("shot_interval_active") is not None:
            self.shot_interval_active = _clamp_float(
                saved.get("shot_interval_active"), 5.0, 600.0, self.shot_interval_active
            )
        if saved.get("shot_interval_idle") is not None:
            self.shot_interval_idle = _clamp_float(
                saved.get("shot_interval_idle"), 10.0, 3600.0, self.shot_interval_idle
            )
        if "auto_detect" in saved:
            self.auto_detect = _safe_bool(saved.get("auto_detect"), self.auto_detect)
        if "prelearn" in saved:
            self.prelearn = _safe_bool(saved.get("prelearn"), self.prelearn)
        if "screen_assist" in saved:
            self.screen_assist = _safe_bool(saved.get("screen_assist"), self.screen_assist)
        if "screen_vision" in saved:
            self.screen_vision = _safe_bool(saved.get("screen_vision"), self.screen_vision)
        if "thinking_mode" in saved:
            self.thinking_mode = _safe_thinking_mode(saved.get("thinking_mode"), self.thinking_mode)

    def _thinking_disabled(self, chain: str) -> bool:
        """该链路要不要关掉思考：chain 取 "chat"（出话）或 "vision"（看画面）。"""
        return self.thinking_mode == "all" or self.thinking_mode == chain

    @lifecycle(id="shutdown")
    def shutdown(self, **_):
        if self._panel_loop is not None and self._panel_loop.is_running():
            self._panel_loop.call_soon_threadsafe(self._panel_loop.stop)
        if self._panel_loop_thread and self._panel_loop_thread.is_alive():
            self._panel_loop_thread.join(timeout=2.0)
        if self._panel_loop is not None:
            self._panel_loop.close()
        if self._pwindow is not None:
            self._pwindow.close()
        if self._panel_server:
            self._panel_server.stop()
        self._stop_event.set()
        self._wake_event.set()
        if self._tick_thread and self._tick_thread.is_alive():
            self._tick_thread.join(timeout=3.0)
        if self._timeline is not None:
            self._timeline.close()
            self._timeline = None
        self.logger.info("[watch_party] 关闭")
        return Ok("stopped")

    # ── B 站数据抓取（匿名） ───────────────────────────────────
    async def _get_cookie(self) -> str:
        """匿名获取 buvid3（B 站风控要求的最低限度 cookie）。"""
        try:
            data = await asyncio.to_thread(
                _http_get_json, "https://api.bilibili.com/x/frontend/finger/spi", "", self.request_timeout
            )
        except Exception:
            data = None
        if data and data.get("code") == 0:
            b3 = _safe_str((data.get("data") or {}).get("b_3"))
            if b3:
                return f"buvid3={b3}"
        return ""

    async def _fetch_video(self, video_id: dict[str, Any]) -> dict[str, Any]:
        cookie = await self._get_cookie()
        if "bvid" in video_id:
            url = f"https://api.bilibili.com/x/web-interface/view?bvid={video_id['bvid']}"
        else:
            url = f"https://api.bilibili.com/x/web-interface/view?aid={video_id['aid']}"
        data = await asyncio.to_thread(_http_get_json, url, cookie, self.request_timeout)
        if not data or data.get("code") != 0:
            # API 被风控（412 等）→ 页面通道：真浏览器开视频页提取
            self.logger.warning("[watch_party] 视频接口失败(code={}), 切换页面通道", (data or {}).get("code"))
            info, _dm, err = await asyncio.to_thread(fetch_via_page, video_id.get("bvid", ""), self.request_timeout)
            if info:
                self.logger.info("[watch_party] 页面通道成功：{}", info.get("title", "")[:30])
                return info
            raise SdkError(
                f"呜…视频信息没拿到喵（B站接口异常；页面通道: {err or '无数据'}）。"
                "大概率是B站对本机临时风控，歇几分钟再试就好。"
            )
        v = data.get("data") or {}
        page = max(1, _safe_int(video_id.get("page"), 1))
        pages = v.get("pages") or []
        page_info = pages[page - 1] if isinstance(pages, list) and len(pages) >= page else {}
        cid = _safe_int(page_info.get("cid") or v.get("cid"), 0)
        duration = _safe_int(page_info.get("duration") or v.get("duration"), 0)
        if not cid or not duration:
            raise SdkError("呜…这个视频拿不到分P信息喵，本喵陪不了它。")
        return {
            "bvid": _safe_str(v.get("bvid")),
            "aid": _safe_int(v.get("aid"), 0),
            "cid": cid,
            "page": page,
            "title": _safe_str(v.get("title"), "未知标题"),
            "desc": _safe_str(v.get("desc")),
            "up": _safe_str((v.get("owner") or {}).get("name"), "未知UP主"),
            "duration": duration,
            "view": _safe_int((v.get("stat") or {}).get("view"), 0),
        }

    async def _fetch_danmaku(self, cid: int, bvid: str = "") -> list[dict[str, Any]]:
        url = f"https://api.bilibili.com/x/v1/dm/list.so?oid={cid}"
        status, body = await asyncio.to_thread(_http_get, url, "", self.request_timeout)
        if status == 200 and body:
            return parse_danmaku_xml(body)
        self.logger.warning("[watch_party] 弹幕接口失败 status={}, 尝试页面通道", status)
        if bvid:
            try:
                _info, danmaku, err = await asyncio.to_thread(fetch_via_page, bvid, self.request_timeout)
                if danmaku:
                    self.logger.info("[watch_party] 页面通道弹幕 {} 条", len(danmaku))
                    return danmaku
                self.logger.warning("[watch_party] 页面通道弹幕失败: {}", err)
            except Exception as exc:
                self.logger.warning("[watch_party] 页面通道弹幕异常: {}", exc)
        return []

    async def _fetch_comments(self, aid: int) -> list[dict[str, Any]]:
        """热评抓取（可降级）：返回 [{user, like, text}]，失败返回空列表。"""
        url = f"https://api.bilibili.com/x/v2/reply/main?type=1&oid={aid}&mode=3"
        try:
            data = await asyncio.to_thread(_http_get_json, url, "", self.request_timeout)
        except Exception:
            return []
        if not data or data.get("code") != 0:
            return []
        replies = (data.get("data") or {}).get("replies") or []
        out: list[dict[str, Any]] = []
        for reply in replies[:20]:
            if not isinstance(reply, dict):
                continue
            msg = _safe_str((reply.get("content") or {}).get("message"))
            if not msg:
                continue
            like = int(reply.get("like") or 0)
            user = _safe_str((reply.get("member") or {}).get("uname"), "匿名")
            out.append({"user": user, "like": like, "text": msg})
        return out

    # ── 预学习（可选辅助）：素材入库，不预写脚本、不含时间点 ──────
    async def _prepare_session(self, video_text: str) -> dict[str, Any]:
        """解析视频 + 可选预学习。**预学习是可选的**：关着就只认标题，不抓素材。"""
        await self._ensure_config_loaded()
        video_id = parse_video_id(video_text)
        if not video_id:
            raise SdkError(
                "没认出这是哪个视频喵…发我B站链接或 BV 号（如 BV1xx411c7mD）就好。"
            )
        t0 = time.time()
        video = await self._fetch_video(video_id)
        self.logger.info(
            "[watch_party] 视频信息 OK：bvid={} 标题={}（{:.1f}s）",
            video.get("bvid"), str(video.get("title") or "")[:30], time.time() - t0,
        )

        danmaku: list[dict[str, Any]] = []
        subtitles: list[dict[str, Any]] = []
        comments: list[dict[str, Any]] = []
        if self.prelearn:
            danmaku = await self._fetch_danmaku(video["cid"], video.get("bvid", ""))
            self.logger.info("[watch_party] 预学习①弹幕 {} 条", len(danmaku))
            await asyncio.sleep(0.3)
            sessdata = self.bili_sessdata or await asyncio.to_thread(self._sessdata_from_window)
            cookie = f"SESSDATA={sessdata}" if sessdata else ""
            if video.get("bvid") and video.get("cid"):
                try:
                    subtitles = await asyncio.to_thread(
                        fetch_subtitles, video["bvid"], video["cid"], cookie, self.request_timeout
                    )
                except Exception as exc:
                    self.logger.warning("[watch_party] 字幕获取失败: {}", exc)
            self.logger.info("[watch_party] 预学习②字幕 {} 条", len(subtitles))
            comments = await self._fetch_comments(video["aid"]) if video.get("aid") else []
            self.logger.info("[watch_party] 预学习③热评 {} 条", len(comments))
            try:
                saved = await asyncio.to_thread(
                    self._ensure_timeline().save_timeline, video, danmaku, subtitles, comments
                )
                self.logger.info(
                    "[watch_party] 预学习素材入库 弹幕{} 字幕{} 热评{}（共 {:.1f}s）",
                    saved.get("danmaku"), saved.get("subtitles"), saved.get("comments"), time.time() - t0,
                )
            except Exception as exc:
                self.logger.warning("[watch_party] 时间轴库写入失败（不影响陪看）：{}", exc)
        else:
            self.logger.info("[watch_party] 预学习关闭，仅记住标题，不抓任何素材")

        return {
            "video": video,
            "danmaku": danmaku,
            "subtitles": subtitles,
            "comments": comments,
            # 整片级背景参考（高频弹幕梗 + 热评），**不含任何时间点**
            "material_ctx": build_material_overview(danmaku, comments),
        }

    def _ensure_timeline(self) -> TimelineDB:
        """懒加载本地时间轴库（仅在开着预学习时才写入，供以后复用）。"""
        if self._timeline is None:
            db_path = Path(self.data_path()) / "timeline.db"
            self._timeline = TimelineDB(str(db_path))
            self.logger.info("[watch_party] 时间轴库就绪：{}", db_path)
        return self._timeline

    def _material_context(self) -> str:
        """当前可用素材背景（整片级、无时间点）；没做预学习就返回空串。"""
        session = self._session
        if not session:
            return ""
        return str(session.get("material_ctx") or "")

    # ── 调度：开关 + 双档位截屏节奏（全程无进度概念）─────────────
    def _tick_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._tick_once()
            except Exception:
                self.logger.exception("[watch_party] tick 异常")
            self._wake_event.clear()
            if self._stop_event.is_set():
                break
            # 睡到"下一件该做的事"可能发生的时间点，保证频率自定义立即生效
            self._wake_event.wait(timeout=self._next_wait())
        self.logger.info("[watch_party] 调度线程退出")

    def _next_wait(self) -> float:
        """按当前开关状态算出下一次醒来前该睡多久（秒）。"""
        if self._watching:
            interval = float(self.shot_interval_active)
            elapsed = time.time() - self._last_comment_ts
        else:
            interval = float(self.shot_interval_idle)
            elapsed = time.time() - self._last_probe_ts
        return max(1.0, min(30.0, interval - elapsed))

    def _tick_once(self) -> None:
        # 开关打开：高频截屏，看到画面就自主评论
        if self._watching:
            if time.time() - self._last_comment_ts < float(self.shot_interval_active):
                return
            self._comment_on_screen()
            return

        # 开关关闭：低频探测要不要自动打开（面板可关掉 auto_detect）
        if not (self.auto_detect and self.screen_assist):
            return
        if time.time() - self._last_probe_ts < float(self.shot_interval_idle):
            return
        self._last_probe_ts = time.time()
        if self._probe_playing():
            self._turn_on_watch(auto=True)

    def _probe_playing(self) -> bool:
        """探测主人现在是否在看视频：视觉模型探针优先，OCR 关键词兜底。"""
        seen = self._describe_sync(frame=self._grab_player_frame(), prompt=PLAYING_PROBE_PROMPT)
        if seen.get("ok") and seen.get("text"):
            text = str(seen.get("text") or "")
            if seen.get("source") == "ocr":
                return looks_like_video_ui(text)
            return parse_playing_answer(text)
        try:
            shot = capture_screen_text()
        except Exception as exc:
            self.logger.info("[watch_party] 待机探测失败：{}", exc)
            return False
        if shot.get("ok"):
            return looks_like_video_ui(str(shot.get("text") or ""))
        return False

    def _turn_on_watch(self, auto: bool = False) -> None:
        """打开陪看开关：清空本轮话术、立刻来一发。"""
        if self._watching:
            return
        self._watching = True
        self._said = []
        self._recent = []
        self._last_comment_ts = 0.0
        self._wake_event.set()
        if auto:
            self._push(f"咦，本喵截屏瞄到{self.master_name}在看视频喵～自己凑过来一起看啦！")
        self.logger.info("[watch_party] 陪看开关打开（{}）", "自动探测" if auto else "主人开启")

    def _turn_off_watch(self) -> None:
        if not self._watching:
            return
        self._watching = False
        self._last_probe_ts = time.time()
        self._wake_event.set()
        self.logger.info("[watch_party] 陪看开关关闭")

    def _comment_on_screen(self) -> None:
        """截一帧主人正在看的画面，看到内容才让猫娘**主动**说一句；看不到就安静。

        这是陪看**唯一**的发言路径：没有预习稿、不按时间轴念稿、不提任何时间点。
        可选预学习素材只作为"整片级背景"帮猫娘认出画面里看不清的梗与笑点。
        """
        if not self.screen_assist:
            self._last_comment_ts = time.time()
            return

        seen = self._describe_sync(frame=self._grab_player_frame())
        self._last_comment_ts = time.time()
        if not (seen.get("ok") and seen.get("text")):
            self.logger.info("[watch_party] 这一轮没看到画面（{}），保持安静", seen.get("error"))
            return

        session = self._session
        video = (session or {}).get("video") or {}
        title = str(video.get("title") or "") or f"{self.master_name}正在看的视频"
        prompt = build_screen_reaction_prompt(
            title=title,
            screen_text=str(seen.get("text") or ""),
            timeline_ctx=self._material_context(),
            catgirl_name=self.catgirl_name,
            master_name=self.master_name,
            recent=list(self._recent),
            source=str(seen.get("source") or "vision"),
        )
        try:
            raw = asyncio.run(
                _call_llm(
                    "你是陪看的猫娘，只对看到的画面做真实反应，看不到就别硬说，别报时间进度。",
                    prompt, self.llm_timeout,
                    disable_thinking=self._thinking_disabled("chat"),
                )
            )
        except SdkError as exc:
            self.logger.warning("[watch_party] 看屏反应生成失败：{}", exc)
            return
        except Exception as exc:
            self.logger.exception("[watch_party] 看屏反应异常：{}", exc)
            return

        reaction = parse_screen_reaction(raw)
        if not reaction:
            self.logger.info("[watch_party] 模型没给出可用反应，本轮跳过")
            return

        self._said.append(reaction)
        self._recent.append(reaction["text"])
        del self._recent[:-5]
        seen_text = str(seen.get("text") or "").replace(chr(10), " ")[:40]
        tag = "看" if seen.get("source") == "vision" else "瞄"
        self._push(f"{format_reaction(reaction)}（本喵{tag}了一眼，画面上是「{seen_text}」）")

    def _push(self, text: str) -> None:
        try:
            self.ctx.push_message(
                source=_PLUGIN_ID,
                visibility=[],
                ai_behavior="respond",
                parts=[{"type": "text", "text": text}],
                priority=3,
                metadata={"description": "🐱 陪看猫娘"},
            )
        except Exception:
            self.logger.exception("[watch_party] push_message 失败")

    # ── 看片：多模态视觉优先，本地 OCR 兜底 ─────────────────────
    def _vision_override(self) -> dict[str, Any]:
        """插件级视觉模型覆盖；四项留空则回退宿主 vision / conversation 通道。"""
        return {
            "vision_model": self.vision_model,
            "vision_base_url": self.vision_base_url,
            "vision_api_key": self.vision_api_key,
            "vision_provider_type": self.vision_provider_type,
        }

    def _grab_player_frame(self) -> Any:
        """优先从同步播放窗口抽帧（画面干净、无桌面隐私）；没有窗口返回 None 走主屏。"""
        window = self._pwindow
        if window is None or not window.is_alive():
            return None
        try:
            return window.grab_frame(timeout=4.0)
        except Exception:
            return None

    async def _describe_screen(self, frame: Any = None, prompt: str = "") -> dict[str, Any]:
        """让猫娘"看"一帧：视觉模型优先，失败降级本地 OCR。

        帧来源（两者都要）：优先传入的播放窗口抽帧，否则抓主屏。
        返回 {"ok","text","source","error"}，source 为 vision/ocr。
        """
        last_err = ""
        if self.screen_vision:
            try:
                if frame is not None:
                    ok, text = await describe_frame(
                        frame, prompt=prompt, override=self._vision_override(),
                        timeout=self.vision_timeout,
                        disable_thinking=self._thinking_disabled("vision"),
                    )
                else:
                    res = await describe_screen(
                        prompt=prompt, override=self._vision_override(),
                        timeout=self.vision_timeout,
                        disable_thinking=self._thinking_disabled("vision"),
                    )
                    ok = bool(res.get("ok"))
                    text = str(res.get("text") or res.get("error") or "")
                if ok and text:
                    return {"ok": True, "text": text.strip(), "source": "vision", "error": ""}
                last_err = text
            except Exception as exc:
                last_err = str(exc)
        try:
            shot = await asyncio.to_thread(capture_screen_text)
        except Exception as exc:
            return {"ok": False, "text": "", "source": "ocr", "error": last_err or str(exc)}
        if shot.get("ok") and shot.get("text"):
            return {"ok": True, "text": shot["text"], "source": "ocr", "error": ""}
        return {"ok": False, "text": "", "source": "ocr", "error": last_err or str(shot.get("error") or "")}

    def _describe_sync(self, frame: Any = None, prompt: str = "") -> dict[str, Any]:
        """同步上下文（调度线程）里跑视觉描述。"""
        try:
            return asyncio.run(self._describe_screen(frame=frame, prompt=prompt))
        except Exception as exc:
            return {"ok": False, "text": "", "source": "vision", "error": str(exc)}

    # ── B站登录态（窗口导出 / 手工粘贴共用一套落盘）────────────
    def _browser_profile_dir(self) -> str:
        """持久化浏览器用户目录：登录一次，之后长期复用。"""
        path = Path(self.data_path()) / "browser_profile"
        try:
            path.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        return str(path)

    def _persist_sessdata(self, value: str) -> None:
        state_path = Path(self.data_path()) / "panel_state.json"
        try:
            state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
        except Exception:
            state = {}
        if value:
            state["bili_sessdata"] = value
        else:
            state.pop("bili_sessdata", None)
        try:
            state_path.parent.mkdir(parents=True, exist_ok=True)
            state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _sessdata_from_window(self) -> str:
        """从同步播放窗口导出登录态（httpOnly 也拿得到），拿到就落盘复用。"""
        window = self._pwindow
        if window is None or not window.is_alive():
            return ""
        try:
            value = window.sessdata(timeout=4.0)
        except Exception:
            return ""
        if value:
            self.bili_sessdata = value
            self._persist_sessdata(value)
            self.logger.info("[watch_party] 已从播放窗口导出登录态（字幕模式可用）")
        return value

    # ── 管理面板 ───────────────────────────────────────────────
    def _panel_status(self, _body: dict[str, Any]) -> dict[str, Any]:
        """面板状态：**只暴露开关与截屏节奏**，没有任何播放进度。"""
        session = self._session
        video = (session or {}).get("video") or {}
        return {
            "watching": self._watching,
            "prepared": bool(session),
            "title": video.get("title", ""),
            "bvid": video.get("bvid", ""),
            "said": len(self._said),
            "shot_interval_active": self.shot_interval_active,
            "shot_interval_idle": self.shot_interval_idle,
            "auto_detect": self.auto_detect,
            "prelearn": self.prelearn,
            "screen_assist": self.screen_assist,
            "screen_vision": self.screen_vision,
            "thinking_mode": self.thinking_mode,
            "vision_model": self.vision_model or "宿主 vision 通道",
            "sessdata_set": bool(self.bili_sessdata),
            "player_alive": bool(self._pwindow is not None and self._pwindow.is_alive()),
        }

    def _panel_stop(self, _body: dict[str, Any]) -> dict[str, Any]:
        self._turn_off_watch()
        session = self._session
        if not session and not self._said:
            return {"ok": True, "message": "本来就没在看喵"}
        summary = build_summary(
            (session or {}).get("video") or {},
            (session or {}).get("danmaku") or [],
            self._said,
            [str(c.get("text") or "") for c in ((session or {}).get("comments") or [])],
        )
        self._said = []
        self._push("好呀，先停在这里喵。" + chr(10) + summary)
        return {"ok": True}

    def _panel_config(self, body: dict[str, Any]) -> dict[str, Any]:
        """面板自定义截屏频率与开关；只改传进来的字段，落盘 panel_state.json。"""
        changed: dict[str, Any] = {}
        if body.get("shot_interval_active") is not None:
            self.shot_interval_active = _clamp_float(
                body.get("shot_interval_active"), 5.0, 600.0, self.shot_interval_active
            )
            changed["shot_interval_active"] = self.shot_interval_active
        if body.get("shot_interval_idle") is not None:
            self.shot_interval_idle = _clamp_float(
                body.get("shot_interval_idle"), 10.0, 3600.0, self.shot_interval_idle
            )
            changed["shot_interval_idle"] = self.shot_interval_idle
        for key in ("auto_detect", "prelearn", "screen_assist", "screen_vision"):
            if key in body:
                value = _safe_bool(body.get(key), bool(getattr(self, key)))
                setattr(self, key, value)
                changed[key] = value
        if "thinking_mode" in body:
            self.thinking_mode = _safe_thinking_mode(body.get("thinking_mode"), self.thinking_mode)
            changed["thinking_mode"] = self.thinking_mode
        if changed:
            state_path = Path(self.data_path()) / "panel_state.json"
            try:
                state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
                if not isinstance(state, dict):
                    state = {}
                state.update(changed)
                state_path.parent.mkdir(parents=True, exist_ok=True)
                state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                pass
            self._wake_event.set()
        return {"ok": True, **changed}

    def _panel_save_sessdata(self, body: dict[str, Any]) -> dict[str, Any]:
        value = str(body.get("sessdata") or "").strip()
        self.bili_sessdata = value
        self._persist_sessdata(value)
        return {"ok": True, "sessdata_set": bool(value)}

    def _panel_open_login(self, _body: dict[str, Any]) -> dict[str, Any]:
        """打开B站登录窗口（持久化上下文）：登录一次，之后自动复用。"""
        if self._pwindow is None:
            self._pwindow = PlayerWindow(self._browser_profile_dir())
        ok, msg = self._pwindow.open_login(timeout=45)
        return {"ok": ok, "message": msg, "error": None if ok else msg}

    def _panel_read_sessdata(self, _body: dict[str, Any]) -> dict[str, Any]:
        """从登录窗口导出 SESSDATA 并落盘，喂给字幕接口。"""
        if self._pwindow is None or not self._pwindow.is_alive():
            return {"ok": False, "error": "登录窗口还没开喵，先点「登录 B 站」"}
        try:
            value = self._pwindow.sessdata(timeout=6.0)
        except Exception as exc:
            return {"ok": False, "error": f"读取登录态失败：{exc}"}
        if not value:
            return {"ok": False, "error": "窗口里还没读到登录态喵，先在窗口里把B站登录成功再试"}
        self.bili_sessdata = value
        self._persist_sessdata(value)
        return {"ok": True, "message": "已经拿到登录态啦喵，字幕功能可以用了！", "sessdata_set": True}

    def _start_panel_loop(self) -> None:
        """面板专用的常驻事件循环（独立线程）。

        宿主对插件入口是"每次触发临时开循环"的模式，startup 里捕获的循环用完即关；
        面板线程要跑异步入口必须有自己的常驻循环。
        """
        self._panel_loop = asyncio.new_event_loop()
        self._panel_loop_thread = threading.Thread(
            target=self._run_panel_loop, daemon=True, name="neko-watch-panel-loop"
        )
        self._panel_loop_thread.start()

    def _run_panel_loop(self) -> None:
        asyncio.set_event_loop(self._panel_loop)
        self._panel_loop.run_forever()

    def _run_async(self, coro, timeout: float):
        if self._panel_loop is None or self._panel_loop.is_closed():
            raise RuntimeError("面板事件循环不可用喵")
        future = asyncio.run_coroutine_threadsafe(coro, self._panel_loop)
        return future.result(timeout=timeout)

    def _panel_start_watch(self, body: dict[str, Any]) -> dict[str, Any]:
        video = str(body.get("video") or "").strip()
        if not video:
            return {"ok": False, "error": "要填视频链接或BV号喵"}
        try:
            intro = self._run_async(self._start(video), timeout=150)
            begin_text = self._run_async(self._begin(), timeout=30)
            return {"ok": True, "message": intro + chr(10) + begin_text}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def _panel_react(self, _body: dict[str, Any]) -> dict[str, Any]:
        try:
            text = self._run_async(self._react_now(), timeout=60)
            return {"ok": True, "message": text}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def _panel_begin(self, _body: dict[str, Any]) -> dict[str, Any]:
        """面板开关：打开陪看（无需先预学习视频，只认截屏画面）。"""
        try:
            text = self._run_async(self._begin(), timeout=30)
            return {"ok": True, "message": text}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def _panel_comments(self, _body: dict[str, Any]) -> dict[str, Any]:
        try:
            text = self._run_async(self._read_comments(), timeout=45)
            return {"ok": True, "message": text}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def _panel_open_player(self, _body: dict[str, Any]) -> dict[str, Any]:
        session = self._session
        bvid = str((session or {}).get("video", {}).get("bvid") or _body.get("bvid") or "").strip()
        if not bvid:
            return {"ok": False, "error": "还没有陪看的视频喵（先开始一次陪看）"}
        if self._pwindow is None:
            self._pwindow = PlayerWindow(self._browser_profile_dir())
        ok, msg = self._pwindow.open(bvid, timeout=45)
        if ok and session is not None:
            with self._lock:
                session["player_window"] = True
        if ok and not self.bili_sessdata:
            self._sessdata_from_window()
        return {"ok": ok, "message": msg, "error": None if ok else msg}

    def _panel_close_player(self, _body: dict[str, Any]) -> dict[str, Any]:
        if self._pwindow is not None:
            self._pwindow.close()
        return {"ok": True, "message": "同步播放窗口已关闭喵"}

    def _panel_html(self) -> str:
        page = Path(__file__).parent / "static" / "index.html"
        try:
            return page.read_text(encoding="utf-8")
        except Exception:
            return "<h1>面板页缺失喵（static/index.html）</h1>"

    def _start_panel(self) -> None:
        endpoints = {
            ("GET", "/api/status"): self._panel_status,
            ("POST", "/api/stop"): self._panel_stop,
            ("POST", "/api/start"): self._panel_start_watch,
            ("POST", "/api/begin"): self._panel_begin,
            ("POST", "/api/react"): self._panel_react,
            ("POST", "/api/config"): self._panel_config,
            ("POST", "/api/sessdata"): self._panel_save_sessdata,
            ("POST", "/api/open_login"): self._panel_open_login,
            ("POST", "/api/read_sessdata"): self._panel_read_sessdata,
            ("POST", "/api/comments"): self._panel_comments,
            ("POST", "/api/open_player"): self._panel_open_player,
            ("POST", "/api/close_player"): self._panel_close_player,
        }
        port = find_open_port(self._panel_port)
        server = PanelServer(port, self._panel_html, endpoints)
        if server.start():
            self._panel_server = server
            self._panel_port = port
            self.logger.info("[watch_party] 管理面板已启动: http://127.0.0.1:{}", port)
            try:
                registered = self.register_static_ui("static")
                self.logger.info("[watch_party] static UI 注册: {}", registered)
            except Exception as exc:
                self.logger.warning("[watch_party] static UI 注册失败: {}", exc)
        else:
            self.logger.warning("[watch_party] 管理面板启动失败")

    # ── 功能入口 ───────────────────────────────────────────────
    async def _start(self, video_text: str) -> str:
        session = await self._prepare_session(video_text)
        with self._lock:
            self._session = session
        video = session["video"]
        intro = format_video_intro(video, len(session["danmaku"]), len(session["comments"]), self.catgirl_name)
        if self.prelearn:
            intro += (
                "\n（预学习素材已经存进本地素材库喵，只帮本喵看懂画面里看不清的梗；"
                "本喵不念稿、不报进度，只有截屏看到你正在放的画面才会开口）"
            )
        else:
            intro += "\n（预学习没开喵，本喵不抓弹幕/字幕/热评，纯靠截屏看画面聊；想开可以在面板里打开）"
        if not self.screen_assist:
            intro += "\nℹ️ 现在没开截屏辅助，本喵看不到画面就说不出话；想一起吐槽请在面板里把「截屏辅助」打开喵。"
        return intro

    async def _begin(self) -> str:
        """打开陪看开关：之后按 shot_interval_active 高频截屏自主评论。

        开关也可以由猫娘**自动**打开（截屏探测到主人在看视频时），
        所以这里允许还没有预学习会话——那时只认画面，不认素材。
        """
        await self._ensure_config_loaded()
        with self._lock:
            session = self._session
            title = str(((session or {}).get("video") or {}).get("title") or "")
        self._turn_on_watch(auto=False)
        if not self.screen_assist:
            return "好喵，开关打开啦——不过截屏辅助没开，本喵看不到画面就开不了口喵，先去面板把「截屏辅助」打开吧。"
        if title:
            return f"本喵搬好小板凳啦喵！从现在开始一起看《{title}》，本喵自己盯着屏幕，看到好笑的就吐槽～"
        return "本喵搬好小板凳啦喵！从现在开始盯着屏幕陪你看，看到好笑的就吐槽～"

    async def _read_comments(self) -> str:
        await self._ensure_config_loaded()
        session = self._session
        if not session:
            raise SdkError("还没有陪看中的视频喵，先发链接开始陪看。")
        aid = int(session["video"].get("aid") or 0)
        comments = session.get("comments") or []
        if not comments and aid:
            comments = await self._fetch_comments(aid)
            session["comments"] = comments
        from ._watch_logic import format_comments

        text = format_comments(comments, top=5)
        self._push(text)
        return text

    async def _react_now(self) -> str:
        """手动来一发：立刻看一眼当前画面并吐槽（不依赖任何进度/时间点）。"""
        session = self._session
        video = (session or {}).get("video") or {}
        title = str(video.get("title") or "") or f"{self.master_name}正在看的视频"
        seen = await self._describe_screen(frame=self._grab_player_frame())
        if not (seen.get("ok") and seen.get("text")):
            self.logger.info("[watch_party] 手动反应没看到画面: {}", seen.get("error"))
            raise SdkError("本喵这会儿没看到你的画面喵，把视频窗口露出来再喊我一次～")
        self.logger.info("[watch_party] 手动看图({})：{} 字", seen.get("source"), len(seen["text"]))
        prompt = build_screen_reaction_prompt(
            title=title,
            screen_text=str(seen.get("text") or ""),
            timeline_ctx=self._material_context(),
            catgirl_name=self.catgirl_name,
            master_name=self.master_name,
            recent=list(self._recent),
            source=str(seen.get("source") or "vision"),
        )
        try:
            raw = await _call_llm(
                "你是陪看的猫娘，只对看到的画面做真实反应，看不到就别硬说，别报时间进度。",
                prompt, self.llm_timeout,
                disable_thinking=self._thinking_disabled("chat"),
            )
        except SdkError:
            raise
        except Exception as exc:
            raise SdkError(f"本喵看画面想词的时候卡住了喵：{exc}")
        reaction = parse_screen_reaction(raw)
        if not reaction:
            raise SdkError("本喵看着画面憋了半天没想出话来喵…再等一下下？")
        self._said.append(reaction)
        self._recent.append(reaction["text"])
        del self._recent[:-5]
        self._last_comment_ts = time.time()
        return format_reaction(reaction)

    async def _stop(self) -> str:
        self._turn_off_watch()
        session = self._session
        if not session and not self._said:
            return "本来就没在看喵～"
        summary = build_summary(
            (session or {}).get("video") or {},
            (session or {}).get("danmaku") or [],
            self._said,
            [str(c.get("text") or "") for c in ((session or {}).get("comments") or [])],
        )
        self._said = []
        return f"好呀，先停在这里喵。\n{summary}"

    @llm_tool(
        name="neko_watch_party",
        description="让猫娘陪用户看B站视频：传入视频链接或BV号，猫娘会把弹幕/字幕/热评存入本地素材库；陪看时截屏看用户正在播放的画面，看到什么才吐槽什么，看不到就安静陪着。",
        parameters={
            "type": "object",
            "properties": {
                "video": {"type": "string", "description": "B站视频链接或BV号，如 BV1xx411c7mD"},
                "begin_now": {"type": "boolean", "description": "是否立即开始同步陪看（默认 true）"},
            },
            "required": ["video"],
        },
        timeout=120.0,
    )
    @plugin_entry(
        id="start_watch",
        name="陪我看B站",
        description="让猫娘抓取B站视频（链接/BV号）的弹幕、字幕、热评存入本地素材库并开始陪看；begin_now=true 时立即开始看屏陪看。",
        input_schema={
            "type": "object",
            "properties": {
                "video": {"type": "string", "description": "视频链接或BV号"},
                "begin_now": {"type": "boolean"},
            },
            "required": ["video"],
        },
    )
    async def start_watch_entry(self, video: str = "", begin_now: bool = True, **_):
        await self._ensure_config_loaded()
        if not _safe_str(video):
            return Err(SdkError("要发我视频链接或BV号喵，比如 BV1xx411c7mD。"))
        self.logger.info("[watch_party] start_watch 触发：video={!r} begin_now={}", video, begin_now)
        try:
            intro = await self._start(video)
            if begin_now:
                intro = f"{intro}\n{await self._begin()}"
            self.logger.info("[watch_party] start_watch 完成，会话就绪")
            return Ok(intro)
        except SdkError as exc:
            self.logger.warning("[watch_party] start_watch 失败: {}", exc)
            return Err(exc)
        except Exception as exc:
            self.logger.exception("预习失败: {}", exc)
            return Err(SdkError(f"预习出错了喵：{exc}"))

    @plugin_entry(
        id="begin_watch",
        name="陪我看（打开开关）",
        description="打开陪看开关：猫娘随后按截屏节奏**自主**高频截屏看画面并主动评论、抓笑点接梗；看不到画面就安静等着。",
        input_schema={"type": "object", "properties": {}},
    )
    async def begin_watch_entry(self, **_):
        await self._ensure_config_loaded()
        try:
            return Ok(await self._begin())
        except SdkError as exc:
            return Err(exc)

    @plugin_entry(
        id="react_now",
        name="聊聊现在这画面",
        description="让猫娘立刻看一眼当前画面并现场吐槽（看不到画面会直接说看不到）。",
        input_schema={"type": "object", "properties": {}},
    )
    async def react_now_entry(self, **_):
        await self._ensure_config_loaded()
        try:
            return Ok(await self._react_now())
        except SdkError as exc:
            return Err(exc)

    @plugin_entry(
        id="stop_watch",
        name="结束陪看",
        description="结束本次陪看并输出总结。",
        input_schema={"type": "object", "properties": {}},
    )
    async def stop_watch_entry(self, **_):
        await self._ensure_config_loaded()
        try:
            return Ok(await self._stop())
        except SdkError as exc:
            return Err(exc)

    @llm_tool(
        name="neko_watch_party_comments",
        description="让猫娘读当前陪看视频的评论区：热评排序展示并点评最戳的一条，同时推送到聊天。",
        parameters={"type": "object", "properties": {}},
        timeout=30.0,
    )
    @plugin_entry(
        id="read_comments",
        name="看评论区",
        description="读取当前陪看视频的热评并点评（需已开始陪看）。",
        input_schema={"type": "object", "properties": {}},
    )
    async def read_comments_entry(self, **_):
        await self._ensure_config_loaded()
        try:
            return Ok(await self._read_comments())
        except SdkError as exc:
            return Err(exc)
        except Exception as exc:
            self.logger.exception("读评论区失败: {}", exc)
            return Err(SdkError(f"评论区读不出来喵：{exc}"))

    @plugin_entry(
        id="status",
        name="陪看状态",
        description="查看当前陪看会话状态。",
        input_schema={"type": "object", "properties": {}},
    )
    async def status_entry(self, **_):
        await self._ensure_config_loaded()
        session = self._session
        video = (session or {}).get("video") or {}
        lines = [
            f"🐱 陪看开关：{'▶️ 已打开（高频自主截屏中）' if self._watching else '⏸️ 关闭'}",
            f"- 截屏节奏：在看时每 {self.shot_interval_active:.0f} 秒 / 待机每 {self.shot_interval_idle:.0f} 秒",
            f"- 自动探测：{'开' if self.auto_detect else '关'}｜预学习：{'开' if self.prelearn else '关'}｜截屏辅助：{'开' if self.screen_assist else '关'}",
        ]
        if video:
            lines.append(f"- 当前视频：《{video.get('title', '')}》（{video.get('bvid', '')}）")
        else:
            lines.append("- 还没预学习任何视频喵，发我链接或 BV 号就能认得标题～")
        lines.append(f"- 已自主吐槽 {len(self._said)} 句")
        return Ok("\n".join(lines))
