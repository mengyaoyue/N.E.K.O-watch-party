from __future__ import annotations
"""陪看猫娘（neko_watch_party）v0.1 · 作者：MENGYAOYUE

陪主人看B站视频：抓视频信息 + 全量弹幕 + 热评，一次 LLM 调用生成"陪看脚本"
（带时间戳的情绪反应），按播放进度到点让猫娘冒出来表达 笑/感动/同情/震惊/吐槽。
支持 /跳到 校准进度、随时问"刚才那段什么梗"、看完自动总结。

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
from ._bili_data import fetch_subtitles, fetch_via_page, subtitle_window, subtitles_to_text
from ._screen import capture_frame, capture_screen_text, ocr_frame, build_screen_context, extract_playback_time
from ._watch_logic import (
    build_cold_script_prompt,
    build_script_prompt_v2,
    build_summary,
    count_for_duration as effective_reaction_count,
    format_reaction,
    format_video_intro,
    gap_for_rpm,
    is_cold_video,
    normalize_reactions,
    parse_danmaku_xml,
    parse_video_id,
    sample_danmaku,
    spontaneous_remark,
)

_PLUGIN_ID = "neko_watch_party"

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


async def _call_llm(system: str, user: str, timeout: float) -> str:
    """调用 N.E.K.O 配置的对话模型（与 neko_natural_command 同一套取数方式）。"""
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
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        base_url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    def _post() -> tuple[int, str]:
        try:
            with urllib.request.urlopen(request, timeout=timeout) as resp:
                return resp.status, resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            return exc.code, ""
        except Exception:
            return 0, ""

    status, text = await asyncio.to_thread(_post)
    if status != 200:
        raise SdkError(f"模型接口返回 {status}，预习失败了喵。")
    try:
        data = json.loads(text)
        return data["choices"][0]["message"]["content"]
    except (json.JSONDecodeError, KeyError, IndexError):
        raise SdkError("模型返回格式不对，预习失败了喵。")


@neko_plugin


class WatchPartyPlugin(NekoPluginBase):
    """陪看猫娘：读取视频内容与弹幕，按进度共情陪伴。"""

    def __init__(self, ctx):
        super().__init__(ctx)
        self.file_logger = self.enable_file_logging(log_level="INFO")
        self.logger = self.file_logger

        self.reaction_count: int = 10
        self.auto_begin: bool = True
        self.heartbeat_minutes: int = 5
        self.bili_sessdata: str = ""
        self.screen_assist: bool = False
        self.screen_on_react: bool = True
        self.request_timeout: float = 15.0
        self.llm_timeout: float = 45.0
        self.poll_interval: float = 2.0
        self.auto_summary: bool = True
        self.catgirl_name: str = "猫娘"
        self.master_name: str = "主人"
        self._config_loaded = False

        # 陪看会话（内存态；重启即视为结束，无需持久化）
        self._lock = threading.Lock()
        self._session: Optional[dict[str, Any]] = None
        self._stop_event = threading.Event()
        self._wake_event = threading.Event()
        self._tick_thread: Optional[threading.Thread] = None
        self._panel_server = None
        self._panel_port: int = 15690
        self._panel_loop: Optional[asyncio.AbstractEventLoop] = None
        self._pwindow: Optional[PlayerWindow] = None
        self._panel_loop_thread: Optional[threading.Thread] = None

    # ── 配置 ───────────────────────────────────────────────────
    async def _load_config(self) -> None:
        try:
            cfg = await self.config.dump(timeout=5.0)
        except Exception as exc:
            self.logger.warning("[watch_party] 读取配置失败：{}", exc)
            cfg = {}
        section = cfg.get(_PLUGIN_ID) if isinstance(cfg, dict) else None
        section = section if isinstance(section, dict) else {}
        self.reaction_count = max(3, _safe_int(section.get("reaction_count"), 10))
        self.request_timeout = max(5.0, float(_safe_int(section.get("request_timeout"), 15)))
        self.llm_timeout = max(10.0, float(_safe_int(section.get("llm_timeout"), 45)))
        self.poll_interval = max(1.0, float(_safe_int(section.get("poll_interval"), 2)))
        self.auto_summary = bool(section.get("auto_summary", True))
        self.auto_begin = _safe_bool(section.get("auto_begin"), True)
        self.auto_density = _safe_bool(section.get("auto_density"), True)
        self.reactions_per_minute = max(0.5, min(10.0, float(_safe_int(section.get("reactions_per_minute"), 3))))
        self.heartbeat_minutes = max(0, _safe_int(section.get("heartbeat_minutes"), 5))
        self.bili_sessdata = _safe_str(section.get("bili_sessdata"))
        self.screen_assist = _safe_bool(section.get("screen_assist"), False)
        self.screen_on_react = _safe_bool(section.get("screen_on_react"), True)
        self.auto_align = _safe_bool(section.get("auto_align"), True)
        self.reaction_lead_seconds = max(0.0, float(_safe_int(section.get("reaction_lead_seconds"), 4)))
        self.reaction_lead_seconds: float = 4.0
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
        if not self.bili_sessdata:
            try:
                state_path = Path(self.data_path()) / "panel_state.json"
                saved = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
                self.bili_sessdata = str(saved.get("bili_sessdata") or "")
            except Exception:
                pass
        try:
            state_path = Path(self.data_path()) / "panel_state.json"
            saved = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
            saved_rpm = saved.get("reactions_per_minute")
            if saved_rpm:
                self.reactions_per_minute = max(0.5, min(10.0, float(saved_rpm)))
        except Exception:
            pass
        self._start_panel()
        self.logger.info("[watch_party] 启动：reaction_count={}", self.reaction_count)
        return Ok({"status": "running", "version": "0.1.0"})

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

    # ── 陪看脚本生成 ───────────────────────────────────────────
    async def _prepare_session(self, video_text: str) -> dict[str, Any]:
        await self._ensure_config_loaded()
        video_id = parse_video_id(video_text)
        if not video_id:
            raise SdkError(
                "没认出这是哪个视频喵…发我B站链接或 BV 号（如 BV1xx411c7mD）就好。"
            )
        t0 = time.time()
        video = await self._fetch_video(video_id)
        self.logger.info(
            "[watch_party] 预习①视频信息 OK：bvid={} cid={} 时长={}s（{:.1f}s）",
            video.get("bvid"), video.get("cid"), video.get("duration"), time.time() - t0,
        )

        danmaku = await self._fetch_danmaku(video["cid"], video.get("bvid", ""))
        self.logger.info("[watch_party] 预习②弹幕 {} 条", len(danmaku))
        await asyncio.sleep(0.3)

        # 字幕（时间轴台词）：配置了 SESSDATA 才拿得到；没有就跳过走弹幕模式
        subtitles: list[dict[str, Any]] = []
        cookie = ""
        if self.bili_sessdata:
            cookie = f"SESSDATA={self.bili_sessdata}"
        if video.get("bvid") and video.get("cid"):
            try:
                subtitles = await asyncio.to_thread(
                    fetch_subtitles, video["bvid"], video["cid"], cookie, self.request_timeout
                )
            except Exception as exc:
                self.logger.warning("[watch_party] 字幕获取失败: {}", exc)
        self.logger.info("[watch_party] 预习③字幕 {} 条（{}）", len(subtitles), "字幕模式" if subtitles else "弹幕模式")

        comments = await self._fetch_comments(video["aid"]) if video.get("aid") else []
        self.logger.info("[watch_party] 预习④热评 {} 条", len(comments))
        comment_texts = [c["text"] for c in comments]

        subtitle_text = subtitles_to_text(subtitles)
        sampled = sample_danmaku(danmaku)
        rpm = self.reactions_per_minute if self.auto_density else max(1.0, 60.0 / max(1, self.reaction_count))
        want = effective_reaction_count(video["duration"], rpm, floor=self.reaction_count)
        gap = gap_for_rpm(rpm)
        prompt = build_script_prompt_v2(
            video["title"], video["desc"], video["up"], video["duration"],
            subtitle_text, sampled, comment_texts, want, min_gap=gap,
        )
        raw = await _call_llm("你是陪看猫娘的脚本引擎。", prompt, self.llm_timeout)
        reactions = normalize_reactions(raw, video["duration"], want, min_gap=gap)
        if not reactions:
            reactions = self._fallback_reactions(video["duration"], danmaku)
            self.logger.warning("[watch_party] LLM 脚本不可用，降级弹幕高能点模式")
        self.logger.info(
            "[watch_party] 预习⑤陪看脚本 {} 条（时长 {}s，密度 {}/分钟 → 目标 {} 条，间隔≥{}s，共 {:.1f}s）",
            len(reactions), video["duration"], rpm, want, int(gap), time.time() - t0,
        )
        return {
            "video": video,
            "danmaku": danmaku,
            "subtitles": subtitles,
            "comments": comments,
            "reactions": reactions,
            "fired": [],
            "start_epoch": time.time() if self.auto_begin else None,
            "offset": 0.0,
            "summarized": False,
            "last_heartbeat": time.time(),
        }

    def _fallback_reactions(self, duration: int, danmaku: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """LLM 不可用时的兜底：在弹幕最密的几个点说"高能预警"式反应。"""
        moments = self._top_moments_local(danmaku, top=min(5, self.reaction_count))
        lines = [
            ("震惊", "弹幕突然爆炸了！前面一定发生了什么喵！"),
            ("好奇", "这一段弹幕刷得飞起，本喵也紧张起来了喵…"),
            ("吐槽", "密集弹幕预警！这段绝对是名场面喵。"),
            ("笑", "弹幕都在哈哈哈，到底有什么好笑的喵？！"),
            ("燃", "弹幕密度拉满了，燃起来了喵！！"),
        ]
        reactions = []
        for idx, moment in enumerate(moments):
            emotion, text = lines[idx % len(lines)]
            reactions.append({"at": max(3, moment["at"]), "emotion": emotion, "text": text, "quote": ""})
        return reactions

    @staticmethod
    def _top_moments_local(danmaku: list[dict[str, Any]], top: int = 3) -> list[dict[str, Any]]:
        from ._watch_logic import top_moments

        return top_moments(danmaku, top=top)

    # ── 时间轴调度 ─────────────────────────────────────────────
    def _tick_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._tick_once()
            except Exception:
                self.logger.exception("[watch_party] tick 异常")
            self._wake_event.clear()
            if self._stop_event.is_set():
                break
            self._wake_event.wait(timeout=max(1.0, self.poll_interval))
        self.logger.info("[watch_party] 调度线程退出")

    def _current_position(self, session: dict[str, Any]) -> float:
        if session.get("start_epoch") is None:
            return float(session.get("offset", 0.0))
        return time.time() - float(session["start_epoch"]) + float(session.get("offset", 0.0))

    def _tick_once(self) -> None:
        session = self._session
        if not session or session.get("start_epoch") is None:
            return
        position = self._current_position(session)
        video = session["video"]
        fired_ids = {r["at"] for r in session["fired"]}

        # 提前量：推送到聊天后猫娘开口需要几秒，提前触发刚好卡在画面节点上
        lead = self.reaction_lead_seconds
        for reaction in session["reactions"]:
            if reaction["at"] in fired_ids or reaction["at"] > position + lead:
                continue
            session["fired"].append(reaction)
            fired_ids.add(reaction["at"])
            text = format_reaction(reaction, reaction["at"])
            # 截屏辅助开启时：以实时画面为主参考（OCR 本地推理），脚本情绪做辅助
            if self.screen_assist and self.screen_on_react:
                try:
                    shot = capture_screen_text()
                except Exception:
                    shot = {"ok": False, "text": ""}
                if shot.get("ok") and shot.get("text"):
                    emoji = {"笑": "😂", "感动": "🥹", "同情": "🫂", "震惊": "😱", "吐槽": "😤",
                             "好奇": "🤔", "心疼": "🥺", "燃": "🔥"}.get(reaction.get("emotion", ""), "🐱")
                    minutes, seconds = divmod(int(reaction["at"]), 60)
                    screen_text = shot["text"][:46]
                    text = f"{emoji} [{minutes:02d}:{seconds:02d}] 画面上是「{screen_text}」，{reaction.get('text', '')[:24]}喵"
            self._push(text)

        # 同步播放窗口：直接读 video.currentTime（毫秒级真实进度，最高优先级）
        if session.get("player_window") and self._pwindow is not None:
            real = self._pwindow.position(timeout=3.0)
            if real is not None:
                if abs(real - position) >= 1.0:
                    with self._lock:
                        session["offset"] += real - position
                position = real
            elif self._pwindow.is_alive() is False:
                session["player_window"] = False
                self._push("同步播放窗口关掉了喵，回到手动校准模式（「跳到 X 分」还能用）")

        # 自发碎碎念：每 N 分钟结合台词/弹幕表达一次看法（不是干巴巴的进度条）
        if (
            self.heartbeat_minutes > 0
            and time.time() - float(session.get("last_heartbeat", time.time())) >= self.heartbeat_minutes * 60
        ):
            session["last_heartbeat"] = time.time()
            minutes, seconds = divmod(int(position), 60)
            # 自动对齐：截屏读播放器进度条（偏差 ≥30 秒才修正）
            aligned_note = ""
            if self.screen_assist and self.auto_align:
                aligned, aligned_note = self._auto_align(position)
                if aligned is not None:
                    position = aligned
                    minutes, seconds = divmod(int(position), 60)
            recent_fired = [r for r in session["fired"] if abs(r["at"] - position) <= 45]
            if not recent_fired:
                remark = spontaneous_remark(
                    session.get("subtitles", []), session.get("danmaku", []),
                    position, video.get("title", ""), seed=f"{video.get('bvid')}|{int(position)}",
                    comments=session.get("comments") or [],
                    rotate=int(session.get("rotate", 0)),
                )
                session["rotate"] = int(session.get("rotate", 0)) + 1
                if self.screen_assist:
                    shot = capture_screen_text()
                    if shot["ok"] and shot["text"]:
                        remark += f"（瞄到你画面上有「{shot['text'][:26]}」喵）"
                self._push(f"💬 {minutes}分{seconds:02d}：{remark}")
            else:
                self._push(f"⏱ 陪看中喵～放到 {minutes} 分{seconds:02d} 秒，要校准就说「跳到 X 分」")

        # 播完自动总结
        if (
            self.auto_summary
            and not session.get("summarized")
            and position >= video["duration"] + 15
        ):
            session["summarized"] = True
            summary = build_summary(video, session["danmaku"], session["fired"], [c["text"] for c in session["comments"]])
            self._push(summary)
            self.logger.info("[watch_party] 播放完毕，自动总结已推送")

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

    # ── 管理面板 ───────────────────────────────────────────────
    def _panel_status(self, _body: dict[str, Any]) -> dict[str, Any]:
        session = self._session
        if not session:
            return {"watching": False, "prepared": False}
        video = session["video"]
        watching = session.get("start_epoch") is not None
        position = self._current_position(session) if watching else float(session.get("offset", 0))
        return {
            "watching": watching, "prepared": True,
            "title": video.get("title", ""), "bvid": video.get("bvid", ""),
            "duration": video.get("duration", 0),
            "position": int(position),
            "fired": len(session["fired"]), "total": len(session["reactions"]),
            "screen_assist": self.screen_assist,
            "reactions_per_minute": self.reactions_per_minute,
            "sessdata_set": bool(self.bili_sessdata),
        }

    def _panel_stop(self, _body: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            session = self._session
            self._session = None
        if not session:
            return {"ok": True, "message": "本来就没在看喵"}
        summary = build_summary(session["video"], session["danmaku"], session["fired"], [c["text"] for c in session["comments"]])
        self._push("好呀，先停在这里喵。" + chr(10) + summary)
        return {"ok": True}

    def _panel_config(self, body: dict[str, Any]) -> dict[str, Any]:
        if "rpm" in body:
            self.reactions_per_minute = max(0.5, min(10.0, float(body["rpm"])))
            state_path = Path(self.data_path()) / "panel_state.json"
            try:
                state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
                state["reactions_per_minute"] = self.reactions_per_minute
                state_path.parent.mkdir(parents=True, exist_ok=True)
                state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                pass
        return {"ok": True, "rpm": self.reactions_per_minute}

    def _panel_save_sessdata(self, body: dict[str, Any]) -> dict[str, Any]:
        value = str(body.get("sessdata") or "").strip()
        state_path = Path(self.data_path()) / "panel_state.json"
        try:
            state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
        except Exception:
            state = {}
        if value:
            self.bili_sessdata = value
            state["bili_sessdata"] = value
        else:
            self.bili_sessdata = ""
            state.pop("bili_sessdata", None)
        try:
            state_path.parent.mkdir(parents=True, exist_ok=True)
            state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass
        return {"ok": True, "sessdata_set": bool(self.bili_sessdata)}

    def _panel_jump(self, body: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            session = self._session
            if not session or session.get("start_epoch") is None:
                return {"ok": False, "error": "还没开始看喵"}
            target = max(0.0, float(body.get("minute", 0)) * 60.0)
            session["offset"] += target - self._current_position(session)
            if self._pwindow is not None and self._pwindow.is_alive():
                self._pwindow.seek(target)
        return {"ok": True, "position": int(target)}

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
            self._pwindow = PlayerWindow()
        ok, msg = self._pwindow.open(bvid, timeout=45)
        if ok and session is not None:
            with self._lock:
                session["player_window"] = True
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
            ("POST", "/api/jump"): self._panel_jump,
            ("POST", "/api/start"): self._panel_start_watch,
            ("POST", "/api/react"): self._panel_react,
            ("POST", "/api/config"): self._panel_config,
            ("POST", "/api/sessdata"): self._panel_save_sessdata,
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
        if session.get("cold"):
            tip = "这个视频弹幕评论都很少喵，本喵只能靠标题「盲看」。"
            if not self.screen_assist:
                tip += "建议开启截屏辅助（配置 screen_assist=true），本喵就能看着画面陪你看了喵！"
            intro += f"\nℹ️ {tip}"
        if session["reactions"]:
            preview = session["reactions"][0]
            intro += f"\n（预习到 {len(session['reactions'])} 个想跟你吐槽的点，第一个在 {preview['at'] // 60}分{preview['at'] % 60:02d}秒喵）"
        return intro

    async def _begin(self) -> str:
        with self._lock:
            session = self._session
            if not session:
                raise SdkError("还没有要陪看的视频喵，先发我链接或 BV 号。")
            session["start_epoch"] = time.time()
            video = dict(session["video"])
        return f"本喵搬好小板凳了喵！从现在开始一起看《{video['title']}》，暂停了就跟本喵说跳到几分几秒～"

    async def _jump(self, minute: float) -> str:
        with self._lock:
            session = self._session
            if not session or session.get("start_epoch") is None:
                raise SdkError("还没有开始看喵，先说开始陪看。")
            target = max(0.0, minute * 60.0)
            session["offset"] += target - self._current_position(session)
        return f"好喵，本喵把进度条拽到 {int(minute)} 分了，跟上了！"

    def _auto_align(self, current_position: float) -> tuple[Optional[float], str]:
        """截屏读取播放器进度条，自动对齐时间轴（偏差 ≥30 秒才修正）。"""
        try:
            ok, frame = capture_frame()
            if not ok:
                return None, f"截屏失败：{frame}"
            ok2, items = ocr_frame(frame, keep_boxes=True)
        except Exception as exc:
            return None, f"截屏失败：{exc}"
        if not ok2:
            return None, str(items)[:60]
        if not isinstance(items, list):
            return None, "OCR 无位置信息"
        info = (self._session or {}).get("video", {}) if self._session else {}
        duration = int(info.get("duration") or 0)
        got = extract_playback_time(items, screen_h=1080)
        if not got or got.get("total", 0) <= 0:
            return None, "画面上没读到进度条时间"
        if duration and abs(got["total"] - duration) > 90:
            return None, f"进度条总时长 {got['total']}s 与视频 {duration}s 不符，跳过"
        drift = got["position"] - current_position
        if abs(drift) < 30:
            return None, "偏差不足 30 秒"
        with self._lock:
            session = self._session
            if not session or session.get("start_epoch") is None:
                return None, "会话已结束"
            session["offset"] += got["position"] - current_position
        self.logger.info(
            "[watch_party] 自动对齐：{}s → {}s（偏差 {:+d}s）",
            int(current_position), got["position"], int(drift),
        )
        minutes, seconds = divmod(got["position"], 60)
        return float(got["position"]), f"已自动对齐到 {minutes} 分{seconds:02d} 秒喵"

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
        with self._lock:
            session = self._session
            if not session:
                raise SdkError("还没有要陪看的视频喵。")
            position = self._current_position(session)
            window = [d for d in session["danmaku"] if abs(d["t"] - position) <= 15]
            fired_ids = {r["at"] for r in session["fired"]}
            upcoming = [r for r in session["reactions"] if r["at"] not in fired_ids and r["at"] >= position - 5]
        screen_ctx = ""
        if self.screen_assist and self.screen_on_react:
            shot = capture_screen_text()
            if shot["ok"]:
                screen_ctx = build_screen_context(shot["text"])
                self.logger.info("[watch_party] 截屏辅助：{} 字", len(shot["text"]))
            else:
                self.logger.info("[watch_party] 截屏不可用: {}", shot["error"])
        subs = session.get("subtitles") or []
        near_subs = subtitle_window(subs, position) if subs else []
        if screen_ctx:
            parts = [f"本喵瞄了一眼你的屏幕喵：{screen_ctx}"]
            if near_subs:
                parts.append(f"台词正说到「{near_subs[0]['text'][:40]}」")
            return "；".join(parts) + "，所以这到底在放什么喵？！"
        if near_subs:
            line = near_subs[len(near_subs) // 2]["text"]
            return f"台词正说到「{line[:40]}」喵，本喵听得很认真！"
        if window:
            burst = random.choice(window)["text"]
            return f"{random.choice(('好奇', '吐槽'))} 咦，这附近弹幕都在说「{burst[:30]}」喵！"
        if upcoming:
            nearest = upcoming[0]
            return f"{format_reaction(nearest, nearest['at'])}"
        return "这一段风平浪静喵，弹幕都在憋大招呢…"

    async def _stop(self) -> str:
        with self._lock:
            session = self._session
            self._session = None
        if not session:
            return "本来就没在看喵～"
        summary = build_summary(session["video"], session["danmaku"], session["fired"], [c["text"] for c in session["comments"]])
        return f"好呀，先停在这里喵。\n{summary}"

    @llm_tool(
        name="neko_watch_party",
        description="让猫娘陪用户看B站视频：传入视频链接或BV号，猫娘会预习视频、读取弹幕与热评，按播放进度表达笑/感动/同情/震惊等情绪反馈。",
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
        description="让猫娘预习B站视频（链接/BV号）并生成陪看脚本；begin_now=true 时立即开始同步陪看。",
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
        name="开始陪看",
        description="开始同步陪看（按时间轴推送猫娘的反应）。",
        input_schema={"type": "object", "properties": {}},
    )
    async def begin_watch_entry(self, **_):
        await self._ensure_config_loaded()
        try:
            return Ok(await self._begin())
        except SdkError as exc:
            return Err(exc)

    @plugin_entry(
        id="jump_to",
        name="校准进度",
        description="校准陪看进度：minute=用户当前看到第几分钟。",
        input_schema={
            "type": "object",
            "properties": {"minute": {"type": "number", "description": "当前看到第几分钟"}},
            "required": ["minute"],
        },
    )
    async def jump_to_entry(self, minute: float = 0, **_):
        await self._ensure_config_loaded()
        try:
            return Ok(await self._jump(float(minute)))
        except SdkError as exc:
            return Err(exc)

    @plugin_entry(
        id="react_now",
        name="聊聊刚才那段",
        description="猫娘针对当前播放位置附近的弹幕现场反应。",
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
        if not session:
            return Ok("现在没有在陪看喵。发我B站链接或 BV 号就可以开始～")
        video = session["video"]
        watching = session.get("start_epoch") is not None
        position = self._current_position(session)
        return Ok(
            f"🐱 陪看中：{'▶️ 进行中' if watching else '⏸️ 已预习未开始'}\n"
            f"- 《{video['title']}》（{video['bvid']}）\n"
            f"- 进度：{int(position) // 60}分{int(position) % 60:02d}秒 / {video['duration'] // 60}分{video['duration'] % 60:02d}秒\n"
            f"- 反应点：{len(session['fired'])}/{len(session['reactions'])} 已触发"
        )
