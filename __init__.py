"""陪看猫娘（neko_watch_party）v0.1 · 作者：MENGYAOYUE

陪主人看B站视频：抓视频信息 + 全量弹幕 + 热评，一次 LLM 调用生成"陪看脚本"
（带时间戳的情绪反应），按播放进度到点让猫娘冒出来表达 笑/感动/同情/震惊/吐槽。
支持 /跳到 校准进度、随时问"刚才那段什么梗"、看完自动总结。

数据全部匿名读取公开接口（UA + buvid3），零第三方依赖；评论接口失败自动降级。
"""

from __future__ import annotations

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

from ._watch_logic import (
    build_script_prompt,
    build_summary,
    format_reaction,
    format_video_intro,
    normalize_reactions,
    parse_danmaku_xml,
    parse_video_id,
    sample_danmaku,
)

_PLUGIN_ID = "neko_watch_party"

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
)


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
        self.logger.info("[watch_party] 启动：reaction_count={}", self.reaction_count)
        return Ok({"status": "running", "version": "0.1.0"})

    @lifecycle(id="shutdown")
    def shutdown(self, **_):
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
            code = (data or {}).get("code")
            raise SdkError(f"呜…视频信息没拿到喵（接口码 {code}）。检查链接对不对，或稍后再试。")
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

    async def _fetch_danmaku(self, cid: int) -> list[dict[str, Any]]:
        url = f"https://api.bilibili.com/x/v1/dm/list.so?oid={cid}"
        status, body = await asyncio.to_thread(_http_get, url, "", self.request_timeout)
        if status != 200 or not body:
            self.logger.warning("[watch_party] 弹幕获取失败 status={}", status)
            return []
        return parse_danmaku_xml(body)

    async def _fetch_comments(self, aid: int) -> list[str]:
        """热评抓取（可降级）：失败返回空列表。"""
        url = f"https://api.bilibili.com/x/v2/reply/main?type=1&oid={aid}&mode=3"
        try:
            data = await asyncio.to_thread(_http_get_json, url, "", self.request_timeout)
        except Exception:
            return []
        if not data or data.get("code") != 0:
            return []
        replies = (data.get("data") or {}).get("replies") or []
        texts: list[str] = []
        for reply in replies[:10]:
            if isinstance(reply, dict):
                msg = _safe_str((reply.get("content") or {}).get("message"))
                if msg:
                    texts.append(msg)
        return texts

    # ── 陪看脚本生成 ───────────────────────────────────────────
    async def _prepare_session(self, video_text: str) -> dict[str, Any]:
        await self._ensure_config_loaded()
        video_id = parse_video_id(video_text)
        if not video_id:
            raise SdkError(
                "没认出这是哪个视频喵…发我B站链接或 BV 号（如 BV1xx411c7mD）就好。"
            )
        video = await self._fetch_video(video_id)
        danmaku = await self._fetch_danmaku(video["cid"])
        await asyncio.sleep(0.3)
        comments = await self._fetch_comments(video["aid"]) if video.get("aid") else []

        sampled = sample_danmaku(danmaku)
        prompt = build_script_prompt(
            video["title"], video["desc"], video["up"], video["duration"],
            sampled, comments, self.reaction_count,
        )
        raw = await _call_llm("你是陪看猫娘的脚本引擎。", prompt, self.llm_timeout)
        reactions = normalize_reactions(raw, video["duration"], self.reaction_count)
        if not reactions:
            # 模型没吐出可用脚本：降级为"高能点机械吐槽"，保证玩法不中断
            reactions = self._fallback_reactions(video["duration"], danmaku)
        return {
            "video": video,
            "danmaku": danmaku,
            "comments": comments,
            "reactions": reactions,
            "fired": [],
            "start_epoch": None,
            "offset": 0.0,
            "summarized": False,
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

        for reaction in session["reactions"]:
            if reaction["at"] in fired_ids or reaction["at"] > position:
                continue
            session["fired"].append(reaction)
            fired_ids.add(reaction["at"])
            self._push(format_reaction(reaction, reaction["at"]))

        # 播完自动总结
        if (
            self.auto_summary
            and not session.get("summarized")
            and position >= video["duration"] + 15
        ):
            session["summarized"] = True
            summary = build_summary(video, session["danmaku"], session["fired"], session["comments"])
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

    # ── 功能入口 ───────────────────────────────────────────────
    async def _start(self, video_text: str) -> str:
        session = await self._prepare_session(video_text)
        with self._lock:
            self._session = session
        video = session["video"]
        intro = format_video_intro(video, len(session["danmaku"]), len(session["comments"]), self.catgirl_name)
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

    async def _react_now(self) -> str:
        with self._lock:
            session = self._session
            if not session:
                raise SdkError("还没有要陪看的视频喵。")
            position = self._current_position(session)
            window = [d for d in session["danmaku"] if abs(d["t"] - position) <= 15]
            fired_ids = {r["at"] for r in session["fired"]}
            upcoming = [r for r in session["reactions"] if r["at"] not in fired_ids and r["at"] >= position - 5]
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
        summary = build_summary(session["video"], session["danmaku"], session["fired"], session["comments"])
        return f"好呀，先停在这里喵。\n{summary}"

    @llm_tool(
        name="neko_watch_party",
        description="让猫娘陪用户看B站视频：传入视频链接或BV号，猫娘会预习视频、读取弹幕与热评，按播放进度表达笑/感动/同情/震惊等情绪反馈。",
        parameters={
            "type": "object",
            "properties": {
                "video": {"type": "string", "description": "B站视频链接或BV号，如 BV1xx411c7mD"},
                "begin_now": {"type": "boolean", "description": "是否立即开始同步陪看（默认 false，等用户说开始）"},
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
    async def start_watch_entry(self, video: str = "", begin_now: bool = False, **_):
        await self._ensure_config_loaded()
        if not _safe_str(video):
            return Err(SdkError("要发我视频链接或BV号喵，比如 BV1xx411c7mD。"))
        try:
            intro = await self._start(video)
            if begin_now:
                intro = f"{intro}\n{await self._begin()}"
            return Ok(intro)
        except SdkError as exc:
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
