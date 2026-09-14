"""陪看猫娘核心逻辑（纯函数，独立于 N.E.K.O SDK 与网络，便于测试）

职责：BV 号/链接解析、B 站弹幕 XML 解压与解析、看屏反应提示词与解析、
渲染与总结。HTTP、截屏与 LLM 调用都在插件层（__init__.py）。

不预习全片、不预写"陪看脚本"：素材（弹幕/字幕/热评）由插件层存进 TimelineDB，
当作背景参考；但猫娘**只根据截屏看到的画面**说话，不报播放进度、不按时间轴念稿。
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
import zlib
from collections import Counter
from typing import Any, Optional

# 猫娘的陪看情绪枚举 → 表情（推送时带上，一眼读懂她在共情什么）
EMOTIONS: dict[str, str] = {
    "笑": "😂",
    "感动": "🥹",
    "同情": "🫂",
    "震惊": "😱",
    "吐槽": "😤",
    "好奇": "🤔",
    "心疼": "🥺",
    "燃": "🔥",
}


# "画面里是不是在看视频"的启发式关键词：视觉模型不可用、只剩 OCR 文字时兜底判断
_VIDEO_UI_HINTS = (
    "播放", "暂停", "弹幕", "全屏", "倍速", "选集", "画质", "投币", "收藏",
    "点赞", "关注", "已关注", "订阅", "推荐", "评论区", "正在看", "播放量",
    "up主", "UP主", "番剧", "直播", "清晰度", "自动连播", "发个弹幕",
)


def looks_like_video_ui(text: str) -> bool:
    """根据屏幕文字判断用户是不是正开着视频页面（纯 OCR 兜底用）。

    只做"像不像视频界面"的粗略判断，命中任意一个播放器/站内 UI 关键词即算在看。
    """
    content = (text or "").strip()
    if not content:
        return False
    return any(hint in content for hint in _VIDEO_UI_HINTS)


def parse_playing_answer(raw: Any) -> bool:
    """解析视觉模型对"用户现在是否在看视频"的回答，归一成 bool。

    约定模型只回「是 / 否」；先排除否定词（"不是"里含"是"，必须先查否定）。
    """
    text = str(raw or "").strip().lower()
    if not text:
        return False
    for neg in ("否", "不是", "没有看", "没在看", "不在看", "未在", "没在", "no"):
        if neg in text:
            return False
    for pos in ("是", "在看", "正在看", "有视频", "yes"):
        if pos in text:
            return True
    return False


_BV_RE = re.compile(r"BV[0-9A-Za-z]{10}")
_AV_RE = re.compile(r"(?:^|[^0-9A-Za-z])av(\d{1,15})(?:[^0-9]|$)", re.IGNORECASE)
_PAGE_RE = re.compile(r"[?&]p=(\d{1,4})")

# 单条反应文本上限（猫娘不写小作文）
_MAX_TEXT_CHARS = 60


def parse_video_id(text: str) -> Optional[dict[str, Any]]:
    """从任意文本里解析 B 站视频标识。

    支持：完整链接（含 ?p=2 分P）、裸 BV 号、av 号。
    返回 ``{"bvid"|"aid", "page": int}``，解析失败返回 ``None``。
    """
    text = (text or "").strip()
    if not text:
        return None
    bv = _BV_RE.search(text)
    if bv:
        page = 1
        page_match = _PAGE_RE.search(text)
        if page_match:
            page = max(1, int(page_match.group(1)))
        return {"bvid": bv.group(0), "page": page}
    av = _AV_RE.search(text)
    if av:
        return {"aid": int(av.group(1)), "page": 1}
    return None


def decompress_danmaku(data: bytes) -> Optional[bytes]:
    """解开 dm/list.so 返回的压缩 XML：先试 zlib（zlib/deflate 两种头）。"""
    if not data:
        return None
    for wbits in (47, 15, -15):
        try:
            return zlib.decompress(data, wbits)
        except zlib.error:
            continue
    # 万一服务器直接给了未压缩的 XML
    if data.lstrip()[:1] == b"<":
        return data
    return None


def parse_danmaku_xml(data: bytes) -> list[dict[str, Any]]:
    """解析弹幕 XML，返回按时间排序的 ``[{"t": 秒, "text": 内容}]``。"""
    xml_bytes = decompress_danmaku(data)
    if not xml_bytes:
        return []
    try:
        root = ET.fromstring(xml_bytes.decode("utf-8", errors="ignore"))
    except ET.ParseError:
        return []
    items: list[dict[str, Any]] = []
    for node in root.iter("d"):
        p_attr = node.get("p") or ""
        text = (node.text or "").strip()
        if not p_attr or not text:
            continue
        try:
            t = float(p_attr.split(",")[0])
        except ValueError:
            continue
        if t < 0:
            t = 0.0
        items.append({"t": t, "text": text})
    items.sort(key=lambda item: item["t"])
    return items


def _bucket_density(danmaku: list[dict[str, Any]], bucket_seconds: float) -> dict[int, int]:
    buckets: dict[int, int] = {}
    for item in danmaku:
        idx = int(item["t"] // bucket_seconds)
        buckets[idx] = buckets.get(idx, 0) + 1
    return buckets


def top_moments(danmaku: list[dict[str, Any]], top: int = 3, window: float = 30.0) -> list[dict[str, Any]]:
    """找出弹幕最密集的 top N "高能时刻"（窗口起点秒 + 弹幕数）。"""
    if not danmaku:
        return []
    bucket_seconds = 5.0
    buckets = _bucket_density(danmaku, bucket_seconds)
    scored: list[dict[str, Any]] = []
    for idx in sorted(buckets):
        start = idx * bucket_seconds
        count = sum(c for i, c in buckets.items() if start <= i * bucket_seconds < start + window)
        scored.append({"at": int(start), "count": count})
    scored.sort(key=lambda item: item["count"], reverse=True)
    picked: list[dict[str, Any]] = []
    for candidate in scored:
        if all(abs(candidate["at"] - p["at"]) >= window for p in picked):
            picked.append(candidate)
        if len(picked) >= top:
            break
    picked.sort(key=lambda item: item["at"])
    return picked


def strip_code_fence(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        first = text.find("\n")
        last = text.rfind("```")
        if first != -1 and last > first:
            text = text[first + 1 : last].strip()
    return text


def extract_json_object(text: str) -> Optional[str]:
    """从模型输出中提取第一个完整 JSON 对象（兼容代码块与前后杂讯）。"""
    if not text:
        return None
    text = strip_code_fence(text)
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


_EMOTION_ALIAS = {
    "开心": "笑", "哈哈": "笑", "大笑": "笑",
    "破防": "感动", "泪目": "感动", "感动了": "感动",
    "可怜": "同情", "惨": "同情",
    "惊": "震惊", "卧槽": "震惊", "惊讶": "震惊",
    "无语": "吐槽", "吐槽一下": "吐槽",
    "疑惑": "好奇", "想问": "好奇",
    "难受": "心疼", "虐": "心疼",
    "热血": "燃", "燃起来了": "燃",
}


def coerce_emotion(value: Any) -> str:
    """把模型给的情绪词归一化到 ``EMOTIONS``；无法识别返回空串。"""
    text = str(value or "").strip()
    if text in EMOTIONS:
        return text
    return _EMOTION_ALIAS.get(text, "")


def format_video_intro(video: dict[str, Any], danmaku_count: int, comment_count: int, catgirl_name: str = "猫娘") -> str:
    """素材就绪后的开场白。"""
    views = video.get("view", 0)
    if not isinstance(views, int):
        views = 0
    duration = int(video.get("duration") or 0)
    minutes, seconds = divmod(duration, 60)
    return (
        f"好呀喵！这是《{video.get('title', '未知标题')}》，UP主：{video.get('up', '未知')}，"
        f"时长 {minutes}分{seconds:02d}秒，播放 {views:,}，弹幕 {danmaku_count:,} 条"
        f"（还扒到了 {comment_count} 条热评）。"
        f"素材已经就绪，随时可以开陪！开始看的时候跟{catgirl_name}说一声喵～"
    )


def format_reaction(reaction: dict[str, Any]) -> str:
    """把一条反应渲染成推送文本（表情 + 内容，**不带任何时间点**）。"""
    emoji = EMOTIONS.get(reaction.get("emotion", ""), "🐱")
    return f"{emoji} {reaction.get('text', '')}"


def build_material_overview(
    danmaku: list[dict[str, Any]],
    comments: list[dict[str, Any]],
    limit_danmaku: int = 15,
    limit_comments: int = 5,
) -> str:
    """把预学习素材压成一段"整片层面的背景参考"，帮猫娘看懂画面里看不清的梗。

    注意：这里**刻意不带任何时间点**（不写第几分几秒），只给全片高频弹幕梗与热评，
    从根上避免模型把素材里的时间当成"当前进度"而产生进度幻觉。
    """
    lines: list[str] = []
    texts = [str(d.get("text") or "").strip() for d in danmaku if str(d.get("text") or "").strip()]
    if texts:
        common = [t for t, _ in Counter(texts).most_common(limit_danmaku)]
        lines.append("【观众高频弹幕/梗】" + "；".join(common))
    if comments:
        top = sorted(comments, key=lambda c: int(c.get("like", 0) or 0), reverse=True)[:limit_comments]
        picked = [str(c.get("text") or "").strip()[:40] for c in top if str(c.get("text") or "").strip()]
        if picked:
            lines.append("【热评】" + " / ".join(picked))
    return "\n".join(lines)


def format_comments(comments: list[dict[str, Any]], top: int = 5) -> str:
    """把评论区渲染成猫娘点评（按点赞排序取 top N）。"""
    if not comments:
        return "这个视频还没有评论喵～"
    sorted_c = sorted(comments, key=lambda c: int(c.get("like", 0) or 0), reverse=True)
    lines = ["💬 评论区精选："]
    medals = ["🥇", "🥈", "🥉"]
    for i, c in enumerate(sorted_c[:top]):
        medal = medals[i] if i < len(medals) else f"{i + 1}."
        like = int(c.get("like", 0) or 0)
        user = str(c.get("user") or "匿名")[:12]
        text = str(c.get("text") or "").replace(chr(10), " ")[:60]
        lines.append(f"{medal} [{user}·👍{like}] {text}")
    best = sorted_c[0]
    lines.append(
        f"本喵觉得「{str(best.get('text') or '')[:40]}」这条最说到点子上了喵"
    )
    return chr(10).join(lines)


def build_summary(
    video: dict[str, Any],
    danmaku: list[dict[str, Any]],
    said: list[dict[str, Any]],
    comments: list[str],
) -> str:
    """看完了（或中途退出）的本地总结：高能时刻 + 情绪分布 + 神评论。"""
    title = video.get("title", "这个视频")
    lines = [f"🎬 陪看总结：《{title}》"]
    moments = top_moments(danmaku, top=3)
    if moments:
        parts = "、".join(f"{m['at'] // 60}分{m['at'] % 60:02d}秒（{m['count']}条/5秒）" for m in moments)
        lines.append(f"⚡ 弹幕高能时刻：{parts}")
    if said:
        counts: dict[str, int] = {}
        for reaction in said:
            counts[reaction["emotion"]] = counts.get(reaction["emotion"], 0) + 1
        dist = "、".join(f"{k}×{v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1]))
        lines.append(f"🎭 本喵的情绪分布：{dist}")
        lines.append(f"💬 本喵最后一句：{said[-1].get('text', '')}")
    if comments:
        lines.append(f"🌟 热评速览：{comments[0][:60]}")
    lines.append("下次再一起看喵～")
    return "\n".join(lines)


def build_screen_reaction_prompt(
    *,
    title: str,
    screen_text: str,
    timeline_ctx: str = "",
    catgirl_name: str = "猫娘",
    master_name: str = "主人",
    recent: Optional[list[str]] = None,
    source: str = "vision",
) -> str:
    """构造"看到画面才反应"的单帧提示词。

    这里**不预习全片、不预写脚本、不谈任何播放进度/时间点**，
    只针对"此刻截到的这一帧"要一句真实反应；
    ``timeline_ctx`` 是可选预学习攒下的**整片级**弹幕梗/热评背景（不含时间），
    用来帮猫娘认出画面里的梗，但绝不能凌驾于画面之上。
    """
    if source == "vision":
        seen = f"{catgirl_name}刚截了一帧{master_name}正在播放的画面，视觉模型对画面的描述是：「{screen_text[:300]}」"
    else:
        seen = f"{catgirl_name}只读到了屏幕上的文字（这次没拿到画面描述）：「{screen_text[:300]}」"
    emotion_list = "/".join(EMOTIONS)
    ctx_block = (
        f"\n[整片素材背景·观众爱刷的梗·不含时间点·仅供参考]\n{timeline_ctx[:800]}\n"
        if timeline_ctx
        else ""
    )
    history = ""
    if recent:
        history = "\n[你刚刚已经说过的话，别重复]\n" + "\n".join(f"- {t}" for t in recent[-3:]) + "\n"
    return f"""你是正在陪{master_name}看B站视频的{catgirl_name}。
{seen}
视频：《{title}》{ctx_block}{history}
请只根据**你此刻看到的这一帧画面**主动说一句真实的猫娘反应，不要念稿、不要复述预习、不要编画面里没有的东西。
你要像真的看进去了：抓到画面里的**笑点、表情包、名场面、梗或弹幕吐槽**就顺势吐槽/狂笑/接梗；画面平淡就轻轻好奇。
情绪只能取：{emotion_list}

严格只返回 JSON，不要解释、不要 markdown 代码块：
{{"emotion": "笑", "text": "哈哈哈哈这个表情太绝了喵"}}

要求：
- text 不超过 {_MAX_TEXT_CHARS} 字，口语化、像一起看片的朋友，句尾可带"喵"
- 不要提"第几分钟/第几秒/进度"这类信息，你没在数时间
- 整片素材背景只是帮你认梗，画面里看不到就别硬套
- 如果这一帧确实看不出什么，就清淡一点（好奇/吐槽），不要硬编剧情
"""


def parse_screen_reaction(raw: Any) -> Optional[dict[str, Any]]:
    """把"看屏反应"的模型输出解析成一条 reaction；无法解析返回 None（宁可不说话）。

    返回的 reaction **不含任何时间点字段**。
    """
    data: Any = None
    if isinstance(raw, str):
        raw_json = extract_json_object(raw)
        if raw_json:
            try:
                data = json.loads(raw_json)
            except json.JSONDecodeError:
                data = None
    elif isinstance(raw, dict):
        data = raw
    if not isinstance(data, dict):
        return None
    emotion = coerce_emotion(data.get("emotion"))
    text = str(data.get("text") or "").strip().replace("\r", " ").replace("\n", " ")
    if not text:
        return None
    if len(text) > _MAX_TEXT_CHARS:
        text = text[:_MAX_TEXT_CHARS] + "…"
    return {"emotion": emotion or "好奇", "text": text, "quote": ""}
