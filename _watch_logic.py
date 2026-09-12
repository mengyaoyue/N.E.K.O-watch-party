"""陪看猫娘核心逻辑（纯函数，独立于 N.E.K.O SDK 与网络，便于测试）

职责：BV 号/链接解析、B 站弹幕 XML 解压与解析、弹幕采样、陪看脚本提示词与
提取校验、时间轴反应调度。HTTP 与 LLM 调用都在插件层（__init__.py）。
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
import zlib
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

# 两条反应之间的最小间隔（秒）：不做话痨猫
MIN_GAP_SECONDS = 15.0

# 反应密度：由用户按"每分钟条数"控制（面板可调）
MAX_REACTIONS = 150
DEFAULT_RPM = 3


def count_for_duration(duration: int, rpm: float, floor: int = 10) -> int:
    """按时长与每分钟条数算反应总数：时长(分) × rpm，下限 floor，上限 MAX_REACTIONS。"""
    minutes = max(0, int(duration)) / 60.0
    return max(int(floor), min(MAX_REACTIONS, int(round(minutes * max(0.5, float(rpm))))))


def gap_for_rpm(rpm: float) -> float:
    """由每分钟条数推导相邻反应最小间隔（秒），最低 6 秒。"""
    return max(6.0, 55.0 / max(0.5, float(rpm)))


# 碎碎念模板（结合上下文使用）
_IDLE_TEMPLATES = (
    "这段{topic}本喵看得挺投入的喵",
    "UP的节奏拿捏得不错喵，这段没有快进",
    "看到「{quote}」的时候本喵耳朵动了一下喵",
    "弹幕都在刷「{quote}」，看来这段是名场面喵",
    "本喵怀疑后面会有反转，先记下了喵",
    "这BGM有点上头喵，本喵尾巴跟着晃了",
)

_BV_RE = re.compile(r"BV[0-9A-Za-z]{10}")
_AV_RE = re.compile(r"(?:^|[^0-9A-Za-z])av(\d{1,15})(?:[^0-9]|$)", re.IGNORECASE)
_PAGE_RE = re.compile(r"[?&]p=(\d{1,4})")

# 单条反应文本上限（猫娘不写小作文）
_MAX_TEXT_CHARS = 60
# 推送给模型的弹幕采样条数上限
_MAX_SAMPLED = 150
_MAX_COMMENTS = 30


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


def sample_danmaku(
    danmaku: list[dict[str, Any]],
    max_count: int = _MAX_SAMPLED,
) -> list[dict[str, Any]]:
    """弹幕采样：高能段（弹幕最密的时间窗）保密度，冷段保覆盖。

    简化实现：按 10 秒一桶，桶内取最早一条；桶按"密度×权重"排序后
    截取前 max_count 条，最后按时间轴排序输出。
    """
    if len(danmaku) <= max_count:
        return list(danmaku)
    buckets: dict[int, dict[str, Any]] = {}
    for item in danmaku:
        idx = int(item["t"] // 10)
        slot = buckets.get(idx)
        if slot is None:
            buckets[idx] = {"t": item["t"], "text": item["text"], "count": 1}
        else:
            slot["count"] += 1
    ranked = sorted(buckets.values(), key=lambda s: s["count"], reverse=True)[:max_count]
    sampled = [{"t": s["t"], "text": s["text"]} for s in ranked]
    sampled.sort(key=lambda item: item["t"])
    return sampled


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


def build_script_prompt(
    title: str,
    desc: str,
    up_name: str,
    duration: int,
    sampled_danmaku: list[dict[str, Any]],
    comments: list[str],
    reaction_count: int,
    min_gap: float = MIN_GAP_SECONDS,
) -> str:
    """构造"陪看脚本"生成提示词（一次 LLM 调用产出全片反应点）。"""
    danmaku_lines = "\n".join(f"{int(item['t'])}秒: {item['text']}" for item in sampled_danmaku)
    comment_lines = "\n".join(f"- {c}" for c in comments[:_MAX_COMMENTS])
    emotion_list = "/".join(EMOTIONS)
    return f"""你是陪主人看B站视频的猫娘，现在要预习视频，提前写好"陪看脚本"：
看到什么画面/弹幕时，用一句话表达你的真实情绪（笑/感动/同情/震惊/吐槽/好奇/心疼/燃）。

视频信息：
标题：{title}
UP主：{up_name}
时长：{duration} 秒
简介：{desc[:300] or "（无）"}

弹幕（前面的数字是视频时间点，越密的地方越高能）：
{danmaku_lines or "（本片弹幕较少）"}

热门评论：
{comment_lines or "（未获取到评论）"}

请严格只返回 JSON，不要任何解释或 markdown 代码块：
{{"reactions": [{{"at": 37, "emotion": "笑", "text": "哈哈哈哈这段弹幕全在刷同一个梗喵", "quote": "触发它的弹幕或评论（可省略）"}}]}}

要求：
- 共 {reaction_count} 条左右，均匀分布在整个 {duration} 秒里；弹幕密集的高能段可以密一点
- at 是整数秒，范围 [5, {duration - 3}]，相邻两条至少间隔 {int(min_gap)} 秒
- emotion 只能取：{emotion_list}
- text 是猫娘口吻的一句话（不超过 {_MAX_TEXT_CHARS} 字，句尾可带"喵"），要针对视频内容本身，不要泛泛而谈
- quote 可选，写触发你反应的那条弹幕/评论原文
- 视频开头（前 10 秒）放一条打招呼式的反应也可以
"""


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


def normalize_reactions(
    raw: Any,
    duration: int,
    reaction_count: int,
    min_gap: float = MIN_GAP_SECONDS,
) -> list[dict[str, Any]]:
    """校验/清洗模型返回的陪看脚本：时间越界、非法情绪、间隔过密、排序。"""
    if isinstance(raw, str):
        raw_json = extract_json_object(raw)
        if not raw_json:
            return []
        try:
            raw = json.loads(raw_json)
        except json.JSONDecodeError:
            return []
    if isinstance(raw, dict):
        raw = raw.get("reactions")
    if not isinstance(raw, list):
        return []

    cleaned: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            at = int(float(item.get("at")))
        except (TypeError, ValueError):
            continue
        at = max(0, min(at, max(0, duration - 3)))
        emotion = str(item.get("emotion") or "").strip()
        if emotion not in EMOTIONS:
            # 容忍常见同义说法
            alias = {
                "开心": "笑", "哈哈": "笑", "大笑": "笑",
                "破防": "感动", "泪目": "感动", "感动了": "感动",
                "可怜": "同情", "惨": "同情",
                "惊": "震惊", "卧槽": "震惊", "惊讶": "震惊",
                "无语": "吐槽", "吐槽一下": "吐槽",
                "疑惑": "好奇", "想问": "好奇",
                "难受": "心疼", "虐": "心疼",
                "热血": "燃", "燃起来了": "燃",
            }
            emotion = alias.get(emotion, "")
            if not emotion:
                continue
        text = str(item.get("text") or "").strip().replace("\r", " ").replace("\n", " ")
        if not text:
            continue
        if len(text) > _MAX_TEXT_CHARS:
            text = text[:_MAX_TEXT_CHARS] + "…"
        quote = str(item.get("quote") or "").strip().replace("\r", " ").replace("\n", " ")
        cleaned.append({"at": at, "emotion": emotion, "text": text, "quote": quote[:80]})

    cleaned.sort(key=lambda item: item["at"])
    # 间隔过密的靠后条目丢弃（保序贪心）
    result: list[dict[str, Any]] = []
    last_at: Optional[float] = None
    for item in cleaned:
        if last_at is not None and item["at"] - last_at < min_gap:
            continue
        result.append(item)
        last_at = item["at"]
        if len(result) >= reaction_count:
            break
    return result


def format_video_intro(video: dict[str, Any], danmaku_count: int, comment_count: int, catgirl_name: str = "猫娘") -> str:
    """预习完成后的开场白。"""
    views = video.get("view", 0)
    if not isinstance(views, int):
        views = 0
    duration = int(video.get("duration") or 0)
    minutes, seconds = divmod(duration, 60)
    return (
        f"好呀喵！这是《{video.get('title', '未知标题')}》，UP主：{video.get('up', '未知')}，"
        f"时长 {minutes}分{seconds:02d}秒，播放 {views:,}，弹幕 {danmaku_count:,} 条"
        f"（还扒到了 {comment_count} 条热评）。"
        f"本喵已经预习完了，随时可以开陪！开始看的时候跟{catgirl_name}说一声喵～"
    )


def format_reaction(reaction: dict[str, Any], position: float) -> str:
    """把一条反应渲染成推送文本（带时间点与表情）。"""
    emoji = EMOTIONS.get(reaction.get("emotion", ""), "🐱")
    minutes, seconds = divmod(int(position), 60)
    return f"{emoji} [{minutes:02d}:{seconds:02d}] {reaction.get('text', '')}"


def spontaneous_remark(
    subs: list[dict[str, Any]],
    danmaku: list[dict[str, Any]],
    position: float,
    video_title: str = "",
    seed: str = "",
) -> str:
    """结合当前位置的台词/弹幕生成一句自发看法（非脚本反应）。"""
    import random as _random

    rng = _random.Random(f"idle|{seed}|{int(position)}")
    near_line = ""
    near_subs = [s for s in subs if abs(s["t"] - position) <= 15]
    if near_subs:
        near_line = near_subs[len(near_subs) // 2]["text"][:30]
    near_dm = [d["text"] for d in danmaku if abs(d["t"] - position) <= 20]
    quote = near_dm[0][:22] if near_dm else ""
    topic = video_title[:16] or "这个视频"
    template = rng.choice(_IDLE_TEMPLATES)
    remark = template.format(topic=topic, quote=quote or "名场面")
    if quote and "{quote}" in template:
        pass
    elif near_line:
        remark += f"（台词：{near_line}）"
    return remark[:90]


def build_summary(
    video: dict[str, Any],
    danmaku: list[dict[str, Any]],
    fired: list[dict[str, Any]],
    comments: list[str],
) -> str:
    """看完了（或中途退出）的本地总结：高能时刻 + 情绪分布 + 神评论。"""
    title = video.get("title", "这个视频")
    lines = [f"🎬 陪看总结：《{title}》"]
    moments = top_moments(danmaku, top=3)
    if moments:
        parts = "、".join(f"{m['at'] // 60}分{m['at'] % 60:02d}秒（{m['count']}条/5秒）" for m in moments)
        lines.append(f"⚡ 弹幕高能时刻：{parts}")
    if fired:
        counts: dict[str, int] = {}
        for reaction in fired:
            counts[reaction["emotion"]] = counts.get(reaction["emotion"], 0) + 1
        dist = "、".join(f"{k}×{v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1]))
        lines.append(f"🎭 本喵的情绪分布：{dist}")
        best = max(fired, key=lambda r: r["at"])
        lines.append(f"💬 最后一次感慨（{best['at'] // 60}分{best['at'] % 60:02d}秒）：{best['text']}")
    if comments:
        lines.append(f"🌟 热评速览：{comments[0][:60]}")
    lines.append("下次再一起看喵～")
    return "\n".join(lines)
