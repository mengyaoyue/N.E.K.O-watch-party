"""陪看猫娘 · 本地时间轴库（SQLite，零第三方依赖）

预习阶段抓到的弹幕 / 字幕 / 热评按时间轴落盘。之后**主进程（截屏）**在任意
播放位置都能按窗口把"这一刻观众在说什么、台词在说什么、热评在聊什么"取回来，
作为视觉模型理解画面的补充信息（画面里看不清的梗，往往弹幕/台词里写着）。

设计要点：
- 纯标准库 sqlite3，WAL 模式，允许跨线程使用（插件调度线程 + 面板线程）
- 同一个 bvid 重复预习 = 覆盖式写入（先删后插），避免数据重复累积
- 查询走 (bvid, t) 索引，按"当前位置前后窗口"取少量条目，控制提示词体积
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    bvid        TEXT PRIMARY KEY,
    title       TEXT DEFAULT '',
    up          TEXT DEFAULT '',
    duration    INTEGER DEFAULT 0,
    updated_at  REAL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS danmaku (
    bvid        TEXT NOT NULL,
    t           REAL NOT NULL,
    text        TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS subtitles (
    bvid        TEXT NOT NULL,
    t           REAL NOT NULL,
    dur         REAL DEFAULT 0,
    text        TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS comments (
    bvid        TEXT NOT NULL,
    rank        INTEGER DEFAULT 0,
    likes       INTEGER DEFAULT 0,
    user        TEXT DEFAULT '',
    text        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_danmaku_bvid_t   ON danmaku(bvid, t);
CREATE INDEX IF NOT EXISTS idx_subtitles_bvid_t ON subtitles(bvid, t);
CREATE INDEX IF NOT EXISTS idx_comments_bvid    ON comments(bvid, rank);
"""


def _clip(text: Any, limit: int) -> str:
    value = str(text or "").replace("\r", " ").replace("\n", " ").strip()
    return value[:limit]


def _fmt_time(seconds: float) -> str:
    total = max(0, int(seconds))
    minutes, secs = divmod(total, 60)
    return f"{minutes:02d}:{secs:02d}"


class TimelineDB:
    """陪看时间轴库：写入一次预习素材，按播放位置反复取用。"""

    def __init__(self, db_path: str):
        self._path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False, timeout=5.0)
        try:
            self._conn.execute("PRAGMA journal_mode=WAL")
        except sqlite3.Error:
            pass
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    # ── 写入 ───────────────────────────────────────────────────
    def save_video(self, video: dict[str, Any]) -> None:
        bvid = _clip(video.get("bvid"), 32)
        if not bvid:
            return
        with self._lock:
            self._conn.execute(
                "INSERT INTO videos(bvid, title, up, duration, updated_at) VALUES(?,?,?,?,?) "
                "ON CONFLICT(bvid) DO UPDATE SET title=excluded.title, up=excluded.up, "
                "duration=excluded.duration, updated_at=excluded.updated_at",
                (
                    bvid,
                    _clip(video.get("title"), 200),
                    _clip(video.get("up"), 80),
                    int(video.get("duration") or 0),
                    time.time(),
                ),
            )
            self._conn.commit()

    def save_danmaku(self, bvid: str, danmaku: list[dict[str, Any]]) -> int:
        bvid = _clip(bvid, 32)
        if not bvid:
            return 0
        rows = []
        for item in danmaku or []:
            try:
                t = float(item.get("t"))
            except (TypeError, ValueError):
                continue
            text = _clip(item.get("text"), 120)
            if text:
                rows.append((bvid, t, text))
        with self._lock:
            self._conn.execute("DELETE FROM danmaku WHERE bvid=?", (bvid,))
            self._conn.executemany("INSERT INTO danmaku(bvid, t, text) VALUES(?,?,?)", rows)
            self._conn.commit()
        return len(rows)

    def save_subtitles(self, bvid: str, subtitles: list[dict[str, Any]]) -> int:
        bvid = _clip(bvid, 32)
        if not bvid:
            return 0
        rows = []
        for item in subtitles or []:
            try:
                t = float(item.get("t"))
            except (TypeError, ValueError):
                continue
            text = _clip(item.get("text"), 200)
            if text:
                rows.append((bvid, t, float(item.get("dur") or 0), text))
        with self._lock:
            self._conn.execute("DELETE FROM subtitles WHERE bvid=?", (bvid,))
            self._conn.executemany(
                "INSERT INTO subtitles(bvid, t, dur, text) VALUES(?,?,?,?)", rows
            )
            self._conn.commit()
        return len(rows)

    def save_comments(self, bvid: str, comments: list[dict[str, Any]]) -> int:
        bvid = _clip(bvid, 32)
        if not bvid:
            return 0
        ordered = sorted(comments or [], key=lambda c: int(c.get("like", 0) or 0), reverse=True)
        rows = []
        for rank, item in enumerate(ordered):
            text = _clip(item.get("text"), 300)
            if text:
                rows.append((bvid, rank, int(item.get("like", 0) or 0), _clip(item.get("user"), 40), text))
        with self._lock:
            self._conn.execute("DELETE FROM comments WHERE bvid=?", (bvid,))
            self._conn.executemany(
                "INSERT INTO comments(bvid, rank, likes, user, text) VALUES(?,?,?,?,?)", rows
            )
            self._conn.commit()
        return len(rows)

    def save_timeline(
        self,
        video: dict[str, Any],
        danmaku: list[dict[str, Any]],
        subtitles: list[dict[str, Any]],
        comments: list[dict[str, Any]],
    ) -> dict[str, int]:
        """一次性把一份预习素材落库。返回各表写入条数。"""
        self.save_video(video)
        bvid = _clip(video.get("bvid"), 32)
        return {
            "danmaku": self.save_danmaku(bvid, danmaku),
            "subtitles": self.save_subtitles(bvid, subtitles),
            "comments": self.save_comments(bvid, comments),
        }

    # ── 读取 ───────────────────────────────────────────────────
    def counts(self, bvid: str) -> dict[str, int]:
        bvid = _clip(bvid, 32)
        if not bvid:
            return {"danmaku": 0, "subtitles": 0, "comments": 0}
        with self._lock:
            out: dict[str, int] = {}
            for table in ("danmaku", "subtitles", "comments"):
                cur = self._conn.execute(f"SELECT COUNT(*) FROM {table} WHERE bvid=?", (bvid,))
                out[table] = int(cur.fetchone()[0])
            return out

    def has(self, bvid: str) -> bool:
        got = self.counts(bvid)
        return bool(got["danmaku"] or got["subtitles"] or got["comments"])

    def window(
        self,
        bvid: str,
        position: float,
        before: float = 20.0,
        after: float = 10.0,
        danmaku_limit: int = 30,
        subtitle_limit: int = 6,
        comment_limit: int = 3,
    ) -> dict[str, list[dict[str, Any]]]:
        """取当前位置附近的弹幕/台词，以及全片热评 top N。"""
        bvid = _clip(bvid, 32)
        if not bvid:
            return {"danmaku": [], "subtitles": [], "comments": []}
        lo, hi = float(position) - max(0.0, before), float(position) + max(0.0, after)
        with self._lock:
            d_cur = self._conn.execute(
                "SELECT t, text FROM danmaku WHERE bvid=? AND t BETWEEN ? AND ? "
                "ORDER BY t LIMIT ?",
                (bvid, lo, hi, max(1, danmaku_limit)),
            )
            s_cur = self._conn.execute(
                "SELECT t, dur, text FROM subtitles WHERE bvid=? AND t BETWEEN ? AND ? "
                "ORDER BY t LIMIT ?",
                (bvid, lo, hi, max(1, subtitle_limit)),
            )
            c_cur = self._conn.execute(
                "SELECT rank, likes, user, text FROM comments WHERE bvid=? "
                "ORDER BY rank LIMIT ?",
                (bvid, max(1, comment_limit)),
            )
            return {
                "danmaku": [{"t": r[0], "text": r[1]} for r in d_cur.fetchall()],
                "subtitles": [{"t": r[0], "dur": r[1], "text": r[2]} for r in s_cur.fetchall()],
                "comments": [
                    {"rank": r[0], "like": r[1], "user": r[2], "text": r[3]} for r in c_cur.fetchall()
                ],
            }

    def context_text(
        self,
        bvid: str,
        position: float,
        before: float = 20.0,
        after: float = 10.0,
    ) -> str:
        """把当前位置附近的时间轴素材压成一段可喂给模型的补充上下文。"""
        win = self.window(bvid, position, before=before, after=after)
        return format_window(win, position)

    # ── 收尾 ───────────────────────────────────────────────────
    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except sqlite3.Error:
                pass


def format_window(win: dict[str, list[dict[str, Any]]], position: float) -> str:
    """把窗口素材渲染成提示词里的补充上下文（无素材返回空串）。"""
    lines: list[str] = []
    danmaku = win.get("danmaku") or []
    if danmaku:
        lines.append(f"【观众弹幕·此刻±窗口内 {len(danmaku)} 条】")
        for item in danmaku[:20]:
            lines.append(f"- {_fmt_time(item.get('t', 0))}「{_clip(item.get('text'), 40)}」")
    subtitles = win.get("subtitles") or []
    if subtitles:
        lines.append("【台词字幕·此刻前后】")
        for item in subtitles[:8]:
            lines.append(f"- {_fmt_time(item.get('t', 0))}「{_clip(item.get('text'), 60)}」")
    comments = win.get("comments") or []
    if comments:
        lines.append("【全片热评（点赞高）】")
        for item in comments[:4]:
            lines.append(f"- 👍{int(item.get('like', 0) or 0)}「{_clip(item.get('text'), 60)}」")
    if not lines:
        return ""
    lines.append(f"（当前播放位置约 {_fmt_time(position)}）")
    return "\n".join(lines)
