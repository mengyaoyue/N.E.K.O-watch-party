"""陪看猫娘：纯标准库独立测试（python tests/test_basic.py，不联网）

覆盖：BV 解析、弹幕解析、高能时刻、**看屏反应**（提示词与解析）、渲染、
WBI 签名、字幕、截屏上下文与待机探测、本地时间轴库 TimelineDB，
以及"开关 + 截屏驱动 + 可选素材"的源码级接线断言
（确认**任何播放进度/时间点/自动对齐机制都已彻底移除**）。
"""

import importlib.util
import json
import sys
import tempfile
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_logic():
    return load_module("neko_watch_logic", "_watch_logic.py")


def assert_eq(actual, expected, msg=""):
    if actual != expected:
        raise AssertionError(f"{msg}: expected {expected!r}, got {actual!r}")


def make_danmaku_xml(items) -> bytes:
    """构造压缩弹幕 XML（items: [(秒, 文本)]）。"""
    rows = "".join(
        f'<d p="{t},1,25,16777215,0,0,0,0">{text}</d>' for t, text in items
    )
    xml = f'<?xml version="1.0"?><i><chatserver>cs</chatserver>{rows}</i>'
    return zlib.compress(xml.encode("utf-8"))


def main():
    print("加载 _watch_logic ...")
    mod = load_logic()

    # 1. 视频标识解析
    cases = [
        ("https://www.bilibili.com/video/BV1xx411c7mD?spm_id_from=x", {"bvid": "BV1xx411c7mD", "page": 1}),
        ("https://www.bilibili.com/video/BV1xx411c7mD?p=3", {"bvid": "BV1xx411c7mD", "page": 3}),
        ("BV1xx411c7mD", {"bvid": "BV1xx411c7mD", "page": 1}),
        ("快看这个 av170001", {"aid": 170001, "page": 1}),
        ("https://b23.tv/BV1xx411c7mD", {"bvid": "BV1xx411c7mD", "page": 1}),
    ]
    for text, expected in cases:
        assert_eq(mod.parse_video_id(text), expected, f"解析 {text[:40]!r}")
    assert mod.parse_video_id("随便说点什么") is None, "无视频标识应返回 None"
    assert mod.parse_video_id("") is None, "空文本应返回 None"

    # 2. 弹幕 XML 解压与解析（含时间排序、非法行跳过）
    raw = make_danmaku_xml([(30.5, "哈哈哈"), (5.0, "前方高能"), ("abc", "坏行"), (120, "名场面"), (30.5, " duplicate")])
    items = mod.parse_danmaku_xml(raw)
    assert_eq([d["t"] for d in items], [5.0, 30.5, 30.5, 120.0], "应按时间排序")
    assert_eq(items[0]["text"], "前方高能", "文本应正确")
    # 未压缩的 XML 也能吃
    plain = mod.decompress_danmaku(b"<i><d p=\"1,1,1,1,1,1,1,1\">hi</d></i>")
    assert_eq(mod.parse_danmaku_xml(plain), [{"t": 1.0, "text": "hi"}], "未压缩 XML 应可直接解析")
    assert mod.parse_danmaku_xml(b"\x00\x01garbage") == [], "垃圾数据应返回空"

    # 3. 高能时刻：窗口密度 top3，且互不重叠
    burst = [{"t": 100.0 + (i % 40) * 0.1, "text": "x"} for i in range(200)]
    quiet = [{"t": float(i * 60), "text": "y"} for i in range(10)]
    moments = mod.top_moments(burst + quiet, top=3)
    assert len(moments) == 3, f"应有 3 个高能时刻: {moments}"
    biggest = max(moments, key=lambda m: m["count"])
    assert biggest["count"] >= 100, f"最高密度应来自爆发区: {moments}"
    assert 95 <= biggest["at"] <= 145, f"爆发区应落在 100 秒附近: {moments}"
    assert [m["at"] for m in moments] == sorted(m["at"] for m in moments), "输出应按时间排序"
    for a, b in zip(moments, moments[1:]):
        assert abs(a["at"] - b["at"]) >= 30, f"高能窗口应互不重叠: {moments}"

    # 4. 看屏反应提示词：只对"此刻这一帧"要一句，不念稿、不带任何时间进度
    prompt = mod.build_screen_reaction_prompt(
        title="测试视频",
        screen_text="画面里一个人被辣到流泪，桌上摆着一碗红油面",
        timeline_ctx="【观众高频弹幕/梗】前方高能；这个真的很辣\n【热评】看着就辣",
        recent=["刚刚说过「这碗面看着就很辣」"],
    )
    assert "测试视频" in prompt, f"应带标题: {prompt[:120]!r}"
    assert "被辣到流泪" in prompt, "应注入视觉模型看到的画面描述"
    assert "观众爱刷的梗" in prompt and "前方高能" in prompt, "应注入可选预学习素材"
    assert "别重复" in prompt and "这碗面看着就很辣" in prompt, "应带走已说过的话以防复读"
    assert "不要念稿" in prompt, "应明确要求不念稿"
    assert "笑/感动/同情/震惊/吐槽/好奇/心疼/燃" in prompt, "应列出情绪枚举"
    assert "不要提" in prompt and "进度" in prompt, "应明确禁止提时间进度"
    # 没拿到画面描述（只 OCR）时的措辞
    prompt_ocr = mod.build_screen_reaction_prompt(title="t", screen_text="标题栏文字", source="ocr")
    assert "只读到了屏幕上的文字" in prompt_ocr, f"OCR 分支措辞: {prompt_ocr[:120]!r}"
    # 没有素材时不应出现补充块
    prompt_plain = mod.build_screen_reaction_prompt(title="t", screen_text="画面")
    assert "观众爱刷的梗" not in prompt_plain, "无素材不应注入素材块"
    assert "你刚刚已经说过的话" not in prompt_plain, "无历史不应注入防复读块"

    # 5. 看屏反应解析：脏输出归 None（宁可不说话），情绪容错，文本截断，且无时间字段
    raw_reaction = '```json\n{"emotion": "开心", "text": "太好笑了喵"}\n```'
    r = mod.parse_screen_reaction(raw_reaction)
    assert r is not None and r["emotion"] == "笑" and r["text"] == "太好笑了喵", f"解析: {r!r}"
    assert "at" not in r and "position" not in r, "反应不应含任何时间字段"
    assert mod.parse_screen_reaction('{"emotion": "笑"}') is None, "缺 text 应返回 None"
    assert mod.parse_screen_reaction("完全不是JSON") is None, "无法解析应返回 None"
    assert mod.parse_screen_reaction(json.dumps({"text": "嗯"}))["emotion"] == "好奇", "情绪缺失应回退好奇"
    long_r = mod.parse_screen_reaction(json.dumps({"emotion": "吐槽", "text": "x" * 100}, ensure_ascii=False))
    assert len(long_r["text"]) <= 61 and long_r["text"].endswith("…"), f"超长应截断: {len(long_r['text'])}"

    # 6. 渲染：开场白 / 反应 / 总结
    video = {"title": "测试视频", "up": "某UP", "duration": 185, "view": 12345}
    intro = mod.format_video_intro(video, 1200, 5, "猫娘")
    assert "测试视频" in intro and "某UP" in intro and "1,200" in intro, f"开场白: {intro!r}"
    minutes, seconds = divmod(185, 60)
    assert f"{minutes}分{seconds:02d}秒" in intro, "时长应格式化"

    reaction_text = mod.format_reaction({"emotion": "笑", "text": "太好笑了喵"})
    assert "😂" in reaction_text and "太好笑了喵" in reaction_text, f"反应渲染: {reaction_text!r}"
    assert "[" not in reaction_text, "反应渲染不应带任何时间点"

    danmaku = [{"t": float(i), "text": "x"} for i in range(50)] + [{"t": 100.0 + i * 0.1, "text": "boom"} for i in range(80)]
    said = [{"at": 20, "emotion": "笑", "text": "a"}, {"at": 100, "emotion": "震惊", "text": "b"}]
    summary = mod.build_summary(video, danmaku, said, ["第一条热评"])
    assert "陪看总结" in summary and "高能时刻" in summary and "情绪分布" in summary, f"总结: {summary!r}"
    assert "笑×1" in summary and "震惊×1" in summary, "情绪统计应出现"
    assert "第一条热评" in summary, "热评速览应出现"

    # 7. WBI 签名 + 字幕工具
    bd = load_module("neko_bili_data", "_bili_data.py")

    # 7.1 WBI 签名：确定性 + 参数规范化 + wts/w_rid 存在
    s1 = bd.sign_wbi({"foo": "1", "bvid": "BV1xx411c7mD"}, "imgkey", "subkey", wts=1700000000)
    s2 = bd.sign_wbi({"bvid": "BV1xx411c7mD", "foo": "1"}, "imgkey", "subkey", wts=1700000000)
    assert s1["w_rid"] == s2["w_rid"], "同参数同 wts 应同签名（与参数顺序无关）"
    assert s1["wts"] == 1700000000 and len(s1["w_rid"]) == 32, "wts 透传 + w_rid 为 md5"
    s3 = bd.sign_wbi({"foo": "1"}, "imgkey", "subkey", wts=1700000001)
    assert s3["w_rid"] != s1["w_rid"], "wts 变化签名应变化"

    # 7.2 mixin key 截断 32 位
    assert_eq(len(bd._mixin_key("a" * 64, "b" * 64)), 32, "mixin key 应为 32 字符")

    # 7.3 字幕正文解析（B站字幕 JSON）
    body = json.dumps({"body": [
        {"from": 1.5, "to": 3.0, "content": "大家好"},
        {"from": 3.2, "to": 5.0, "content": "今天讲原神"},
        {"bad": 1},
    ]}).encode()
    subs = bd.parse_subtitle_json(json.loads(body.decode("utf-8")))
    assert_eq(len(subs), 2, "应解析 2 条有效字幕")
    assert_eq(subs[0], {"t": 1.5, "dur": 1.5, "text": "大家好"}, "字幕字段")
    assert bd.parse_subtitle_json([{"bad": 1}]) == [], "坏数据返回空"

    # 7.4 时间轴窗口 + 采样文本
    near = bd.subtitle_window(subs, 2.0, window=5.0)
    assert_eq(len(near), 2, "窗口应覆盖两条")
    text = bd.subtitles_to_text(subs)
    assert "[00:01] 大家好" in text and "[00:03] 今天讲原神" in text, f"时间轴文本: {text!r}"

    # 8. 截屏辅助纯逻辑：画面上下文 + 待机探测提示词
    scr = load_module("neko_screen", "_screen.py")

    # 8.1 build_screen_context：空文本不产出、超长截断
    assert_eq(scr.build_screen_context(""), "", "空 OCR 不产出上下文")
    ctx = scr.build_screen_context("原神 7.1 前瞻直播")
    assert "【画面上的文字】" in ctx and "原神" in ctx, f"上下文: {ctx!r}"
    long_ctx = scr.build_screen_context("x" * 500, max_chars=100)
    assert len(long_ctx) <= 120 and "…" in long_ctx, "超长应截断"

    # 8.2 待机探测提示词：只问"是不是在看视频"，要求回是/否，越短越好
    assert "是" in scr.PLAYING_PROBE_PROMPT and "否" in scr.PLAYING_PROBE_PROMPT, "探测提示词应要求是/否"
    assert "不" in scr.PLAYING_PROBE_PROMPT, "探测提示词应说明不要解释/不要描述"

    # 8.3 截屏层不得再具备"从画面里抠播放进度"的能力（那正是幻觉根源）
    assert not hasattr(scr, "extract_playback_time"), "截屏层不得再解析播放进度"

    # 8.4 视觉链路要支持透传"关思考"，且必须是防御式的（宿主接口不认就退回原样）
    src_scr = (ROOT / "_screen.py").read_text(encoding="utf-8")
    assert "disable_thinking" in src_scr, "视觉描述应支持透传关思考开关"
    assert 'extra_body' in src_scr and '"type": "disabled"' in src_scr, "关思考应走 extra_body 的 thinking:{type:disabled}"
    assert "async def _attempt" in src_scr, "视觉带参数调用失败后应能退回原样重试一次"

    # 9. 源码级断言：开关 + 截屏驱动；任何时间进度/对齐/跳转机制都应彻底移除
    src_init = (ROOT / "__init__.py").read_text(encoding="utf-8")
    assert "auto_detect" in src_init, "应有自动探测开关"
    assert "shot_interval_active" in src_init and "shot_interval_idle" in src_init, "应有可自定义截屏节奏"
    assert "PLAYING_PROBE_PROMPT" in src_init and "looks_like_video_ui" in src_init and "parse_playing_answer" in src_init, "应接待机探测"
    assert "TimelineDB" in src_init and "save_timeline" in src_init, "可选预学习素材应落本地时间轴库"
    assert "build_material_overview" in src_init, "预学习素材应压成整片级背景"
    assert "build_screen_reaction_prompt" in src_init and "parse_screen_reaction" in src_init, "应接看屏反应"
    assert "画面上是「" in src_init, "推送应以画面内容为主参考"
    assert "def _panel_config" in src_init, "面板应能自定义截屏频率"

    # 9.1 思考模式：默认可关、分链路生效，且不认这个字段的模型会自动退回普通请求
    assert "thinking_mode" in src_init, "应有可选的思考模式开关（关掉能明显降延迟）"
    assert "_safe_thinking_mode" in src_init and "def _thinking_disabled" in src_init, "思考模式应有取值收敛与分链路判断"
    for mode in ("none", "chat", "vision", "all"):
        assert f'"{mode}"' in src_init, f"思考模式应含取值 {mode}"
    assert '"type": "disabled"' in src_init and "thinking=" in src_init, "对话链路关思考应写 thinking:{type:disabled}"
    assert src_init.count("disable_thinking=self._thinking_disabled") == 4, "出话两处 + 看画面两处都要带上思考开关"
    src_html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    assert "thinking_mode" in src_html and "saveThinking" in src_html, "面板应能改思考模式"
    for dead in (
        "_current_position", "reactions_per_minute", "_auto_align", "_panel_jump",
        "api/jump", "jump_to", "extract_playback_time", "start_epoch",
        "heartbeat_minutes", "reaction_lead_seconds", "reaction_count",
        "auto_density", "screen_on_react", "build_script_prompt",
        "normalize_reactions", "spontaneous_remark", "is_cold_video",
        "build_cold_script_prompt", "recent_fired", "position + lead", "盲看",
    ):
        assert dead not in src_init, f"时间进度/念稿机制残留：{dead}"
    for gone in (
        "build_script_prompt", "build_script_prompt_v2", "normalize_reactions",
        "spontaneous_remark", "is_cold_video", "build_cold_script_prompt",
        "sample_danmaku", "count_for_duration", "fmt_pos", "format_window",
    ):
        assert not hasattr(mod, gone), f"_watch_logic 应移除时间进度相关函数 {gone}"
    print("源码断言通过：开关 + 截屏驱动 + 可选素材，无时间进度残留")

    # 10. 本地时间轴库 TimelineDB：写入一次、覆盖不累积、按位置取用
    db_mod = load_module("neko_timeline_db", "_timeline_db.py")
    with tempfile.TemporaryDirectory() as td:
        db = db_mod.TimelineDB(str(Path(td) / "timeline.db"))
        try:
            video = {"bvid": "BV1xx411c7mD", "title": "测试视频", "up": "某UP", "duration": 300}
            danmaku = [{"t": 10.0, "text": "前方高能"}, {"t": 90.0, "text": "名场面"}]
            subtitles = [{"t": 8.0, "dur": 2.0, "text": "大家好"}]
            comments = [{"user": "a", "like": 5, "text": "好看"}, {"user": "b", "like": 99, "text": "神评"}]
            counts = db.save_timeline(video, danmaku, subtitles, comments)
            assert_eq(counts, {"danmaku": 2, "subtitles": 1, "comments": 2}, "入库条数")
            assert db.has("BV1xx411c7mD"), "入库后 has 应为真"
            assert_eq(db.counts("BV1xx411c7mD"), {"danmaku": 2, "subtitles": 1, "comments": 2}, "counts")
            assert_eq(db.counts("BVnone"), {"danmaku": 0, "subtitles": 0, "comments": 0}, "未知视频为零")
            assert not db.has("BVnone"), "未知视频 has 为假"

            # 重复预习 = 覆盖式写入，不累积
            db.save_timeline(video, danmaku[:1], [], [])
            assert_eq(db.counts("BV1xx411c7mD"), {"danmaku": 1, "subtitles": 0, "comments": 0}, "覆盖写入不累积")

            # 完整重写后再验证按位置取用
            db.save_timeline(video, danmaku, subtitles, comments)
            win = db.window("BV1xx411c7mD", 10.0, before=5.0, after=5.0)
            assert_eq([d["text"] for d in win["danmaku"]], ["前方高能"], "窗口只取当前位置附近弹幕")
            assert_eq(win["comments"][0]["text"], "神评", "热评按点赞排序")
            ctx = db.context_text("BV1xx411c7mD", 9.0)
            assert "前方高能" in ctx and "大家好" in ctx, f"上下文: {ctx!r}"
            assert "前方高能" not in db.context_text("BV1xx411c7mD", 100000.0), "远处位置不应取到开场弹幕"
            assert_eq(db_mod.format_window({"danmaku": [], "subtitles": [], "comments": []}, 0), "", "空窗口不产出")
        finally:
            db.close()
    print("时间轴库通过：写入/覆盖/按位置取用")

    print("全部测试通过 ✅")


if __name__ == "__main__":
    main()
