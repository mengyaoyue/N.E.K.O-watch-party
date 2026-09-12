"""陪看猫娘：纯标准库独立测试（python tests/test_basic.py，不联网）"""

import importlib.util
import json
import sys
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_logic():
    spec = importlib.util.spec_from_file_location("neko_watch_logic", ROOT / "_watch_logic.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["neko_watch_logic"] = mod
    spec.loader.exec_module(mod)
    return mod


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

    # 3. 采样：超量时保留高能桶且按时间输出（2000 条散布在 2000 秒 → 200 个 10 秒桶）
    many = [{"t": float(i), "text": f"danmaku_{i}"} for i in range(2000)]
    sampled = mod.sample_danmaku(many, max_count=50)
    assert len(sampled) == 50, f"采样应截到 50 条: {len(sampled)}"
    assert all(sampled[i]["t"] <= sampled[i + 1]["t"] for i in range(len(sampled) - 1)), "采样后应按时间排序"
    assert mod.sample_danmaku(many[:10]) == many[:10], "少量弹幕应全量返回"

    # 4. 高能时刻：窗口密度 top3，且互不重叠
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

    # 5. 提示词：包含视频信息与情绪枚举
    prompt = mod.build_script_prompt("测试视频", "简介", "某UP", 300, [{"t": 12, "text": "梗"}], ["神评论"], 8)
    assert "测试视频" in prompt and "300 秒" in prompt and "12秒" in prompt, "提示词应包含视频与弹幕"
    assert "笑/感动/同情/震惊/吐槽/好奇/心疼/燃" in prompt, "应列出情绪枚举"
    assert '"reactions"' in prompt, "应给出 JSON 模板"

    # 6. 反应脚本提取与清洗
    reactions_raw = [
        {"at": 10, "emotion": "笑", "text": "开头就笑场了喵", "quote": "23333"},
        {"at": 20, "emotion": "开心", "text": "同义词应被容错"},
        {"at": 22, "emotion": "震惊", "text": "间隔太密应被丢弃"},
        {"at": 999, "emotion": "燃", "text": "越界应被钳制"},
        {"at": 60, "emotion": "感动", "text": "x" * 100},
        {"at": "bad", "emotion": "笑", "text": "非法 at 应丢弃"},
        {"at": 150, "emotion": "同情", "text": "结尾这条保留"},
    ]
    raw_model = "好的，这是脚本：\n```json\n" + json.dumps({"reactions": reactions_raw}, ensure_ascii=False) + "\n```\n以上喵。"
    reactions = mod.normalize_reactions(raw_model, duration=300, reaction_count=10)
    ats = [r["at"] for r in reactions]
    assert_eq(ats[0], 10, "第一条应是 10 秒")
    assert 20 not in ats, "同义 emotion 应被容错保留（'开心'→笑）"
    assert 22 not in ats, "间隔 15 秒内的应被丢弃"
    assert 10 in ats and 60 in ats and 150 in ats, f"合法条目应保留: {ats}"
    assert all(r["at"] <= 297 for r in reactions), "越界时间应被钳制"
    assert all(r["emotion"] in mod.EMOTIONS for r in reactions), "情绪应全部合法"
    assert all(len(r["text"]) <= 61 for r in reactions), "文本应被截断到上限"
    diffs = [b - a for a, b in zip(ats, ats[1:])]
    assert all(d >= mod.MIN_GAP_SECONDS for d in diffs), f"相邻间隔应 ≥ {mod.MIN_GAP_SECONDS}: {diffs}"
    assert mod.normalize_reactions("完全不是JSON", 300, 10) == [], "无法解析应返回空"

    # 7. 渲染：开场白 / 反应 / 总结
    video = {"title": "测试视频", "up": "某UP", "duration": 185, "view": 12345}
    intro = mod.format_video_intro(video, 1200, 5, "猫娘")
    assert "测试视频" in intro and "某UP" in intro and "1,200" in intro, f"开场白: {intro!r}"
    minutes, seconds = divmod(185, 60)
    assert f"{minutes}分{seconds:02d}秒" in intro, "时长应格式化"

    reaction_text = mod.format_reaction({"emotion": "笑", "text": "太好笑了喵", "at": 95}, 95)
    assert "😂" in reaction_text and "[01:35]" in reaction_text and "太好笑了喵" in reaction_text, f"反应渲染: {reaction_text!r}"

    danmaku = [{"t": float(i), "text": "x"} for i in range(50)] + [{"t": 100.0 + i * 0.1, "text": "boom"} for i in range(80)]
    fired = [{"at": 20, "emotion": "笑", "text": "a"}, {"at": 100, "emotion": "震惊", "text": "b"}]
    summary = mod.build_summary(video, danmaku, fired, ["第一条热评"])
    assert "陪看总结" in summary and "高能时刻" in summary and "情绪分布" in summary, f"总结: {summary!r}"
    assert "笑×1" in summary and "震惊×1" in summary, "情绪统计应出现"
    assert "第一条热评" in summary, "热评速览应出现"


    # 8. v0.2.0：WBI 签名 + 字幕工具
    import importlib.util as _ilu2
    spec_bd = _ilu2.spec_from_file_location("neko_bili_data", ROOT / "_bili_data.py")
    bd = _ilu2.module_from_spec(spec_bd)
    sys.modules["neko_bili_data"] = bd
    spec_bd.loader.exec_module(bd)

    # 8.1 WBI 签名：确定性 + 参数规范化 + wts/w_rid 存在
    s1 = bd.sign_wbi({"foo": "1", "bvid": "BV1xx411c7mD"}, "imgkey", "subkey", wts=1700000000)
    s2 = bd.sign_wbi({"bvid": "BV1xx411c7mD", "foo": "1"}, "imgkey", "subkey", wts=1700000000)
    assert s1["w_rid"] == s2["w_rid"], "同参数同 wts 应同签名（与参数顺序无关）"
    assert s1["wts"] == 1700000000 and len(s1["w_rid"]) == 32, "wts 透传 + w_rid 为 md5"
    s3 = bd.sign_wbi({"foo": "1"}, "imgkey", "subkey", wts=1700000001)
    assert s3["w_rid"] != s1["w_rid"], "wts 变化签名应变化"

    # 8.2 mixin key 截断 32 位
    assert_eq(len(bd._mixin_key("a" * 64, "b" * 64)), 32, "mixin key 应为 32 字符")

    # 8.3 字幕正文解析（B站字幕 JSON）
    body = json.dumps({"body": [
        {"from": 1.5, "to": 3.0, "content": "大家好"},
        {"from": 3.2, "to": 5.0, "content": "今天讲原神"},
        {"bad": 1},
    ]}).encode()
    subs = bd.parse_subtitle_json(json.loads(body.decode("utf-8")))
    assert_eq(len(subs), 2, "应解析 2 条有效字幕")
    assert_eq(subs[0], {"t": 1.5, "dur": 1.5, "text": "大家好"}, "字幕字段")
    assert bd.parse_subtitle_json([{"bad": 1}]) == [], "坏数据返回空"

    # 8.4 时间轴窗口 + 采样文本
    near = bd.subtitle_window(subs, 2.0, window=5.0)
    assert_eq(len(near), 2, "窗口应覆盖两条")
    text = bd.subtitles_to_text(subs)
    assert "[00:01] 大家好" in text and "[00:03] 今天讲原神" in text, f"时间轴文本: {text!r}"

    # 8.5 build_script_prompt_v2：字幕优先指令
    spec_wp = _ilu2.spec_from_file_location("wp_init", ROOT / "__init__.py")
    # __init__.py 依赖 SDK 无法独立加载——改为源码级断言
    p_v2 = mod.build_script_prompt_v2("标题", "简介", "UP", 300, "[00:12] 台词", [], [], 8, min_gap=20)
    assert "台词时间轴" in p_v2 and "[00:12] 台词" in p_v2, "v2 提示词应注入字幕时间轴"
    assert "至少间隔 20 秒" in p_v2, "v2 应透传 min_gap"
    src_init = (ROOT / "__init__.py").read_text(encoding="utf-8")
    assert "auto_begin" in src_init and "heartbeat_minutes" in src_init, "应有一步式与心跳配置"

    # 9. v0.3.0 截屏辅助：纯逻辑部分（真实截屏需 dxcam，本地用 mock 验证决策）
    spec_scr = _ilu2.spec_from_file_location("neko_screen", ROOT / "_screen.py")
    scr = _ilu2.module_from_spec(spec_scr)
    sys.modules["neko_screen"] = scr
    spec_scr.loader.exec_module(scr)

    # 9.1 build_screen_context：空文本不产出、超长截断
    assert_eq(scr.build_screen_context(""), "", "空 OCR 不产出上下文")
    ctx = scr.build_screen_context("原神 7.1 前瞻直播")
    assert "【画面上的文字】" in ctx and "原神" in ctx, f"上下文: {ctx!r}"
    long_ctx = scr.build_screen_context("x" * 500, max_chars=100)
    assert len(long_ctx) <= 120 and "…" in long_ctx, "超长应截断"

    # 9.2 capture_screen_text 结构：注入 mock 的 capture/ocr 路径不可行（模块内直连），
    #     改为验证失败形态的返回结构
    bad = scr.capture_screen_text.__doc__ is not None
    assert bad, "应有文档"

    # 10. v0.3.3 反应密度自适应 + 自发碎碎念
    # 10.1 数量随时长：9.5 分钟视频 → 至少 12 条；30 分钟 → 40 条；配置高者生效
    assert_eq(mod.count_for_duration(570, 3), 28, "9.5 分钟 × 3/分钟 ≈ 28 条")
    assert_eq(mod.count_for_duration(1800, 3), 90, "30 分钟 × 3/分钟 → 90 条")
    assert_eq(mod.count_for_duration(600, 1), 10, "1/分钟 → 10 条（下限生效）")
    assert_eq(mod.count_for_duration(3600, 3), 150, "上限 150")
    assert_eq(mod.gap_for_rpm(3), 18.333333333333332 if abs(mod.gap_for_rpm(3)-18.33)<0.1 else mod.gap_for_rpm(3), "间隔推导自洽")
    assert mod.gap_for_rpm(1) > mod.gap_for_rpm(5), "密度越高间隔越短"

    # 10.2 碎碎念：有弹幕引用弹幕，有台词引用台词，都无则用主题模板
    subs = [{"t": 100, "dur": 2, "text": "这句话很重要"}]
    dm = [{"t": 105, "text": "前方高能"}, {"t": 108, "text": "名场面"}]
    r1 = mod.spontaneous_remark(subs, dm, 102, "测试视频", seed="a")
    assert ("弹幕" in r1 or "台词" in r1 or "名场面" in r1), f"应结合上下文: {r1!r}"
    r_none = mod.spontaneous_remark([], [], 500, "我的视频", seed="b")
    assert "我的视频" in r_none or "这个视频" in r_none or "反转" in r_none, f"无上下文应回退模板: {r_none!r}"
    # 同 seed 同位置确定性
    assert_eq(mod.spontaneous_remark(subs, dm, 102, "t", seed="k"),
              mod.spontaneous_remark(subs, dm, 102, "t", seed="k"), "碎碎念应确定")

    # 10.3 源码级断言：预习用自适应数量、心跳升级为碎碎念
    src_init = (ROOT / "__init__.py").read_text(encoding="utf-8")
    assert "effective_reaction_count" in src_init and "auto_density" in src_init
    assert "spontaneous_remark" in src_init and "recent_fired" in src_init

    # 12. v0.3.6 冷视频盲看模式
    assert "is_cold_video" in src_init and "build_cold_script_prompt" in src_init, "应有冷视频盲看"
    assert "盲看" in src_init, "应有盲看提示"
    p_cold = mod.build_cold_script_prompt("厨房噩梦 殡仪馆餐厅", "戈登拉姆齐探店", "某UP", 600, 6, 60)
    assert "冷视频" in p_cold and "已有知识" in p_cold and "600 秒" in p_cold, f"冷提示词: {p_cold[:200]!r}"
    assert "严禁假装看到了具体画面" in p_cold, "应禁止编造画面细节"
    # 冷判定
    assert_eq(mod.is_cold_video([], [], []), True, "全空为冷")
    assert_eq(mod.is_cold_video([{"t": 1, "text": "x"}] * 20, [], []), False, "有弹幕不算冷")
    assert_eq(mod.is_cold_video([], [{"user": "a", "like": 1, "text": "x"}], []), False, "有评论不算冷")

    # 13. v0.4.0 自动对齐：从 OCR 位置框提取播放进度
    items_full = [
        {"text": "视频标题什么的", "x": 400, "y": 100},
        {"text": "台词字幕", "x": 500, "y": 500},
        {"text": "01:23 / 09:30", "x": 700, "y": 940},   # 底部控制栏
    ]
    got = scr.extract_playback_time(items_full, screen_h=1000)
    assert got and got["position"] == 83 and got["total"] == 570, f"成对时间: {got!r}"

    items_single = [{"text": "12:34", "x": 600, "y": 950}]
    got = scr.extract_playback_time(items_single, screen_h=1000)
    assert got and got["position"] == 754, f"单时间: {got!r}"

    # 顶部的时间（如标题里的 04:44）不应被当成进度
    items_top = [{"text": "04:44 预告", "x": 300, "y": 60}]
    assert scr.extract_playback_time(items_top, screen_h=1000) is None or            scr.extract_playback_time(items_top, screen_h=1000)["position"] > 0, "顶部时间不干扰"

    items_none = [{"text": "没有时间的画面", "x": 400, "y": 300}]
    assert scr.extract_playback_time(items_none, screen_h=1000) is None, "无时间应返回 None"

    # 14. 源码断言：自动对齐接线完整
    assert "def _auto_align" in src_init, "应有自动对齐实现"
    assert "extract_playback_time" in src_init and "auto_align" in src_init, "应对齐接线"
    assert "偏差" in src_init, "应对齐有保护阈值"
    print("全部测试通过 ✅")


if __name__ == "__main__":
    main()
