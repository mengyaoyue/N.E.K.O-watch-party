"""陪看猫娘 · 宿主语音输入监听（Windows 侧信道）

宿主的「屏幕共享」和「语音输入」是绑定的：用户把屏幕共享给猫娘看的时候，
语音输入也会一起开着。但宿主没给插件任何读取麦克风状态的接口
（语音状态只活在 Electron 渲染进程的内存里，插件 SDK / HTTP / 磁盘都摸不到），
所以这里绕一条系统侧信道——

Windows 的隐私授权记录（CapabilityAccessManager\\ConsentStore）在某个程序
“真正开始占用麦克风”时，会把 LastUsedTimeStart 写成此刻、并把
LastUsedTimeStop 落成 0；等它释放麦克风，才把 LastUsedTimeStop 写回真实时间。
于是 **LastUsedTimeStart > 0 且 LastUsedTimeStop == 0** 就等于“此刻正在用麦克风”。

只认 N.E.K.O.exe 自己的记录（按可执行文件路径匹配子键名），
Chrome / 微信 / 豆包等其它任何 App 用麦克风都不会误判。
非 Windows 或读不到记录时一律返回 False，静默降级，不影响插件其它功能。

注意：这是“最近一次占用是否还没结束”的近似判断。若宿主在使用麦克风时被强杀
（没来得及写回停止时间），记录会停留在“占用中”，下次它重新占用时才会刷新。
"""

from __future__ import annotations

import threading
from typing import Callable, Optional

_MIC_KEY = (
    r"SOFTWARE\Microsoft\Windows\CurrentVersion\CapabilityAccessManager"
    r"\ConsentStore\microphone"
)
_DEFAULT_KEYWORD = "N.E.K.O"
_SUBKEYS = ("NonPackaged", "Packaged")


def _decode_app(key: str) -> str:
    """注册表子键名里的路径用 # 代替反斜杠，还原成可读路径。"""
    return key.replace("#", "\\")


def mic_in_use(keyword: str = _DEFAULT_KEYWORD) -> bool:
    """宿主（默认 N.E.K.O.exe）此刻是否正占着麦克风。"""
    try:
        import winreg
    except Exception:
        return False

    want = str(keyword).upper()
    if not want:
        return False

    for sub in _SUBKEYS:
        try:
            root = winreg.OpenKey(winreg.HKEY_CURRENT_USER, _MIC_KEY + "\\" + sub)
        except OSError:
            continue
        try:
            index = 0
            while True:
                try:
                    name = winreg.EnumKey(root, index)
                except OSError:
                    break
                index += 1
                if want not in _decode_app(name).upper():
                    continue
                try:
                    app = winreg.OpenKey(root, name)
                except OSError:
                    continue
                try:
                    start = int(winreg.QueryValueEx(app, "LastUsedTimeStart")[0] or 0)
                    stop = int(winreg.QueryValueEx(app, "LastUsedTimeStop")[0] or 0)
                except OSError:
                    continue
                finally:
                    winreg.CloseKey(app)
                if start > 0 and stop == 0:
                    return True
        finally:
            winreg.CloseKey(root)
    return False


def voice_input_on(keyword: str = _DEFAULT_KEYWORD) -> bool:
    """语义化别名：宿主语音输入是否开着（等价于麦克风被宿主占着）。"""
    return mic_in_use(keyword)


class HostVoiceWatcher:
    """后台线程轮询宿主麦克风占用，状态真变化时才回调一次。

    回调跑在监听线程里，只用来改标记 / 叫醒调度，别在回调里做重活。
    """

    def __init__(
        self,
        on_change: Optional[Callable[[bool], None]] = None,
        interval: float = 3.0,
        keyword: str = _DEFAULT_KEYWORD,
    ) -> None:
        self._on_change = on_change
        self._interval = max(1.0, float(interval))
        self._keyword = keyword
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._active = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        # 先同步采一次：启动瞬间状态就是准的，不用等第一个轮询周期
        self._set(mic_in_use(self._keyword))
        self._thread = threading.Thread(target=self._run, daemon=True, name="neko-host-voice")
        self._thread.start()

    def _set(self, active: bool) -> None:
        with self._lock:
            self._active = bool(active)

    def is_active(self) -> bool:
        with self._lock:
            return self._active

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            active = mic_in_use(self._keyword)
            if active == self.is_active():
                continue
            self._set(active)
            cb = self._on_change
            if cb is None:
                continue
            try:
                cb(active)
            except Exception:
                pass

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        self._thread = None
