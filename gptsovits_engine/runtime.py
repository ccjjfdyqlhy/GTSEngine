"""
运行时引导 —— 让 GPT-SoVITS 从任意目录/任意机器加载

这是整个包最关键的一层。上游 GPT-SoVITS 有三处硬依赖当前工作目录(CWD):
  1. TTS.py 模块顶部执行 now_dir = os.getcwd(), 声码器路径基于它拼接
  2. sv.py 用 sys.path.append(f"{os.getcwd()}/GPT_SoVITS/eres2net") 做扁平导入
  3. 项目内部大量 from module.x / from AR.x / from tools.x 的扁平导入,
     只有在项目根位于 sys.path 时才成立

本模块的做法: 在导入上游模块前,
  - 把项目根加入 sys.path
  - 临时把 CWD 切到项目根 (保证 os.getcwd() 拿到正确值)
  - 补上 GPT_SoVITS / GPT_SoVITS/eres2net 等子目录
并提供 cwd_guard() 上下文管理器, 在推理期间保持 CWD 正确, 结束后恢复,
从而不干扰宿主程序自己的相对路径逻辑。
"""
from __future__ import annotations

import contextlib
import os
import sys
import threading
from typing import Optional

__all__ = ["RuntimeBootstrap", "bootstrap", "cwd_guard", "get_runtime"]

_lock = threading.RLock()
_runtime: Optional["RuntimeBootstrap"] = None


class RuntimeBootstrapError(RuntimeError):
    """运行时引导失败"""


class RuntimeBootstrap:
    """管理 GPT-SoVITS 项目根的 sys.path / CWD 注入与还原

    Parameters
    ----------
    project_root:
        GPT-SoVITS 项目根目录 (含 GPT_SoVITS/ 子目录与 config.py)。
        也可以指向一个只放了 GPT_SoVITS/ 和 pretrained_models/ 的精简部署目录。
    extra_paths:
        额外需要加入 sys.path 的目录。
    """

    #: 项目内部需要额外加入 sys.path 的子目录 (扁平导入所需)
    INTERNAL_SUBDIRS = (
        "GPT_SoVITS",
        "GPT_SoVITS/eres2net",
    )

    def __init__(self, project_root: str, extra_paths=()):
        root = os.path.abspath(os.path.expanduser(project_root))
        if not os.path.isdir(root):
            raise RuntimeBootstrapError(f"GPT-SoVITS 项目根目录不存在: {root}")
        self.project_root = root
        self.extra_paths = [os.path.abspath(os.path.expanduser(p)) for p in extra_paths]
        self._applied = False

    # ------------------------------------------------------ 校验
    def validate(self) -> list:
        """检查目录结构是否完整, 返回问题列表 (空表示通过)"""
        problems = []
        required = [
            "GPT_SoVITS",
            "GPT_SoVITS/TTS_infer_pack",
            "GPT_SoVITS/pretrained_models",
        ]
        for rel in required:
            if not os.path.isdir(os.path.join(self.project_root, rel)):
                problems.append(f"缺少目录: {rel}")
        return problems

    # ------------------------------------------------------ 应用
    def apply(self) -> "RuntimeBootstrap":
        """幂等地注入 sys.path 并把 CWD 切到项目根"""
        with _lock:
            if self._applied:
                return self
            problems = self.validate()
            if problems:
                raise RuntimeBootstrapError(
                    "GPT-SoVITS 目录结构不完整:\n  " + "\n  ".join(problems)
                )

            # 1) sys.path: 项目根 + 内部子目录 + 额外路径
            paths = [self.project_root]
            paths += [os.path.join(self.project_root, s) for s in self.INTERNAL_SUBDIRS]
            paths += self.extra_paths
            for p in reversed(paths):
                if os.path.isdir(p) and p not in sys.path:
                    sys.path.insert(0, p)

            # 2) 上游用 os.getcwd() 拼模型路径。这里做一次 chdir 验证可用性,
            #    然后立即恢复 —— 真正的切换由 cwd_guard() 在推理期间按需完成,
            #    这样宿主程序的 CWD 不会被引擎永久劫持。
            saved = os.getcwd()
            try:
                os.chdir(self.project_root)
            except OSError as e:
                raise RuntimeBootstrapError(f"无法切换到项目根 {self.project_root}: {e}")
            finally:
                try:
                    os.chdir(saved)
                except OSError:
                    pass

            self._applied = True
            return self

    def restore(self):
        """标记运行时失效 (sys.path 不回滚, 避免已导入模块失效)

        apply() 本身已不再修改宿主 CWD, 因此这里无需切换目录 ——
        CWD 的正确性完全由 cwd_guard() 在需要时保证。
        """
        with _lock:
            self._applied = False

    # ------------------------------------------------------ 上下文
    @contextlib.contextmanager
    def cwd_guard(self):
        """在 with 块内保证 CWD 位于项目根, 退出后恢复命中前的 CWD

        用于推理调用: 上游某些路径在运行时才读取 CWD (如声码器加载),
        宿主程序可能在这期间切换过目录。
        """
        with _lock:
            saved = os.getcwd()
            need = os.path.abspath(saved) != os.path.abspath(self.project_root)
            if need:
                os.chdir(self.project_root)
            try:
                yield
            finally:
                if need and os.path.isdir(saved):
                    try:
                        os.chdir(saved)
                    except OSError:
                        pass

    def __repr__(self):
        return f"RuntimeBootstrap(project_root={self.project_root!r}, applied={self._applied})"


def bootstrap(project_root: str, extra_paths=()) -> RuntimeBootstrap:
    """全局唯一入口: 完成引导并缓存 (多次调用幂等)"""
    global _runtime
    with _lock:
        root = os.path.abspath(os.path.expanduser(project_root))
        if _runtime is not None:
            # 同一个根就复用; 换根则重建
            if os.path.abspath(_runtime.project_root) == root:
                return _runtime.apply()
            _runtime.restore()
        _runtime = RuntimeBootstrap(root, extra_paths)
        return _runtime.apply()


def get_runtime() -> RuntimeBootstrap:
    """获取已初始化的运行时 (未初始化则抛错)"""
    if _runtime is None:
        raise RuntimeBootstrapError("尚未调用 bootstrap(); 请先初始化 GTSEngine")
    return _runtime


def cwd_guard():
    """便捷函数: 用当前运行时的 cwd_guard"""
    return get_runtime().cwd_guard()