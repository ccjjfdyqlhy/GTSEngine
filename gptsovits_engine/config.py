"""
配置模型：用显式路径描述一次可移植的 GPT-SoVITS 部署

设计目标：只给 5 类信息就能在任何机器上跑推理
  1. GPT-SoVITS 项目根目录
  2. SoVITS 权重 (.pth)
  3. GPT 权重 (.ckpt)
  4. 参考音频 + 参考文本 (+ 语言)
  5. 可选：运行时 Python 解释器

除此之外的公共预训练资源 (BERT / CNHuBERT) 都从项目根自动发现，
也可以用 pretrained_root 指向独立存放的目录，实现代码与模型分离部署。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional

__all__ = ["EngineConfig", "ReferenceAudio", "SynthParams", "ConfigError"]


class ConfigError(ValueError):
    """配置有误"""


def _expand(p):
    if p is None or p == "":
        return None
    return os.path.abspath(os.path.expanduser(str(p)))


@dataclass
class ReferenceAudio:
    """参考音频 (音色克隆样本)。上游要求时长 3~10 秒。"""
    path: str
    text: str
    language: str = "zh"

    def __post_init__(self):
        self.path = _expand(self.path) or self.path
        if not self.text or not str(self.text).strip():
            raise ConfigError("reference.text 不能为空 (v3/v4/v5 模型必需)")
        if self.language not in ("zh", "en", "ja", "ko", "yue"):
            raise ConfigError("不支持的参考语言: %s" % self.language)

    def validate(self):
        if not os.path.isfile(self.path):
            raise ConfigError("参考音频不存在: %s" % self.path)


@dataclass
class EngineConfig:
    """一次推理所需的全部路径配置 (支持绝对、相对、~ 路径)"""

    project_root: str
    sovits_weights: str
    gpt_weights: str
    reference: Optional[ReferenceAudio] = None

    pretrained_root: Optional[str] = None
    python_executable: Optional[str] = None
    device: str = "cuda:0"
    is_half: bool = True
    version: Optional[str] = None
    extra_sys_path: list = field(default_factory=list)
    #: 冻结的合成参数 (None 表示使用引擎默认值)
    params: Optional["SynthParams"] = None

    def __post_init__(self):
        self.project_root = _expand(self.project_root)
        self.sovits_weights = _expand(self.sovits_weights)
        self.gpt_weights = _expand(self.gpt_weights)
        self.pretrained_root = _expand(self.pretrained_root)
        if isinstance(self.reference, dict):
            self.reference = ReferenceAudio(**self.reference)
        if isinstance(self.params, dict):
            self.params = SynthParams.from_dict(self.params)

    @property
    def pretrained_dir(self):
        """公共预训练资源根目录"""
        if self.pretrained_root:
            return self.pretrained_root
        return os.path.join(self.project_root, "GPT_SoVITS", "pretrained_models")

    def resolve_pretrained(self, *rel):
        """在 pretrained 根下解析资源，兼容两种常见布局"""
        p = os.path.join(self.pretrained_dir, *rel)
        if os.path.exists(p):
            return p
        alt = os.path.join(self.project_root, "GPT_SoVITS", "pretrained_models", *rel)
        return alt if os.path.exists(alt) else p

    @property
    def bert_path(self):
        return self.resolve_pretrained("chinese-roberta-wwm-ext-large")

    @property
    def cnhubert_path(self):
        return self.resolve_pretrained("chinese-hubert-base")

    def validate(self):
        """校验全部必需路径，失败抛 ConfigError 并列出所有问题"""
        problems = []
        if not os.path.isdir(self.project_root):
            problems.append("project_root 不存在: %s" % self.project_root)
        for name, p in (("sovits_weights", self.sovits_weights),
                        ("gpt_weights", self.gpt_weights)):
            if not os.path.isfile(p):
                problems.append("%s 不存在: %s" % (name, p))
        if self.reference is not None:
            try:
                self.reference.validate()
            except ConfigError as e:
                problems.append(str(e))
        for name, p in (("bert", self.bert_path), ("cnhubert", self.cnhubert_path)):
            if not os.path.isdir(p):
                problems.append("%s 预训练模型缺失: %s" % (name, p))
        if problems:
            raise ConfigError("配置校验失败:\n  " + "\n  ".join(problems))
        return self

    def to_dict(self):
        d = {
            "project_root": self.project_root,
            "sovits_weights": self.sovits_weights,
            "gpt_weights": self.gpt_weights,
            "pretrained_root": self.pretrained_root,
            "python_executable": self.python_executable,
            "device": self.device,
            "is_half": self.is_half,
            "version": self.version,
            "extra_sys_path": list(self.extra_sys_path),
        }
        if self.reference is not None:
            d["reference"] = {"path": self.reference.path,
                              "text": self.reference.text,
                              "language": self.reference.language}
        if self.params is not None:
            d["params"] = self.params.to_dict()
        return d

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        ref = d.pop("reference", None)
        if isinstance(ref, dict):
            ref = ReferenceAudio(**ref)
        # params 由 __post_init__ 解析
        return cls(reference=ref, **d)


    # ------------------------------------------------------ 配置文件读写
    #: 配置文件格式版本, 便于将来迁移
    CONFIG_VERSION = 1

    def save(self, path, portable=False, indent=2):
        """把当前配置(含冻结的合成参数)写入 JSON 配置文件

        Parameters
        ----------
        portable :
            True 时把所有路径转为相对于配置文件所在目录的相对路径，
            便于连同模型一起打包搬迁到别的机器；False 保留绝对路径。
        """
        path = os.path.abspath(os.path.expanduser(str(path)))
        payload = self.to_dict()
        payload["config_version"] = self.CONFIG_VERSION

        if portable:
            base = os.path.dirname(path) or "."

            def rel(p):
                if not p:
                    return p
                try:
                    return os.path.relpath(p, base)
                except ValueError:
                    return p

            for k in ("project_root", "sovits_weights", "gpt_weights",
                      "pretrained_root"):
                if payload.get(k):
                    payload[k] = rel(payload[k])
            if payload.get("reference"):
                payload["reference"]["path"] = rel(payload["reference"]["path"])
            payload["_portable"] = True

        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        # 先写临时文件再替换, 避免写一半断电留下坏配置
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=indent)
            f.write("\n")
        os.replace(tmp, path)
        return path

    @classmethod
    def load(cls, path, overrides=None):
        """从 JSON 配置文件加载配置

        portable 配置中的相对路径会基于配置文件所在目录解析。
        `overrides` 可以覆盖任意顶层字段(如临时换 device)，
        但不会写回文件。
        """
        path = os.path.abspath(os.path.expanduser(str(path)))
        if not os.path.isfile(path):
            raise ConfigError("配置文件不存在: %s" % path)
        with open(path, "r", encoding="utf-8") as f:
            try:
                payload = json.load(f)
            except ValueError as e:
                raise ConfigError("配置文件不是合法 JSON (%s): %s" % (path, e))
        if not isinstance(payload, dict):
            raise ConfigError("配置文件顶层必须是对象: %s" % path)

        ver = payload.pop("config_version", None)
        if ver is not None and ver > cls.CONFIG_VERSION:
            raise ConfigError(
                "配置版本 %s 高于本引擎支持的 %s，请升级 gptsovits-engine" % (
                    ver, cls.CONFIG_VERSION))
        is_portable = payload.pop("_portable", False)

        # 便携配置: 相对路径 -> 基于配置文件目录的绝对路径
        if is_portable:
            base = os.path.dirname(path)

            def abs_(p):
                if not p or os.path.isabs(p):
                    return p
                return os.path.normpath(os.path.join(base, p))

            for k in ("project_root", "sovits_weights", "gpt_weights",
                      "pretrained_root"):
                if payload.get(k):
                    payload[k] = abs_(payload[k])
            if payload.get("reference"):
                payload["reference"]["path"] = abs_(payload["reference"]["path"])

        if overrides:
            unknown = [k for k in overrides if k not in cls.__dataclass_fields__]
            if unknown:
                raise ConfigError("未知的覆盖字段: %s" % ", ".join(unknown))
            payload.update({k: v for k, v in overrides.items() if v is not None})

        known = [k for k in payload if k in cls.__dataclass_fields__]
        ignored = [k for k in payload if k not in cls.__dataclass_fields__]
        if ignored:
            raise ConfigError("配置文件包含未知字段: %s" % ", ".join(sorted(ignored)))
        return cls(**{k: payload[k] for k in known})

    def with_params(self, **overrides):
        """返回一个派生配置：冻结的合成参数被部分覆盖"""
        base = self.params or SynthParams()
        merged = base.merged(**overrides)
        return EngineConfig(
            project_root=self.project_root,
            sovits_weights=self.sovits_weights,
            gpt_weights=self.gpt_weights,
            reference=self.reference,
            pretrained_root=self.pretrained_root,
            python_executable=self.python_executable,
            device=self.device,
            is_half=self.is_half,
            version=self.version,
            extra_sys_path=list(self.extra_sys_path),
            params=merged,
        )

    def __repr__(self):
        return ("EngineConfig(version=%r, device=%r, sovits=%s, gpt=%s, params=%r)" % (
            self.version, self.device,
            os.path.basename(self.sovits_weights),
            os.path.basename(self.gpt_weights), self.params))

# ==================================================================== 合成参数冻结
@dataclass
class SynthParams:
    """合成参数集 —— 可冻结到配置文件后复用

    这些参数不涉及"用什么模型/什么音色"，只影响"这一次怎么合成"，
    例如采样步数、切句方式、语速、随机种子等。

    典型用法：调好一组参数后冻结成 JSON，之后直接加载复用，
    不用每次在代码里重复写一堆关键字参数。
    """

    #: 切句方式: cut0(不切) / cut1 / cut5 等，影响批处理效果
    text_split_method: str = "cut5"
    #: 推理批次; None 表示沿用引擎锁定的 batch
    batch_size: Optional[int] = None
    #: 声码器采样步数 (v5dev 默认 32, v5turbo 默认 4)
    sample_steps: Optional[int] = None
    #: CFG 强度 (v5dev 默认 1.30, v5turbo 默认 0.0)
    cfg_rate: Optional[float] = None
    #: 采样参数
    top_k: int = 15
    top_p: float = 1.0
    temperature: float = 1.0
    #: 语速倍率, 1.0 为原速
    speed_factor: float = 1.0
    #: 随机种子
    seed: int = 42
    #: 重复惩罚
    repetition_penalty: float = 1.35
    #: 是否流式输出 (仅 synthesize 录制时会忽略)
    streaming: bool = False

    #: 允许出现在配置文件中的键, 用于校验未知字段
    FIELDS = (
        "text_split_method", "batch_size", "sample_steps", "cfg_rate",
        "top_k", "top_p", "temperature", "speed_factor", "seed",
        "repetition_penalty", "streaming",
    )

    def to_dict(self):
        """只输出非默认值的字段，保持配置文件简洁"""
        from dataclasses import fields as _fields
        out = {}
        defaults = SynthParams()
        for f in _fields(self):
            val = getattr(self, f.name)
            if val != getattr(defaults, f.name):
                out[f.name] = val
        return out

    @classmethod
    def from_dict(cls, d):
        """从 dict 构造，遇到未知字段直接报错，避免拼写错误被静默忽略"""
        if not isinstance(d, dict):
            raise ConfigError("合成参数必须是对象(dict)")
        unknown = [k for k in d if k not in cls.FIELDS]
        if unknown:
            raise ConfigError(
                "未知的合成参数: %s (可用: %s)" % (", ".join(sorted(unknown)),
                                                   ", ".join(cls.FIELDS)))
        return cls(**d)

    def merged(self, **overrides):
        """基于当前参数派生一份，只覆盖显式给出的非 None 值"""
        from dataclasses import replace
        clean = {k: v for k, v in overrides.items()
                 if v is not None and k in self.FIELDS}
        return replace(self, **clean) if clean else self

    def __repr__(self):
        diff = self.to_dict()
        inner = ", ".join("%s=%r" % (k, v) for k, v in diff.items())
        return "SynthParams(%s)" % inner if inner else "SynthParams()"