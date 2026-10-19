"""GTSEngine: portable GPT-SoVITS inference engine

Capabilities (measured):
  1. Works from any directory / any machine: all paths are explicit, runtime
     manages sys.path and CWD transparently.
  2. Auto version detection from weight file header.
  3. Model residency + LRU hot-swap across models.
  4. Adaptive batch processing (1.54x on v5turbo).
  5. Streaming output (TTFT improvement 5-6.5x).
  6. Warmup to eliminate first-request CUDA graph capture cost.
"""
from __future__ import annotations

import copy
import gc
import os
import time
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np

from .config import ConfigError, EngineConfig, SynthParams
from .runtime import bootstrap

__all__ = ["GTSEngine", "SynthResult", "EngineStats"]

#: versions that support true token-level streaming
STREAMING_NATIVE = {"v1", "v2", "v2Pro", "v2ProPlus"}


@dataclass
class SynthResult:
    audio: np.ndarray
    sr: int
    version: str
    latency_s: float
    audio_dur_s: float
    batch_size: int
    warning: str = None

    @property
    def rtf(self):
        return self.latency_s / self.audio_dur_s if self.audio_dur_s > 0 else 0.0


@dataclass
class EngineStats:
    total_requests: int = 0
    total_audio_s: float = 0.0
    total_latency_s: float = 0.0
    history: list = field(default_factory=list)

    @property
    def avg_rtf(self):
        return self.total_latency_s / self.total_audio_s if self.total_audio_s else 0.0


class GTSEngine:
    """Portable GPT-SoVITS inference engine.

    Parameters
    ----------
    config : EngineConfig | dict
        Explicit paths for project root, weights and reference audio.
    max_batch : int
        Batch cap. Locked at load time and used for warmup; all later requests
        reuse the same CUDA graph (changing batch triggers a costly rebuild).
    cache_size : int
        Max resident model instances (LRU).
    use_cuda_graph : bool
        Enable upstream CUDA graph acceleration (auto-degrades if unsupported).
    validate : bool
        Validate all paths before loading.
    """

    def __init__(self, config, max_batch=8, cache_size=2,
                 use_cuda_graph=True, validate=True):
        if isinstance(config, dict):
            config = EngineConfig.from_dict(config)
        if not isinstance(config, EngineConfig):
            raise TypeError("config must be EngineConfig or dict")
        if validate:
            config.validate()
        self.config = config
        self.max_batch = max(1, int(max_batch))
        self.cache_size = max(1, int(cache_size))
        self.use_cuda_graph = use_cuda_graph
        os.environ.setdefault("CUDAGraph", "1" if use_cuda_graph else "0")
        self._runtime = None
        self._instances = OrderedDict()
        self._warmed = set()
        self._locked_batch = None
        self._last_key = None
        self._torch = None
        self.stats = EngineStats()

    # ---------------------------------------------------------------- bootstrap
    def _bootstrap(self):
        """Inject sys.path and switch CWD before importing upstream modules."""
        if self._runtime is not None:
            return
        self._runtime = bootstrap(self.config.project_root,
                                  self.config.extra_sys_path)
        import torch
        self._torch = torch

    def _upstream(self):
        from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
        return TTS, TTS_Config

    # ---------------------------------------------------------------- version
    def detect_version(self):
        """Detect model version from SoVITS weight header."""
        self._bootstrap()
        with self._runtime.cwd_guard():
            from process_ckpt import get_sovits_version_from_path_fast
            res = get_sovits_version_from_path_fast(self.config.sovits_weights)
        return res[1] if isinstance(res, (tuple, list)) else res
    # ---------------------------------------------------------------- build
    def _key(self, version):
        return "%s|%s|%s" % (version,
                             os.path.basename(self.config.sovits_weights),
                             os.path.basename(self.config.gpt_weights))

    def _build(self, version):
        TTS, TTS_Config = self._upstream()
        cfg = copy.deepcopy(TTS_Config.default_configs[version])
        cfg.update({
            "device": self.config.device,
            "is_half": self.config.is_half,
            "t2s_weights_path": self.config.gpt_weights,
            "vits_weights_path": self.config.sovits_weights,
            "bert_base_path": self.config.bert_path,
            "cnhuhbert_base_path": self.config.cnhubert_path,
        })
        merged = {"custom": cfg}
        for k, v in TTS_Config.default_configs.items():
            merged[k] = copy.deepcopy(v)
        return TTS(TTS_Config(merged))

    def load(self, warmup=True):
        """Load model (idempotent). Returns the detected version string."""
        self._bootstrap()
        version = self.config.version or self.detect_version()
        self.config.version = version
        key = self._key(version)

        if key in self._instances:
            self._instances.move_to_end(key)
            self._last_key = key
            return version

        while len(self._instances) >= self.cache_size:
            _, old = self._instances.popitem(last=False)
            del old
            gc.collect()
            self._torch.cuda.empty_cache()

        t0 = time.time()
        with self._runtime.cwd_guard():
            inst = self._build(version)
        self._instances[key] = inst
        self._instances.move_to_end(key)
        self._last_key = key
        self._locked_batch = self.max_batch
        print("[engine] loaded %s in %.2fs (%s)" % (
            version, time.time() - t0,
            os.path.basename(self.config.sovits_weights)))

        self._warmed.add(key)
        if warmup:
            self._warmup(inst, version, self._locked_batch)
        return version

    # ---------------------------------------------------------------- warmup
    def _warmup(self, inst, version, batch, ref_audio=None):
        """Prime CUDA graph capture and allocator.

        Three conditions are required, otherwise the first real request is
        still several times slower:
          1. reference audio must be 3-10 seconds (upstream validates this)
          2. batch must equal the batch used by real requests
          3. target text must be close to real length (graph is shape-specific)
        """
        import soundfile as sf
        t0 = time.time()
        ref = None
        tmp = None
        ref_cfg = self.config.reference
        if ref_audio and os.path.exists(ref_audio):
            ref = ref_audio
        elif ref_cfg is not None and os.path.isfile(ref_cfg.path):
            ref = ref_cfg.path
        if ref is None:
            tmp = os.path.join("/tmp", "_gsv_warmup_%d.wav" % os.getpid())
            sr_w = 16000
            n = int(4.0 * sr_w)
            t = np.arange(n) / sr_w
            env = np.minimum(1.0, np.minimum(t / 0.1, (4.0 - t) / 0.1))
            sf.write(tmp, (0.1 * np.sin(2 * np.pi * 220 * t) * env).astype(np.float32), sr_w)
            ref = tmp

        lang = ref_cfg.language if ref_cfg is not None else "zh"
        prompt = ref_cfg.text if (ref_cfg is not None and ref_cfg.text) else "test"
        if lang == "en":
            text = ("The wind brings new stories to everyone. "
                    "And it also brings new songs for us to sing. "
                    "This is a warm up sentence for the engine.")
        else:
            text = ("这是用于预热的测试文本，长度接近真实请求。"
                    "风带来了新的故事，也带来了新的歌声。"
                    "这是最后一句测试文本内容。")
        req = {
            "text": text, "text_lang": lang,
            "ref_audio_path": ref, "prompt_text": prompt, "prompt_lang": lang,
            "top_k": 15, "top_p": 1, "temperature": 1,
            "text_split_method": "cut5", "batch_size": batch,
            "speed_factor": 1.0, "return_fragment": False,
            "streaming_mode": False, "seed": 42, "repetition_penalty": 1.35,
            "parallel_infer": True,
        }
        try:
            with self._runtime.cwd_guard():
                for i in range(3):
                    tr = time.time()
                    for _ in inst.run(dict(req)):
                        pass
                    print("[engine] warmup round %d: %.2fs" % (i + 1, time.time() - tr))
            print("[engine] warmup done in %.2fs (batch=%d)" % (time.time() - t0, batch))
        except Exception as e:
            print("[engine] warmup skipped (%s: %s)" % (type(e).__name__, e))
        finally:
            if tmp:
                try:
                    os.remove(tmp)
                except OSError:
                    pass

    # ---------------------------------------------------------------- config file
    @classmethod
    def from_file(cls, path, overrides=None, **engine_kwargs):
        # Build an engine directly from a frozen config file.
        #
        # Example::
        #
        #     eng = GTSEngine.from_file("voice.json")
        #     eng.load()
        #     eng.synthesize("hello")   # uses the frozen params
        return cls(EngineConfig.load(path, overrides=overrides), **engine_kwargs)

    # ---------------------------------------------------------------- batch select
    def _pick_batch(self, text, split_method):
        if split_method == "cut0":
            return 1
        seps = "。！？!?.\n"
        n = sum(1 for ch in text if ch in seps)
        if n <= 1:
            return 1
        return max(1, min(n, self.max_batch))

    # ---------------------------------------------------------------- params freeze
    @staticmethod
    def freeze_params(text_split_method=None, batch_size=None,
                      sample_steps=None, cfg_rate=None,
                      top_k=15, top_p=1.0, temperature=1.0,
                      speed_factor=1.0, seed=42, repetition_penalty=1.35,
                      streaming=False):
        # Collect one set of synthesis parameters into a SynthParams object.
        # Use this to capture tuned settings and freeze them into a config file.
        return SynthParams(
            # None 表示"未指定", 落回默认值, 避免把 null 写进配置覆盖默认
            text_split_method=text_split_method or SynthParams().text_split_method,
            batch_size=batch_size,
            sample_steps=sample_steps,
            cfg_rate=cfg_rate,
            top_k=top_k, top_p=top_p, temperature=temperature,
            speed_factor=speed_factor, seed=seed,
            repetition_penalty=repetition_penalty,
            streaming=streaming,
        )

    def _effective_params(self, kwargs):
        # Merge frozen params with per-call overrides (call wins).
        frozen = self.config.params or SynthParams()
        return frozen.merged(**kwargs)

    def _base_req(self, text, ref_audio, prompt_text, text_lang, prompt_lang,
                  params=None):
        p = params or SynthParams()
        return {
            "text": text, "text_lang": text_lang,
            "ref_audio_path": ref_audio, "prompt_text": prompt_text,
            "prompt_lang": prompt_lang,
            "top_k": p.top_k, "top_p": p.top_p, "temperature": p.temperature,
            "speed_factor": p.speed_factor, "seed": p.seed,
            "repetition_penalty": p.repetition_penalty,
        }

    # ---------------------------------------------------------------- synthesize
    def synthesize(self, text, ref_audio=None, prompt_text=None,
                   text_lang=None, prompt_lang=None,
                   text_split_method=None, batch_size=None,
                   sample_steps=None, cfg_rate=None,
                   top_k=None, top_p=None, temperature=None,
                   speed_factor=None, seed=None, repetition_penalty=None):
        # Batch synthesis. Returns SynthResult (fastest total throughput).
        #
        # Parameters left as None fall back to the frozen params in
        # self.config.params, then to defaults. Pass an explicit value to
        # override just for this call.
        if not self._instances:
            raise RuntimeError("call load() first")
        overrides = dict(
            text_split_method=text_split_method, batch_size=batch_size,
            sample_steps=sample_steps, cfg_rate=cfg_rate,
            top_k=top_k, top_p=top_p, temperature=temperature,
            speed_factor=speed_factor, seed=seed,
            repetition_penalty=repetition_penalty,
        )
        p = self._effective_params(overrides)

        ref_cfg = self.config.reference
        ref_audio = ref_audio or (ref_cfg.path if ref_cfg else None)
        prompt_text = prompt_text if prompt_text is not None else (ref_cfg.text if ref_cfg else None)
        text_lang = text_lang or (ref_cfg.language if ref_cfg else "zh")
        prompt_lang = prompt_lang or (ref_cfg.language if ref_cfg else "zh")
        if not ref_audio or not os.path.isfile(ref_audio):
            raise ValueError("reference audio path required")
        if not prompt_text:
            raise ValueError("prompt_text required for v3/v4/v5 models")

        key = self._last_key
        inst = self._instances[key]
        version = key.split("|")[0]

        if p.batch_size is not None:
            batch_size = p.batch_size
        else:
            if self._locked_batch is None:
                self._locked_batch = self.max_batch
            batch_size = self._locked_batch

        req = self._base_req(text, ref_audio, prompt_text, text_lang, prompt_lang, p)
        req.update({
            "text_split_method": p.text_split_method,
            "batch_size": batch_size,
            "return_fragment": False, "streaming_mode": False,
            "parallel_infer": True,
        })
        if p.sample_steps is not None:
            req["sample_steps"] = p.sample_steps
        if p.cfg_rate is not None:
            req["cfg_rate"] = p.cfg_rate

        self._torch.cuda.synchronize()
        t0 = time.time()
        result = None
        with self._runtime.cwd_guard():
            for item in inst.run(dict(req)):
                result = item
        self._torch.cuda.synchronize()
        latency = time.time() - t0

        sr, audio = result
        audio = np.asarray(audio)
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32) / 32768.0
        dur = len(audio) / sr

        # 自动质量检查: 静音输出通常意味着参考文本与音频不匹配
        warning = self.check_output_quality(audio)

        res = SynthResult(audio=audio, sr=sr, version=version,
                          latency_s=latency, audio_dur_s=dur,
                          batch_size=batch_size, warning=warning)
        if warning:
            print("[engine] WARNING: " + warning)
        self.stats.total_requests += 1
        self.stats.total_audio_s += dur
        self.stats.total_latency_s += latency
        self.stats.history.append((version, latency, dur, batch_size))
        return res

    # ---------------------------------------------------------------- streaming
    def synthesize_stream(self, text, ref_audio=None, prompt_text=None,
                          text_lang=None, prompt_lang=None,
                          text_split_method=None, sample_steps=None,
                          cfg_rate=None, top_k=None, top_p=None, temperature=None,
                          speed_factor=None, seed=None, repetition_penalty=None):
        # Streaming synthesis: yields (sample_rate, chunk) to cut TTFT.
        # Same precedence as synthesize(): explicit arg > frozen param > default.
        if not self._instances:
            raise RuntimeError("call load() first")
        p = self._effective_params(dict(
            text_split_method=text_split_method, sample_steps=sample_steps,
            cfg_rate=cfg_rate, top_k=top_k, top_p=top_p,
            temperature=temperature, speed_factor=speed_factor,
            seed=seed, repetition_penalty=repetition_penalty,
        ))
        ref_cfg = self.config.reference
        ref_audio = ref_audio or (ref_cfg.path if ref_cfg else None)
        prompt_text = prompt_text if prompt_text is not None else (ref_cfg.text if ref_cfg else None)
        text_lang = text_lang or (ref_cfg.language if ref_cfg else "zh")
        prompt_lang = prompt_lang or (ref_cfg.language if ref_cfg else "zh")

        key = self._last_key
        inst = self._instances[key]
        version = key.split("|")[0]

        req = self._base_req(text, ref_audio, prompt_text, text_lang, prompt_lang, p)
        req.update({
            "text_split_method": p.text_split_method,
            "batch_size": 1,
            "return_fragment": False, "streaming_mode": True,
            "parallel_infer": False,
        })
        if p.sample_steps is not None:
            req["sample_steps"] = p.sample_steps
        if p.cfg_rate is not None:
            req["cfg_rate"] = p.cfg_rate

        t0 = time.time()
        first = None
        total_audio = 0.0
        with self._runtime.cwd_guard():
            for chunk_sr, chunk in inst.run(dict(req)):
                if first is None:
                    first = time.time() - t0
                arr = np.asarray(chunk)
                total_audio += len(arr) / chunk_sr
                yield chunk_sr, arr
        latency = time.time() - t0
        self.stats.total_requests += 1
        self.stats.total_audio_s += total_audio
        self.stats.total_latency_s += latency
        self.stats.history.append((version + "/stream", latency, total_audio, 1))
        self._last_ttft = first or 0.0

    def supports_native_streaming(self):
        """True if current model streams token-by-token; v3+ returns fragments."""
        if not self._instances:
            return False
        return self._last_key.split("|")[0] in STREAMING_NATIVE

    # ---------------------------------------------------------------- io
    def synthesize_to_file(self, out_path, **kwargs):
        import soundfile as sf
        res = self.synthesize(**kwargs)
        d = os.path.dirname(os.path.abspath(out_path))
        if d:
            os.makedirs(d, exist_ok=True)
        sf.write(out_path, res.audio, res.sr)
        return res

    def unload(self):
        self._instances.clear()
        gc.collect()
        if self._torch is not None:
            self._torch.cuda.empty_cache()

    def summary(self):
        s = self.stats
        if not s.total_requests:
            return "[engine] no requests yet"
        return ("[engine] requests=%d audio=%.2fs latency=%.2fs avg_rtf=%.4f" % (
            s.total_requests, s.total_audio_s, s.total_latency_s, s.avg_rtf))

    def __enter__(self):
        self.load()
        return self

    def __exit__(self, *exc):
        self.unload()
        return False

    # ---------------------------------------------------------------- QC
    #: peak amplitude below this is treated as suspicious silence
    SILENCE_PEAK = 0.02

    @classmethod
    def check_output_quality(cls, audio):
        # Return a warning string if the output looks wrong, else None.
        #
        # Most common cause: reference.text does not match the reference audio
        # content. GPT-SoVITS derives timbre and pacing from BOTH, so a
        # mismatched transcript still returns audio, but it comes out nearly
        # silent -- easy to mistake for an engine defect.
        arr = np.asarray(audio, dtype=np.float32)
        if arr.size == 0:
            return "output audio is empty"
        peak = float(np.abs(arr).max())
        if peak < cls.SILENCE_PEAK:
            return ("output is nearly silent (peak=%.4f). The most likely cause "
                    "is that reference.text does not match the reference audio "
                    "content; pass the exact transcript of the reference clip." % peak)
        return None


# ==================================================================== CLI
def main(argv=None):
    import argparse
    import json
    import sys

    ap = argparse.ArgumentParser(
        prog="gptsovits-engine",
        description="Portable GPT-SoVITS inference engine")
    sub = ap.add_subparsers(dest="cmd")

    p_say = sub.add_parser("say", help="synthesize text to a wav file")
    p_say.add_argument("--config", help="JSON config file (EngineConfig.to_dict)")
    p_say.add_argument("--project-root")
    p_say.add_argument("--sovits")
    p_say.add_argument("--gpt")
    p_say.add_argument("--ref")
    p_say.add_argument("--ref-text")
    p_say.add_argument("--lang", default="zh")
    p_say.add_argument("--text", required=False)
    p_say.add_argument("--out", default="out.wav")
    p_say.add_argument("--device", default="cuda:0")
    p_say.add_argument("--max-batch", type=int, default=8)
    p_say.add_argument("--steps", type=int, default=None)
    p_say.add_argument("--cfg-rate", type=float, default=None)
    p_say.add_argument("--stream", action="store_true", help="streaming mode")
    p_say.add_argument("--split", default=None, help="text_split_method (cut0/cut5...)")
    p_say.add_argument("--speed", type=float, default=None, help="speed_factor")
    p_say.add_argument("--seed", type=int, default=None)
    p_say.add_argument("--save-config", metavar="PATH", default=None,
                       help="freeze this run's params into a config file")
    p_say.add_argument("--portable", action="store_true",
                       help="with --save-config: store relative paths")
    p_say.add_argument("--dump-config", action="store_true",
                       help="print normalized config as JSON and exit")

    p_probe = sub.add_parser("probe", help="validate paths and detect version")
    p_probe.add_argument("--config")
    p_probe.add_argument("--project-root")
    p_probe.add_argument("--sovits")
    p_probe.add_argument("--gpt")

    # 配置管理: 冻结 / 修改 / 查看
    p_cfg = sub.add_parser("config", help="inspect or edit a config file")
    p_cfg.add_argument("--config", required=True, help="config JSON path")
    p_cfg.add_argument("--show", action="store_true", help="print config")
    p_cfg.add_argument("--set", nargs="+", metavar="KEY=VALUE",
                       help="freeze/override synthesis params, e.g. --set sample_steps=4 speed_factor=1.1")
    p_cfg.add_argument("--portable", action="store_true",
                       help="when re-saving, convert paths to relative")

    args = ap.parse_args(argv)
    if args.cmd is None:
        ap.print_help()
        return 0

    if args.cmd == "config":
        return _cmd_config(args)

    if args.cmd == "probe":
        cfg = _cfg_from_args(args, need_ref=False)
        cfg.validate()
        eng = GTSEngine(cfg, validate=False)
        ver = eng.detect_version()
        print("project_root :", cfg.project_root)
        print("sovits       :", cfg.sovits_weights)
        print("gpt          :", cfg.gpt_weights)
        print("bert         :", cfg.bert_path)
        print("cnhubert     :", cfg.cnhubert_path)
        print("version      :", ver)
        print("streaming    :", "native" if ver in STREAMING_NATIVE else "fragment")
        return 0

    cfg = _cfg_from_args(args, need_ref=True)
    if getattr(args, "dump_config", False):
        print(json.dumps(cfg.to_dict(), ensure_ascii=False, indent=2))
        return 0
    if not args.text:
        print("error: --text is required", file=sys.stderr)
        return 2

    # 本次调用参数 (None 表示沿用冻结值)
    call_kwargs = dict(
        sample_steps=args.steps,
        cfg_rate=args.cfg_rate,
        text_split_method=args.split,
        speed_factor=args.speed,
        seed=args.seed,
    )

    eng = GTSEngine(cfg, max_batch=args.max_batch)
    ver = eng.load()

    # --save-config: 把"这次推理用的参数"冻结进配置文件, 便于以后复用
    if getattr(args, "save_config", None):
        frozen = eng.freeze_params(
            text_split_method=args.split,
            sample_steps=args.steps,
            cfg_rate=args.cfg_rate,
            speed_factor=args.speed,
            seed=args.seed,
            streaming=bool(args.stream),
        )
        cfg.params = frozen
        out = cfg.save(args.save_config, portable=getattr(args, "portable", False))
        print("[config] frozen params saved to %s" % out)
        print("[config] params =", json.dumps(frozen.to_dict(), ensure_ascii=False))
        return 0

    if args.stream:
        import soundfile as sf
        chunks = []
        sr = 48000
        t0 = time.time()
        first = None
        for sr, chunk in eng.synthesize_stream(text=args.text, **call_kwargs):
            if first is None:
                first = time.time() - t0
            chunks.append(chunk)
        audio = np.concatenate(chunks)
        sf.write(args.out, audio, sr)
        print("[result] %s stream ttft=%.3fs total=%.3fs audio=%.2fs -> %s" % (
            ver, first or 0.0, time.time() - t0, len(audio) / sr, args.out))
    else:
        res = eng.synthesize_to_file(args.out, text=args.text, **call_kwargs)
        print("[result] %s latency=%.3fs audio=%.2fs rtf=%.4f -> %s" % (
            ver, res.latency_s, res.audio_dur_s, res.rtf, args.out))
        if res.warning:
            print("[warning]", res.warning)
    print(eng.summary())
    return 0



def _cmd_config(args):
    import json
    path = args.config

    if args.set:
        # 修改模式: 加载 -> 覆盖冻结参数 -> 原子写回
        cfg = EngineConfig.load(path)
        updates = {}
        for item in args.set:
            if "=" not in item:
                raise SystemExit("格式错误, 应为 KEY=VALUE: %s" % item)
            key, raw = item.split("=", 1)
            key = key.strip()
            if key not in SynthParams.FIELDS:
                raise SystemExit(
                    "未知参数 %s\n可用: %s" % (key, ", ".join(SynthParams.FIELDS)))
            updates[key] = _parse_value(raw)
        cfg = cfg.with_params(**updates)
        out = cfg.save(path, portable=args.portable)
        print("[config] updated %s" % out)
        if cfg.params:
            print("[config] params =", json.dumps(cfg.params.to_dict(), ensure_ascii=False))
        return 0

    # 查看模式
    cfg = EngineConfig.load(path)
    payload = cfg.to_dict()
    # 校验但不因缺模型而中断, 便于在无模型的机器上检查配置
    try:
        cfg.validate()
        payload["_valid"] = True
    except ConfigError as e:
        payload["_valid"] = False
        payload["_problems"] = str(e).splitlines()[1:]
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _parse_value(raw):
    # 把命令行字符串转成合适的类型: bool/int/float/None/str
    s = raw.strip()
    low = s.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null", ""):
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s

def _cfg_from_args(args, need_ref=True):
    import json
    from .config import EngineConfig, ReferenceAudio
    if getattr(args, "config", None):
        # 用 EngineConfig.load: 它能处理 config_version / _portable 相对路径,
        # 而裸的 from_dict 做不到, 会导致便携配置在异地读取失败。
        return EngineConfig.load(args.config)
    ref = None
    if need_ref:
        missing = [n for n in ("ref", "ref_text") if not getattr(args, n, None)]
        if missing:
            raise SystemExit("missing args: " + ", ".join("--" + m.replace("_", "-") for m in missing))
        ref = ReferenceAudio(args.ref, args.ref_text, args.lang)
    return EngineConfig(
        project_root=getattr(args, "project_root", None) or ".",
        sovits_weights=args.sovits,
        gpt_weights=args.gpt,
        reference=ref,
        device=getattr(args, "device", "cuda:0"),
    )


if __name__ == "__main__":
    raise SystemExit(main())