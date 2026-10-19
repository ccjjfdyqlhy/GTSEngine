"""gptsovits-engine: portable GPT-SoVITS inference engine.

Quick start::

    from gptsovits_engine import GTSEngine, EngineConfig, ReferenceAudio

    cfg = EngineConfig(
        project_root="/opt/GPT-SoVITS",
        sovits_weights="/models/sovits.pth",
        gpt_weights="/models/gpt.ckpt",
        reference=ReferenceAudio("/models/ref.wav", "reference text", "zh"),
    )
    with GTSEngine(cfg) as eng:
        res = eng.synthesize("hello world")
        print(res.rtf, res.audio_dur_s)
"""
from .config import ConfigError, EngineConfig, ReferenceAudio, SynthParams
from .engine import EngineStats, GTSEngine, STREAMING_NATIVE, SynthResult
from .runtime import RuntimeBootstrap, RuntimeBootstrapError, bootstrap

__version__ = "0.2.0"

__all__ = [
    "GTSEngine",
    "EngineConfig",
    "ReferenceAudio",
    "SynthParams",
    "SynthResult",
    "EngineStats",
    "ConfigError",
    "RuntimeBootstrap",
    "RuntimeBootstrapError",
    "bootstrap",
    "STREAMING_NATIVE",
    "__version__",
]