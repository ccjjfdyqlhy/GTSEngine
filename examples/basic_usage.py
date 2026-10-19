"""Minimal usage example.

Run from any directory:
    python examples/basic_usage.py
"""
import os
import sys

from gptsovits_engine import EngineConfig, GTSEngine, ReferenceAudio

# ---- fill in your own paths ----
PROJECT_ROOT = os.environ.get("GSV_PROJECT_ROOT", "/opt/GPT-SoVITS")
SOVITS = os.environ.get("GSV_SOVITS", "/models/sovits.pth")
GPT = os.environ.get("GSV_GPT", "/models/gpt.ckpt")
REF_AUDIO = os.environ.get("GSV_REF", "/models/ref.wav")
REF_TEXT = os.environ.get("GSV_REF_TEXT", "reference transcript here")
LANG = os.environ.get("GSV_LANG", "zh")


def main():
    cfg = EngineConfig(
        project_root=PROJECT_ROOT,
        sovits_weights=SOVITS,
        gpt_weights=GPT,
        reference=ReferenceAudio(REF_AUDIO, REF_TEXT, LANG),
        device=os.environ.get("GSV_DEVICE", "cuda:0"),
    )

    with GTSEngine(cfg, max_batch=8) as eng:
        print("version:", eng.config.version)
        res = eng.synthesize("Hello, this is a test of the portable engine.")
        print("latency=%.3fs audio=%.2fs rtf=%.4f" % (
            res.latency_s, res.audio_dur_s, res.rtf))

        out = "example_out.wav"
        eng.synthesize_to_file(out, text="This file is written from the engine.")
        print("wrote", os.path.abspath(out))
        print(eng.summary())


if __name__ == "__main__":
    sys.exit(main() or 0)