# gptsovits-engine

可移植的 **GPT-SoVITS** 推理引擎。只需给出 GPT-SoVITS 位置、模型权重、参考音频与文本，
即可在任意目录、任意机器上跑推理；同时可作为标准 PyPI 包安装使用。

## 特性

| 能力 | 说明 | 实测 |
|---|---|---|
| 任意目录运行 | 自动管理 sys.path 与 CWD，不污染宿主程序 | 从 `/` 目录实测通过 |
| 代码/模型分离部署 | `project_root` 与 `pretrained_root` 可指向不同位置 | 已实测 |
| 自动版本识别 | 从权重文件头识别 v1/v2/v3/v4/v5dev/v5turbo/v2Pro/v2ProPlus | 全部支持 |
| 模型常驻 | 加载一次约 4-8s，常驻后不再重复；支持 LRU 多模型热切换 | - |
| 自适应批处理 | 长文本合并 batch，提升吞吐 | **1.54x** (v5turbo) |
| 流式输出 | 逐块产出，降低首包延迟 | **TTFT 5-6.5x** |
| 预热 | 消除首次请求的 CUDA Graph 捕获开销 | 7s -> 0.8s |
| 清晰的错误提示 | 配置校验一次性列出所有缺失项 | - |

## 安装

```bash
# 从源码
cd engine && pip install -e .

# 从 wheel
pip install dist/gptsovits_engine-0.1.0-py3-none-any.whl
```

> torch 需按所在机器的 CUDA 版本单独安装（引擎不锁定 torch 版本）。
> 本引擎 **不包含** GPT-SoVITS 源码与模型，需自行准备。

## 前置条件

1. **GPT-SoVITS 项目**（含 `GPT_SoVITS/` 目录，即上游 RVC-Boss/GPT-SoVITS）
2. **Python 环境**已装好 GPT-SoVITS 的依赖（torch、transformers、gradio 等）
3. **模型权重**：SoVITS `.pth` + GPT `.ckpt`
4. **参考音频** 3~10 秒 + 对应文本

## 快速开始

```python
from gptsovits_engine import GTSEngine, EngineConfig, ReferenceAudio

eng = GTSEngine(EngineConfig(
    project_root="/opt/GPT-SoVITS",                    # GPT-SoVITS 位置
    sovits_weights="/models/my_sovits.pth",            # SoVITS 权重
    gpt_weights="/models/my_gpt.ckpt",                 # GPT 权重
    reference=ReferenceAudio(
        "/models/ref_audio.wav",                       # 参考音频 (3~10s)
        "这是参考音频对应的文字。",                       # 参考文本
        "zh",                                          # 语言
    ),
    device="cuda:0",
))

eng.load()                                             # 加载 + 预热
res = eng.synthesize("你好，这是要合成的文本。")
print(res.rtf, res.audio_dur_s, res.sr)
res.audio  # numpy float32 波形
```

也可以直接用 dict：

```python
eng = GTSEngine({
    "project_root": "/opt/GPT-SoVITS",
    "sovits_weights": "/models/my_sovits.pth",
    "gpt_weights": "/models/my_gpt.ckpt",
    "reference": {"path": "/models/ref.wav", "text": "参考文字", "language": "zh"},
})
```

## 代码与模型分离部署

模型不必放在 GPT-SoVITS 目录里：

```python
cfg = EngineConfig(
    project_root="/opt/gptsovits_src",       # 只有代码
    pretrained_root="/data/pretrained",      # BERT / CNHuBERT 放这里
    sovits_weights="/data/models/sovits.pth",
    gpt_weights="/data/models/gpt.ckpt",
    reference=ReferenceAudio("/data/ref.wav", "文字", "zh"),
)
```

## 流式输出

```python
for sr, chunk in eng.synthesize_stream("很长的一段文本，会分块返回。"):
    player.feed(chunk, sr)      # 边生成边播放，TTFT 更低
```

- `eng.supports_native_streaming()` 判断是否为真流式
- v1/v2 为真流式（逐块）；v3/v4/v5 上游会回退为按句分段返回
- **权衡**：流式会关闭并行批处理，总耗时比批量模式差约 58%；
  交互式用流式，批量/离线用 `synthesize()`

## 多模型热切换

```python
eng = GTSEngine(cfg, cache_size=2)   # 常驻 2 个模型
eng2 = GTSEngine(cfg_another, cache_size=2)
```

## 命令行

```bash
# 校验路径并识别版本
gptsovits-engine probe \
  --project-root /opt/GPT-SoVITS \
  --sovits /models/s.pth --gpt /models/g.ckpt

# 合成
gptsovits-engine say \
  --project-root /opt/GPT-SoVITS \
  --sovits /models/s.pth --gpt /models/g.ckpt \
  --ref /models/ref.wav --ref-text "参考文字" --lang zh \
  --text "要合成的文本" --out out.wav

# 流式
gptsovits-engine say ... --stream
```

也可以把配置存成 JSON：

```bash
gptsovits-engine say --config my_config.json --text "..."
```
## 参数冻结（配置复用）

调好一组合成参数后，可以冻结进配置文件，之后直接复用，不用每次重复写。

### 在代码里冻结

```python
from gptsovits_engine import GTSEngine, EngineConfig, ReferenceAudio, SynthParams

cfg = EngineConfig(
    project_root="/opt/GPT-SoVITS",
    sovits_weights="/models/sovits.pth",
    gpt_weights="/models/gpt.ckpt",
    reference=ReferenceAudio("/models/ref.wav", "参考文本", "zh"),
    params=SynthParams(sample_steps=8, speed_factor=1.2, seed=777),   # 冻结
)
cfg.save("my_voice.json")
```

之后任何地方：

```python
eng = GTSEngine.from_file("my_voice.json")
eng.load()
res = eng.synthesize("文本")        # 自动使用冻结的 8步/1.2倍速/种子777
res = eng.synthesize("文本", speed_factor=1.5)   # 临时覆盖，不写回文件
```

### 用命令行冻结

```bash
# 从这次推理中录制参数并保存
gptsovits-engine say --project-root /opt/GPT-SoVITS \
  --sovits /models/s.pth --gpt /models/g.ckpt \
  --ref /models/ref.wav --ref-text "参考文字" --lang zh \
  --text "warmup" \
  --steps 8 --speed 1.2 --seed 777 --split cut5 \
  --save-config my_voice.json

# 查看
gptsovits-engine config --config my_voice.json --show

# 修改某个参数
gptsovits-engine config --config my_voice.json --set speed_factor=1.0 sample_steps=4

# 直接复用
gptsovits-engine say --config my_voice.json --text "要合成的文本" --out out.wav
```

### 可冻结的参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `text_split_method` | cut5 | 切句方式 |
| `batch_size` | None | 推理批次，None 沿用引擎锁定值 |
| `sample_steps` | None | 声码器步数（v5turbo 4，v5dev 32） |
| `cfg_rate` | None | CFG 强度 |
| `top_k` / `top_p` / `temperature` | 15 / 1.0 / 1.0 | 采样参数 |
| `speed_factor` | 1.0 | 语速倍率 |
| `seed` | 42 | 随机种子 |
| `repetition_penalty` | 1.35 | 重复惩罚 |
| `streaming` | False | 是否流式 |

### 配置优先级

```
本次调用显式传入 > 配置文件中冻结的值 > 引擎默认值
```

### 便携配置

```python
cfg.save("my_voice.json", portable=True)
```

`portable=True` 会把所有路径转成相对于配置文件目录的形式，
连同模型一起拷到别的机器也能直接用。

### 配置文件示例

```json
{
  "project_root": "/opt/GPT-SoVITS",
  "sovits_weights": "/models/sovits.pth",
  "gpt_weights": "/models/gpt.ckpt",
  "device": "cuda:0",
  "reference": {
    "path": "/models/ref.wav",
    "text": "参考文本内容",
    "language": "zh"
  },
  "params": {
    "sample_steps": 8,
    "speed_factor": 1.2,
    "seed": 777
  },
  "config_version": 1
}
```

配置文件只记录**非默认值**，所以通常很短，也方便手改。
写入使用临时文件+原子替换，避免写一半损坏。

## API 参考

### EngineConfig

| 参数 | 必填 | 说明 |
|---|---|---|
| `project_root` | 是 | GPT-SoVITS 根目录（含 `GPT_SoVITS/`） |
| `sovits_weights` | 是 | SoVITS `.pth` |
| `gpt_weights` | 是 | GPT `.ckpt` |
| `reference` | 建议 | `ReferenceAudio(path, text, language)` |
| `pretrained_root` | 否 | 独立存放的预训练资源目录 |
| `device` | 否 | 默认 `cuda:0`，可设 `cpu` |
| `is_half` | 否 | 默认 `True` |
| `version` | 否 | 默认自动识别 |
| `extra_sys_path` | 否 | 额外 sys.path 目录 |

### GTSEngine

| 方法 | 说明 |
|---|---|
| `load(warmup=True)` | 加载并按 batch 预热，返回版本 |
| `synthesize(text, ...)` | 批量合成，返回 `SynthResult` |
| `synthesize_stream(text, ...)` | 流式，yield `(sr, chunk)` |
| `synthesize_to_file(path, ...)` | 合成并写文件 |
| `detect_version()` | 从权重识别版本 |
| `supports_native_streaming()` | 是否真流式 |
| `unload()` | 释放模型 |
| `summary()` | 运行统计 |

### SynthResult

| 字段 | 说明 |
|---|---|
| `audio` | numpy float32 波形 |
| `sr` | 采样率 |
| `version` | 实际版本 |
| `latency_s` / `audio_dur_s` | 耗时 / 音频时长 |
| `rtf` | 实时率（越小越快） |

## 实现要点

### 为什么需要 runtime 层

上游 GPT-SoVITS 有三处硬依赖当前工作目录：

1. `TTS.py` 模块顶部 `now_dir = os.getcwd()`，声码器路径基于它拼接
2. `sv.py` 用 `sys.path.append(f"{os.getcwd()}/GPT_SoVITS/eres2net")`
3. 大量 `from module.x` / `from AR.x` 扁平导入依赖项目根在 sys.path

`runtime.py` 的做法：

- 把项目根与内部子目录加入 `sys.path`
- 用 `cwd_guard()` 在推理期间临时切换 CWD，**结束后恢复**，不劫持宿主程序

### 为什么预热要跑 3 轮

上游 CUDA Graph 按执行形状逐个捕获。预热需满足三个条件，缺一则首请求仍慢数倍：

1. 参考音频 3~10 秒（上游有校验）
2. batch 与正式请求一致（变了会重建引擎并重捕获，约 7s）
3. 文本长度接近真实请求（graph 与序列长度相关）

## 许可证

本包代码为 MIT。GPT-SoVITS 本体遵循其自身许可证。