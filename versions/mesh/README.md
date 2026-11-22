# CalmWheel \[mesh\] — 使用说明

车载情绪感知 AI 陪伴系统。摄像头实时检测司机面部情绪，结合语音输入，由 LLM 生成情绪化回应。

---

## 环境要求

- Python 3.10 或以上
- 摄像头（默认 camera 0）
- 麦克风
- Windows / macOS / Linux 均可

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

> 首次运行 STT 时会自动下载 Whisper `base` 模型（约 150 MB），需要等待约 30 秒。

### 2. 配置 API Key

编辑项目根目录下的 `.env` 文件，填入 OpenRouter API Key：

```
OPENROUTER_API_KEY=你的key
```

### 3. 放置模型文件

确认以下文件/文件夹存在于项目根目录：

```
face_landmarker.task          # MediaPipe 人脸关键点模型
blendshape_mlp_ckplus/        # 情绪识别模型（新版）
    checkpoints/
        best.pt
gifs/                         # 情绪头像动画
    neutral.gif / happy.gif / angry.gif ...
```

`face_landmarker.task` 如果没有，从这里下载并放到根目录：
https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker

### 4. 启动系统

```bash
python run.py
```

启动后会打开一个三栏界面：左侧摄像头画面、中间情绪头像、右侧对话记录。

---

## 界面操作

| 操作 | 说明 |
|------|------|
| 直接对着摄像头 | FER 模块自动检测面部情绪并更新头像 |
| 对着麦克风说话 | STT 转录后触发 LLM 生成回应，自动播放语音 |
| 右上角下拉菜单 | 切换 LLM 模型（Claude Haiku / GPT-4o Mini / Llama 70B / Gemma 26B） |

---

## 命令行参数

```bash
python run.py [选项]

  --checkpoint PATH   情绪模型路径（默认 ./blendshape_mlp_ckplus/checkpoints/best.pt）
  --camera INT        摄像头编号（默认 0）
  --conf FLOAT        情绪置信度阈值（默认 0.3）
  --no-fer            禁用摄像头 FER，只用 CLI 模拟
  --no-bar            隐藏摄像头画面中的概率条
  --landmarker PATH   face_landmarker.task 路径（默认 ./face_landmarker.task）
```

---

## 没有摄像头？用 CLI 模拟

启动时加 `--no-fer`，然后在终端直接输入情绪和文本：

```bash
python run.py --no-fer
```

```
>> angry 前面那个车怎么开的！      # 情绪 + 语音 → 触发 LLM
>> sad                              # 只更新情绪状态，不触发 LLM
>> reset                            # 清空对话记忆，开始新旅程
>> mute                             # 开关 TTS 语音
>> quit                             # 退出
```

支持的情绪标签：`angry` `sad` `fear` `happy` `neutral` `disgust` `surprise`

---

## 模块说明（对接用）

### FER → LLM 接口

```python
from interface import LLMInput
from main import CalmWheelLLM

llm = CalmWheelLLM()
output = llm.process(LLMInput(emotion="angry", speech_text="怎么这么堵！"))

if output:
    print(output.response_text)       # LLM 回应文本
    print(output.emotion_timeline)    # 头像情绪时间轴，格式见下
```

`emotion_timeline` 格式：
```python
[{"t": 0, "emotion": "angry"}, {"t": 2500, "emotion": "neutral"}]
# t 单位为毫秒，从 TTS 播放开始计时
```

### 新旅程（清空记忆）

```python
llm.new_trip()
```

---

## 常见问题

**摄像头画面显示 "No face detected"**
确认 `realtime_fer.py` 已替换为最新修复版本（blendshape 提取 Bug 已修复）。

**LLM 不响应**
检查 `.env` 中的 `OPENROUTER_API_KEY` 是否有效，或使用 `--no-fer` 后在 CLI 输入 `neutral 你好` 测试。

**STT 识别不到声音**
系统默认麦克风如果不正确，在 `stt.py` 顶部修改 `sounddevice` 的设备编号。

**想换回 attention 模型**
```bash
python run.py --checkpoint ./blendshape_attn_ckplus/checkpoints/best.pt
```
`load_model` 会自动识别模型类型，无需改代码。
