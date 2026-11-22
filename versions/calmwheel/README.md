# CalmWheel — LLM 模块

## 文件结构

```
llm_module/
├── interface.py          # 对外接口定义（LLMInput / LLMOutput）
├── emotion_strategy.py   # 7种情绪的应对策略和prompt
├── prompt_builder.py     # 把情绪策略 + 语音 融合成messages
├── llm_client.py         # 调用 OpenRouter / Ollama
├── trigger_control.py    # 冷却时间、触发逻辑
├── conversation.py       # 多轮对话记忆
├── benchmark.py          # 多模型对比测试
├── main.py               # 主入口 + 交互式测试
├── .env.example          # API key模板
└── README.md
```

## 快速开始

```bash
# 1. 安装依赖
pip install requests python-dotenv

# 2. 配置API key
cp .env.example .env
# 编辑 .env，填入你的 OpenRouter API key

# 3. 交互式测试
python main.py

# 4. 运行 Benchmark
python -c "from benchmark import run_benchmark; run_benchmark()"
```

## 和队友对接

### FER组 + STT组 → 传给你

```python
from interface import LLMInput

# 有语音
inp = LLMInput(emotion="angry", speech_text="前面那个车怎么开的！")

# 只有情绪标签（没说话）
inp = LLMInput(emotion="sad")
```

### 你 → 传给TTS组

```python
from main import CalmWheelLLM
from interface import LLMInput

llm = CalmWheelLLM()
output = llm.process(inp)

if output:
    tts.speak(output.response_text)   # 传给TTS组
```

## 支持的情绪标签

`angry` / `sad` / `fear` / `happy` / `neutral` / `disgust` / `surprise`

## 模型选择

| 模型 | mode | model_key |
|------|------|-----------|
| Claude Haiku（默认） | online | claude-haiku |
| GPT-4o-mini | online | gpt-4o-mini |
| Gemma 4B（免费） | online | gemma-4b |
| Llama 3.1 8B（免费） | online | llama-3-8b |
| Gemma 本地（需Ollama）| offline | gemma-local |

```python
# 指定模型
llm = CalmWheelLLM(mode="online", model_key="gpt-4o-mini")

# 自动fallback（online失败切本地）
llm = CalmWheelLLM(mode="auto")
```
