# 入口点模块

> 模块路径: `vllm/entrypoints/`
> 最后更新: 2026-03-21

## 模块概述

入口点模块提供各种 API 和服务接口，是用户与 vLLM 交互的主要方式。

## 目录结构

```
vllm/entrypoints/
├── __init__.py
├── llm.py                 # 高级 LLM API
├── api_server.py          # OpenAI API 服务器
├── launcher.py            # 服务启动器
├── cli/                   # 命令行接口
│   └── main.py
├── openai/                # OpenAI 兼容 API
│   ├── protocol.py        # API 协议
│   ├── api_server.py      # API 服务器
│   └── chat_utils.py      # 聊天工具
├── serve/                 # 服务组件
│   ├── grpc_server.py     # gRPC 服务器
│   └── ...
├── anthropic/             # Anthropic API 兼容
├── pooling/               # 池化服务（嵌入模型）
├── sagemaker/             # SageMaker 集成
├── mcp/                   # MCP 协议支持
├── utils.py               # 工具函数
├── logger.py              # 日志配置
├── ssl.py                 # SSL/TLS 支持
├── constants.py           # 常量定义
└── protocol.py            # 通用协议定义
```

## 核心组件

### 1. 高级 LLM API (`llm.py`)

提供简单的 Python API 用于文本生成。

```python
from vllm import LLM, SamplingParams

# 初始化 LLM
llm = LLM(model="meta-llama/Llama-2-7b-hf")

# 生成参数
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=100
)

# 生成文本
outputs = llm.generate(["Hello, my name is"], sampling_params)

for output in outputs:
    print(f"输出: {output.outputs[0].text}")
```

### 2. OpenAI API 服务器 (`openai/`)

提供与 OpenAI API 兼容的服务器接口。

**启动服务器**：
```bash
vllm serve meta-llama/Llama-2-7b-hf --port 8000
```

**使用客户端**：
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"
)

completion = client.completions.create(
    model="meta-llama/Llama-2-7b-hf",
    prompt="Hello, my name is",
    max_tokens=100
)

print(completion.choices[0].text)
```

### 3. 命令行接口 (`cli/`)

提供命令行工具来启动和管理服务。

**主要命令**：
```bash
# 启动服务
vllm serve <model_name>

# 交互式聊天
vllm chat <model_name>

# 完成文本
vllm complete <model_name>
```

### 4. 池化服务 (`pooling/`)

为嵌入模型提供池化服务。

```python
from vllm.entrypoints.openai.protocol import EmbeddingRequest

# 创建嵌入请求
request = EmbeddingRequest(
    model="intfloat/e5-mistral-7b-instruct",
    input=["Hello, world!", "How are you?"]
)
```

## API 协议

### Chat API

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"
)

response = client.chat.completions.create(
    model="meta-llama/Llama-2-7b-hf",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ],
    temperature=0.7,
    max_tokens=100
)
```

### Completion API

```python
response = client.completions.create(
    model="meta-llama/Llama-2-7b-hf",
    prompt="Once upon a time",
    max_tokens=100,
    temperature=0.8
)
```

### Embedding API

```python
response = client.embeddings.create(
    model="intfloat/e5-mistral-7b-instruct",
    input="Hello, world!"
)

embedding = response.data[0].embedding
```

## 采样参数

### SamplingParams

```python
from vllm import SamplingParams

params = SamplingParams(
    # 生成参数
    max_tokens=100,              # 最大 token 数
    temperature=0.8,             # 温度
    top_p=0.95,                  # Top-p 采样
    top_k=-1,                    # Top-k 采样（-1 表示禁用）
    presence_penalty=0.0,        # 存在惩罚
    frequency_penalty=0.0,       # 频率惩罚
    repetition_penalty=1.0,      # 重复惩罚

    # 停止条件
    stop=["\n", "USER:"],       # 停止序列
    stop_token_ids=[],           # 停止 token ID

    # 输出选项
    n=1,                         # 生成候选数
    logprobs=None,               # 返回 log probabilities
    prompt_logprobs=None,        # 返回 prompt log probabilities

    # 其他
    seed=None,                   # 随机种子
    best_of=1,                   # 采样候选数
)
```

## 服务配置

### API 服务器选项

```python
from vllm.entrypoints.openai.cli import serve_parser

args = serve_parser.parse_args([
    "meta-llama/Llama-2-7b-hf",
    "--host", "0.0.0.0",
    "--port", "8000",
    "--tensor-parallel-size", "2",
    "--gpu-memory-utilization", "0.9",
    "--enable-prefix-caching",
])
```

### 常用启动参数

```bash
vllm serve <model> \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 4096 \
    --enable-prefix-caching \
    --dtype auto \
    --quantization awq
```

## 高级功能

### 1. 流式输出

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"
)

stream = client.chat.completions.create(
    model="meta-llama/Llama-2-7b-hf",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### 2. 多轮对话

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
]

response = client.chat.completions.create(
    model="meta-llama/Llama-2-7b-hf",
    messages=messages
)

assistant_message = response.choices[0].message.content
messages.append({"role": "assistant", "content": assistant_message})
messages.append({"role": "user", "content": "And what about Germany?"})

# 继续对话...
```

### 3. 批量处理

```python
prompts = [
    "Explain quantum computing",
    "What is machine learning?",
    "Describe the solar system"
]

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"Prompt: {output.prompt}")
    print(f"Output: {output.outputs[0].text}\n")
```

### 4. LoRA 适配器

```python
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    enable_lora=True,
    max_loras=4
)

# 使用特定 LoRA 适配器生成
outputs = llm.generate(
    prompts,
    sampling_params,
    lora_request="path/to/lora/adapter"
)
```

## API 认证

### API Key 认证

```bash
vllm serve <model> --api-key your-secret-key
```

```python
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="your-secret-key"
)
```

### Token 认证

```bash
vllm serve <model> --token-based-auth
```

## 监控和日志

### 健康检查

```bash
curl http://localhost:8000/health
```

### 模型信息

```bash
curl http://localhost:8000/v1/models
```

### 日志配置

```python
import logging

# 配置日志级别
logging.getLogger("vllm").setLevel(logging.DEBUG)
```

## 部署选项

### Docker 部署

```dockerfile
FROM vllm/vllm-openai:latest

CMD ["--model", "meta-llama/Llama-2-7b-hf"]
```

### Kubernetes 部署

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-deployment
spec:
  replicas: 1
  selector:
    matchLabels:
      app: vllm
  template:
    metadata:
      labels:
        app: vllm
    spec:
      containers:
      - name: vllm
        image: vllm/vllm-openai:latest
        args:
        - "--model"
        - "meta-llama/Llama-2-7b-hf"
        resources:
          limits:
            nvidia.com/gpu: 1
```

## 性能优化

### 1. 批处理

```python
# 批量请求提高吞吐量
prompts = [...] * 100  # 100 个提示
outputs = llm.generate(prompts, sampling_params)
```

### 2. 前缀缓存

```bash
vllm serve <model> --enable-prefix-caching
```

### 3. 推测解码

```bash
vllm serve <model> \
    --speculative-model <draft_model> \
    --num-speculative-tokens 5
```

## 常见问题

### Q: 如何处理并发请求？

A: vLLM 自动处理并发请求，使用连续批处理优化吞吐量。

### Q: 如何限制并发请求数？

A: 使用 `--max-num-seqs` 参数：
```bash
vllm serve <model> --max-num-seqs 128
```

### Q: 如何启用 SSL/TLS？

A: 使用 `--ssl-keyfile` 和 `--ssl-certfile`：
```bash
vllm serve <model> \
    --ssl-keyfile key.pem \
    --ssl-certfile cert.pem
```

## 相关文档

- [根级文档](../../CLAUDE.md)
- [v1 架构](../v1/CLAUDE.md)
- [配置系统](../config/CLAUDE.md)
