# 配置系统模块

> 模块路径: `vllm/config/`
> 最后更新: 2026-03-21

## 模块概述

配置系统提供统一的配置管理，支持各种 vLLM 组件的配置选项。所有配置都是类型安全的，并支持验证和默认值。

## 目录结构

```
vllm/config/
├── __init__.py
├── vllm.py              # 主配置类 VllmConfig
├── model.py             # 模型配置
├── cache.py             # 缓存配置
├── attention.py         # 注意力配置
├── parallel.py          # 并行配置
├── compilation.py       # 编译配置
├── device.py            # 设备配置
├── kernel.py            # 内核配置
├── scheduler.py         # 调度器配置
├── speculative.py       # 推测解码配置
├── lora.py              # LoRA 配置
├── multimodal.py        # 多模态配置
├── load.py              # 加载配置
├── pooler.py            # 池化配置
├── profiler.py          # 性能分析配置
├── offload.py           # 卸载配置
├── structured_outputs.py # 结构化输出配置
├── kv_transfer.py       # KV 传输配置
├── kv_events.py         # KV 事件配置
├── ec_transfer.py       # EC 传输配置
├── observability.py     # 可观测性配置
├── speech_to_text.py    # 语音转文本配置
├── weight_transfer.py   # 权重传输配置
├── model_arch.py        # 模型架构配置
└── utils.py             # 配置工具
```

## 核心配置类

### VllmConfig (`vllm.py`)

主配置类，聚合所有子配置。

```python
@dataclass
class VllmConfig:
    # 模型配置
    model: str
    model_config: ModelConfig
    # 缓存配置
    cache_config: CacheConfig
    # 并行配置
    parallel_config: ParallelConfig
    # 调度器配置
    scheduler_config: SchedulerConfig
    # 设备配置
    device_config: DeviceConfig
    # 编译配置
    compilation_config: CompilationConfig
    # LoRA 配置
    lora_config: Optional[LoRAConfig]
    # 推测解码配置
    speculative_config: Optional[SpeculativeConfig]
    # ... 其他配置
```

## 主要配置详解

### 1. 模型配置 (`model.py`)

定义模型的基本属性和参数。

```python
@dataclass
class ModelConfig:
    # 模型标识
    model: str
    # 信任远程代码
    trust_remote_code: bool = False
    # 分词器
    tokenizer: Optional[str] = None
    # 分词器模式
    tokenizer_mode: str = "auto"
    # 跳过 tokenizer 初始化
    skip_tokenizer_init: bool = False
    # 限制分词器并行
    limit_mm_per_prompt: Optional[Dict[str, int]] = None
    # 最大模型长度
    max_model_len: Optional[int] = None
    # 张量并行大小
    tensor_parallel_size: int = 1
    # 数据类型
    dtype: str = "auto"
    # 量化
    quantization: Optional[str] = None
    # KV 缓存数据类型
    kv_cache_dtype: str = "auto"
    # ... 更多配置
```

### 2. 缓存配置 (`cache.py`)

控制 KV 缓存的行为。

```python
@dataclass
class CacheConfig:
    # 块大小（token 数）
    block_size: int = 16
    # GPU 内存利用率
    gpu_memory_utilization: float = 0.9
    # 交换空间（字节）
    swap_space: int = 4
    # 缓存引擎
    cache_dtype: str = "auto"
    # 启用前缀缓存
    enable_prefix_caching: bool = False
    # ...
```

### 3. 并行配置 (`parallel.py`)

定义并行策略。

```python
@dataclass
class ParallelConfig:
    # 张量并行大小
    tensor_parallel_size: int = 1
    # 流水线并行大小
    pipeline_parallel_size: int = 1
    # 数据并行大小
    data_parallel_size: int = 1
    # 最大并行工作线程
    max_parallel_loading_workers: Optional[int] = None
    # 分布式执行后端
    distributed_executor_backend: Optional[str] = None
    # ...
```

### 4. 调度器配置 (`scheduler.py`)

控制请求调度策略。

```python
@dataclass
class SchedulerConfig:
    # 最大序列长度
    max_num_batched_tokens: int
    # 最大序列数
    max_num_seqs: int = 256
    # 最大等待时间（秒）
    max_model_len: Optional[int] = None
    # 调度策略
    scheduling_policy: str = "max_utilization"
    # ...
```

### 5. 编译配置 (`compilation.py`)

控制模型编译优化。

```python
@dataclass
class CompilationConfig:
    # 启用编译
    enable_compilation: bool = False
    # 编译后端
    backend: str = "inductor"
    # 编译缓存大小
    cache_size: int = 8
    # ...
```

### 6. LoRA 配置 (`lora.py`)

LoRA 适配器配置。

```python
@dataclass
class LoRAConfig:
    # 启用 LoRA
    enabled: bool = False
    # 最大 LoRA 数量
    max_loras: int = 1
    # 最大 LoRA Rank
    max_lora_rank: int = 16
    # LoRA 模型路径
    lora_modules: Optional[List[str]] = None
    # ...
```

### 7. 推测解码配置 (`speculative.py`)

推测解码参数。

```python
@dataclass
class SpeculativeConfig:
    # 启用推测解码
    enabled: bool = False
    # 草稿模型
    draft_model: str
    # 草稿模型张量并行大小
    draft_tensor_parallel_size: int = 1
    # 草稿模型数据类型
    draft_dtype: str = "auto"
    # 草稿模型目标
    draft_model_target: str = "auto"
    # ...
```

## 配置加载

### 从命令行参数加载

```python
from vllm.config import VllmConfig
from vllm.engine.arg_utils import EngineArgs

# 解析命令行参数
engine_args = EngineArgs.from_cli(args)

# 创建配置
config = VllmConfig.from_engine_args(engine_args)
```

### 从字典加载

```python
from vllm.config import VllmConfig

config_dict = {
    "model": "meta-llama/Llama-2-7b-hf",
    "tensor_parallel_size": 2,
    "gpu_memory_utilization": 0.9,
}

config = VllmConfig(**config_dict)
```

### 从环境变量加载

```bash
# 设置环境变量
export VLLM_TENSOR_PARALLEL_SIZE=2
export VLLM_GPU_MEMORY_UTILIZATION=0.9

# 环境变量会覆盖默认值
```

## 配置验证

配置系统会自动验证配置参数：

```python
from vllm.config import CacheConfig

try:
    # 无效配置
    config = CacheConfig(block_size=-1)
except ValueError as e:
    print(f"配置错误: {e}")
```

## 配置优先级

配置参数的优先级（从高到低）：

1. 显式传递的参数
2. 环境变量
3. 配置文件
4. 默认值

## 自定义配置

### 创建自定义配置类

```python
from dataclasses import dataclass
from vllm.config import CacheConfig

@dataclass
class MyCacheConfig(CacheConfig):
    # 添加自定义参数
    my_custom_param: int = 42

    # 重写验证逻辑
    def __post_init__(self):
        super().__post_init__()
        if self.my_custom_param < 0:
            raise ValueError("my_custom_param 必须 >= 0")
```

## 配置文件支持

vLLM 支持 JSON/YAML 配置文件：

```json
{
    "model": "meta-llama/Llama-2-7b-hf",
    "tensor_parallel_size": 2,
    "gpu_memory_utilization": 0.9,
    "enable_prefix_caching": true
}
```

使用配置文件：

```bash
vllm serve --config config.json
```

## 环境变量覆盖

常用环境变量：

```bash
# 张量并行
VLLM_TENSOR_PARALLEL_SIZE=2

# GPU 内存利用率
VLLM_GPU_MEMORY_UTILIZATION=0.9

# 日志级别
VLLM_LOGGING_LEVEL=DEBUG

# 启用跟踪
VLLM_USE_TRACING=1
```

## 调试配置

### 检查当前配置

```python
from vllm.config import VllmConfig

config = VllmConfig.from_engine_args(engine_args)
print(config)
```

### 验证配置

```python
from vllm.config import validate_config

errors = validate_config(config)
if errors:
    for error in errors:
        print(f"配置错误: {error}")
```

## 常见问题

### Q: 如何设置不同的数据类型？

A: 使用 `dtype` 参数：
```python
config = ModelConfig(
    model="meta-llama/Llama-2-7b-hf",
    dtype="float16"  # 或 "bfloat16", "float32"
)
```

### Q: 如何启用前缀缓存？

A: 在 CacheConfig 中设置：
```python
cache_config = CacheConfig(
    enable_prefix_caching=True
)
```

### Q: 如何配置多 GPU？

A: 设置张量并行大小：
```python
parallel_config = ParallelConfig(
    tensor_parallel_size=4  # 使用 4 个 GPU
)
```

## 相关文档

- [根级文档](../../CLAUDE.md)
- [v1 架构](../v1/CLAUDE.md)
- [模型执行器](../model_executor/CLAUDE.md)
