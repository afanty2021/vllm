# 模型执行器模块

> 模块路径: `vllm/model_executor/`
> 最后更新: 2026-03-21

## 模块概述

模型执行器负责加载和执行各种大语言模型。它是 vLLM 的核心组件之一，支持 100+ 种不同的模型架构。

## 目录结构

```
vllm/model_executor/
├── layers/             # 模型层实现
│   ├── activation.py
│   ├── attention.py
│   ├── ffn.py
│   ├── layernorm.py
│   ├── linear.py
│   └── embedding.py
├── models/             # 具体模型实现（100+ 模型）
│   ├── llama.py
│   ├── mixtral.py
│   ├── qwen2.py
│   └── ...
├── model_loader/       # 模型加载器
│   ├── loader.py
│   ├── weight_loader.py
│   └── utils.py
├── kernels/            # 自定义 CUDA 内核
├── offloader/          # 模型卸载器
├── sampling_metadata.py
├── parameter.py        # 参数管理
└── utils.py
```

## 核心组件

### 1. 模型层 (`layers/`)

通用的模型层实现，被多个模型共享。

#### 主要层类型

**线性层** (`linear.py`)
- `RowParallelLinear` - 行并行线性层
- `ColumnParallelLinear` - 列并行线性层
- `ReplicatedLinear` - 复制线性层
- `QKVParallelLinear` - QKV 并行线性层

**注意力层** (`attention.py`)
- `Attention` - 基础注意力
- `PagedAttention` - 分页注意力
- `PagedAttentionWithRoPE` - 带 RoPE 的分页注意力

**FFN 层** (`ffn.py`)
- `FFN` - 前馈网络
- `MergedColumnParallelFFN` - 合并列并行 FFN

**归一化层** (`layernorm.py`)
- `LayerNorm` - 层归一化
- `RMSNorm` - RMS 归一化

**激活函数** (`activation.py`)
- `SiluAndMul` - SiLU 激活
- `GatedSiluAndMul` - 门控 SiLU
- `FatgeluAndMul` - FatGELU 激活

### 2. 模型实现 (`models/`)

具体模型的实现，每个文件对应一个模型系列。

#### 主要模型类别

**Transformer LLM**
- `llama.py` - LLaMA/Llama-2/Llama-3 系列
- `qwen2.py` - Qwen2 系列
- `mistral.py` - Mistral 系列
- `phi3.py` - Phi-3 系列

**混合专家模型**
- `mixtral.py` - Mixtral MoE
- `deepseek.py` - DeepSeek MoE
- `qwen.py` - Qwen MoE

**多模态模型**
- `llava.py` - LLaVA 视觉语言模型
- `paligemma.py` - PaliGemma
- `chameleon.py` - Chameleon 多模态

**嵌入模型**
- `bert.py` - BERT 系列
- `e5.py` - E5 嵌入模型
- `jina.py` - Jina 嵌入模型

### 3. 模型加载器 (`model_loader/`)

负责从各种来源加载模型权重。

**关键文件**:
- `loader.py` - 主加载器
- `weight_loader.py` - 权重加载
- `utils.py` - 加载工具

### 4. 参数管理 (`parameter.py`)

管理模型参数的内存布局和访问。

**关键类**:
- `PerTensorWeightParameter` - 每张量权重参数
- `PerTensorScaleParameter` - 每张量缩放参数
- `ChannelScaleParameter` - 通道缩放参数

### 5. 自定义内核 (`kernels/`)

优化的 CUDA/HIP 内核实现。

**主要内核**:
- 注意力内核
- 量化内核
- 激活内核
- 归一化内核

## 添加新模型支持

### 步骤 1: 创建模型文件

在 `models/` 目录创建新模型文件：

```python
# models/my_model.py
from vllm.model_executor.models.llama import LlamaForCausalLM

class MyModelForCausalLM(LlamaForCausalLM):
    # 实现模型特定的配置和加载逻辑
    pass
```

### 步骤 2: 实现模型配置

```python
@dataclass
class MyModelConfig(ModelConfig):
    # 模型特定的配置参数
    my_param: int = 128
```

### 步骤 3: 注册模型

在 `models/__init__.py` 中注册：

```python
from .my_model import MyModelForCausalLM

_MODEL_REGISTRY.register_model("MyModel", MyModelForCausalLM)
```

### 步骤 4: 添加测试

在 `tests/models/` 创建对应测试：

```python
# tests/models/test_my_model.py
@pytest.mark.parametrize("model", ["my-model-small"])
def test_my_model(vllm_runner, model):
    # 测试逻辑
    pass
```

## 模型接口

### 基础模型接口

```python
class ModelInterface:
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        """初始化模型"""

    def forward(self, input_ids: torch.Tensor,
                positions: torch.Tensor,
                kv_caches: List[KVCache],
                **kwargs) -> torch.Tensor:
        """前向传播"""
        pass

    def compute_logits(self, hidden_states: torch.Tensor,
                      sampling_metadata: SamplingMetadata) -> torch.Tensor:
        """计算 logits"""
        pass

    def supported_load_methods(self) -> List[str]:
        """支持的加载方法"""
        pass
```

## 量化支持

### 支持的量化格式

1. **GPTQ** - 4-bit 量化
2. **AWQ** - 激活感知量化
3. **INT8** - 8-bit 整数量化
4. **FP8** - 8-bit 浮点量化
5. **INT4** - 4-bit 整数量化

### 量化层实现

量化层在 `layers/quantization/` 中实现。

## 性能优化

### 1. 内核融合

将多个操作融合为单个内核：

```python
# 融合 QKV 投影
qkv = fused_qkv_proj(hidden_state)

# 融合 FFN
hidden = fused_ffn(hidden_state)
```

### 2. 内存优化

- 权重预分配
- KV 缓存共享
- 梯度检查点

### 3. 并行策略

- 张量并行
- 流水线并行
- 数据并行

## 调试技巧

### 检查模型权重

```python
from vllm.model_executor.model_loader import weight_loader

# 打印权重形状
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")
```

### 性能分析

```python
import torch

# 启用性能分析
with torch.profiler.profile() as prof:
    outputs = model(inputs)

print(prof.key_averages().table())
```

## 常见问题

### Q: 如何添加新的激活函数？

A: 在 `layers/activation.py` 中实现新的激活函数，并在 `ffn.py` 中使用。

### Q: 如何支持新的量化格式？

A: 在 `layers/quantization/` 中实现量化层，并在模型加载器中添加加载逻辑。

### Q: 如何调试模型加载问题？

A: 使用 `VLLM_LOGGING_LEVEL=DEBUG` 获取详细日志，检查权重加载过程。

## 相关文档

- [根级文档](../../CLAUDE.md)
- [v1 架构](../v1/CLAUDE.md)
- [配置系统](../config/CLAUDE.md)
