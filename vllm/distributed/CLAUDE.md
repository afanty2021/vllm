# 分布式执行模块

> 模块路径: `vllm/distributed/`
> 最后更新: 2026-03-21

## 模块概述

分布式执行模块提供跨多个 GPU 和节点的并行推理能力，支持张量并行、流水线并行和数据并行。

## 目录结构

```
vllm/distributed/
├── __init__.py
├── parallel_state.py         # 并行状态管理
├── communication_op.py       # 通信操作
├── utils.py                  # 分布式工具
├── stateless_coordinator.py  # 无状态协调器
├── kv_events.py              # KV 事件处理
├── device_communicators/     # 设备通信器
│   ├── __init__.py
│   ├── mpi.py               # MPI 通信
│   ├── pynccl.py            # NCCL 通信
│   └── torch.py             # PyTorch 通信
├── kv_transfer/             # KV 传输
├── weight_transfer/         # 权重传输
├── ec_transfer/             # EC 传输
├── elastic_ep/              # 弹性专家并行
└── eplb/                    # EP 负载均衡
```

## 并行策略

### 1. 张量并行 (Tensor Parallelism, TP)

将模型权重分割到多个 GPU 上，每个 GPU 存储模型的一部分。

**配置**:
```python
from vllm import LLM

llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    tensor_parallel_size=4  # 使用 4 个 GPU
)
```

**工作原理**:
- 将注意力头的 Q、K、V 投影分割
- 将 FFN 层的中间激活分割
- 使用 all-reduce 聚合结果

### 2. 流水线并行 (Pipeline Parallelism, PP)

将模型层分割到多个 GPU 上，每个 GPU 处理模型的一部分层。

**配置**:
```python
llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    pipeline_parallel_size=4  # 使用 4 个 GPU
)
```

**工作原理**:
- 将模型层分段
- 每个阶段处理一部分层
- 微批处理以提高 GPU 利用率

### 3. 数据并行 (Data Parallelism, DP)

在不同 GPU 上处理不同的请求。

**配置**:
```python
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    data_parallel_size=2  # 使用 2 个数据并行副本
)
```

### 4. 混合并行

结合多种并行策略：

```python
llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    tensor_parallel_size=4,     # 4 个 GPU 张量并行
    pipeline_parallel_size=2,   # 2 个 GPU 流水线并行
    data_parallel_size=2,       # 2 个数据并行副本
    # 总共需要 4 × 2 × 2 = 16 个 GPU
)
```

## 核心组件

### 1. 并行状态 (`parallel_state.py`)

管理分布式执行的初始化和状态。

**关键类**:
```python
class ParallelState:
    """管理并行执行状态"""

    def init_model_parallel(
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        data_parallel_size: int = 1,
        backend: str = "nccl"
    ):
        """初始化模型并行"""
        pass

    def destroy_model_parallel():
        """销毁模型并行"""
        pass

    @property
    def tensor_parallel_rank(self) -> int:
        """当前张量并行 rank"""
        pass

    @property
    def pipeline_parallel_rank(self) -> int:
        """当前流水线并行 rank"""
        pass
```

### 2. 设备通信器 (`device_communicators/`)

提供不同后端的通信实现。

**NCCL 通信器** (`pynccl.py`):
- NVIDIA GPU 上的高性能通信
- 支持 all-reduce、all-gather 等操作

**MPI 通信器** (`mpi.py`):
- 支持 MPI 后端
- 适用于 CPU 和异构集群

**PyTorch 通信器** (`torch.py`):
- 基于 PyTorch distributed
- 跨平台支持

### 3. 通信操作 (`communication_op.py`)

提供各种通信原语。

```python
def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce 操作（张量并行）"""
    pass

def tensor_model_parallel_all_gather(
    input_: torch.Tensor,
    dim: int = 0
) -> torch.Tensor:
    """All-gather 操作"""
    pass

def pipeline_model_parallel_send(
    input_: torch.Tensor,
    dst_rank: int
) -> None:
    """流水线并行发送"""
    pass

def pipeline_model_parallel_recv(
    src_rank: int,
    shape: torch.Size
) -> torch.Tensor:
    """流水线并行接收"""
    pass
```

## 使用示例

### 单机多卡

```bash
# 使用 4 个 GPU
vllm serve meta-llama/Llama-2-7b-hf --tensor-parallel-size 4
```

```python
from vllm import LLM

llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    tensor_parallel_size=4
)
```

### 多机多卡

**节点 0**:
```bash
VLLM_HOST_IP=10.0.0.1 \
vllm serve meta-llama/Llama-2-70b-hf \
    --tensor-parallel-size 8 \
    --pipeline-parallel-size 2 \
    --distributed-executor-backend mp \
    --host 0.0.0.0 \
    --port 8000
```

**节点 1**:
```bash
VLLM_HOST_IP=10.0.0.1 \
vllm serve meta-llama/Llama-2-70b-hf \
    --tensor-parallel-size 8 \
    --pipeline-parallel-size 2 \
    --distributed-executor-backend mp
```

### Ray 分布式

```bash
vllm serve meta-llama/Llama-2-70b-hf \
    --tensor-parallel-size 8 \
    --pipeline-parallel-size 2 \
    --distributed-executor-backend ray
```

## 通信后端

### NCCL (推荐)

```python
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    tensor_parallel_size=4,
    distributed_executor_backend="nccl"
)
```

### MPI

```bash
# 使用 MPI 启动
mpirun -np 4 vllm serve meta-llama/Llama-2-7b-hf \
    --tensor-parallel-size 4 \
    --distributed-executor-backend mpi
```

### Ray

```python
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    tensor_parallel_size=4,
    distributed_executor_backend="ray"
)
```

## 高级功能

### 1. KV 传输 (`kv_transfer/`)

在不同设备间传输 KV 缓存。

```python
from vllm.distributed.kv_transfer import KVTransferAgent

agent = KVTransferAgent(
    kv_rank=0,
    kv_parallel_size=2
)
```

### 2. 权重传输 (`weight_transfer/`)

优化权重加载和传输。

```python
from vllm.distributed.weight_transfer import load_weights

weights = load_weights(
    model_config,
    tensor_parallel_rank=0,
    tensor_parallel_size=4
)
```

### 3. 弹性专家并行 (`elastic_ep/`)

动态调整 MoE 专家分布。

```python
from vllm.distributed.elastic_ep import ElasticExpertParallel

ep = ElasticExpertParallel(
    num_experts=8,
    num_experts_per_rank=2
)
```

## 性能优化

### 1. 通信重叠

```python
# 启用通信与计算重叠
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    tensor_parallel_size=4,
    overlap_comm_compute=True
)
```

### 2. 梯度检查点

```python
# 减少内存使用
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    enforce_eager=True,
    gradient_checkpointing=True
)
```

### 3. 通信压缩

```python
# 使用量化通信
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    tensor_parallel_size=4,
    comm_quantization_bits=8
)
```

## 调试技巧

### 检查并行状态

```python
from vllm.distributed import parallel_state

print(f"TP Rank: {parallel_state.get_tensor_parallel_rank()}")
print(f"TP Size: {parallel_state.get_tensor_parallel_world_size()}")
print(f"PP Rank: {parallel_state.get_pipeline_parallel_rank()}")
print(f"PP Size: {parallel_state.get_pipeline_parallel_world_size()}")
```

### 通信分析

```bash
# 启用通信分析
NCCL_DEBUG=INFO vllm serve <model> --tensor-parallel-size 4
```

### 性能监控

```python
from vllm.distributed import monitoring

# 启用性能监控
monitoring.enable_profiling()
```

## 常见问题

### Q: 如何选择并行策略？

A:
- **小模型** (< 7B): 张量并行
- **中等模型** (7B-70B): 张量并行 + 流水线并行
- **大模型** (> 70B): 张量并行 + 流水线并行 + 数据并行

### Q: 如何处理通信瓶颈？

A:
1. 使用高速互连（InfiniBand、NVLink）
2. 减少跨节点通信
3. 使用通信重叠
4. 增加批大小

### Q: 如何调试分布式问题？

A:
1. 使用 `VLLM_LOGGING_LEVEL=DEBUG`
2. 检查 NCCL 环境变量
3. 验证网络连接
4. 使用单节点测试

## 环境变量

### NCCL 相关

```bash
# NCCL 调试
export NCCL_DEBUG=INFO

# NCCL 使用的网络接口
export NCCL_SOCKET_IFNAME=eth0

# 禁用 NCCL IB
export NCCL_IB_DISABLE=1

# NCCL 超时
export NCCL_BLOCKING_WAIT=1
```

### vLLM 相关

```bash
# 主节点 IP
export VLLM_HOST_IP=10.0.0.1

# 端口范围
export VLLM_PORT_RANGE=10000-11000
```

## 相关文档

- [根级文档](../../CLAUDE.md)
- [v1 架构](../v1/CLAUDE.md)
- [配置系统](../config/CLAUDE.md)
- [入口点](../entrypoints/CLAUDE.md)
