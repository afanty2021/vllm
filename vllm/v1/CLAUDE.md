# vLLM v1 架构模块

> 模块路径: `vllm/v1/`
> 最后更新: 2026-03-21

## 模块概述

v1 是 vLLM 的下一代架构，旨在提供更高的性能和更好的可扩展性。它是对原有引擎的重新设计，采用更模块化的架构。

## 目录结构

```
vllm/v1/
├── attention/          # 注意力机制实现
├── core/               # 核心调度和管理
├── engine/             # LLM 引擎实现
├── executor/           # 执行器（GPU/CPU）
├── worker/             # 工作进程
├── metrics/            # 性能指标
├── kv_offload/         # KV 缓存卸载
├── spec_decode/        # 推测解码
├── structured_output/  # 结构化输出
├── sample/             # 采样策略
└── pool/               # 对象池
```

## 核心组件

### 1. 注意力机制 (`attention/`)

实现 PagedAttention 和其他注意力变体。

**关键文件**:
- `attention.py` - 注意力接口定义
- `paged_attn.py` - PagedAttention 实现
- `prefix_prefill.py` - 前缀预填充

### 2. 核心组件 (`core/`)

调度器和块管理器的核心实现。

**关键文件**:
- `scheduler.py` - 请求调度器
- `block_manager.py` - 块管理器
- `cache_manager.py` - 缓存管理器

### 3. 引擎 (`engine/`)

LLM 推理引擎的核心实现。

**关键文件**:
- `llm_engine.py` - 主引擎类
- `async_llm_engine.py` - 异步引擎
- `processor.py` - 请求处理器

### 4. 执行器 (`executor/`)

负责在 GPU/CPU 上执行模型计算。

**关键文件**:
- `gpu_executor.py` - GPU 执行器
- `cpu_executor.py` - CPU 执行器
- `neuron_executor.py` - AWS Neuron 执行器

### 5. 工作进程 (`worker/`)

工作进程的实现和管理。

**关键文件**:
- `worker.py` - 工作进程基类
- `gpu_worker.py` - GPU 工作进程
- `model_runner.py` - 模型运行器

### 6. 推测解码 (`spec_decode/`)

使用小模型加速大模型推理。

**关键文件**:
- `spec_decode_worker.py` - 推测解码工作进程
- `draft_model_runner.py` - 草稿模型运行器
- `mqa_scoring.py` - MQA 评分

### 7. KV 缓存卸载 (`kv_offload/`)

将 KV 缓存卸载到 CPU 或其他存储。

**关键文件**:
- `offload_manager.py` - 卸载管理器
- `cpu_offload.py` - CPU 卸载实现

## 架构设计原则

### 1. 模块化

v1 架构采用更清晰的模块分离，每个组件有明确的职责。

### 2. 可扩展性

- 支持新的执行器后端
- 支持新的调度策略
- 支持新的注意力实现

### 3. 性能优化

- CUDA 图支持
- 内核融合
- 内存预分配

## 关键接口

### Scheduler

```python
class Scheduler:
    def schedule(self) -> SchedulerOutput:
        """调度请求，返回执行计划"""
        pass

    def block_manager(self) -> BlockManager:
        """获取块管理器"""
        pass
```

### Executor

```python
class Executor:
    def execute_model(self, execute_model_req) -> List[SamplerOutput]:
        """执行模型计算"""
        pass

    def profile_run(self):
        """性能分析运行"""
        pass
```

### Worker

```python
class Worker:
    def execute_model(self, execute_model_req) -> List[SamplerOutput]:
        """在工作进程上执行模型"""
        pass

    def profile_num_gpu_blocks(self) -> int:
        """分析 GPU 块数量"""
        pass
```

## 开发指南

### 添加新的注意力实现

1. 在 `attention/` 创建新文件
2. 实现注意力接口
3. 在调度器中注册
4. 添加测试

### 添加新的执行器

1. 在 `executor/` 创建新文件
2. 继承 `ExecutorBase`
3. 实现必需方法
4. 在引擎工厂中注册

## 与旧版引擎的区别

| 特性 | 旧版引擎 | v1 引擎 |
|------|----------|---------|
| 架构 | 单体式 | 模块化 |
| 可扩展性 | 有限 | 高 |
| 性能 | 优化较少 | 深度优化 |
| 代码组织 | 混合 | 清晰分层 |

## 性能特性

1. **CUDA 图**: 减少内核启动开销
2. **内核融合**: 减少内存访问
3. **内存池**: 减少分配开销
4. **异步执行**: 提高吞吐量

## 调试技巧

```bash
# 启用详细日志
VLLM_LOGGING_LEVEL=DEBUG python -m vllm serve ...

# 性能分析
VLLM_USE_TRACING=1 python -m vllm serve ...
```

## 相关文档

- [根级文档](../../CLAUDE.md)
- [模型执行器](../model_executor/CLAUDE.md)
- [配置系统](../config/CLAUDE.md)
