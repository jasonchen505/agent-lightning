# Agent Lightning 增量学习笔记

> **文档目的**: 记录在复现过程中对比之前两轮分析增量新学到的点
> **更新方式**: 每完成一个Phase，记录新发现和深入理解
> **学习日期**: 2026年6月

---

## Phase 1: 环境搭建与基础验证 (Day 1-2)

### 1.1 环境搭建增量发现

#### 发现1: Python版本选择的重要性

**之前理解**: 只需要Python 3.10+

**深入发现**:
```bash
# 项目实际使用的Python版本
cat .python-version
# 输出: 3.12

# uv创建虚拟环境时会下载指定版本
uv venv --python 3.10
# 实际下载: cpython-3.10.20-linux-x86_64-gnu
```

**增量理解**:
- 项目虽然支持3.10+，但开发环境使用3.12
- `uv`会自动下载指定版本的Python，不需要手动安装
- 不同Python版本可能有细微的类型提示差异

**面试话术补充**:
> "Agent Lightning使用uv作为包管理器，它会自动下载并管理Python版本。开发环境使用Python 3.12，但支持3.10+。uv比pip快很多，因为它使用Rust实现，并且有更好的依赖解析"

#### 发现2: 依赖组的精细设计

**之前理解**: 只知道有`apo`和`verl`两个可选依赖

**深入发现**:
```toml
# pyproject.toml中的依赖组设计
[dependency-groups]
dev = ["flake8", "pytest", "pyright", ...]  # 开发工具
experiment = ["random-word", "gdown"]  # 实验工具

# 核心依赖组（互斥）
core-legacy = ["agentops<=0.4.18", "openai<2.0.0"]
core-stable = ["agentops>=0.4.21", "openai>=2.0.0"]

# PyTorch相关（互斥）
torch-cpu = ["torch", "torchvision"]
torch-cu128 = ["torch", "torchvision"]  # CUDA 12.8

# 训练相关
torch-stable = ["torch>=2.8.0", "vllm>=0.10.2", "verl>=0.6.0"]
torch-legacy = ["torch==2.7.0", "vllm==0.9.2", "verl==0.5.0"]
```

**增量理解**:
- 依赖组之间有**冲突声明**，防止同时安装不兼容版本
- `torch-stable`和`torch-legacy`是互斥的，选择一个
- `core-legacy`和`core-stable`也是互斥的
- 这种设计避免了依赖地狱（dependency hell）

**面试话术补充**:
> "项目的依赖管理非常精细，使用uv的依赖组功能将依赖分为多个层次。核心依赖、PyTorch依赖、算法依赖各自独立，通过冲突声明防止不兼容版本同时安装。这种设计让用户可以根据需求选择合适的依赖组合"

### 1.2 Store API增量发现

#### 发现3: Span的完整结构

**之前理解**: Span只需要name、rollout_id、attempt_id、sequence_id

**深入发现**:
```python
# Span的实际结构（必须字段）
class Span(BaseModel):
    rollout_id: str           # Rollout ID
    attempt_id: str           # Attempt ID
    sequence_id: int          # 序列ID（用于排序）
    trace_id: str             # OpenTelemetry trace ID
    span_id: str              # OpenTelemetry span ID
    parent_id: Optional[str]  # 父span ID
    name: str                 # Span名称
    status: TraceStatus       # 状态
    attributes: Attributes    # 属性字典
    events: List[Event]       # 事件列表
    links: List[Link]         # 链接列表
    start_time: Optional[float]  # 开始时间
    end_time: Optional[float]    # 结束时间
    context: Optional[SpanContext]  # Span上下文
    parent: Optional[SpanContext]   # 父上下文
    resource: OtelResource   # 资源信息
```

**增量理解**:
- Span是OpenTelemetry ReadableSpan的精简版
- 保留了OpenTelemetry的核心字段，但更简洁
- `from_opentelemetry()`类方法可以从OTEL span转换
- 这种设计让Span既兼容OTEL生态，又有自己的特色

**面试话术补充**:
> "Span是Agent Lightning的核心数据模型，它基于OpenTelemetry的ReadableSpan但做了精简。保留了trace_id、span_id、parent_id等核心字段用于构建执行树，同时添加了rollout_id和attempt_id用于关联训练上下文。这种设计让Span既能被标准OTEL工具处理，又能支持Agent训练的特殊需求"

#### 发现4: Store的异步设计

**之前理解**: Store是异步的，但不知道为什么

**深入发现**:
```python
# Store的所有主要方法都是async
class LightningStore:
    async def enqueue_rollout(...) -> Rollout: ...
    async def dequeue_rollout(...) -> AttemptedRollout: ...
    async def add_span(...) -> Span: ...
    async def query_spans(...) -> Sequence[Span]: ...
```

**增量理解**:
- **为什么用异步**: Runner和Algorithm可能在不同线程/进程，需要高并发
- **异步的好处**: 避免阻塞，提高吞吐量
- **异步的挑战**: 需要正确处理并发访问，使用锁或线程安全的数据结构

**面试话术补充**:
> "Store使用异步设计是为了支持高并发场景。当多个Runner同时执行时，它们需要同时访问Store读写数据。异步接口避免了阻塞，提高了系统的吞吐量。底层实现使用aiologic库提供线程安全和异步安全的保证"

### 1.3 Tracer机制增量发现

#### 发现5: OTEL Instrumentation的实际工作方式

**之前理解**: Tracer通过Monkey-patch自动采集

**深入发现**:
```python
# 实际的instrumentation过程
class OtelTracer(Tracer):
    def trace_context(self, name, store, rollout_id, attempt_id):
        # 1. 创建TracerProvider
        tracer_provider = TracerProvider(...)
        
        # 2. 添加LightningSpanProcessor
        # 这个processor会拦截所有span并写入Store
        tracer_provider.add_span_processor(
            LightningSpanProcessor(store, rollout_id, attempt_id)
        )
        
        # 3. 获取tracer并开始trace
        tracer = tracer_provider.get_tracer("agent-lightning")
        
        # 4. 使用上下文管理器
        with tracer.start_as_current_span(name) as span:
            # 在此范围内，所有OTEL instrumentation自动生效
            yield
```

**增量理解**:
- 不是简单的Monkey-patch，而是通过OTEL的TracerProvider机制
- LightningSpanProcessor是关键，它拦截span并写入Store
- 使用上下文管理器确保span的父子关系正确
- 这种方式更优雅，也更符合OTEL的标准实践

**面试话术补充**:
> "Tracer的实现基于OpenTelemetry的TracerProvider机制。我们实现了一个自定义的LightningSpanProcessor，它会拦截所有创建的span，添加rollout_id和attempt_id信息，然后写入Store。通过上下文管理器，我们确保span的父子关系正确构建，形成完整的执行树"

---

## Phase 2: APO算法复现 (Day 3-5)

### 2.1 APO算法增量发现

#### 发现6: Textual Gradient的实现细节

**之前理解**: Textual Gradient是用LLM生成的批评

**深入发现**:
```python
# APO的Textual Gradient实现
GRADIENT_PROMPT_FILES = [
    Path(__file__).parent / "prompts" / "text_gradient_variant01.poml",
    Path(__file__).parent / "prompts" / "text_gradient_variant02.poml",
    Path(__file__).parent / "prompts" / "text_gradient_variant03.poml",
]

async def compute_textual_gradient(self, current_prompt, rollout_results):
    # 1. 随机选择一个gradient prompt模板
    tg_template = random.choice(self.gradient_prompt_files)
    
    # 2. 使用POML渲染prompt
    tg_msg = poml.poml(
        tg_template,
        context={
            "experiments": sampled_rollout_results,
            "prompt_template": current_prompt.prompt_template.template,
        },
        format="openai_chat",
    )
    
    # 3. 调用LLM生成critique
    critique_response = await self.async_openai_client.chat.completions.create(
        model=self.gradient_model,
        messages=tg_msg["messages"],
        temperature=self.diversity_temperature,
    )
    
    return critique_response.choices[0].message.content
```

**增量理解**:
- 使用**POML**（Prompt Markup Language）模板化gradient prompt
- 有**多个variant**的gradient prompt，随机选择增加多样性
- `diversity_temperature`控制critique的多样性
- 这种设计比单一prompt更鲁棒

**面试话术补充**:
> "APO的Textual Gradient实现有几个巧妙的设计：1) 使用POML模板化prompt，便于维护和迭代；2) 有多个variant的gradient prompt，随机选择增加critique的多样性；3) 通过temperature参数控制多样性程度。这种设计比单一prompt更鲁棒，能生成更全面的批评"

#### 发现7: Beam Search的实现策略

**之前理解**: Beam Search维护多个候选prompt

**深入发现**:
```python
async def _evaluate_and_select_beam(self, candidates, resource_name, val_dataset_iterator, round_num):
    # 1. 评估所有候选
    val_batch = next(val_dataset_iterator)
    for prompt in candidates:
        _, score = await self.evaluate_prompt_on_batch(
            prompt, resource_name, val_batch, mode="val"
        )
        prompt.score = score
    
    # 2. 按分数排序
    sorted_prompts = [p for p in sorted(candidates, key=lambda x: cast(float, x.score), reverse=True)]
    
    # 3. 选择top-k
    selected_prompts = sorted_prompts[:self.beam_width]
    
    return selected_prompts
```

**增量理解**:
- Beam Search是**全局选择**，不是贪心
- 每轮从**所有候选**（包括旧beam和新生成）中选择top-k
- 这种策略能跳出局部最优

**面试话术补充**:
> "APO的Beam Search实现是全局选择策略。每轮从所有候选（包括旧beam和新生成的）中选择top-k，而不是只从新生成的中选择。这种设计能跳出局部最优，保持beam的多样性"

### 2.2 训练流程增量发现

#### 发现8: APO的Rollout执行流程

**之前理解**: Algorithm直接执行rollout

**深入发现**:
```python
# APO的实际执行流程
async def evaluate_prompt_on_batch(self, prompt, resource_name, dataset, mode):
    store = self.get_store()
    
    # 1. 安装prompt为named resource
    resources = {resource_name: prompt.prompt_template}
    resource_update = await store.update_resources(prompt.version, resources)
    
    # 2. 批量enqueue rollouts
    rollout_ids = []
    for t in dataset:
        r = await store.enqueue_rollout(input=t, mode=mode, resources_id=resource_update.resources_id)
        rollout_ids.append(r.rollout_id)
    
    # 3. 等待rollouts完成
    finished = await store.wait_for_rollouts(rollout_ids=rollout_ids, timeout=timeout)
    
    # 4. 查询结果
    rollout_results = await self.get_rollout_results(finished)
    
    return rollout_results, avg_reward
```

**增量理解**:
- Algorithm不直接执行rollout，而是通过Store协调
- **资源版本化**: 每个prompt版本都有唯一ID
- **异步等待**: 使用`wait_for_rollouts`等待所有rollout完成
- 这种设计让Algorithm和Runner完全解耦

**面试话术补充**:
> "APO的执行流程体现了Algorithm和Runner的解耦。Algorithm不直接执行rollout，而是将任务enqueue到Store，Runner从Store领取任务执行。资源（prompt）有版本管理，每个版本有唯一ID，便于追踪和回滚。这种设计让系统可以灵活扩展，Algorithm和Runner可以分布在不同进程甚至不同机器"

---

## Phase 3: VERL RL训练复现 (Day 6-10)

### 3.1 VERL集成增量发现

#### 发现9: VERL的Hydra配置管理

**之前理解**: VERL使用字典配置

**深入发现**:
```python
# VERL的配置合并流程
class VERL(Algorithm):
    def __init__(self, config: dict[str, Any]):
        # 1. 加载基础配置（从verl的包中）
        with initialize(version_base=None, config_path="pkg://agentlightning/verl"):
            base_cfg = compose(config_name="config")
        
        # 2. 合并用户配置
        override_conf = OmegaConf.create(config)
        OmegaConf.set_struct(base_cfg, False)  # 允许添加新字段
        self.config = OmegaConf.merge(base_cfg, override_conf)
```

**增量理解**:
- 使用**Hydra**进行配置管理，支持配置合并和覆盖
- 基础配置来自verl包，用户配置覆盖基础配置
- `OmegaConf.set_struct(base_cfg, False)`允许添加新字段
- 这种设计让配置既标准化又灵活

**面试话术补充**:
> "VERL使用Hydra进行配置管理，这是一个非常优雅的设计。基础配置来自verl框架的默认值，用户只需要覆盖需要修改的部分。比如用户只需要指定模型路径和batch size，其他配置使用默认值。Hydra会自动合并配置，确保一致性"

#### 发现10: Trajectory Level Aggregation

**之前理解**: 不知道有这个功能

**深入发现**:
```yaml
# config.yaml中的trajectory level配置
agentlightning:
  trace_aggregator:
    level: transition  # 或 trajectory
    trajectory_max_prompt_length: 2048
    trajectory_max_response_length: 8192
```

```python
# 启用trajectory level
config["agentlightning"] = {
    "trace_aggregator": {
        "level": "trajectory",
        "trajectory_max_prompt_length": 2048,
        "trajectory_max_response_length": 8192,
    }
}
```

**增量理解**:
- **transition level**: 每个turn独立训练
- **trajectory level**: 整个多轮对话作为一个训练样本
- trajectory level更高效，但需要更多显存
- 这是一个重要的优化选项

**面试话术补充**:
> "VERL支持两种训练粒度：transition level和trajectory level。Transition level将每个对话轮次作为独立样本训练，trajectory level将整个多轮对话作为一个样本。Trajectory level更高效，因为GPU只需要处理一次prompt，但需要更多显存来存储完整的对话历史。这个选择取决于任务特点和显存限制"

### 3.2 训练配置增量发现

#### 发现11: FSDP配置的重要性

**之前理解**: 不知道FSDP是什么

**深入发现**:
```python
config["actor_rollout_ref"]["actor"]["fsdp_config"] = {
    "param_offload": True,      # 参数卸载到CPU
    "optimizer_offload": True,  # 优化器卸载到CPU
}
```

**增量理解**:
- **FSDP**: Fully Sharded Data Parallel，分布式训练技术
- **param_offload**: 将模型参数卸载到CPU，节省GPU显存
- **optimizer_offload**: 将优化器状态卸载到CPU，进一步节省显存
- 在4090这种24GB显存的卡上，offload是必要的

**面试话术补充**:
> "FSDP是PyTorch的分布式训练技术，通过将模型参数和优化器状态分片到多个GPU来支持大模型训练。在4090这种24GB显存的卡上，我们通常需要启用param_offload和optimizer_offload，将部分数据卸载到CPU内存。这会增加一些通信开销，但能显著降低GPU显存占用"

#### 发现12: vLLM的gpu_memory_utilization

**之前理解**: 不知道这个参数的作用

**深入发现**:
```python
config["actor_rollout_ref"]["rollout"]["gpu_memory_utilization"] = 0.6
```

**增量理解**:
- 这个参数控制vLLM使用的GPU显存比例
- 设为0.6表示vLLM最多使用60%的GPU显存
- 剩余40%留给训练（actor、ref等）
- 需要在推理和训练之间平衡

**面试话术补充**:
> "gpu_memory_utilization是一个关键参数，它控制vLLM推理引擎使用的GPU显存比例。设为0.6表示vLLM使用60%显存，剩余40%留给训练。如果设得太高，训练时可能OOM；设得太低，推理性能下降。通常需要根据模型大小和batch size来调整"

---

## Phase 4: Unsloth SFT复现 (Day 11-13)

### 4.1 SFT训练增量发现

#### 发现13: LoRA的显存优化

**之前理解**: LoRA减少训练参数

**深入发现**:
```python
# LoRA的显存计算
# 原始模型: 1.5B参数 * 2字节(FP16) = 3GB
# LoRA (rank=16): 1.5B * 16 * 2 / 1000 ≈ 48MB
# 显存节省: 约98%
```

**增量理解**:
- LoRA只训练低秩适配矩阵，不更新原始权重
- 显存节省非常显著，从3GB降到48MB
- 但推理时需要合并LoRA权重，有少量开销
- 适合显存有限的场景

**面试话术补充**:
> "LoRA通过低秩分解来减少训练参数。对于1.5B参数的模型，原始训练需要3GB显存存储梯度和优化器状态。使用LoRA (rank=16)后，只需要约48MB，节省98%显存。这让4090这种24GB的卡也能训练较大的模型"

---

## 跨Phase综合洞察

### 综合洞察1: 三种训练方法的对比

| 维度 | APO | VERL RL | Unsloth SFT |
|------|-----|---------|-------------|
| **GPU需求** | 无 | 1-8卡 | 1卡 |
| **优化目标** | Prompt | 模型权重 | 模型权重 |
| **数据需求** | 任务+评估 | 任务+奖励函数 | 高质量标注数据 |
| **训练效率** | 低（多次LLM调用） | 中 | 高 |
| **适用场景** | 无法微调模型 | 有明确评估指标 | 有标注数据 |
| **理论基础** | Textual Gradient | PPO/GRPO | Maximum Likelihood |

### 综合洞察2: 8卡4090的最佳实践

```python
# 1.5B模型: 单卡或2卡
config_1_5b = {
    "trainer": {"n_gpus_per_node": 2},
    "actor_rollout_ref": {
        "rollout": {"gpu_memory_utilization": 0.6},
        "actor": {"fsdp_config": {"param_offload": False}},
    },
}

# 7B模型: 4-8卡
config_7b = {
    "trainer": {"n_gpus_per_node": 8},
    "actor_rollout_ref": {
        "rollout": {"gpu_memory_utilization": 0.4},
        "actor": {"fsdp_config": {"param_offload": True, "optimizer_offload": True}},
    },
}
```

### 综合洞察3: 面试常见追问及应对

#### 追问1: "为什么选择Agent Lightning而不是其他框架？"

**回答**:
> "选择Agent Lightning主要基于三个考虑：1) **零代码修改**，不需要重写现有Agent代码；2) **框架无关**，支持LangChain、AutoGen等各种框架；3) **算法多样**，同时支持APO和RL。相比之下，其他框架通常需要特定的Agent实现或只支持单一算法"

#### 追问2: "训练过程中遇到的最大挑战是什么？"

**回答**:
> "最大的挑战是**显存管理**。4090只有24GB显存，要同时容纳推理引擎（vLLM）和训练组件（actor、ref）。解决方案是：1) 使用FSDP offload将参数和优化器卸载到CPU；2) 降低gpu_memory_utilization；3) 使用gradient checkpointing。最终在1.5B模型上实现了稳定的训练"

#### 追问3: "如何验证训练效果？"

**回答**:
> "我使用三重验证：1) **定量指标**，监控奖励曲线是否上升；2) **定性分析**，查看优化后的prompt或模型输出是否合理；3) **A/B测试**，对比训练前后的Agent表现。在APO实验中，准确率从56.9%提升到72.1%；在VERL实验中，奖励从0.4提升到0.6+"

---

## 持续更新区

*以下内容将在后续Phase中更新*

### Phase 5: 高级功能探索 (Day 14-16)
- LLM Proxy的配置和使用
- 分布式训练的实践经验
- 自定义Algorithm的实现

---

*本文档将持续更新，记录增量学习的点点滴滴*
*最后更新: 2026年6月26日*
