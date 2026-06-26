# Agent Lightning 8卡4090复现计划

> **硬件资源**: 8 x NVIDIA RTX 4090 (24GB VRAM each)
> **目标**: 完整复现Agent Lightning项目的核心功能
> **时间估算**: 约2-3周

---

## 目录

1. [资源评估与可行性分析](#1-资源评估与可行性分析)
2. [环境搭建计划](#2-环境搭建计划)
3. [分阶段复现计划](#3-分阶段复现计划)
4. [详细执行步骤](#4-详细执行步骤)
5. [预期挑战与解决方案](#5-预期挑战与解决方案)
6. [验证检查点](#6-验证检查点)

---

## 1. 资源评估与可行性分析

### 1.1 硬件资源评估

| 资源 | 规格 | 项目需求 | 可行性 |
|------|------|---------|--------|
| GPU | 8x RTX 4090 (24GB) | 最少1卡，推荐4-8卡 | ✅ 充足 |
| 显存 | 24GB/卡 | 1.5B模型约需8-12GB | ✅ 充足 |
| 内存 | 建议128GB+ | Runner并行需要大内存 | ⚠️ 需确认 |
| 存储 | 建议500GB+ SSD | 模型检查点、数据集 | ⚠️ 需确认 |

### 1.2 模型选择策略

**首选模型**: Qwen2.5-1.5B-Instruct
- 显存需求: ~8GB (FP16), ~4GB (INT4)
- 4090可以轻松支持
- 项目示例默认使用此模型

**可选扩展**: Qwen2.5-7B-Instruct
- 显存需求: ~16GB (FP16), ~8GB (INT4)
- 4090可以支持，但需要更谨慎的显存管理
- 性能更好，适合验证效果

### 1.3 算法可行性

| 算法 | GPU需求 | 8卡4090可行性 | 备注 |
|------|---------|--------------|------|
| APO | 无GPU需求（仅需API） | ✅ 完全可行 | 依赖OpenAI API |
| VERL (1.5B) | 1-2卡 | ✅ 完全可行 | 默认配置 |
| VERL (7B) | 4-8卡 | ✅ 可行 | 需要优化配置 |
| Unsloth SFT | 1卡 | ✅ 完全可行 | 4bit量化 |

---

## 2. 环境搭建计划

### 2.1 基础环境

```bash
# 1. 系统要求
# - Ubuntu 22.04 LTS (推荐)
# - Python 3.10+
# - CUDA 12.1+
# - NVIDIA Driver 535+

# 2. 检查GPU
nvidia-smi

# 3. 安装uv (推荐的包管理器)
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
```

### 2.2 项目安装

```bash
# 1. 克隆项目
cd /data/home/yizhou
git clone https://github.com/microsoft/agent-lightning.git
cd agent-lightning

# 2. 创建虚拟环境
uv venv --python 3.10
source .venv/bin/activate

# 3. 安装基础依赖 (CPU)
uv sync --group dev --extra apo

# 4. 安装GPU依赖 (用于VERL和SFT)
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pip install flash-attn==2.8.1 --no-build-isolation
pip install vllm==0.10.2
pip install verl==0.6.0

# 5. 安装Agent框架依赖
uv sync --group agents

# 6. 验证安装
python -c "import agentlightning; print(agentlightning.__version__)"
```

### 2.3 API配置

```bash
# APO算法需要OpenAI API
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="https://api.openai.com/v1"  # 或其他兼容API

# 如果使用本地LLM代理（如Ollama）
export OPENAI_BASE_URL="http://localhost:11434/v1"
export OPENAI_API_KEY="dummy"
```

---

## 3. 分阶段复现计划

### Phase 1: 基础组件验证 (Day 1-2)

**目标**: 验证环境搭建正确，理解核心组件

| 任务 | 时间 | 验证标准 |
|------|------|---------|
| 1.1 环境搭建 | 2小时 | `import agentlightning` 成功 |
| 1.2 运行minimal示例 | 1小时 | write_traces.py成功执行 |
| 1.3 理解Store接口 | 2小时 | 能解释主要API的作用 |
| 1.4 理解Tracer机制 | 2小时 | 能解释Span的产生过程 |

**执行命令**:
```bash
# 1.1 验证安装
python -c "import agentlightning as agl; print(f'Version: {agl.__version__}')"

# 1.2 运行minimal示例
cd examples/minimal
python write_traces.py

# 1.3 测试Store
python -c "
import asyncio
from agentlightning.store.memory import InMemoryLightningStore

async def test_store():
    store = InMemoryLightningStore()
    # 测试enqueue和dequeue
    rollout = await store.enqueue_rollout(input={'test': 'data'})
    print(f'Enqueued rollout: {rollout.rollout_id}')
    
    attempted = await store.dequeue_rollout()
    print(f'Deqpple attempted: {attempted.rollout_id}')

asyncio.run(test_store())
"
```

### Phase 2: APO算法复现 (Day 3-5)

**目标**: 完整运行APO算法，优化Room Selector Agent

| 任务 | 时间 | 验证标准 |
|------|------|---------|
| 2.1 理解APO原理 | 2小时 | 能解释Textual Gradient |
| 2.2 运行room_selector示例 | 2小时 | 成功完成2轮优化 |
| 2.3 分析优化结果 | 1小时 | 准确率有提升 |
| 2.4 尝试自定义Agent | 3小时 | 编写并训练自己的Agent |

**执行命令**:
```bash
# 2.1 运行APO示例
cd examples/apo
python room_selector_apo.py

# 预期输出:
# Round 1/2...
#   Evaluating prompt...
#   Score: 0.569 -> 0.638
# Round 2/2...
#   Score: 0.638 -> 0.721

# 2.2 查看优化日志
cat apo.log
```

### Phase 3: VERL RL训练复现 (Day 6-10)

**目标**: 使用VERL进行RL训练，优化SQL Agent或Calc Agent

| 任务 | 时间 | 验证标准 |
|------|------|---------|
| 3.1 理解VERL架构 | 3小时 | 能解释GRPO算法 |
| 3.2 准备数据集 | 2小时 | 下载并验证数据集 |
| 3.3 单卡训练测试 | 3小时 | 成功完成1个epoch |
| 3.4 多卡训练测试 | 3小时 | 8卡并行训练成功 |
| 3.5 分析训练结果 | 2小时 | 奖励曲线有提升 |

**执行命令**:
```bash
# 3.1 准备Calc-X数据集
cd examples/calc_x
# 下载数据集（从README中的Google Drive链接）
unzip calc-x-data.zip -d data

# 3.2 单卡测试（快速验证）
python train_calc_agent.py --ci-fast --n-runners 2

# 3.3 多卡训练
bash ../../scripts/restart_ray.sh
python train_calc_agent.py --n-runners 8

# 3.4 监控训练
# 在另一个终端查看GPU使用
watch -n 1 nvidia-smi
```

### Phase 4: Unsloth SFT复现 (Day 11-13)

**目标**: 使用Unsloth进行SFT训练

| 任务 | 时间 | 验证标准 |
|------|------|---------|
| 4.1 理解SFT流程 | 2小时 | 能解释LoRA原理 |
| 4.2 安装Unsloth | 1小时 | 安装成功 |
| 4.3 运行SFT训练 | 3小时 | 成功完成训练 |
| 4.4 对比RL和SFT | 2小时 | 理解两种方法的区别 |

**执行命令**:
```bash
# 4.1 安装Unsloth
pip install unsloth==2025.10.1 unsloth_zoo==2025.10.1 bitsandbytes peft datasets transformers trl kernels

# 4.2 运行SFT示例
cd examples/unsloth
python sft_allinone.py
```

### Phase 5: 高级功能探索 (Day 14-16)

**目标**: 探索分布式训练、LLM Proxy等高级功能

| 任务 | 时间 | 验证标准 |
|------|------|---------|
| 5.1 LLM Proxy使用 | 2小时 | 成功配置Proxy |
| 5.2 分布式训练 | 3小时 | 多进程训练成功 |
| 5.3 自定义Algorithm | 3小时 | 实现简单自定义算法 |
| 5.4 Dashboard使用 | 2小时 | 可视化训练过程 |

---

## 4. 详细执行步骤

### 4.1 Day 1: 环境搭建与基础验证

**上午 (3小时)**:
```bash
# 1. 检查系统环境
uname -a
python3 --version
nvidia-smi

# 2. 安装uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. 克隆并安装项目
cd /data/home/yizhou
git clone https://github.com/microsoft/agent-lightning.git
cd agent-lightning
uv venv --python 3.10
source .venv/bin/activate

# 4. 安装依赖
uv sync --group dev --extra apo
```

**下午 (3小时)**:
```bash
# 1. 运行minimal示例
cd examples/minimal
python write_traces.py

# 2. 阅读源码
# 重点理解:
# - agentlightning/store/base.py (Store接口)
# - agentlightning/tracer/base.py (Tracer接口)
# - agentlightning/types/core.py (核心数据类型)

# 3. 尝试Store API
python -c "
import asyncio
from agentlightning import InMemoryLightningStore, Span

async def test():
    store = InMemoryLightningStore()
    
    # 创建rollout
    rollout = await store.enqueue_rollout(input={'question': 'What is 2+2?'})
    print(f'Rollout ID: {rollout.rollout_id}')
    
    # 领取任务
    attempted = await store.dequeue_rollout()
    print(f'Attempt ID: {attempted.attempt.attempt_id}')
    
    # 添加span
    span = Span(
        name='test_span',
        rollout_id=attempted.rollout_id,
        attempt_id=attempted.attempt.attempt_id,
        sequence_id=1,
    )
    await store.add_span(span)
    print('Span added successfully')

asyncio.run(test())
"
```

### 4.2 Day 2: 理解核心架构

**任务**: 深入理解三组件架构

```bash
# 1. 阅读架构文档
cat docs/deep-dive/birds-eye-view.md

# 2. 理解数据流
# Algorithm -> Store -> Runner -> Agent -> Store -> Algorithm

# 3. 画出架构图（面试准备）
# 参考文档中的Mermaid图，用自己的话解释

# 4. 理解关键概念
# - Rollout: 一次完整的Agent执行
# - Attempt: Rollout的一次尝试（可能有重试）
# - Span: 执行过程中的一个操作记录
# - Resource: 可优化的资源（Prompt、LLM等）
```

### 4.3 Day 3-4: APO算法深入

**Day 3: 理解APO原理**

```bash
# 1. 阅读APO源码
cat agentlightning/algorithm/apo/apo.py | head -200

# 2. 理解Textual Gradient
# 核心思想: 用LLM生成对prompt的批评，然后用另一个LLM修改prompt

# 3. 理解Beam Search
# 维护多个候选prompt，每轮生成新候选，保留最优的

# 4. 运行示例并观察
cd examples/apo
python room_selector_apo.py 2>&1 | tee apo_output.log
```

**Day 4: 分析APO结果**

```bash
# 1. 查看优化日志
cat apo.log

# 2. 分析每轮的prompt变化
# 关注:
# - Textual Gradient的内容
# - Prompt的修改方向
# - 准确率的变化

# 3. 理解局限性
# - 依赖LLM质量
# - 计算成本高
# - 无理论收敛保证
```

### 4.4 Day 5-7: VERL RL训练

**Day 5: 准备环境**

```bash
# 1. 安装VERL依赖
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pip install flash-attn==2.8.1 --no-build-isolation
pip install vllm==0.10.2
pip install verl==0.6.0

# 2. 准备数据集
cd examples/calc_x
# 下载calc-x-data.zip并解压到data目录

# 3. 验证数据
python -c "
import pandas as pd
df = pd.read_parquet('data/train.parquet')
print(f'Training samples: {len(df)}')
print(df.head())
"
```

**Day 6: 单卡训练**

```bash
# 1. 启动Ray
bash ../../scripts/restart_ray.sh

# 2. 快速测试（CI模式）
python train_calc_agent.py --ci-fast --n-runners 2

# 3. 观察训练过程
# 关注:
# - vLLM推理服务启动
# - Runner执行rollout
# - 奖励计算
# - 模型更新

# 4. 查看GPU使用
watch -n 1 nvidia-smi
```

**Day 7: 多卡训练**

```bash
# 1. 配置多卡训练
# 修改train_calc_agent.py中的配置:
config["trainer"]["n_gpus_per_node"] = 4  # 使用4卡

# 2. 启动训练
python train_calc_agent.py --n-runners 8

# 3. 监控训练
# 使用wandb或tensorboard查看训练曲线

# 4. 分析结果
# - 奖励是否提升？
# - 训练是否稳定？
# - GPU利用率如何？
```

### 4.5 Day 8-10: SQL Agent训练

**Day 8: 理解SQL Agent**

```bash
# 1. 阅读SQL Agent代码
cd examples/spider
cat sql_agent.py

# 2. 理解Agent逻辑
# - 使用LangChain/LangGraph
# - 调用工具执行SQL
# - 评估SQL正确性

# 3. 准备数据集
# 下载Spider数据集
```

**Day 9: 训练SQL Agent**

```bash
# 1. 快速测试
python train_sql_agent.py fast

# 2. 完整训练
python train_sql_agent.py qwen

# 3. 监控训练
# 使用wandb查看训练曲线
```

**Day 10: 分析结果**

```bash
# 1. 评估模型
# 在测试集上评估训练后的模型

# 2. 对比分析
# - 训练前后的准确率
# - 不同prompt的效果
# - 错误案例分析

# 3. 总结经验
# - RL训练的关键因素
# - 奖励设计的重要性
# - 超参数的影响
```

### 4.6 Day 11-13: SFT训练对比

**Day 11: 安装Unsloth**

```bash
# 1. 安装依赖
pip install unsloth==2025.10.1 unsloth_zoo==2025.10.1 bitsandbytes peft datasets transformers trl kernels

# 2. 验证安装
python -c "from unsloth import FastLanguageModel; print('Unsloth installed')"
```

**Day 12: 运行SFT**

```bash
# 1. 运行SFT示例
cd examples/unsloth
python sft_allinone.py

# 2. 观察训练过程
# 关注:
# - 数据收集
# - LoRA配置
# - 训练loss

# 3. 对比RL和SFT
# - 训练效率
# - 最终效果
# - 适用场景
```

**Day 13: 总结对比**

```bash
# 1. 整理实验结果
# - APO: 准确率提升
# - VERL RL: 奖励提升
# - Unsloth SFT: loss下降

# 2. 分析各方法优缺点
# APO: 不需要GPU，但依赖API
# VERL: 需要GPU，但效果更好
# SFT: 简单高效，但需要高质量数据

# 3. 准备面试话术
# 能清楚解释每种方法的原理和适用场景
```

### 4.7 Day 14-16: 高级功能

**Day 14: LLM Proxy**

```bash
# 1. 理解LLM Proxy的作用
cat docs/deep-dive/serving-llm.md

# 2. 配置LLM Proxy
python examples/minimal/llm_proxy.py

# 3. 使用Proxy进行训练
python train_calc_agent.py --llm-proxy
```

**Day 15: 分布式训练**

```bash
# 1. 理解执行策略
# - SharedMemoryExecutionStrategy: 单进程多线程
# - ClientServerExecutionStrategy: 多进程

# 2. 配置分布式训练
# 使用ClientServer策略

# 3. 测试分布式训练
```

**Day 16: 自定义Algorithm**

```bash
# 1. 阅读自定义Algorithm示例
cat examples/apo/apo_custom_algorithm.py

# 2. 实现简单Algorithm
# 例如: 随机搜索prompt

# 3. 测试自定义Algorithm
```

---

## 5. 预期挑战与解决方案

### 5.1 显存不足

**问题**: 7B模型在4090上显存不足

**解决方案**:
```python
# 方案1: 使用4bit量化
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
)

# 方案2: 使用LoRA
config["actor_rollout_ref"]["model"]["lora_rank"] = 16

# 方案3: 降低batch size
config["data"]["train_batch_size"] = 8
config["actor_rollout_ref"]["rollout"]["log_prob_micro_batch_size_per_gpu"] = 2

# 方案4: 启用gradient checkpointing
config["actor_rollout_ref"]["model"]["enable_gradient_checkpointing"] = True

# 方案5: 参数offload
config["actor_rollout_ref"]["actor"]["fsdp_config"]["param_offload"] = True
config["actor_rollout_ref"]["actor"]["fsdp_config"]["optimizer_offload"] = True
```

### 5.2 训练不稳定

**问题**: 奖励曲线震荡或下降

**解决方案**:
```python
# 方案1: 降低学习率
config["actor_rollout_ref"]["actor"]["optim"]["lr"] = 5e-7

# 方案2: 增加KL惩罚
config["algorithm"]["use_kl_in_reward"] = True
config["algorithm"]["kl_penalty_coef"] = 0.1

# 方案3: 使用更大的batch size
config["data"]["train_batch_size"] = 64

# 方案4: 增加训练步数
config["trainer"]["total_training_steps"] = 200
```

### 5.3 API限流

**问题**: OpenAI API被限流

**解决方案**:
```python
# 方案1: 使用本地LLM
export OPENAI_BASE_URL="http://localhost:11434/v1"

# 方案2: 降低并发
trainer = Trainer(n_runners=2)  # 从8降到2

# 方案3: 添加重试逻辑
# 在Agent中添加tenacity重试

# 方案4: 使用多个API key
# 轮询使用不同的key
```

### 5.4 数据集问题

**问题**: 数据集下载失败或格式错误

**解决方案**:
```bash
# 方案1: 使用镜像源
export HF_ENDPOINT="https://hf-mirror.com"

# 方案2: 手动下载
# 从Google Drive或其他源下载

# 方案3: 使用小数据集
# 先用100条数据测试，确认流程正确后再用完整数据集
```

### 5.5 依赖冲突

**问题**: 不同库版本冲突

**解决方案**:
```bash
# 方案1: 使用uv的依赖组
uv sync --group dev --extra apo --group torch-stable

# 方案2: 创建独立环境
# 为不同示例创建独立的虚拟环境

# 方案3: 手动解决冲突
pip install --no-deps vllm==0.10.2
pip install --no-deps verl==0.6.0
```

---

## 6. 验证检查点

### 6.1 Phase 1 检查点

```bash
# 检查1: 环境安装成功
python -c "import agentlightning; print(agentlightning.__version__)"
# 预期输出: 0.3.1

# 检查2: minimal示例运行成功
cd examples/minimal
python write_traces.py
# 预期输出: 成功创建rollout和span

# 检查3: Store API工作正常
python -c "
import asyncio
from agentlightning import InMemoryLightningStore

async def test():
    store = InMemoryLightningStore()
    rollout = await store.enqueue_rollout(input={'test': 'data'})
    print(f'✓ Store API works: {rollout.rollout_id}')

asyncio.run(test())
"
```

### 6.2 Phase 2 检查点

```bash
# 检查1: APO示例运行成功
cd examples/apo
python room_selector_apo.py

# 检查2: 准确率有提升
# 预期: 0.569 -> 0.721

# 检查3: 能解释APO原理
# 面试问题: "APO的Textual Gradient是什么？"
# 回答: "用LLM生成对prompt的自然语言批评，然后用另一个LLM根据批评修改prompt"
```

### 6.3 Phase 3 检查点

```bash
# 检查1: VERL训练成功
cd examples/calc_x
python train_calc_agent.py --ci-fast

# 检查2: 奖励有提升
# 预期: 奖励从0.4提升到0.6+

# 检查3: 多卡训练成功
python train_calc_agent.py --n-runners 8

# 检查4: 能解释GRPO算法
# 面试问题: "GRPO和PPO有什么区别？"
# 回答: "GRPO不需要Critic网络，直接用组内相对优势估计，节省显存"
```

### 6.4 Phase 4 检查点

```bash
# 检查1: SFT训练成功
cd examples/unsloth
python sft_allinone.py

# 检查2: 能解释LoRA原理
# 面试问题: "LoRA为什么能节省显存？"
# 回答: "LoRA只训练低秩适配矩阵，不更新原始权重，大幅减少训练参数量"

# 检查3: 能对比RL和SFT
# 面试问题: "什么时候用RL，什么时候用SFT？"
# 回答: "有明确评估指标时用RL，有高质量标注数据时用SFT"
```

---

## 附录: 常用命令速查

### 环境管理
```bash
# 激活虚拟环境
source .venv/bin/activate

# 安装依赖
uv sync --group dev --extra apo

# 更新依赖
uv lock
```

### 训练命令
```bash
# APO训练
cd examples/apo && python room_selector_apo.py

# VERL训练 (Calc-X)
cd examples/calc_x && python train_calc_agent.py

# VERL训练 (Spider)
cd examples/spider && python train_sql_agent.py qwen

# SFT训练
cd examples/unsloth && python sft_allinone.py
```

### 监控命令
```bash
# GPU监控
watch -n 1 nvidia-smi

# Ray监控
ray status

# 训练日志
tail -f apo.log
```

### 调试命令
```bash
# 快速测试
python train_calc_agent.py --ci-fast

# 详细日志
python train_calc_agent.py --debug

# 单步执行
python -c "
import agentlightning as agl
agl.setup_logging('DEBUG')
# ... 你的代码
"
```

---

## 学习资源

### 必读文档
1. [README.md](README.md) - 项目概述
2. [docs/deep-dive/birds-eye-view.md](docs/deep-dive/birds-eye-view.md) - 架构详解
3. [docs/tutorials/write-agents.md](docs/tutorials/write-agents.md) - Agent编写指南
4. [docs/algorithm-zoo/verl.md](docs/algorithm-zoo/verl.md) - VERL算法详解

### 论文阅读
1. [Agent Lightning论文](https://arxiv.org/abs/2508.03680) - 核心论文
2. [ProTeGi](https://aclanthology.org/2023.emnlp-main.494.pdf) - APO算法基础
3. [TextGrad](https://github.com/zou-group/textgrad) - Textual Gradient
4. [VERL](https://github.com/volcengine/verl) - RL训练框架

### 代码阅读顺序
1. `agentlightning/types/core.py` - 核心数据类型
2. `agentlightning/store/base.py` - Store接口
3. `agentlightning/trainer/trainer.py` - Trainer实现
4. `agentlightning/algorithm/apo/apo.py` - APO算法
5. `agentlightning/verl/entrypoint.py` - VERL入口

---

*本文档基于8卡4090资源配置编写*
*最后更新: 2026年6月*
