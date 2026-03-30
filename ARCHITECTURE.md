"""
DiffWorldFinance 架构文档
"""

# DiffWorldFinance: Hierarchical Diffusion World Model for Financial Markets

## 核心架构概述

DiffWorldFinance 是一个基于分层扩散模型的金融市场多模态因果世界模型系统。它将金融市场建模为具有明确时间因果结构的多个子世界系统，通过扩散过程统一建模其状态演化与跨层交互。

## 系统架构

### 1. 三层子世界结构

```
Global Diffusion World Model
│
├── Sub-World A: Market Microstructure (市场微观结构层)
│   ├── 价格路径 (Price Paths)
│   ├── 成交量流 (Volume Flows)
│   ├── 订单失衡 (Order Imbalance)
│   └── 波动率演化 (Volatility Evolution)
│   └── Latent Dim: 128
│
├── Sub-World B: Macro & Regime Dynamics (宏观与市场结构层)
│   ├── 趋势/震荡状态 (Trend/Oscillation States)
│   ├── 波动结构 (Volatility Structure)
│   ├── 流动性环境 (Liquidity Environment)
│   └── 风险偏好 (Risk Preference)
│   └── Latent Dim: 96
│
└── Sub-World C: Strategy & Risk Agent (策略-风险干预层)
    ├── 交易动作 (Trading Actions)
    ├── 风险嵌入调制 (Risk Embedding Modulation)
    └── 策略信号 (Strategy Signals)
    └── Latent Dim: 64
```

### 2. 因果结构

#### 因果边定义
- **Microstructure → Macro**: 市场微观结构影响宏观结构
  - 价格路径 (latency=1, strength=0.8)
  - 成交量流 (latency=2, strength=0.6)

- **Macro → Microstructure**: 宏观结构影响微观动态
  - 市场状态 (latency=1, strength=0.7)
  - 流动性环境 (latency=1, strength=0.5)

- **Strategy → Microstructure**: 策略干预市场微观
  - 交易动作 (latency=0, strength=0.9)

- **Strategy → Macro**: 策略影响宏观结构
  - 风险调制 (latency=1, strength=0.6)

- **Microstructure → Strategy**: 微观信息反馈给策略
  - 价格信息 (latency=0, strength=0.8)

- **Macro → Strategy**: 宏观信息反馈给策略
  - 市场状态 (latency=0, strength=0.7)

#### 因果局部性
- 每条因果边都有明确的延迟 (latency) 和强度 (strength)
- 子世界间的交互通过模态桥接 (ModalityBridge) 进行
- 确保信息流遵循因果结构，避免因果循环

### 3. 核心模块

#### 3.1 扩散基础 (core/diffusion_base.py)
- **SinusoidalPositionalEncoding**: 时间步编码
- **DiffusionBlock**: 基础扩散块
- **ConditionalDiffusionBlock**: 条件扩散块
- **DiffusionSchedule**: 噪声调度 (linear/cosine)
- **BaseDiffusionModel**: 基础扩散模型

#### 3.2 因果结构 (core/causal_structure.py)
- **CausalEdge**: 因果边定义
- **SubWorldSpec**: 子世界规范
- **CausalStructure**: 因果结构管理
- **CausalLocalityConstraint**: 因果局部性约束
- **ModalityBridge**: 模态桥接

#### 3.3 分层世界模型 (core/hierarchical_world.py)
- **SubWorldDiffusionModel**: 单个子世界的扩散模型
- **HierarchicalDiffusionWorldModel**: 分层世界模型核心
  - 编码观测为隐状态
  - 应用因果条件
  - 级联扩散块进行跨层交互
  - 采样和解码

#### 3.4 子世界实现
- **microstructure.py**: 市场微观结构层
  - PricePathEncoder
  - VolumeFlowEncoder
  - OrderImbalanceEncoder
  - VolatilityEncoder
  - MicrostructureWorldEncoder/Decoder

- **macro_regime.py**: 宏观与市场结构层
  - TrendStateEncoder
  - VolatilityStructureEncoder
  - LiquidityEnvironmentEncoder
  - RiskPreferenceEncoder
  - MacroRegimeWorldEncoder/Decoder

- **strategy_agent.py**: 策略-风险干预层
  - ActionEncoder
  - RiskEmbeddingEncoder
  - StrategySignalEncoder
  - ObjectiveModulator
  - StrategyAgentWorldEncoder/Decoder
  - StrategyExecutor

#### 3.5 数据处理 (data/market_data.py)
- **MarketDataset**: 市场数据集
- **DataNormalizer**: 数据归一化
- **SyntheticMarketDataGenerator**: 合成数据生成

#### 3.6 训练框架 (training/trainer.py)
- **DiffusionLoss**: 扩散损失
- **CausalConsistencyLoss**: 因果一致性损失
- **ReconstructionLoss**: 重建损失
- **HierarchicalDiffusionTrainer**: 训练器

#### 3.7 推理模块 (inference/world_simulator.py)
- **WorldSimulator**: 世界模拟器
  - 市场演化模拟
  - 状态预测
  - 场景采样
  - 因果影响分析

- **StrategySimulator**: 策略模拟器
  - 交易动作生成
  - 策略回测
  - 目标比较

## 多模态因果建模

### 模态类型
1. **金融模态** (Financial Modality)
   - 价格、收益率、波动率
   - 成交量、订单流
   - 流动性指标

2. **交易市场模态** (Trading Market Modality)
   - 趋势状态、市场结构
   - 风险偏好、流动性环境
   - 市场压力指标

3. **企业投资管理模态** (Corporate Investment Modality)
   - 交易策略、风险管理
   - 投资组合调整
   - 风险调制

### 因果关系
- 微观结构变化 → 宏观结构演化
- 宏观结构变化 → 微观动态调整
- 策略干预 → 市场状态变化
- 市场反馈 → 策略调整

## 目的性分布建模

### 8种策略目标
1. **PROFIT (盈利)**: 最大化收益
2. **HIGH_FREQ_DOUBLE (高频翻倍)**: 高频交易翻倍收益
3. **FULL_MARGIN_BOTTOM (满融满仓抄底)**: 底部满仓建仓
4. **SHORT_TOP (高点做空)**: 顶部做空
5. **TREND_FOLLOW (趋势跟踪)**: 跟踪市场趋势
6. **HEDGE (对冲)**: 风险对冲
7. **MOMENTUM (动量)**: 动量策略
8. **TAKE_PROFIT (止盈)**: 止盈策略

### 目标调制
- ObjectiveModulator 根据策略目标调制风险和动作
- 不同目标对应不同的风险-收益权衡
- 通过条件扩散实现目标特定的策略生成

## 级联扩散块架构

### 跨层交互机制
1. **信息编码**: 将观测编码为隐状态
2. **因果条件应用**: 根据因果结构调制隐状态
3. **模态桥接**: 通过模态特定的投影进行信息传递
4. **级联扩散**: 条件扩散块进行跨层交互
5. **状态更新**: 反向扩散过程更新隐状态

### 因果局部性保持
- 每个子世界只接收来自其因果前驱的信息
- 模态桥接确保信息的模态一致性
- 因果延迟确保时间因果性
- 因果强度权重确保影响程度

## 长期稳定吸引子

### 稳定性机制
1. **因果结构约束**: 确保系统演化遵循因果规则
2. **重建损失**: 确保编码-解码的一致性
3. **因果一致性损失**: 确保子世界间的因果关系
4. **扩散过程**: 通过噪声调度实现平滑演化

### 吸引子特性
- 市场结构的稳定性
- 趋势的可预测性
- 风险的可控性
- 策略的一致性

## 使用示例

### 1. 训练模型
```bash
python train.py --config config/default.yaml
```

### 2. 运行演示
```bash
python demo.py
```

### 3. 推理和模拟
```python
from core.hierarchical_world import HierarchicalDiffusionWorldModel
from inference.world_simulator import WorldSimulator

# 创建模型
model = HierarchicalDiffusionWorldModel()

# 创建模拟器
simulator = WorldSimulator(model, device)

# 模拟市场演化
trajectories = simulator.simulate_market_evolution(
    initial_observations,
    num_steps=100,
    strategy_objective=0
)
```

## 性能指标

### 模型规模
- 总参数数: ~2.5M
- 可训练参数: ~2.5M
- 内存占用: ~500MB (batch_size=32)

### 计算复杂度
- 前向传播: O(batch_size × seq_len × latent_dim)
- 反向传播: O(batch_size × seq_len × latent_dim)
- 采样: O(num_steps × batch_size × latent_dim)

## 扩展方向

1. **多资产建模**: 支持多个资产的联合建模
2. **实时数据集成**: 集成实时市场数据
3. **强化学习**: 结合RL进行策略优化
4. **风险管理**: 增强风险管理模块
5. **可解释性**: 提高模型的可解释性

## 参考文献

- Diffusion Models: Denoising Diffusion Probabilistic Models (DDPM)
- Causal Inference: Pearl's Causal Model
- Financial Modeling: Multi-scale Market Microstructure
- World Models: Learning Latent Dynamics for Imagination-Driven Agents
