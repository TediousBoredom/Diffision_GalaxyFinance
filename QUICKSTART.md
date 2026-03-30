"""
快速开始指南
"""

# DiffWorldFinance 快速开始指南

## 安装

### 1. 克隆项目
```bash
cd /inspire/ssd/project/video-generation/public/openveo3/openveo3_dmd/Diffision_GalaxyFinance
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

## 快速演示

### 运行完整演示
```bash
python demo.py
```

这将演示：
- 分层世界模型的创建
- 因果结构的展示
- 合成数据的生成
- 世界状态的编码
- 噪声预测
- 市场演化模拟
- 场景采样
- 因果影响分析
- 8种策略目标的交易动作生成

## 训练模型

### 1. 准备配置文件
编辑 `config/default.yaml` 调整超参数：
```yaml
data:
  n_samples: 10000      # 数据样本数
  seq_len: 60           # 序列长度
  seed: 42

model:
  num_steps: 1000       # 扩散步数
  schedule_type: "cosine"

training:
  batch_size: 32
  num_workers: 4
  learning_rate: 1.0e-4
  weight_decay: 1.0e-5
  num_epochs: 100
  save_path: "checkpoints/model.pt"
```

### 2. 开始训练
```bash
python train.py --config config/default.yaml --inference
```

参数说明：
- `--config`: 配置文件路径
- `--inference`: 训练后运行推理演示

### 3. 训练输出
- 模型检查点: `checkpoints/model.pt`
- 训练日志: 控制台输出

## 推理和模拟

### 1. 加载预训练模型
```python
import torch
from core.hierarchical_world import HierarchicalDiffusionWorldModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = HierarchicalDiffusionWorldModel()
model.load_state_dict(torch.load('checkpoints/model.pt'))
model = model.to(device)
```

### 2. 创建模拟器
```python
from inference.world_simulator import WorldSimulator, StrategySimulator
from subworlds.strategy_agent import StrategyExecutor

simulator = WorldSimulator(model, device)
strategy_executor = StrategyExecutor(128, 96, 64)
strategy_simulator = StrategySimulator(model, strategy_executor, device)
```

### 3. 模拟市场演化
```python
# 创建初始观测
initial_obs = {
    'microstructure': torch.randn(4, 128, device=device),
    'macro_regime': torch.randn(4, 96, device=device),
    'strategy_agent': torch.randn(4, 64, device=device)
}

# 模拟50步
trajectories = simulator.simulate_market_evolution(
    initial_obs,
    num_steps=50,
    strategy_objective=0
)
```

### 4. 采样市场场景
```python
scenarios = simulator.sample_market_scenarios(batch_size=4, num_scenarios=10)
```

### 5. 分析因果影响
```python
influence = simulator.analyze_causal_influence(initial_obs)
for source, targets in influence.items():
    for target, strength in targets.items():
        print(f"{source} → {target}: {strength:.4f}")
```

### 6. 生成交易动作
```python
# 对于不同的策略目标
objectives = [
    "PROFIT",
    "HIGH_FREQ_DOUBLE",
    "FULL_MARGIN_BOTTOM",
    "SHORT_TOP",
    "TREND_FOLLOW",
    "HEDGE",
    "MOMENTUM",
    "TAKE_PROFIT"
]

for obj_idx in range(8):
    actions = strategy_simulator.generate_trading_actions(
        initial_obs,
        objective=obj_idx
    )
    print(f"{objectives[obj_idx]}: {actions['action']}")
```

### 7. 策略回测
```python
# 生成价格序列
price_series = np.random.randn(100).cumsum() + 100

# 生成观测序列
observations_series = [initial_obs for _ in range(100)]

# 回测
performance = strategy_simulator.backtest_strategy(
    price_series,
    observations_series,
    objective=0,
    initial_capital=100000.0
)

print(f"Total Return: {performance['total_return']:.4f}")
print(f"Sharpe Ratio: {performance['sharpe_ratio']:.4f}")
print(f"Max Drawdown: {performance['max_drawdown']:.4f}")
print(f"Win Rate: {performance['win_rate']:.4f}")
```

## 项目结构

```
DiffWorldFinance/
├── core/                          # 核心模块
│   ├── diffusion_base.py         # 基础扩散模型
│   ├── causal_structure.py       # 因果结构
│   └── hierarchical_world.py     # 分层世界模型
├── subworlds/                     # 子世界实现
│   ├── microstructure.py         # 市场微观结构
│   ├── macro_regime.py           # 宏观与市场结构
│   └── strategy_agent.py         # 策略-风险干预
├── data/                          # 数据处理
│   └── market_data.py            # 市场数据
├── training/                      # 训练框架
│   └── trainer.py                # 训练器
├── inference/                     # 推理模块
│   └── world_simulator.py        # 世界模拟器
├── config/                        # 配置文件
│   └── default.yaml              # 默认配置
├── checkpoints/                   # 模型检查点
├── train.py                       # 训练脚本
├── demo.py                        # 演示脚本
├── requirements.txt               # 依赖
├── README.md                      # 项目说明
└── ARCHITECTURE.md                # 架构文档
```

## 关键概念

### 分层子世界
- **Sub-World A (微观结构)**: 建模价格、成交量、订单、波动率
- **Sub-World B (宏观结构)**: 建模趋势、波动结构、流动性、风险偏好
- **Sub-World C (策略-风险)**: 建模交易动作、风险调制、策略信号

### 因果结构
- 子世界间通过明确的因果边进行交互
- 每条因果边有延迟和强度参数
- 确保信息流遵循因果规则

### 扩散过程
- 前向过程: 从数据添加噪声
- 反向过程: 从噪声恢复数据
- 条件扩散: 根据其他子世界的信息进行条件生成

### 目的性分布
- 8种策略目标对应8种不同的分布
- 通过ObjectiveModulator进行目标调制
- 实现目标特定的策略生成

## 常见问题

### Q: 如何使用真实市场数据？
A: 修改 `data/market_data.py` 中的数据加载部分，替换为真实数据源。

### Q: 如何调整模型大小？
A: 修改 `core/hierarchical_world.py` 中各子世界的 `latent_dim` 参数。

### Q: 如何添加新的策略目标？
A: 在 `subworlds/strategy_agent.py` 中的 `StrategyObjective` 枚举中添加新目标。

### Q: 如何改进模型性能？
A: 
1. 增加训练数据量
2. 调整学习率和权重衰减
3. 增加训练轮数
4. 调整损失函数权重

## 性能优化

### GPU 加速
```python
device = torch.device('cuda')
model = model.to(device)
```

### 批处理
```python
batch_size = 64  # 增加批大小
```

### 混合精度训练
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    loss = model(...)
scaler.scale(loss).backward()
```

## 故障排除

### 内存不足
- 减少 `batch_size`
- 减少 `seq_len`
- 使用梯度累积

### 训练不收敛
- 降低学习率
- 增加预热步数
- 检查数据质量

### 推理速度慢
- 使用 GPU
- 减少采样步数
- 使用批处理

## 下一步

1. 阅读 `ARCHITECTURE.md` 了解详细架构
2. 查看 `demo.py` 了解使用示例
3. 修改配置进行自定义训练
4. 集成真实市场数据
5. 开发自定义策略

## 联系和支持

如有问题或建议，请提交 Issue 或 Pull Request。

## 许可证

MIT License
