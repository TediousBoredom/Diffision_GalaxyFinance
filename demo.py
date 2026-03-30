"""
演示脚本 - DiffWorldFinance 分层扩散世界模型的完整演示
"""
import torch
import numpy as np
from pathlib import Path
import sys

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from core.hierarchical_world import HierarchicalDiffusionWorldModel
from data.market_data import SyntheticMarketDataGenerator, DataNormalizer, MarketDataset
from inference.world_simulator import WorldSimulator, StrategySimulator
from subworlds.strategy_agent import StrategyExecutor, StrategyObjective


def demo_hierarchical_world_model():
    """演示分层扩散世界模型"""
    print("=" * 80)
    print("DiffWorldFinance: Hierarchical Diffusion World Model for Financial Markets")
    print("=" * 80)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[1] Device: {device}")
    
    # 创建模型
    print("\n[2] Creating Hierarchical Diffusion World Model...")
    model = HierarchicalDiffusionWorldModel(num_steps=1000)
    model = model.to(device)
    print(f"    - Microstructure Sub-World: latent_dim=128")
    print(f"    - Macro & Regime Sub-World: latent_dim=96")
    print(f"    - Strategy & Risk Sub-World: latent_dim=64")
    
    # 显示因果结构
    print("\n[3] Causal Structure:")
    causal_graph = model.causal_constraint.get_causal_graph()
    for source, targets in causal_graph.items():
        for target in targets:
            print(f"    - {source} → {target}")
    
    # 生成合成数据
    print("\n[4] Generating Synthetic Market Data...")
    data_generator = SyntheticMarketDataGenerator(n_samples=5000, seed=42)
    data_dict = data_generator.generate()
    print(f"    - Generated {data_dict['price'].shape[0]} time steps")
    print(f"    - Modalities: price, volume, order, volatility, trend, regime, liquidity, risk, action, signal")
    
    # 数据归一化
    normalizer = DataNormalizer()
    normalizer.fit(data_dict)
    data_dict = normalizer.normalize(data_dict)
    
    # 创建数据集
    dataset = MarketDataset(
        price_data=data_dict['price'],
        volume_data=data_dict['volume'],
        order_data=data_dict['order'],
        volatility_data=data_dict['volatility'],
        trend_data=data_dict['trend'],
        vol_structure_data=data_dict['vol_structure'],
        liquidity_data=data_dict['liquidity'],
        risk_data=data_dict['risk_preference'],
        action_data=data_dict['action'],
        risk_embedding_data=data_dict['risk_embedding'],
        signal_data=data_dict['strategy_signal'],
        seq_len=60
    )
    
    # 获取样本
    sample = dataset[0]
    print(f"    - Sample sequence length: {sample['price'].shape[0]}")
    
    # 创建观测
    batch_size = 4
    observations = {
        'microstructure': torch.cat([
            torch.randn(batch_size, 60, 5),  # price
            torch.randn(batch_size, 60, 3),  # volume
            torch.randn(batch_size, 60, 4),  # order
            torch.randn(batch_size, 60, 3)   # volatility
        ], dim=-1).mean(dim=1).to(device),
        'macro_regime': torch.cat([
            torch.randn(batch_size, 60, 5),  # trend
            torch.randn(batch_size, 60, 4),  # vol_structure
            torch.randn(batch_size, 60, 4),  # liquidity
            torch.randn(batch_size, 60, 4)   # risk_preference
        ], dim=-1).mean(dim=1).to(device),
        'strategy_agent': torch.cat([
            torch.randn(batch_size, 60, 6),  # action
            torch.randn(batch_size, 60, 5),  # risk_embedding
            torch.randn(batch_size, 60, 8)   # strategy_signal
        ], dim=-1).mean(dim=1).to(device)
    }
    
    # 获取世界状态
    print("\n[5] Encoding Market Observations to World State...")
    world_state = model.get_world_state(observations)
    for name, latent in world_state.items():
        print(f"    - {name}: {latent.shape}")
    
    # 前向传播
    print("\n[6] Forward Pass (Noise Prediction)...")
    t = torch.randint(0, 100, (batch_size,), device=device)
    noise_predictions = model(observations, t)
    for name, noise in noise_predictions.items():
        print(f"    - {name} noise prediction: {noise.shape}")
    
    # 创建世界模拟器
    print("\n[7] Creating World Simulator...")
    simulator = WorldSimulator(model, device)
    
    # 模拟市场演化
    print("\n[8] Simulating Market Evolution (50 steps)...")
    trajectories = simulator.simulate_market_evolution(
        observations,
        num_steps=50,
        strategy_objective=0
    )
    print(f"    - Generated trajectories for {len(trajectories)} sub-worlds")
    for name, traj in trajectories.items():
        print(f"    - {name}: {len(traj)} steps")
    
    # 采样市场场景
    print("\n[9] Sampling Market Scenarios...")
    scenarios = simulator.sample_market_scenarios(batch_size=4, num_scenarios=5)
    print(f"    - Generated {len(scenarios)} scenarios")
    
    # 分析因果影响
    print("\n[10] Analyzing Causal Influence...")
    influence = simulator.analyze_causal_influence(observations)
    print("     Causal Influence Matrix:")
    for source, targets in influence.items():
        for target, strength in targets.items():
            print(f"     - {source} → {target}: {strength:.4f}")
    
    # 创建策略执行器
    print("\n[11] Creating Strategy Executor...")
    strategy_executor = StrategyExecutor(
        microstructure_dim=128,
        macro_dim=96,
        strategy_dim=64
    )
    strategy_executor = strategy_executor.to(device)
    
    # 创建策略模拟器
    print("\n[12] Creating Strategy Simulator...")
    strategy_simulator = StrategySimulator(model, strategy_executor, device)
    
    # 生成交易动作
    print("\n[13] Generating Trading Actions for Different Objectives:")
    objectives = [
        "PROFIT (盈利)",
        "HIGH_FREQ_DOUBLE (高频翻倍)",
        "FULL_MARGIN_BOTTOM (满融满仓抄底)",
        "SHORT_TOP (高点做空)",
        "TREND_FOLLOW (趋势跟踪)",
        "HEDGE (对冲)",
        "MOMENTUM (动量)",
        "TAKE_PROFIT (止盈)"
    ]
    
    for objective_idx in range(8):
        actions = strategy_simulator.generate_trading_actions(observations, objective=objective_idx)
        print(f"\n     [{objective_idx}] {objectives[objective_idx]}")
        print(f"         - Action: {actions['action'][0].cpu().numpy()}")
        print(f"         - Risk Adjustment: {actions['risk_adjustment'][0].cpu().numpy()}")
        print(f"         - Confidence: {actions['confidence'][0].item():.4f}")
    
    # 模型统计
    print("\n[14] Model Statistics:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"     - Total Parameters: {total_params:,}")
    print(f"     - Trainable Parameters: {trainable_params:,}")
    
    print("\n" + "=" * 80)
    print("Demo completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    demo_hierarchical_world_model()
