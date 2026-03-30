"""
主训练脚本 - DiffWorldFinance 分层扩散世界模型
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import argparse
import yaml
from pathlib import Path
import numpy as np

from core.hierarchical_world import HierarchicalDiffusionWorldModel
from data.market_data import MarketDataset, SyntheticMarketDataGenerator, DataNormalizer
from training.trainer import HierarchicalDiffusionTrainer
from inference.world_simulator import WorldSimulator, StrategySimulator
from subworlds.strategy_agent import StrategyExecutor


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main(args):
    # 加载配置
    config = load_config(args.config)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 生成合成数据
    print("Generating synthetic market data...")
    data_generator = SyntheticMarketDataGenerator(
        n_samples=config['data']['n_samples'],
        seed=config['data']['seed']
    )
    data_dict = data_generator.generate()
    
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
        seq_len=config['data']['seq_len']
    )
    
    # 划分训练/验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers']
    )
    
    # 创建模型
    print("Creating hierarchical diffusion world model...")
    model = HierarchicalDiffusionWorldModel(
        num_steps=config['model']['num_steps']
    )
    
    # 创建训练器
    trainer = HierarchicalDiffusionTrainer(
        model=model,
        device=device,
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # 训练
    print("Starting training...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['training']['num_epochs'],
        save_path=config['training']['save_path']
    )
    
    # 保存模型
    model_path = Path(config['training']['save_path'])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # 推理演示
    if args.inference:
        print("\nRunning inference demonstrations...")
        
        # 创建世界模拟器
        simulator = WorldSimulator(model, device)
        
        # 创建策略执行器
        strategy_executor = StrategyExecutor(
            microstructure_dim=128,
            macro_dim=96,
            strategy_dim=64
        )
        
        # 创建策略模拟器
        strategy_simulator = StrategySimulator(model, strategy_executor, device)
        
        # 采样初始观测
        batch_size = 4
        initial_obs = {
            'microstructure': torch.randn(batch_size, 128, device=device),
            'macro_regime': torch.randn(batch_size, 96, device=device),
            'strategy_agent': torch.randn(batch_size, 64, device=device)
        }
        
        # 模拟市场演化
        print("Simulating market evolution...")
        trajectories = simulator.simulate_market_evolution(
            initial_obs,
            num_steps=50,
            strategy_objective=0
        )
        print(f"Generated trajectories for {len(trajectories)} subworlds")
        
        # 采样市场场景
        print("Sampling market scenarios...")
        scenarios = simulator.sample_market_scenarios(batch_size=4, num_scenarios=5)
        print(f"Generated {len(scenarios)} market scenarios")
        
        # 分析因果影响
        print("Analyzing causal influence...")
        influence = simulator.analyze_causal_influence(initial_obs)
        print("Causal influence matrix:")
        for source, targets in influence.items():
            for target, strength in targets.items():
                print(f"  {source} -> {target}: {strength:.4f}")
        
        # 生成交易动作
        print("Generating trading actions...")
        for objective in range(3):
            actions = strategy_simulator.generate_trading_actions(initial_obs, objective=objective)
            print(f"Objective {objective}:")
            print(f"  Action shape: {actions['action'].shape}")
            print(f"  Confidence: {actions['confidence'].mean().item():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DiffWorldFinance model")
    parser.add_argument('--config', type=str, default='config/default.yaml',
                       help='Path to config file')
    parser.add_argument('--inference', action='store_true',
                       help='Run inference demonstrations')
    
    args = parser.parse_args()
    main(args)
