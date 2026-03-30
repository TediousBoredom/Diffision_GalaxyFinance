"""
世界模拟器 - 使用分层扩散世界模型进行市场模拟和预测
"""
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import numpy as np
from subworlds.strategy_agent import StrategyExecutor, StrategyObjective


class WorldSimulator:
    """分层扩散世界模拟器"""
    
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.model.eval()
    
    def simulate_market_evolution(self,
                                  initial_observations: Dict[str, torch.Tensor],
                                  num_steps: int = 100,
                                  strategy_objective: int = 0) -> Dict[str, List[torch.Tensor]]:
        """
        模拟市场演化
        
        Args:
            initial_observations: 初始观测 {subworld_name: [batch_size, obs_dim]}
            num_steps: 模拟步数
            strategy_objective: 策略目标索引 (0-7)
        
        Returns:
            {subworld_name: [num_steps, batch_size, obs_dim]} 模拟轨迹
        """
        trajectories = {name: [] for name in self.model.subworlds.keys()}
        
        # 初始化世界状态
        current_state = self.model.get_world_state(initial_observations)
        
        with torch.no_grad():
            for step in range(num_steps):
                # 保存当前状态
                for name, latent in current_state.items():
                    obs = self.model.subworlds[name].decode(latent)
                    trajectories[name].append(obs.cpu())
                
                # 生成下一步状态
                # 使用条件扩散进行一步演化
                t = torch.full((current_state['microstructure'].shape[0],), 
                              step % self.model.num_steps, 
                              dtype=torch.long, 
                              device=self.device)
                
                # 前向传播获取噪声预测
                noise_pred = self.model(current_state, t, step=step)
                
                # 更新状态（简化的演化规则）
                for name in current_state.keys():
                    # 添加预测的噪声作为演化
                    current_state[name] = current_state[name] + 0.01 * noise_pred[name]
        
        return trajectories
    
    def predict_next_state(self,
                          observations: Dict[str, torch.Tensor],
                          num_steps: int = 1) -> Dict[str, torch.Tensor]:
        """
        预测下一个状态
        
        Args:
            observations: 当前观测
            num_steps: 预测步数
        
        Returns:
            {subworld_name: [batch_size, obs_dim]} 预测的观测
        """
        current_state = self.model.get_world_state(observations)
        
        with torch.no_grad():
            for step in range(num_steps):
                t = torch.full((current_state['microstructure'].shape[0],), 
                              step % self.model.num_steps, 
                              dtype=torch.long, 
                              device=self.device)
                
                noise_pred = self.model(current_state, t, step=step)
                
                for name in current_state.keys():
                    current_state[name] = current_state[name] + 0.01 * noise_pred[name]
        
        return self.model.decode_world_state(current_state)
    
    def sample_market_scenarios(self,
                               batch_size: int,
                               num_scenarios: int = 10) -> List[Dict[str, torch.Tensor]]:
        """
        采样多个市场场景
        
        Args:
            batch_size: 批大小
            num_scenarios: 场景数量
        
        Returns:
            [num_scenarios] 场景列表，每个场景包含 {subworld_name: [batch_size, obs_dim]}
        """
        scenarios = []
        
        with torch.no_grad():
            for _ in range(num_scenarios):
                # 从模型采样
                latents = self.model.sample(batch_size, self.device)
                
                # 解码为观测
                observations = self.model.decode_world_state(latents)
                scenarios.append(observations)
        
        return scenarios
    
    def analyze_causal_influence(self,
                                observations: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, float]]:
        """
        分析因果影响 - 计算子世界间的因果强度
        
        Args:
            observations: 观测数据
        
        Returns:
            {source: {target: influence_strength}}
        """
        latents = self.model.get_world_state(observations)
        
        influence_matrix = {}
        for source_name in self.model.subworlds.keys():
            influence_matrix[source_name] = {}
            
            for edge in self.model.causal_structure.get_outgoing_edges(source_name):
                target_name = edge.target
                
                # 计算源和目标的相关性
                source_latent = latents[source_name]
                target_latent = latents[target_name]
                
                correlation = torch.nn.functional.cosine_similarity(
                    source_latent, target_latent, dim=-1
                ).mean().item()
                
                influence_matrix[source_name][target_name] = correlation * edge.strength
        
        return influence_matrix


class StrategySimulator:
    """策略模拟器 - 基于世界状态生成交易策略"""
    
    def __init__(self,
                 model: nn.Module,
                 strategy_executor: StrategyExecutor,
                 device: torch.device):
        self.model = model
        self.strategy_executor = strategy_executor.to(device)
        self.device = device
        self.model.eval()
        self.strategy_executor.eval()
    
    def generate_trading_actions(self,
                                observations: Dict[str, torch.Tensor],
                                objective: int = 0) -> Dict[str, torch.Tensor]:
        """
        生成交易动作
        
        Args:
            observations: 市场观测
            objective: 策略目标 (0-7)
        
        Returns:
            {
                'action': [batch_size, 6],
                'risk_adjustment': [batch_size, 5],
                'confidence': [batch_size, 1]
            }
        """
        with torch.no_grad():
            # 获取世界状态
            latents = self.model.get_world_state(observations)
            
            # 执行策略
            actions = self.strategy_executor(
                latents['microstructure'],
                latents['macro_regime'],
                latents['strategy_agent'],
                objective_idx=objective
            )
        
        return actions
    
    def backtest_strategy(self,
                         price_series: np.ndarray,
                         observations_series: List[Dict[str, torch.Tensor]],
                         objective: int = 0,
                         initial_capital: float = 100000.0) -> Dict[str, float]:
        """
        回测策略
        
        Args:
            price_series: [num_steps] 价格序列
            observations_series: 观测序列
            objective: 策略目标
            initial_capital: 初始资本
        
        Returns:
            {
                'total_return': 总收益率,
                'sharpe_ratio': 夏普比率,
                'max_drawdown': 最大回撤,
                'win_rate': 胜率
            }
        """
        portfolio_value = initial_capital
        position = 0.0
        entry_price = 0.0
        trades = []
        
        with torch.no_grad():
            for t, obs in enumerate(observations_series):
                if t >= len(price_series):
                    break
                
                current_price = price_series[t]
                
                # 生成交易信号
                actions = self.generate_trading_actions(obs, objective)
                
                action = actions['action'][0].cpu().numpy()
                confidence = actions['confidence'][0].item()
                
                position_signal = action[0]  # position_size
                
                # 执行交易
                if position_signal > 0.5 and position == 0:
                    # 开多头
                    position = 1.0
                    entry_price = current_price
                elif position_signal < -0.5 and position == 0:
                    # 开空头
                    position = -1.0
                    entry_price = current_price
                elif abs(position_signal) < 0.1 and position != 0:
                    # 平仓
                    pnl = position * (current_price - entry_price)
                    portfolio_value += pnl
                    trades.append({
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'pnl': pnl,
                        'return': pnl / (entry_price * abs(position))
                    })
                    position = 0.0
        
        # 计算性能指标
        total_return = (portfolio_value - initial_capital) / initial_capital
        
        if trades:
            returns = np.array([t['return'] for t in trades])
            win_rate = np.sum(returns > 0) / len(returns)
            sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
        else:
            win_rate = 0.0
            sharpe_ratio = 0.0
        
        # 计算最大回撤
        max_drawdown = 0.0
        peak = initial_capital
        for trade in trades:
            peak = max(peak, portfolio_value)
            drawdown = (peak - portfolio_value) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'num_trades': len(trades)
        }
    
    def compare_objectives(self,
                          observations_series: List[Dict[str, torch.Tensor]],
                          price_series: np.ndarray) -> Dict[int, Dict[str, float]]:
        """
        比较不同策略目标的性能
        
        Args:
            observations_series: 观测序列
            price_series: 价格序列
        
        Returns:
            {objective_idx: performance_metrics}
        """
        results = {}
        
        for objective in range(8):
            print(f"Backtesting objective {objective}...")
            performance = self.backtest_strategy(
                price_series,
                observations_series,
                objective=objective
            )
            results[objective] = performance
        
        return results
