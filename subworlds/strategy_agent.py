"""
Sub-World C: Strategy & Risk Agent (策略-风险干预层)
建模交易动作、风险嵌入调制、策略信号
支持多种目的性策略: 盈利、高频翻倍、满融满仓抄底、高点做空、趋势、对冲、动量、止盈
"""
import torch
import torch.nn as nn
from typing import Dict, List
from enum import Enum


class StrategyObjective(Enum):
    """策略目标枚举"""
    PROFIT = "profit"  # 盈利
    HIGH_FREQ_DOUBLE = "high_freq_double"  # 高频翻倍
    FULL_MARGIN_BOTTOM = "full_margin_bottom"  # 满融满仓抄底
    SHORT_TOP = "short_top"  # 高点做空
    TREND_FOLLOW = "trend_follow"  # 趋势跟踪
    HEDGE = "hedge"  # 对冲
    MOMENTUM = "momentum"  # 动量
    TAKE_PROFIT = "take_profit"  # 止盈


class ActionEncoder(nn.Module):
    """交易动作编码器"""
    
    def __init__(self, input_dim: int = 6, hidden_dim: int = 64, output_dim: int = 20):
        super().__init__()
        # 输入: [position_size, entry_price, exit_price, stop_loss, take_profit, leverage]
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, action_data: torch.Tensor) -> torch.Tensor:
        """
        Args:
            action_data: [batch_size, seq_len, 6] 交易动作数据
        Returns:
            [batch_size, output_dim] 编码的动作特征
        """
        encoded = self.net(action_data)
        return encoded.mean(dim=1)


class RiskEmbeddingEncoder(nn.Module):
    """风险嵌入编码器"""
    
    def __init__(self, input_dim: int = 5, hidden_dim: int = 64, output_dim: int = 22):
        super().__init__()
        # 输入: [var_95, cvar_95, max_drawdown, sharpe_ratio, sortino_ratio]
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, risk_data: torch.Tensor) -> torch.Tensor:
        """
        Args:
            risk_data: [batch_size, seq_len, 5] 风险数据
        Returns:
            [batch_size, output_dim] 编码的风险特征
        """
        encoded = self.net(risk_data)
        return encoded.mean(dim=1)


class StrategySignalEncoder(nn.Module):
    """策略信号编码器"""
    
    def __init__(self, input_dim: int = 8, hidden_dim: int = 64, output_dim: int = 22):
        super().__init__()
        # 输入: [signal_strength, confidence, entry_signal, exit_signal, 
        #        hedge_signal, momentum_signal, trend_signal, regime_signal]
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, signal_data: torch.Tensor) -> torch.Tensor:
        """
        Args:
            signal_data: [batch_size, seq_len, 8] 策略信号数据
        Returns:
            [batch_size, output_dim] 编码的策略信号特征
        """
        encoded = self.net(signal_data)
        return encoded.mean(dim=1)


class ObjectiveModulator(nn.Module):
    """目标调制器 - 根据策略目标调制风险与动作"""
    
    def __init__(self, latent_dim: int = 64, num_objectives: int = 8):
        super().__init__()
        self.num_objectives = num_objectives
        
        # 为每个目标创建调制参数
        self.objective_embeddings = nn.Embedding(num_objectives, latent_dim)
        
        # 调制网络
        self.modulation_net = nn.Sequential(
            nn.Linear(latent_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
    
    def forward(self, latent: torch.Tensor, objective_idx: int) -> torch.Tensor:
        """
        Args:
            latent: [batch_size, latent_dim] 策略隐状态
            objective_idx: 目标索引 (0-7)
        Returns:
            [batch_size, latent_dim] 调制后的隐状态
        """
        batch_size = latent.shape[0]
        objective_emb = self.objective_embeddings(
            torch.full((batch_size,), objective_idx, dtype=torch.long, device=latent.device)
        )
        
        combined = torch.cat([latent, objective_emb], dim=-1)
        return self.modulation_net(combined)


class StrategyAgentWorldEncoder(nn.Module):
    """策略-风险干预世界编码器 - 整合所有策略特征"""
    
    def __init__(self, output_dim: int = 64):
        super().__init__()
        self.action_encoder = ActionEncoder(output_dim=20)
        self.risk_encoder = RiskEmbeddingEncoder(output_dim=22)
        self.signal_encoder = StrategySignalEncoder(output_dim=22)
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, strategy_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            strategy_data: {
                'action': [batch_size, seq_len, 6],
                'risk_embedding': [batch_size, seq_len, 5],
                'strategy_signal': [batch_size, seq_len, 8]
            }
        Returns:
            [batch_size, output_dim] 策略隐状态
        """
        action_feat = self.action_encoder(strategy_data['action'])
        risk_feat = self.risk_encoder(strategy_data['risk_embedding'])
        signal_feat = self.signal_encoder(strategy_data['strategy_signal'])
        
        # 拼接所有特征
        combined = torch.cat([action_feat, risk_feat, signal_feat], dim=-1)
        return self.fusion(combined)


class StrategyAgentWorldDecoder(nn.Module):
    """策略-风险干预世界解码器"""
    
    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.action_decoder = nn.Linear(latent_dim, 6)
        self.risk_decoder = nn.Linear(latent_dim, 5)
        self.signal_decoder = nn.Linear(latent_dim, 8)
    
    def forward(self, latent: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            latent: [batch_size, latent_dim]
        Returns:
            {
                'action': [batch_size, 6],
                'risk_embedding': [batch_size, 5],
                'strategy_signal': [batch_size, 8]
            }
        """
        return {
            'action': self.action_decoder(latent),
            'risk_embedding': self.risk_decoder(latent),
            'strategy_signal': self.signal_decoder(latent)
        }


class StrategyExecutor(nn.Module):
    """策略执行器 - 根据世界状态生成交易动作"""
    
    def __init__(self, 
                 microstructure_dim: int = 128,
                 macro_dim: int = 96,
                 strategy_dim: int = 64):
        super().__init__()
        
        # 目标调制器
        self.objective_modulator = ObjectiveModulator(strategy_dim, num_objectives=8)
        
        # 决策网络 - 整合所有子世界信息
        total_dim = microstructure_dim + macro_dim + strategy_dim
        self.decision_net = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        # 动作输出头
        self.action_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 6)  # [position_size, entry_price, exit_price, stop_loss, take_profit, leverage]
        )
        
        # 风险调制输出头
        self.risk_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 5)  # [var_95, cvar_95, max_drawdown, sharpe_ratio, sortino_ratio]
        )
        
        # 置信度输出头
        self.confidence_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self,
                microstructure_latent: torch.Tensor,
                macro_latent: torch.Tensor,
                strategy_latent: torch.Tensor,
                objective_idx: int = 0) -> Dict[str, torch.Tensor]:
        """
        执行策略 - 生成交易动作与风险调制
        
        Args:
            microstructure_latent: [batch_size, 128] 微观结构隐状态
            macro_latent: [batch_size, 96] 宏观结构隐状态
            strategy_latent: [batch_size, 64] 策略隐状态
            objective_idx: 策略目标索引 (0-7)
        
        Returns:
            {
                'action': [batch_size, 6] 交易动作,
                'risk_adjustment': [batch_size, 5] 风险调制,
                'confidence': [batch_size, 1] 置信度
            }
        """
        # 应用目标调制
        modulated_strategy = self.objective_modulator(strategy_latent, objective_idx)
        
        # 整合所有信息
        combined = torch.cat([microstructure_latent, macro_latent, modulated_strategy], dim=-1)
        
        # 决策
        decision = self.decision_net(combined)
        
        # 生成输出
        return {
            'action': self.action_head(decision),
            'risk_adjustment': self.risk_head(decision),
            'confidence': self.confidence_head(decision)
        }
