"""
Sub-World B: Macro & Regime Dynamics (宏观与市场结构层)
建模趋势/震荡状态、波动结构、流动性环境、风险偏好
"""
import torch
import torch.nn as nn
from typing import Dict


class TrendStateEncoder(nn.Module):
    """趋势/震荡状态编码器"""
    
    def __init__(self, input_dim: int = 5, hidden_dim: int = 64, output_dim: int = 24):
        super().__init__()
        # 输入: [trend_strength, trend_direction, mean_reversion, momentum, regime_prob]
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, trend_data: torch.Tensor) -> torch.Tensor:
        """
        Args:
            trend_data: [batch_size, seq_len, 5] 趋势数据
        Returns:
            [batch_size, output_dim] 编码的趋势特征
        """
        encoded = self.net(trend_data)
        return encoded.mean(dim=1)


class VolatilityStructureEncoder(nn.Module):
    """波动结构编码器"""
    
    def __init__(self, input_dim: int = 4, hidden_dim: int = 64, output_dim: int = 24):
        super().__init__()
        # 输入: [vol_level, vol_skew, vol_term_structure, vol_clustering]
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, vol_struct_data: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vol_struct_data: [batch_size, seq_len, 4] 波动结构数据
        Returns:
            [batch_size, output_dim] 编码的波动结构特征
        """
        encoded = self.net(vol_struct_data)
        return encoded.mean(dim=1)


class LiquidityEnvironmentEncoder(nn.Module):
    """流动性环境编码器"""
    
    def __init__(self, input_dim: int = 4, hidden_dim: int = 64, output_dim: int = 24):
        super().__init__()
        # 输入: [bid_ask_spread, market_depth, liquidity_ratio, stress_indicator]
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, liquidity_data: torch.Tensor) -> torch.Tensor:
        """
        Args:
            liquidity_data: [batch_size, seq_len, 4] 流动性数据
        Returns:
            [batch_size, output_dim] 编码的流动性特征
        """
        encoded = self.net(liquidity_data)
        return encoded.mean(dim=1)


class RiskPreferenceEncoder(nn.Module):
    """风险偏好编码器"""
    
    def __init__(self, input_dim: int = 4, hidden_dim: int = 64, output_dim: int = 24):
        super().__init__()
        # 输入: [risk_appetite, vix_level, credit_spread, equity_premium]
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
            risk_data: [batch_size, seq_len, 4] 风险偏好数据
        Returns:
            [batch_size, output_dim] 编码的风险偏好特征
        """
        encoded = self.net(risk_data)
        return encoded.mean(dim=1)


class MacroRegimeWorldEncoder(nn.Module):
    """宏观与市场结构世界编码器 - 整合所有宏观特征"""
    
    def __init__(self, output_dim: int = 96):
        super().__init__()
        self.trend_encoder = TrendStateEncoder(output_dim=24)
        self.vol_struct_encoder = VolatilityStructureEncoder(output_dim=24)
        self.liquidity_encoder = LiquidityEnvironmentEncoder(output_dim=24)
        self.risk_encoder = RiskPreferenceEncoder(output_dim=24)
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(96, 96),
            nn.ReLU(),
            nn.Linear(96, output_dim)
        )
    
    def forward(self, macro_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            macro_data: {
                'trend': [batch_size, seq_len, 5],
                'vol_structure': [batch_size, seq_len, 4],
                'liquidity': [batch_size, seq_len, 4],
                'risk_preference': [batch_size, seq_len, 4]
            }
        Returns:
            [batch_size, output_dim] 宏观结构隐状态
        """
        trend_feat = self.trend_encoder(macro_data['trend'])
        vol_struct_feat = self.vol_struct_encoder(macro_data['vol_structure'])
        liquidity_feat = self.liquidity_encoder(macro_data['liquidity'])
        risk_feat = self.risk_encoder(macro_data['risk_preference'])
        
        # 拼接所有特征
        combined = torch.cat([trend_feat, vol_struct_feat, liquidity_feat, risk_feat], dim=-1)
        return self.fusion(combined)


class MacroRegimeWorldDecoder(nn.Module):
    """宏观与市场结构世界解码器"""
    
    def __init__(self, latent_dim: int = 96):
        super().__init__()
        self.trend_decoder = nn.Linear(latent_dim, 5)
        self.vol_struct_decoder = nn.Linear(latent_dim, 4)
        self.liquidity_decoder = nn.Linear(latent_dim, 4)
        self.risk_decoder = nn.Linear(latent_dim, 4)
    
    def forward(self, latent: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            latent: [batch_size, latent_dim]
        Returns:
            {
                'trend': [batch_size, 5],
                'vol_structure': [batch_size, 4],
                'liquidity': [batch_size, 4],
                'risk_preference': [batch_size, 4]
            }
        """
        return {
            'trend': self.trend_decoder(latent),
            'vol_structure': self.vol_struct_decoder(latent),
            'liquidity': self.liquidity_decoder(latent),
            'risk_preference': self.risk_decoder(latent)
        }
