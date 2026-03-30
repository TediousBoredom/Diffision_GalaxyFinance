"""
Sub-World A: Market Microstructure (市场微观结构层)
建模价格路径、成交量流、订单失衡、波动率演化
"""
import torch
import torch.nn as nn
from typing import Dict, Tuple


class PricePathEncoder(nn.Module):
    """价格路径编码器"""
    
    def __init__(self, input_dim: int = 5, hidden_dim: int = 64, output_dim: int = 32):
        super().__init__()
        # 输入: [open, high, low, close, returns]
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, price_data: torch.Tensor) -> torch.Tensor:
        """
        Args:
            price_data: [batch_size, seq_len, 5] 价格数据
        Returns:
            [batch_size, output_dim] 编码的价格特征
        """
        # 时间序列聚合
        batch_size = price_data.shape[0]
        encoded = self.net(price_data)  # [batch_size, seq_len, output_dim]
        return encoded.mean(dim=1)  # [batch_size, output_dim]


class VolumeFlowEncoder(nn.Module):
    """成交量流编码器"""
    
    def __init__(self, input_dim: int = 3, hidden_dim: int = 64, output_dim: int = 32):
        super().__init__()
        # 输入: [volume, buy_volume, sell_volume]
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, volume_data: torch.Tensor) -> torch.Tensor:
        """
        Args:
            volume_data: [batch_size, seq_len, 3] 成交量数据
        Returns:
            [batch_size, output_dim] 编码的成交量特征
        """
        encoded = self.net(volume_data)
        return encoded.mean(dim=1)


class OrderImbalanceEncoder(nn.Module):
    """订单失衡编码器"""
    
    def __init__(self, input_dim: int = 4, hidden_dim: int = 64, output_dim: int = 32):
        super().__init__()
        # 输入: [bid_volume, ask_volume, bid_count, ask_count]
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, order_data: torch.Tensor) -> torch.Tensor:
        """
        Args:
            order_data: [batch_size, seq_len, 4] 订单数据
        Returns:
            [batch_size, output_dim] 编码的订单失衡特征
        """
        encoded = self.net(order_data)
        return encoded.mean(dim=1)


class VolatilityEncoder(nn.Module):
    """波动率演化编码器"""
    
    def __init__(self, input_dim: int = 3, hidden_dim: int = 64, output_dim: int = 32):
        super().__init__()
        # 输入: [realized_vol, implied_vol, vol_of_vol]
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, vol_data: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vol_data: [batch_size, seq_len, 3] 波动率数据
        Returns:
            [batch_size, output_dim] 编码的波动率特征
        """
        encoded = self.net(vol_data)
        return encoded.mean(dim=1)


class MicrostructureWorldEncoder(nn.Module):
    """市场微观结构世界编码器 - 整合所有微观特征"""
    
    def __init__(self, output_dim: int = 128):
        super().__init__()
        self.price_encoder = PricePathEncoder(output_dim=32)
        self.volume_encoder = VolumeFlowEncoder(output_dim=32)
        self.order_encoder = OrderImbalanceEncoder(output_dim=32)
        self.vol_encoder = VolatilityEncoder(output_dim=32)
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, market_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            market_data: {
                'price': [batch_size, seq_len, 5],
                'volume': [batch_size, seq_len, 3],
                'order': [batch_size, seq_len, 4],
                'volatility': [batch_size, seq_len, 3]
            }
        Returns:
            [batch_size, output_dim] 微观结构隐状态
        """
        price_feat = self.price_encoder(market_data['price'])
        volume_feat = self.volume_encoder(market_data['volume'])
        order_feat = self.order_encoder(market_data['order'])
        vol_feat = self.vol_encoder(market_data['volatility'])
        
        # 拼接所有特征
        combined = torch.cat([price_feat, volume_feat, order_feat, vol_feat], dim=-1)
        return self.fusion(combined)


class MicrostructureWorldDecoder(nn.Module):
    """市场微观结构世界解码器"""
    
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.price_decoder = nn.Linear(latent_dim, 5)
        self.volume_decoder = nn.Linear(latent_dim, 3)
        self.order_decoder = nn.Linear(latent_dim, 4)
        self.vol_decoder = nn.Linear(latent_dim, 3)
    
    def forward(self, latent: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            latent: [batch_size, latent_dim]
        Returns:
            {
                'price': [batch_size, 5],
                'volume': [batch_size, 3],
                'order': [batch_size, 4],
                'volatility': [batch_size, 3]
            }
        """
        return {
            'price': self.price_decoder(latent),
            'volume': self.volume_decoder(latent),
            'order': self.order_decoder(latent),
            'volatility': self.vol_decoder(latent)
        }
