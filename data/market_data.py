"""
市场数据处理 - 加载和预处理金融市场数据
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, Tuple, Optional
import pandas as pd


class MarketDataset(Dataset):
    """市场数据集"""
    
    def __init__(self,
                 price_data: np.ndarray,
                 volume_data: np.ndarray,
                 order_data: np.ndarray,
                 volatility_data: np.ndarray,
                 trend_data: np.ndarray,
                 vol_structure_data: np.ndarray,
                 liquidity_data: np.ndarray,
                 risk_data: np.ndarray,
                 action_data: np.ndarray,
                 risk_embedding_data: np.ndarray,
                 signal_data: np.ndarray,
                 seq_len: int = 60):
        """
        Args:
            price_data: [N, 5] OHLCR数据
            volume_data: [N, 3] 成交量数据
            order_data: [N, 4] 订单数据
            volatility_data: [N, 3] 波动率数据
            trend_data: [N, 5] 趋势数据
            vol_structure_data: [N, 4] 波动结构数据
            liquidity_data: [N, 4] 流动性数据
            risk_data: [N, 4] 风险偏好数据
            action_data: [N, 6] 交易动作数据
            risk_embedding_data: [N, 5] 风险嵌入数据
            signal_data: [N, 8] 策略信号数据
            seq_len: 序列长度
        """
        self.seq_len = seq_len
        self.price_data = torch.from_numpy(price_data).float()
        self.volume_data = torch.from_numpy(volume_data).float()
        self.order_data = torch.from_numpy(order_data).float()
        self.volatility_data = torch.from_numpy(volatility_data).float()
        self.trend_data = torch.from_numpy(trend_data).float()
        self.vol_structure_data = torch.from_numpy(vol_structure_data).float()
        self.liquidity_data = torch.from_numpy(liquidity_data).float()
        self.risk_data = torch.from_numpy(risk_data).float()
        self.action_data = torch.from_numpy(action_data).float()
        self.risk_embedding_data = torch.from_numpy(risk_embedding_data).float()
        self.signal_data = torch.from_numpy(signal_data).float()
        
        self.n_samples = len(price_data) - seq_len
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取单个样本"""
        end_idx = idx + self.seq_len
        
        return {
            'price': self.price_data[idx:end_idx],
            'volume': self.volume_data[idx:end_idx],
            'order': self.order_data[idx:end_idx],
            'volatility': self.volatility_data[idx:end_idx],
            'trend': self.trend_data[idx:end_idx],
            'vol_structure': self.vol_structure_data[idx:end_idx],
            'liquidity': self.liquidity_data[idx:end_idx],
            'risk_preference': self.risk_data[idx:end_idx],
            'action': self.action_data[idx:end_idx],
            'risk_embedding': self.risk_embedding_data[idx:end_idx],
            'strategy_signal': self.signal_data[idx:end_idx]
        }


class DataNormalizer:
    """数据归一化器"""
    
    def __init__(self):
        self.means = {}
        self.stds = {}
        self.fitted = False
    
    def fit(self, data_dict: Dict[str, np.ndarray]):
        """拟合归一化参数"""
        for key, data in data_dict.items():
            self.means[key] = np.mean(data, axis=0)
            self.stds[key] = np.std(data, axis=0) + 1e-8
        self.fitted = True
    
    def normalize(self, data_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """归一化数据"""
        if not self.fitted:
            raise ValueError("Normalizer not fitted yet")
        
        normalized = {}
        for key, data in data_dict.items():
            normalized[key] = (data - self.means[key]) / self.stds[key]
        return normalized
    
    def denormalize(self, data_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """反归一化数据"""
        if not self.fitted:
            raise ValueError("Normalizer not fitted yet")
        
        denormalized = {}
        for key, data in data_dict.items():
            denormalized[key] = data * self.stds[key] + self.means[key]
        return denormalized


class SyntheticMarketDataGenerator:
    """合成市场数据生成器 - 用于演示和测试"""
    
    def __init__(self, n_samples: int = 10000, seed: int = 42):
        np.random.seed(seed)
        self.n_samples = n_samples
    
    def generate(self) -> Dict[str, np.ndarray]:
        """生成合成市场数据"""
        
        # 生成基础价格路径 (几何布朗运动)
        returns = np.random.normal(0.0005, 0.02, self.n_samples)
        prices = 100 * np.exp(np.cumsum(returns))
        
        # OHLCR数据
        price_data = np.zeros((self.n_samples, 5))
        price_data[:, 0] = prices  # open
        price_data[:, 1] = prices + np.abs(np.random.normal(0, 1, self.n_samples))  # high
        price_data[:, 2] = prices - np.abs(np.random.normal(0, 1, self.n_samples))  # low
        price_data[:, 3] = prices  # close
        price_data[:, 4] = returns  # returns
        
        # 成交量数据
        volume_data = np.zeros((self.n_samples, 3))
        volume_data[:, 0] = np.random.lognormal(10, 1, self.n_samples)  # volume
        volume_data[:, 1] = volume_data[:, 0] * np.random.uniform(0.4, 0.6, self.n_samples)  # buy_volume
        volume_data[:, 2] = volume_data[:, 0] * np.random.uniform(0.4, 0.6, self.n_samples)  # sell_volume
        
        # 订单数据
        order_data = np.zeros((self.n_samples, 4))
        order_data[:, 0] = np.random.lognormal(8, 1, self.n_samples)  # bid_volume
        order_data[:, 1] = np.random.lognormal(8, 1, self.n_samples)  # ask_volume
        order_data[:, 2] = np.random.poisson(100, self.n_samples)  # bid_count
        order_data[:, 3] = np.random.poisson(100, self.n_samples)  # ask_count
        
        # 波动率数据
        volatility_data = np.zeros((self.n_samples, 3))
        realized_vol = np.abs(returns) * 100
        volatility_data[:, 0] = realized_vol  # realized_vol
        volatility_data[:, 1] = realized_vol * (1 + np.random.normal(0, 0.1, self.n_samples))  # implied_vol
        volatility_data[:, 2] = np.abs(np.diff(realized_vol, prepend=realized_vol[0])) * 10  # vol_of_vol
        
        # 趋势数据
        trend_data = np.zeros((self.n_samples, 5))
        trend_data[:, 0] = np.abs(returns) * 100  # trend_strength
        trend_data[:, 1] = np.sign(returns)  # trend_direction
        trend_data[:, 2] = np.random.uniform(0, 1, self.n_samples)  # mean_reversion
        trend_data[:, 3] = np.random.uniform(-1, 1, self.n_samples)  # momentum
        trend_data[:, 4] = np.random.uniform(0, 1, self.n_samples)  # regime_prob
        
        # 波动结构数据
        vol_structure_data = np.zeros((self.n_samples, 4))
        vol_structure_data[:, 0] = realized_vol  # vol_level
        vol_structure_data[:, 1] = np.random.uniform(-0.5, 0.5, self.n_samples)  # vol_skew
        vol_structure_data[:, 2] = np.random.uniform(0, 1, self.n_samples)  # vol_term_structure
        vol_structure_data[:, 3] = np.abs(np.diff(realized_vol, prepend=realized_vol[0]))  # vol_clustering
        
        # 流动性数据
        liquidity_data = np.zeros((self.n_samples, 4))
        liquidity_data[:, 0] = np.random.uniform(0.01, 0.1, self.n_samples)  # bid_ask_spread
        liquidity_data[:, 1] = np.random.lognormal(8, 1, self.n_samples)  # market_depth
        liquidity_data[:, 2] = np.random.uniform(0, 1, self.n_samples)  # liquidity_ratio
        liquidity_data[:, 3] = np.random.uniform(0, 1, self.n_samples)  # stress_indicator
        
        # 风险偏好数据
        risk_data = np.zeros((self.n_samples, 4))
        risk_data[:, 0] = np.random.uniform(0, 1, self.n_samples)  # risk_appetite
        risk_data[:, 1] = 20 + np.random.normal(0, 5, self.n_samples)  # vix_level
        risk_data[:, 2] = np.random.uniform(0, 5, self.n_samples)  # credit_spread
        risk_data[:, 3] = np.random.uniform(0, 10, self.n_samples)  # equity_premium
        
        # 交易动作数据
        action_data = np.zeros((self.n_samples, 6))
        action_data[:, 0] = np.random.uniform(-1, 1, self.n_samples)  # position_size
        action_data[:, 1] = prices * np.random.uniform(0.95, 1.05, self.n_samples)  # entry_price
        action_data[:, 2] = prices * np.random.uniform(0.95, 1.05, self.n_samples)  # exit_price
        action_data[:, 3] = prices * np.random.uniform(0.9, 0.99, self.n_samples)  # stop_loss
        action_data[:, 4] = prices * np.random.uniform(1.01, 1.1, self.n_samples)  # take_profit
        action_data[:, 5] = np.random.uniform(1, 5, self.n_samples)  # leverage
        
        # 风险嵌入数据
        risk_embedding_data = np.zeros((self.n_samples, 5))
        risk_embedding_data[:, 0] = np.random.uniform(0, 5, self.n_samples)  # var_95
        risk_embedding_data[:, 1] = np.random.uniform(0, 10, self.n_samples)  # cvar_95
        risk_embedding_data[:, 2] = np.random.uniform(0, 0.5, self.n_samples)  # max_drawdown
        risk_embedding_data[:, 3] = np.random.uniform(0, 2, self.n_samples)  # sharpe_ratio
        risk_embedding_data[:, 4] = np.random.uniform(0, 3, self.n_samples)  # sortino_ratio
        
        # 策略信号数据
        signal_data = np.zeros((self.n_samples, 8))
        signal_data[:, 0] = np.random.uniform(0, 1, self.n_samples)  # signal_strength
        signal_data[:, 1] = np.random.uniform(0, 1, self.n_samples)  # confidence
        signal_data[:, 2] = np.random.uniform(-1, 1, self.n_samples)  # entry_signal
        signal_data[:, 3] = np.random.uniform(-1, 1, self.n_samples)  # exit_signal
        signal_data[:, 4] = np.random.uniform(-1, 1, self.n_samples)  # hedge_signal
        signal_data[:, 5] = np.random.uniform(-1, 1, self.n_samples)  # momentum_signal
        signal_data[:, 6] = np.random.uniform(-1, 1, self.n_samples)  # trend_signal
        signal_data[:, 7] = np.random.uniform(-1, 1, self.n_samples)  # regime_signal
        
        return {
            'price': price_data,
            'volume': volume_data,
            'order': order_data,
            'volatility': volatility_data,
            'trend': trend_data,
            'vol_structure': vol_structure_data,
            'liquidity': liquidity_data,
            'risk_preference': risk_data,
            'action': action_data,
            'risk_embedding': risk_embedding_data,
            'strategy_signal': signal_data
        }
