"""
因果结构定义 - 定义子世界间的因果关系与边界
"""
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class CausalEdge:
    """因果边定义"""
    source: str  # 源子世界
    target: str  # 目标子世界
    latency: int  # 因果延迟（时间步）
    strength: float  # 因果强度 [0, 1]
    modality: str  # 模态类型: 'price', 'volume', 'regime', 'risk', etc.


@dataclass
class SubWorldSpec:
    """子世界规范"""
    name: str
    latent_dim: int
    modalities: List[str]  # 该子世界包含的模态
    description: str


class CausalStructure:
    """定义金融市场的因果结构"""
    
    def __init__(self):
        # 定义三个子世界
        self.subworlds: Dict[str, SubWorldSpec] = {
            "microstructure": SubWorldSpec(
                name="Market Microstructure",
                latent_dim=128,
                modalities=["price_path", "volume_flow", "order_imbalance", "volatility"],
                description="市场微观结构层 - 价格、成交量、订单动态"
            ),
            "macro_regime": SubWorldSpec(
                name="Macro & Regime Dynamics",
                latent_dim=96,
                modalities=["trend_state", "volatility_structure", "liquidity", "risk_preference"],
                description="宏观与市场结构层 - 趋势、波动、流动性、风险偏好"
            ),
            "strategy_agent": SubWorldSpec(
                name="Strategy & Risk Agent",
                latent_dim=64,
                modalities=["action", "risk_embedding", "strategy_signal"],
                description="策略-风险干预层 - 交易动作、风险调制、策略信号"
            )
        }
        
        # 定义因果边 - 实现因果局部性
        self.causal_edges: List[CausalEdge] = [
            # Microstructure -> Macro (市场微观结构影响宏观结构)
            CausalEdge("microstructure", "macro_regime", latency=1, strength=0.8, modality="price_path"),
            CausalEdge("microstructure", "macro_regime", latency=2, strength=0.6, modality="volume_flow"),
            
            # Macro -> Microstructure (宏观结构影响微观动态)
            CausalEdge("macro_regime", "microstructure", latency=1, strength=0.7, modality="regime_state"),
            CausalEdge("macro_regime", "microstructure", latency=1, strength=0.5, modality="liquidity"),
            
            # Strategy -> Microstructure (策略干预市场微观)
            CausalEdge("strategy_agent", "microstructure", latency=0, strength=0.9, modality="action"),
            
            # Strategy -> Macro (策略影响宏观结构)
            CausalEdge("strategy_agent", "macro_regime", latency=1, strength=0.6, modality="risk_embedding"),
            
            # Microstructure -> Strategy (微观结构信息反馈给策略)
            CausalEdge("microstructure", "strategy_agent", latency=0, strength=0.8, modality="price_path"),
            
            # Macro -> Strategy (宏观结构信息反馈给策略)
            CausalEdge("macro_regime", "strategy_agent", latency=0, strength=0.7, modality="regime_state"),
        ]
    
    def get_subworld(self, name: str) -> SubWorldSpec:
        """获取子世界规范"""
        return self.subworlds[name]
    
    def get_incoming_edges(self, target: str) -> List[CausalEdge]:
        """获取指向目标子世界的因果边"""
        return [e for e in self.causal_edges if e.target == target]
    
    def get_outgoing_edges(self, source: str) -> List[CausalEdge]:
        """获取从源子世界出发的因果边"""
        return [e for e in self.causal_edges if e.source == source]
    
    def get_causal_path(self, source: str, target: str) -> Optional[List[str]]:
        """获取从源到目标的因果路径"""
        visited = set()
        path = []
        
        def dfs(current: str) -> bool:
            if current == target:
                path.append(current)
                return True
            
            if current in visited:
                return False
            
            visited.add(current)
            
            for edge in self.get_outgoing_edges(current):
                if dfs(edge.target):
                    path.append(current)
                    return True
            
            return False
        
        if dfs(source):
            return list(reversed(path))
        return None
    
    def get_max_latency(self) -> int:
        """获取最大因果延迟"""
        return max(e.latency for e in self.causal_edges) if self.causal_edges else 0


class CausalLocalityConstraint(nn.Module):
    """因果局部性约束 - 确保子世界间的交互遵循因果结构"""
    
    def __init__(self, causal_structure: CausalStructure):
        super().__init__()
        self.causal_structure = causal_structure
        
        # 为每条因果边创建调制权重
        self.edge_weights = nn.ParameterDict()
        for i, edge in enumerate(causal_structure.causal_edges):
            key = f"{edge.source}_{edge.target}_{i}"
            self.edge_weights[key] = nn.Parameter(
                torch.tensor(edge.strength, dtype=torch.float32)
            )
    
    def apply_causal_mask(self, 
                         source_latent: torch.Tensor,
                         target_name: str,
                         current_step: int) -> torch.Tensor:
        """
        应用因果掩码 - 根据因果结构过滤信息流
        
        Args:
            source_latent: [batch_size, latent_dim] 源子世界的隐状态
            target_name: 目标子世界名称
            current_step: 当前时间步
        
        Returns:
            [batch_size, latent_dim] 经过因果掩码的隐状态
        """
        incoming_edges = self.causal_structure.get_incoming_edges(target_name)
        
        if not incoming_edges:
            return torch.zeros_like(source_latent)
        
        # 计算加权的因果影响
        weighted_influence = torch.zeros_like(source_latent)
        
        for edge in incoming_edges:
            # 检查因果延迟是否满足
            if current_step >= edge.latency:
                key = f"{edge.source}_{edge.target}_{self.causal_structure.causal_edges.index(edge)}"
                weight = torch.sigmoid(self.edge_weights[key])
                weighted_influence = weighted_influence + weight * source_latent
        
        return weighted_influence
    
    def get_causal_graph(self) -> Dict[str, List[str]]:
        """获取因果图表示"""
        graph = {name: [] for name in self.causal_structure.subworlds.keys()}
        for edge in self.causal_structure.causal_edges:
            if edge.target not in graph[edge.source]:
                graph[edge.source].append(edge.target)
        return graph


class ModalityBridge(nn.Module):
    """模态桥接 - 在子世界间传递多模态信息"""
    
    def __init__(self, source_dim: int, target_dim: int, modality: str):
        super().__init__()
        self.source_dim = source_dim
        self.target_dim = target_dim
        self.modality = modality
        
        # 模态特定的投影网络
        self.projection = nn.Sequential(
            nn.Linear(source_dim, 128),
            nn.ReLU(),
            nn.Linear(128, target_dim)
        )
        
        # 模态特定的注意力权重
        self.attention = nn.Sequential(
            nn.Linear(source_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, source_latent: torch.Tensor) -> torch.Tensor:
        """
        跨模态投影与注意力加权
        
        Args:
            source_latent: [batch_size, source_dim]
        
        Returns:
            [batch_size, target_dim] 投影后的隐状态
        """
        # 计算注意力权重
        attention_weight = self.attention(source_latent)
        
        # 投影到目标维度
        projected = self.projection(source_latent)
        
        # 应用注意力加权
        return attention_weight * projected
