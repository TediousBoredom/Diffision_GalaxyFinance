"""
分层世界模型 - 实现Hierarchical Sub-World扩散系统
"""
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, List
from core.diffusion_base import (
    BaseDiffusionModel, ConditionalDiffusionBlock, DiffusionBlock, DiffusionSchedule
)
from core.causal_structure import CausalStructure, CausalLocalityConstraint, ModalityBridge


class SubWorldDiffusionModel(nn.Module):
    """单个子世界的扩散模型"""
    
    def __init__(self, 
                 name: str,
                 latent_dim: int,
                 num_steps: int = 1000,
                 hidden_dim: int = 256):
        super().__init__()
        self.name = name
        self.latent_dim = latent_dim
        self.num_steps = num_steps
        
        # 基础扩散块
        self.diffusion_block = DiffusionBlock(latent_dim, hidden_dim)
        
        # 扩散时间表
        self.schedule = DiffusionSchedule(num_steps, schedule_type="cosine")
        
        # 编码器 - 将观测编码为隐状态
        self.encoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # 解码器 - 将隐状态解码为观测
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
    
    def encode(self, observation: torch.Tensor) -> torch.Tensor:
        """编码观测为隐状态"""
        return self.encoder(observation)
    
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """解码隐状态为观测"""
        return self.decoder(latent)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """预测噪声"""
        return self.diffusion_block(x, t)
    
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """前向扩散过程"""
        return self.schedule.q_sample(x0, t, noise)


class HierarchicalDiffusionWorldModel(nn.Module):
    """分层扩散世界模型 - 核心架构"""
    
    def __init__(self, num_steps: int = 1000):
        super().__init__()
        self.num_steps = num_steps
        
        # 初始化因果结构
        self.causal_structure = CausalStructure()
        
        # 创建三个子世界的扩散模型
        self.subworlds = nn.ModuleDict()
        for name, spec in self.causal_structure.subworlds.items():
            self.subworlds[name] = SubWorldDiffusionModel(
                name=name,
                latent_dim=spec.latent_dim,
                num_steps=num_steps
            )
        
        # 因果局部性约束
        self.causal_constraint = CausalLocalityConstraint(self.causal_structure)
        
        # 模态桥接 - 连接子世界间的信息流
        self.modality_bridges = nn.ModuleDict()
        self._init_modality_bridges()
        
        # 级联扩散块 - 实现跨层交互
        self.cascade_blocks = nn.ModuleDict()
        self._init_cascade_blocks()
    
    def _init_modality_bridges(self):
        """初始化模态桥接"""
        for edge in self.causal_structure.causal_edges:
            source_spec = self.causal_structure.get_subworld(edge.source)
            target_spec = self.causal_structure.get_subworld(edge.target)
            
            key = f"{edge.source}_{edge.target}_{edge.modality}"
            self.modality_bridges[key] = ModalityBridge(
                source_dim=source_spec.latent_dim,
                target_dim=target_spec.latent_dim,
                modality=edge.modality
            )
    
    def _init_cascade_blocks(self):
        """初始化级联扩散块"""
        # 为每个子世界创建条件扩散块，用于接收来自其他子世界的信息
        for target_name, target_spec in self.causal_structure.subworlds.items():
            incoming_edges = self.causal_structure.get_incoming_edges(target_name)
            
            if incoming_edges:
                # 计算总的条件维度
                condition_dim = sum(
                    self.causal_structure.get_subworld(edge.source).latent_dim
                    for edge in incoming_edges
                )
                
                # 创建条件扩散块
                key = f"cascade_{target_name}"
                self.cascade_blocks[key] = ConditionalDiffusionBlock(
                    latent_dim=target_spec.latent_dim,
                    condition_dim=condition_dim,
                    hidden_dim=256
                )
    
    def forward(self, 
                observations: Dict[str, torch.Tensor],
                t: torch.Tensor,
                step: int = 0) -> Dict[str, torch.Tensor]:
        """
        前向传播 - 计算所有子世界的噪声预测
        
        Args:
            observations: {subworld_name: [batch_size, obs_dim]}
            t: [batch_size] 时间步
            step: 当前扩散步数
        
        Returns:
            {subworld_name: [batch_size, latent_dim]} 噪声预测
        """
        # 第一步：编码所有观测为隐状态
        latents = {}
        for name, obs in observations.items():
            latents[name] = self.subworlds[name].encode(obs)
        
        # 第二步：应用因果约束与模态桥接
        conditioned_latents = self._apply_causal_conditioning(latents, step)
        
        # 第三步：通过级联扩散块进行跨层交互
        noise_predictions = {}
        for name in self.subworlds.keys():
            cascade_key = f"cascade_{name}"
            
            if cascade_key in self.cascade_blocks:
                # 获取条件信息
                incoming_edges = self.causal_structure.get_incoming_edges(name)
                condition_parts = []
                
                for edge in incoming_edges:
                    bridge_key = f"{edge.source}_{name}_{edge.modality}"
                    if bridge_key in self.modality_bridges:
                        bridged = self.modality_bridges[bridge_key](conditioned_latents[edge.source])
                        condition_parts.append(bridged)
                
                if condition_parts:
                    condition = torch.cat(condition_parts, dim=-1)
                    noise_pred = self.cascade_blocks[cascade_key](
                        conditioned_latents[name], t, condition
                    )
                else:
                    noise_pred = self.subworlds[name](conditioned_latents[name], t)
            else:
                # 没有条件信息，使用基础扩散块
                noise_pred = self.subworlds[name](conditioned_latents[name], t)
            
            noise_predictions[name] = noise_pred
        
        return noise_predictions
    
    def _apply_causal_conditioning(self, 
                                   latents: Dict[str, torch.Tensor],
                                   step: int) -> Dict[str, torch.Tensor]:
        """
        应用因果条件 - 根据因果结构调制隐状态
        
        Args:
            latents: {subworld_name: [batch_size, latent_dim]}
            step: 当前扩散步数
        
        Returns:
            {subworld_name: [batch_size, latent_dim]} 调制后的隐状态
        """
        conditioned = {}
        
        for target_name in self.subworlds.keys():
            incoming_edges = self.causal_structure.get_incoming_edges(target_name)
            
            # 初始化为原始隐状态
            conditioned_latent = latents[target_name].clone()
            
            # 应用来自其他子世界的因果影响
            for edge in incoming_edges:
                if step >= edge.latency:
                    # 通过模态桥接传递信息
                    bridge_key = f"{edge.source}_{target_name}_{edge.modality}"
                    if bridge_key in self.modality_bridges:
                        causal_influence = self.modality_bridges[bridge_key](latents[edge.source])
                        # 加权融合
                        conditioned_latent = conditioned_latent + edge.strength * causal_influence
            
            conditioned[target_name] = conditioned_latent
        
        return conditioned
    
    def sample(self, 
               batch_size: int,
               device: torch.device,
               num_steps: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        从分层世界模型采样
        
        Args:
            batch_size: 批大小
            device: 设备
            num_steps: 采样步数（默认使用模型的num_steps）
        
        Returns:
            {subworld_name: [batch_size, latent_dim]} 采样的隐状态
        """
        if num_steps is None:
            num_steps = self.num_steps
        
        # 初始化为纯噪声
        latents = {}
        for name, subworld in self.subworlds.items():
            latents[name] = torch.randn(
                batch_size, subworld.latent_dim, device=device
            )
        
        # 反向扩散过程
        for t in reversed(range(num_steps)):
            t_tensor = torch.full((batch_size,), t, dtype=torch.long, device=device)
            
            # 创建虚拟观测（用于前向传播）
            observations = {name: latents[name] for name in latents.keys()}
            
            with torch.no_grad():
                noise_predictions = self.forward(observations, t_tensor, step=t)
            
            # 更新每个子世界的隐状态
            for name in latents.keys():
                latents[name] = self._reverse_diffusion_step(
                    latents[name],
                    noise_predictions[name],
                    t,
                    name,
                    device
                )
        
        return latents
    
    def _reverse_diffusion_step(self,
                               x: torch.Tensor,
                               pred_noise: torch.Tensor,
                               t: int,
                               subworld_name: str,
                               device: torch.device) -> torch.Tensor:
        """执行单步反向扩散"""
        schedule = self.subworlds[subworld_name].schedule
        
        alpha_t = schedule.alphas[t]
        alpha_cumprod_t = schedule.alphas_cumprod[t]
        alpha_cumprod_prev_t = schedule.alphas_cumprod_prev[t]
        beta_t = schedule.betas[t]
        
        # 计算后向均值
        coeff = beta_t / torch.sqrt(1 - alpha_cumprod_t)
        mean = (1 / torch.sqrt(alpha_t)) * (x - coeff * pred_noise)
        
        # 计算方差
        variance = schedule.get_variance(torch.tensor([t], device=device))
        
        # 添加噪声
        if t > 0:
            noise = torch.randn_like(x)
            x = mean + torch.sqrt(variance) * noise
        else:
            x = mean
        
        return x
    
    def get_world_state(self, 
                       observations: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        获取世界状态 - 编码所有观测为隐状态
        
        Args:
            observations: {subworld_name: [batch_size, obs_dim]}
        
        Returns:
            {subworld_name: [batch_size, latent_dim]} 世界隐状态
        """
        world_state = {}
        for name, obs in observations.items():
            world_state[name] = self.subworlds[name].encode(obs)
        return world_state
    
    def decode_world_state(self, 
                          world_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        解码世界状态为观测
        
        Args:
            world_state: {subworld_name: [batch_size, latent_dim]}
        
        Returns:
            {subworld_name: [batch_size, obs_dim]} 解码的观测
        """
        observations = {}
        for name, latent in world_state.items():
            observations[name] = self.subworlds[name].decode(latent)
        return observations
