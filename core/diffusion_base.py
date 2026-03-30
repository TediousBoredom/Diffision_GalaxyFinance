"""
基础扩散模型 - 用于建模市场多模态因果世界
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import math


class SinusoidalPositionalEncoding(nn.Module):
    """时间步的正弦位置编码"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: [batch_size] 时间步张量
        Returns:
            [batch_size, dim] 位置编码
        """
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class DiffusionBlock(nn.Module):
    """基础扩散块 - 用于建模状态演化"""
    
    def __init__(self, latent_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        # 时间编码
        self.time_emb = SinusoidalPositionalEncoding(hidden_dim)
        
        # 主网络
        self.net = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, latent_dim] 当前状态
            t: [batch_size] 时间步
        Returns:
            [batch_size, latent_dim] 预测的噪声或速度
        """
        t_emb = self.time_emb(t)
        x_t = torch.cat([x, t_emb], dim=-1)
        return self.net(x_t)


class ConditionalDiffusionBlock(nn.Module):
    """条件扩散块 - 用于策略-风险干预层"""
    
    def __init__(self, latent_dim: int, condition_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.hidden_dim = hidden_dim
        
        # 时间编码
        self.time_emb = SinusoidalPositionalEncoding(hidden_dim)
        
        # 条件投影
        self.condition_proj = nn.Linear(condition_dim, hidden_dim)
        
        # 主网络
        self.net = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, latent_dim] 当前状态
            t: [batch_size] 时间步
            condition: [batch_size, condition_dim] 条件信息
        Returns:
            [batch_size, latent_dim] 预测的噪声或速度
        """
        t_emb = self.time_emb(t)
        c_emb = self.condition_proj(condition)
        x_t_c = torch.cat([x, t_emb, c_emb], dim=-1)
        return self.net(x_t_c)


class DiffusionSchedule:
    """扩散时间表 - 定义噪声调度"""
    
    def __init__(self, num_steps: int = 1000, schedule_type: str = "linear"):
        self.num_steps = num_steps
        self.schedule_type = schedule_type
        
        if schedule_type == "linear":
            self.betas = torch.linspace(0.0001, 0.02, num_steps)
        elif schedule_type == "cosine":
            s = 0.008
            steps = torch.arange(num_steps + 1)
            alphas_cumprod = torch.cos(((steps / num_steps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clip(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
        
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1), self.alphas_cumprod[:-1]])
        
        # 预计算常用量
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1.0)
    
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """前向扩散过程: q(x_t | x_0)"""
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
        
        # 调整形状以支持广播
        while len(sqrt_alphas_cumprod_t.shape) < len(x0.shape):
            sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.unsqueeze(-1)
            sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.unsqueeze(-1)
        
        return sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise
    
    def get_variance(self, t: torch.Tensor) -> torch.Tensor:
        """计算后向过程的方差"""
        posterior_variance = (
            self.betas[t] * (1.0 - self.alphas_cumprod_prev[t]) / (1.0 - self.alphas_cumprod[t])
        )
        return posterior_variance


class BaseDiffusionModel(nn.Module):
    """基础扩散模型"""
    
    def __init__(self, latent_dim: int, num_steps: int = 1000, schedule_type: str = "linear"):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_steps = num_steps
        
        self.diffusion_block = DiffusionBlock(latent_dim)
        self.schedule = DiffusionSchedule(num_steps, schedule_type)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """预测噪声"""
        return self.diffusion_block(x, t)
    
    def sample(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """从纯噪声采样"""
        x = torch.randn(batch_size, self.latent_dim, device=device)
        
        for t in reversed(range(self.num_steps)):
            t_tensor = torch.full((batch_size,), t, dtype=torch.long, device=device)
            
            with torch.no_grad():
                pred_noise = self(x, t_tensor)
            
            # 后向扩散步
            alpha_t = self.schedule.alphas[t]
            alpha_cumprod_t = self.schedule.alphas_cumprod[t]
            alpha_cumprod_prev_t = self.schedule.alphas_cumprod_prev[t]
            
            beta_t = self.schedule.betas[t]
            variance = self.schedule.get_variance(t_tensor)
            
            # 重参数化技巧
            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
            
            x = (1 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1 - alpha_cumprod_t)) * pred_noise)
            x = x + torch.sqrt(variance) * noise
        
        return x
