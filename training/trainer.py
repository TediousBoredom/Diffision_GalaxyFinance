"""
训练框架 - 分层扩散世界模型的训练
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple
import numpy as np
from tqdm import tqdm


class DiffusionLoss(nn.Module):
    """扩散模型损失函数"""
    
    def __init__(self, loss_type: str = "mse"):
        super().__init__()
        self.loss_type = loss_type
        if loss_type == "mse":
            self.criterion = nn.MSELoss()
        elif loss_type == "l1":
            self.criterion = nn.L1Loss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def forward(self, pred_noise: torch.Tensor, target_noise: torch.Tensor) -> torch.Tensor:
        """计算扩散损失"""
        return self.criterion(pred_noise, target_noise)


class CausalConsistencyLoss(nn.Module):
    """因果一致性损失 - 确保子世界间的因果关系"""
    
    def __init__(self, weight: float = 0.1):
        super().__init__()
        self.weight = weight
    
    def forward(self, 
                latents: Dict[str, torch.Tensor],
                causal_edges: list) -> torch.Tensor:
        """
        计算因果一致性损失
        
        Args:
            latents: {subworld_name: [batch_size, latent_dim]}
            causal_edges: 因果边列表
        
        Returns:
            因果一致性损失
        """
        loss = 0.0
        
        for edge in causal_edges:
            source_latent = latents[edge.source]
            target_latent = latents[edge.target]
            
            # 计算源和目标之间的相关性
            # 高相关性表示因果关系强
            correlation = torch.nn.functional.cosine_similarity(
                source_latent, target_latent, dim=-1
            ).mean()
            
            # 目标相关性应该接近因果强度
            target_correlation = edge.strength
            loss = loss + torch.abs(correlation - target_correlation)
        
        return self.weight * loss


class ReconstructionLoss(nn.Module):
    """重建损失 - 确保编码-解码的一致性"""
    
    def __init__(self, weight: float = 0.1):
        super().__init__()
        self.weight = weight
        self.criterion = nn.MSELoss()
    
    def forward(self,
                original: Dict[str, torch.Tensor],
                reconstructed: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算重建损失"""
        loss = 0.0
        for key in original.keys():
            loss = loss + self.criterion(original[key], reconstructed[key])
        return self.weight * loss


class HierarchicalDiffusionTrainer:
    """分层扩散世界模型训练器"""
    
    def __init__(self,
                 model: nn.Module,
                 device: torch.device,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-5):
        self.model = model.to(device)
        self.device = device
        
        # 优化器
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100,
            eta_min=1e-6
        )
        
        # 损失函数
        self.diffusion_loss = DiffusionLoss(loss_type="mse")
        self.causal_loss = CausalConsistencyLoss(weight=0.1)
        self.reconstruction_loss = ReconstructionLoss(weight=0.05)
        
        # 训练统计
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(train_loader, desc="Training")
        for batch in pbar:
            # 将数据移到设备
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # 创建观测字典
            observations = {
                'microstructure': torch.cat([
                    batch['price'],
                    batch['volume'],
                    batch['order'],
                    batch['volatility']
                ], dim=-1).mean(dim=1),  # [batch_size, total_dim]
                'macro_regime': torch.cat([
                    batch['trend'],
                    batch['vol_structure'],
                    batch['liquidity'],
                    batch['risk_preference']
                ], dim=-1).mean(dim=1),
                'strategy_agent': torch.cat([
                    batch['action'],
                    batch['risk_embedding'],
                    batch['strategy_signal']
                ], dim=-1).mean(dim=1)
            }
            
            # 随机采样时间步
            batch_size = batch['price'].shape[0]
            t = torch.randint(0, self.model.num_steps, (batch_size,), device=self.device)
            
            # 生成随机噪声
            noise = {
                name: torch.randn_like(obs)
                for name, obs in observations.items()
            }
            
            # 前向扩散
            noisy_obs = {}
            for name in observations.keys():
                noisy_obs[name] = self.model.subworlds[name].q_sample(
                    observations[name], t, noise[name]
                )
            
            # 模型预测
            pred_noise = self.model(noisy_obs, t)
            
            # 计算损失
            loss = 0.0
            
            # 扩散损失
            for name in observations.keys():
                loss = loss + self.diffusion_loss(pred_noise[name], noise[name])
            
            # 因果一致性损失
            latents = self.model.get_world_state(observations)
            loss = loss + self.causal_loss(latents, self.model.causal_structure.causal_edges)
            
            # 重建损失
            reconstructed = self.model.decode_world_state(latents)
            loss = loss + self.reconstruction_loss(observations, reconstructed)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(train_loader)
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def validate(self, val_loader: DataLoader) -> float:
        """验证"""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                observations = {
                    'microstructure': torch.cat([
                        batch['price'],
                        batch['volume'],
                        batch['order'],
                        batch['volatility']
                    ], dim=-1).mean(dim=1),
                    'macro_regime': torch.cat([
                        batch['trend'],
                        batch['vol_structure'],
                        batch['liquidity'],
                        batch['risk_preference']
                    ], dim=-1).mean(dim=1),
                    'strategy_agent': torch.cat([
                        batch['action'],
                        batch['risk_embedding'],
                        batch['strategy_signal']
                    ], dim=-1).mean(dim=1)
                }
                
                batch_size = batch['price'].shape[0]
                t = torch.randint(0, self.model.num_steps, (batch_size,), device=self.device)
                
                noise = {
                    name: torch.randn_like(obs)
                    for name, obs in observations.items()
                }
                
                noisy_obs = {}
                for name in observations.keys():
                    noisy_obs[name] = self.model.subworlds[name].q_sample(
                        observations[name], t, noise[name]
                    )
                
                pred_noise = self.model(noisy_obs, t)
                
                loss = 0.0
                for name in observations.keys():
                    loss = loss + self.diffusion_loss(pred_noise[name], noise[name])
                
                latents = self.model.get_world_state(observations)
                loss = loss + self.causal_loss(latents, self.model.causal_structure.causal_edges)
                
                reconstructed = self.model.decode_world_state(latents)
                loss = loss + self.reconstruction_loss(observations, reconstructed)
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader)
        self.val_losses.append(avg_loss)
        return avg_loss
    
    def train(self,
              train_loader: DataLoader,
              val_loader: DataLoader,
              num_epochs: int = 100,
              save_path: Optional[str] = None) -> Dict[str, list]:
        """完整训练循环"""
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # 训练
            train_loss = self.train_epoch(train_loader)
            print(f"Train Loss: {train_loss:.6f}")
            
            # 验证
            val_loss = self.validate(val_loader)
            print(f"Val Loss: {val_loss:.6f}")
            
            # 学习率调度
            self.scheduler.step()
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if save_path:
                    torch.save(self.model.state_dict(), save_path)
                    print(f"Model saved to {save_path}")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
