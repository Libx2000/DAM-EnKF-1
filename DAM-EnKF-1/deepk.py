import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


def _prepare_obs_operator(obs_operator: Optional[torch.Tensor], batch_size: int,
                          height: int, width: int, obs_dim: int, state_dim: int,
                          device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """将观测算子统一为 (batch, height, width, obs_dim, state_dim)。"""
    if obs_operator is None:
        if obs_dim > state_dim:
            raise ValueError("obs_dim > state_dim 时必须显式提供观测算子 H")
        h_matrix = torch.zeros(obs_dim, state_dim, device=device, dtype=dtype)
        h_matrix[:, :obs_dim] = torch.eye(obs_dim, device=device, dtype=dtype)
        obs_operator = h_matrix.view(1, 1, 1, obs_dim, state_dim)
    else:
        obs_operator = torch.as_tensor(obs_operator, device=device, dtype=dtype)
        if obs_operator.ndim == 2:
            obs_operator = obs_operator.view(1, 1, 1, obs_dim, state_dim)
        elif obs_operator.ndim == 4:
            obs_operator = obs_operator.unsqueeze(0)
    expected = (height, width, obs_dim, state_dim)
    if obs_operator.ndim != 5 or obs_operator.shape[1:] != expected:
        raise ValueError(f"H 的形状应为 (O,S)、(H,W,O,S) 或 (B,H,W,O,S)，当前为 {tuple(obs_operator.shape)}")
    if obs_operator.shape[0] not in (1, batch_size):
        raise ValueError("H 的 batch 维必须为 1 或 batch_size")
    return obs_operator.expand(batch_size, -1, -1, -1, -1)


def _prepare_obs_error(obs_error: Optional[torch.Tensor], batch_size: int,
                       height: int, width: int, obs_dim: int,
                       device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """将 R 统一为逐格点矩阵 (batch, height, width, obs_dim, obs_dim)。"""
    if obs_error is None:
        obs_error = torch.eye(obs_dim, device=device, dtype=dtype) * 0.1
    else:
        obs_error = torch.as_tensor(obs_error, device=device, dtype=dtype)
    if obs_error.ndim == 0:
        obs_error = torch.eye(obs_dim, device=device, dtype=dtype) * obs_error
    if obs_error.ndim == 1:
        obs_error = torch.diag(obs_error)
    if obs_error.ndim == 2:
        obs_error = obs_error.view(1, 1, 1, obs_dim, obs_dim)
    elif obs_error.ndim == 4:
        if obs_error.shape == (height, width, obs_dim, obs_dim):
            obs_error = obs_error.unsqueeze(0)
        elif obs_error.shape[1:] == (height, width, obs_dim):
            obs_error = torch.diag_embed(obs_error)
        else:
            raise ValueError("四维 R 应为 (H,W,O,O) 或 (B,H,W,O)")
    expected = (height, width, obs_dim, obs_dim)
    if obs_error.ndim != 5 or obs_error.shape[1:] != expected:
        raise ValueError(f"R 的形状应为 (O,O)、(B,H,W,O) 或 (B,H,W,O,O)，当前为 {tuple(obs_error.shape)}")
    if obs_error.shape[0] not in (1, batch_size):
        raise ValueError("R 的 batch 维必须为 1 或 batch_size")
    obs_error = obs_error.expand(batch_size, -1, -1, -1, -1)
    if torch.max(torch.abs(obs_error - obs_error.transpose(-1, -2))) > 1e-5:
        raise ValueError("观测误差协方差 R 必须对称")
    return obs_error


def _prepare_obs_mask(obs_mask: Optional[torch.Tensor], batch_size: int,
                      height: int, width: int, obs_dim: int,
                      device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """将显式观测掩码统一为 (batch, height, width, obs_dim)。"""
    if obs_mask is None:
        return torch.ones(batch_size, height, width, obs_dim,
                          device=device, dtype=dtype)
    obs_mask = torch.as_tensor(obs_mask, device=device)
    if obs_mask.ndim == 3:
        obs_mask = obs_mask.unsqueeze(0)
    if obs_mask.ndim != 4 or obs_mask.shape[1:] != (height, width, obs_dim):
        raise ValueError("obs_mask 的形状应为 (H,W,O) 或 (B,H,W,O)")
    if obs_mask.shape[0] not in (1, batch_size):
        raise ValueError("obs_mask 的 batch 维必须为 1 或 batch_size")
    return obs_mask.expand(batch_size, -1, -1, -1).to(dtype=dtype)


class ObservationConditioner(nn.Module):
    """用轻量 1x1 卷积分支把 H、R 和 mask 注入原集合特征。"""

    def __init__(self, state_dim: int, obs_dim: int, hidden_channels: int):
        super(ObservationConditioner, self).__init__()
        in_channels = obs_dim * state_dim + obs_dim * obs_dim + obs_dim
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1)
        )

    def forward(self, obs_operator: torch.Tensor, obs_error: torch.Tensor,
                obs_mask: torch.Tensor) -> torch.Tensor:
        batch_size, height, width = obs_mask.shape[:3]
        condition = torch.cat([
            obs_operator.reshape(batch_size, height, width, -1),
            obs_error.reshape(batch_size, height, width, -1),
            obs_mask
        ], dim=-1)
        return self.layers(condition.permute(0, 3, 1, 2))


class KalmanGainLearner(nn.Module):
    """
    学习从集合扰动到卡尔曼增益矩阵K的深度学习模型
    输入: 集合扰动矩阵 (ensemble_size, 64, 128, state_dim)
    输出: 近似的卡尔曼增益矩阵K (64, 128, state_dim, obs_dim)
    """
    
    def __init__(self, 
                 state_dim: int = 1,
                 obs_dim: int = 1,
                 ensemble_size: int = 20,
                 hidden_channels: int = 64,
                 num_encoder_layers: int = 3,
                 use_attention: bool = True):
        """
        初始化模型
        
        Args:
            state_dim: 状态变量维度（如温度、湿度、气压等）
            obs_dim: 观测变量维度
            ensemble_size: 集合成员数量
            hidden_channels: 隐藏层通道数
            num_encoder_layers: 编码器层数
            use_attention: 是否使用注意力机制
        """
        super(KalmanGainLearner, self).__init__()
        
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.ensemble_size = ensemble_size
        self.hidden_channels = hidden_channels
        self.use_attention = use_attention
        
        # 1. 集合扰动编码器 - 处理每个集合成员
        self.ensemble_encoder = EnsembleEncoder(
            ensemble_size=ensemble_size,
            state_dim=state_dim,
            hidden_channels=hidden_channels,
            num_layers=num_encoder_layers
        )

        # 轻量条件分支：不改变原主干，只把 H、R 和 mask 加到集合编码特征上
        self.observation_conditioner = ObservationConditioner(
            state_dim=state_dim,
            obs_dim=obs_dim,
            hidden_channels=hidden_channels
        )
        
        # 2. 空间特征提取器 - 使用卷积网络提取空间特征
        self.spatial_encoder = SpatialEncoder(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels * 2,
            out_channels=hidden_channels * 4
        )
        
        # 3. 卡尔曼增益解码器 - 生成K矩阵
        self.kalman_decoder = KalmanGainDecoder(
            in_channels=hidden_channels * 4,
            hidden_channels=hidden_channels * 2,
            state_dim=state_dim,
            obs_dim=obs_dim
        )
        
        # 4. 可选的注意力机制 - 捕捉长距离依赖
        if use_attention:
            self.attention = SpatialAttention(
                channels=hidden_channels * 4,
                grid_size=(64, 128)
            )
        
        # 5. 保留旧接口；不再对 K 施加协方差对称或单位范数约束
        self.covariance_reg = CovarianceRegularization()
        
    def forward(self, ensemble_perturbations: torch.Tensor,
                obs_operator: Optional[torch.Tensor] = None,
                obs_error: Optional[torch.Tensor] = None,
                obs_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            ensemble_perturbations: 形状为 (batch_size, ensemble_size, 64, 128, state_dim)
        
        Returns:
            kalman_gain: 形状为 (batch_size, 64, 128, state_dim, obs_dim)
        """
        batch_size, _, height, width, state_dim = ensemble_perturbations.shape
        if state_dim != self.state_dim:
            raise ValueError("输入集合的 state_dim 与模型配置不一致")

        obs_operator = _prepare_obs_operator(
            obs_operator, batch_size, height, width, self.obs_dim,
            self.state_dim, ensemble_perturbations.device,
            ensemble_perturbations.dtype
        )
        obs_error = _prepare_obs_error(
            obs_error, batch_size, height, width, self.obs_dim,
            ensemble_perturbations.device, ensemble_perturbations.dtype
        )
        obs_mask = _prepare_obs_mask(
            obs_mask, batch_size, height, width, self.obs_dim,
            ensemble_perturbations.device, ensemble_perturbations.dtype
        )
        
        # 编码集合扰动
        encoded = self.ensemble_encoder(ensemble_perturbations)
        encoded = encoded + self.observation_conditioner(
            obs_operator, obs_error, obs_mask
        )
        
        # 提取空间特征
        spatial_features = self.spatial_encoder(encoded)
        
        # 应用注意力机制
        if self.use_attention:
            spatial_features = self.attention(spatial_features)
        
        # 解码为卡尔曼增益矩阵
        kalman_gain = self.kalman_decoder(spatial_features)
        
        # 应用协方差正则化
        kalman_gain = self.covariance_reg(kalman_gain)
        # 缺测变量对应的增益列必须严格为零
        kalman_gain = kalman_gain * obs_mask.unsqueeze(-2)
        
        return kalman_gain
    
    def compute_kalman_update(self, 
                             background: torch.Tensor,
                             observations: torch.Tensor,
                             ensemble_perturbations: torch.Tensor,
                             obs_operator: Optional[torch.Tensor] = None,
                             obs_error: Optional[torch.Tensor] = None,
                             obs_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算分析场更新
        
        Args:
            background: 背景场 (batch_size, 64, 128, state_dim)
            observations: 网格化观测场 (batch_size, 64, 128, obs_dim)
            ensemble_perturbations: 集合扰动 (batch_size, ensemble_size, 64, 128, state_dim)
            obs_operator: 观测算子 (obs_dim, state_dim) 或逐格点形式
            obs_error: 观测误差协方差 (obs_dim, obs_dim) 或逐格点形式
            obs_mask: 显式观测掩码 (batch_size, 64, 128, obs_dim)
            
        Returns:
            analysis: 分析场 (batch_size, 64, 128, state_dim)
        """
        batch_size, height, width, state_dim = background.shape
        obs_operator = _prepare_obs_operator(
            obs_operator, batch_size, height, width, self.obs_dim,
            state_dim, background.device, background.dtype
        )
        obs_error = _prepare_obs_error(
            obs_error, batch_size, height, width, self.obs_dim,
            background.device, background.dtype
        )
        obs_mask = _prepare_obs_mask(
            obs_mask, batch_size, height, width, self.obs_dim,
            background.device, background.dtype
        )
        if observations.shape == (batch_size, self.obs_dim, height, width):
            observations = observations.permute(0, 2, 3, 1)
        if observations.shape != (batch_size, height, width, self.obs_dim):
            raise ValueError("observations 应为 (B,H,W,O) 或 (B,O,H,W)")

        # K 已经包含 R 的影响，因此这里只应用显式掩码后的创新
        kalman_gain = self.forward(
            ensemble_perturbations, obs_operator, obs_error, obs_mask
        )
        Hx_b = torch.einsum('bhwos,bhws->bhwo', obs_operator, background)
        obs_increment = obs_mask * (observations - Hx_b)
        analysis_increment = torch.einsum(
            'bhwso,bhwo->bhws', kalman_gain, obs_increment
        )
        
        # 计算分析场
        analysis = background + analysis_increment
        
        return analysis


class EnsembleEncoder(nn.Module):
    """编码集合扰动特征的模块"""
    
    def __init__(self, ensemble_size: int, state_dim: int, 
                 hidden_channels: int, num_layers: int):
        super(EnsembleEncoder, self).__init__()
        
        # 集合统计特征计算层
        self.ensemble_stats = EnsembleStatistics(ensemble_size)
        
        # 多层感知机处理统计特征
        self.mlp = nn.Sequential()
        input_size = state_dim * 4  # 均值、方差、斜度、峰度
        
        for i in range(num_layers):
            output_size = hidden_channels if i == num_layers - 1 else hidden_channels // 2
            self.mlp.add_module(f'linear_{i}', nn.Linear(input_size, output_size))
            self.mlp.add_module(f'relu_{i}', nn.ReLU(inplace=True))
            input_size = output_size
            
        # 最后的卷积层处理空间信息
        self.conv = nn.Conv2d(hidden_channels, hidden_channels, 
                             kernel_size=3, padding=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, ensemble_size, height, width, state_dim)
        """
        # 计算集合统计特征
        stats = self.ensemble_stats(x)  # (batch_size, height, width, state_dim*4)
        
        batch_size, height, width, stats_dim = stats.shape
        
        # 通过MLP处理每个网格点的特征
        stats_flat = stats.reshape(batch_size * height * width, stats_dim)
        features_flat = self.mlp(stats_flat)
        features = features_flat.reshape(batch_size, height, width, -1)
        
        # 调整维度并应用卷积
        features = features.permute(0, 3, 1, 2)
        features = self.conv(features)
        
        return features


class EnsembleStatistics(nn.Module):
    """计算集合扰动统计特征的模块"""
    
    def __init__(self, ensemble_size: int):
        super(EnsembleStatistics, self).__init__()
        self.ensemble_size = ensemble_size
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, ensemble_size, height, width, state_dim)
        """
        # 计算均值
        mean = torch.mean(x, dim=1)
        
        # 计算方差
        variance = torch.var(x, dim=1, unbiased=True)
        
        # 计算斜度（三阶矩）
        centered = x - mean.unsqueeze(1)
        skewness = torch.mean(centered ** 3, dim=1) / (variance ** 1.5 + 1e-8)
        
        # 计算峰度（四阶矩）
        kurtosis = torch.mean(centered ** 4, dim=1) / (variance ** 2 + 1e-8) - 3
        
        # 拼接所有统计特征
        stats = torch.cat([mean, variance, skewness, kurtosis], dim=-1)
        
        return stats


class SpatialEncoder(nn.Module):
    """空间特征提取器 - 使用卷积网络"""
    
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super(SpatialEncoder, self).__init__()
        
        # 编码器部分 - 下采样
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels * 2, hidden_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        
        # 瓶颈层
        self.bottleneck = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, hidden_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels * 4, hidden_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels * 4),
            nn.ReLU(inplace=True),
        )
        
        # 解码器部分 - 上采样回原始尺寸
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_channels * 4, hidden_channels * 2, 
                              kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(hidden_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels * 2, hidden_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels * 2),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(hidden_channels * 2, out_channels, 
                              kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.bottleneck(self.encoder(x)))


class SpatialAttention(nn.Module):
    """空间注意力机制 - 捕捉长距离空间依赖"""
    
    def __init__(self, channels: int, grid_size: Tuple[int, int]):
        super(SpatialAttention, self).__init__()
        self.channels = channels
        self.grid_size = grid_size
        
        # 查询、键、值投影
        self.query = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.key = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)
        
        # 可学习的位置编码
        self.position_encoding = nn.Parameter(
            torch.randn(1, channels // 8, grid_size[0], grid_size[1])
        )
        
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = x.shape
        
        # 添加位置编码
        pos_enc = self.position_encoding.expand(batch_size, -1, -1, -1)
        
        # 计算查询、键、值
        query = self.query(x) + pos_enc
        key = self.key(x) + pos_enc
        value = self.value(x)
        
        # 重塑为矩阵形式
        query = query.view(batch_size, -1, height * width).permute(0, 2, 1)
        key = key.view(batch_size, -1, height * width)
        value = value.view(batch_size, -1, height * width).permute(0, 2, 1)
        
        # 计算注意力分数
        attention = torch.bmm(query, key) / (self.channels // 8) ** 0.5
        attention = self.softmax(attention)
        
        # 应用注意力
        out = torch.bmm(attention, value)
        out = out.permute(0, 2, 1).view(batch_size, -1, height, width)
        
        # 残差连接
        out = out + x
        
        return out


class KalmanGainDecoder(nn.Module):
    """卡尔曼增益矩阵解码器"""
    
    def __init__(self, in_channels: int, hidden_channels: int, 
                 state_dim: int, obs_dim: int):
        super(KalmanGainDecoder, self).__init__()
        
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        
        # 生成卡尔曼增益矩阵的卷积层
        self.gain_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, state_dim * obs_dim, kernel_size=3, padding=1),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, in_channels, height, width)
        返回: (batch_size, height, width, state_dim, obs_dim)
        """
        batch_size, _, height, width = x.shape
        
        # 生成卡尔曼增益矩阵
        gain = self.gain_conv(x)  # (batch_size, state_dim*obs_dim, height, width)
        
        # 重塑为最终格式
        gain = gain.view(batch_size, self.state_dim, self.obs_dim, height, width)
        gain = gain.permute(0, 3, 4, 1, 2)  # (batch_size, height, width, state_dim, obs_dim)
        
        return gain


class CovarianceRegularization(nn.Module):
    """兼容旧模型接口的增益输出层；不对 Kalman gain 施加协方差约束。"""
    
    def __init__(self, epsilon: float = 1e-6):
        super(CovarianceRegularization, self).__init__()
        self.epsilon = epsilon
        
    def forward(self, kalman_gain: torch.Tensor) -> torch.Tensor:
        """
        Kalman gain 通常是矩形矩阵，不具有协方差对称性。
        同时不能强制单位范数，否则会破坏由 H 和 R 决定的增益尺度。
        空间平滑通过 KalmanLoss 中的正则项实现，此处保持原值。
        kalman_gain: (batch_size, height, width, state_dim, obs_dim)
        """
        return kalman_gain


class KalmanLoss(nn.Module):
    """用于训练卡尔曼增益学习器的损失函数"""
    
    def __init__(self, alpha: float = 1.0, beta: float = 0.1, gamma: float = 0.01):
        super(KalmanLoss, self).__init__()
        self.alpha = alpha  # 分析场误差权重
        self.beta = beta    # 卡尔曼增益误差权重
        self.gamma = gamma  # 正则化权重
        
        self.mse_loss = nn.MSELoss()
        
    def forward(self, 
                predicted_k: torch.Tensor,
                target_k: torch.Tensor,
                predicted_analysis: torch.Tensor,
                target_analysis: torch.Tensor) -> torch.Tensor:
        """
        计算多任务损失
        """
        # 1. 分析场误差
        analysis_loss = self.mse_loss(predicted_analysis, target_analysis)
        
        # 2. 卡尔曼增益矩阵误差
        kalman_loss = self.mse_loss(predicted_k, target_k)
        
        # 3. 正则化项 - 鼓励平滑和稳定的增益矩阵
        # 空间平滑性
        spatial_grad = torch.mean(torch.abs(predicted_k[:, 1:, :, :, :] - predicted_k[:, :-1, :, :, :])) + \
                      torch.mean(torch.abs(predicted_k[:, :, 1:, :, :] - predicted_k[:, :, :-1, :, :]))
        
        # 增益矩阵的Frobenius范数（防止过大）
        norm_loss = torch.mean(torch.norm(predicted_k, dim=(3, 4)))
        
        regularization_loss = spatial_grad + norm_loss
        
        # 总损失
        total_loss = self.alpha * analysis_loss + \
                    self.beta * kalman_loss + \
                    self.gamma * regularization_loss
        
        return total_loss, {
            'total': total_loss.item(),
            'analysis': analysis_loss.item(),
            'kalman': kalman_loss.item(),
            'regularization': regularization_loss.item()
        }


# 数据生成器 - 用于训练和测试
class DataGenerator:
    """生成模拟的大气资料同化数据"""
    
    def __init__(self, grid_size: Tuple[int, int] = (64, 128),
                 state_dim: int = 1, obs_dim: int = 1,
                 ensemble_size: int = 20):
        if obs_dim > state_dim:
            raise ValueError("默认观测算子要求 obs_dim <= state_dim")
        self.grid_size = grid_size
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.ensemble_size = ensemble_size
        
    def generate_batch(self, batch_size: int = 32) -> dict:
        """
        生成一个批次的训练数据
        """
        # 生成背景场和集合扰动
        background = self._generate_background(batch_size)
        ensemble_perturbations = self._generate_ensemble_perturbations(
            batch_size, background
        )
        
        # 生成观测数据
        observations, obs_operator, obs_error, obs_mask = self._generate_observations(
            batch_size, background
        )
        
        # 计算包含局地 H、R 和显式观测掩码的真实逐格点 Kalman gain
        kalman_gain = self._compute_kalman_gain(
            ensemble_perturbations, obs_operator, obs_error, obs_mask
        )
        
        # 计算分析场
        analysis = self._compute_analysis(
            background, observations, kalman_gain, obs_operator, obs_mask
        )
        
        return {
            'background': torch.FloatTensor(background),
            'ensemble_perturbations': torch.FloatTensor(ensemble_perturbations),
            'observations': torch.FloatTensor(observations),
            'obs_operator': torch.FloatTensor(obs_operator),
            'obs_error': torch.FloatTensor(obs_error),
            'obs_mask': torch.BoolTensor(obs_mask),
            'kalman_gain': torch.FloatTensor(kalman_gain),
            'analysis': torch.FloatTensor(analysis)
        }
    
    def _generate_background(self, batch_size: int) -> np.ndarray:
        """生成背景场（模拟大气状态）"""
        background = np.random.randn(batch_size, *self.grid_size, self.state_dim) * 0.5
        
        # 添加空间相关性（模拟大气场的空间结构）
        from scipy.ndimage import gaussian_filter
        for b in range(batch_size):
            for d in range(self.state_dim):
                background[b, ..., d] = gaussian_filter(
                    background[b, ..., d], sigma=2.0
                )
        
        return background
    
    def _generate_ensemble_perturbations(self, batch_size: int, 
                                        background: np.ndarray) -> np.ndarray:
        """生成集合扰动"""
        perturbations = np.zeros((batch_size, self.ensemble_size, 
                                 *self.grid_size, self.state_dim))
        
        for b in range(batch_size):
            # 生成均值为0的扰动
            for e in range(self.ensemble_size):
                pert = np.random.randn(*self.grid_size, self.state_dim) * 0.1
                
                # 添加与背景场相关的扰动结构
                pert = pert * (1 + 0.5 * np.abs(background[b]))
                
                # 添加空间平滑性
                from scipy.ndimage import gaussian_filter
                for d in range(self.state_dim):
                    pert[..., d] = gaussian_filter(pert[..., d], sigma=1.0)
                
                perturbations[b, e] = pert
        
        return perturbations
    
    def _generate_observations(self, batch_size: int, 
                              background: np.ndarray) -> Tuple[np.ndarray, ...]:
        """生成与模型网格一致的观测、H、R 和显式 mask。"""
        height, width = self.grid_size
        obs_operator = np.zeros((self.obs_dim, self.state_dim), dtype=np.float64)
        obs_operator[:, :self.obs_dim] = np.eye(self.obs_dim)
        # 这里存储的是误差方差；标准差为 0.1
        obs_error = np.eye(self.obs_dim, dtype=np.float64) * 0.01
        obs_mask = np.random.rand(
            batch_size, height, width, self.obs_dim
        ) < 0.30

        noiseless = np.einsum('os,bhws->bhwo', obs_operator, background)
        noise = np.random.randn(*noiseless.shape) * 0.1
        observations = np.where(obs_mask, noiseless + noise, 0.0)
        return observations, obs_operator, obs_error, obs_mask
    
    def _compute_kalman_gain(self, ensemble_perturbations: np.ndarray,
                             obs_operator: np.ndarray,
                             obs_error: np.ndarray,
                             obs_mask: np.ndarray) -> np.ndarray:
        """逐格点计算 K=P H^T (H P H^T+R)^-1，不构造全局协方差。"""
        batch_size, ensemble_size, height, width, state_dim = ensemble_perturbations.shape
        obs_dim = obs_operator.shape[-2]
        if obs_operator.ndim == 2:
            h_operator = np.broadcast_to(
                obs_operator.reshape(1, 1, 1, obs_dim, state_dim),
                (batch_size, height, width, obs_dim, state_dim)
            )
        elif obs_operator.ndim == 4:
            h_operator = np.broadcast_to(
                obs_operator[None],
                (batch_size, height, width, obs_dim, state_dim)
            )
        elif obs_operator.shape == (batch_size, height, width, obs_dim, state_dim):
            h_operator = obs_operator
        else:
            raise ValueError("obs_operator 的形状与集合不一致")

        if obs_error.ndim == 2:
            r_error = np.broadcast_to(
                obs_error.reshape(1, 1, 1, obs_dim, obs_dim),
                (batch_size, height, width, obs_dim, obs_dim)
            )
        elif obs_error.ndim == 4:
            r_error = np.broadcast_to(
                obs_error[None],
                (batch_size, height, width, obs_dim, obs_dim)
            )
        elif obs_error.shape == (batch_size, height, width, obs_dim, obs_dim):
            r_error = obs_error
        else:
            raise ValueError("obs_error 的形状与集合不一致")
        mask = obs_mask.astype(ensemble_perturbations.dtype)
        masked_h = h_operator * mask[..., None]

        anomalies = ensemble_perturbations - np.mean(
            ensemble_perturbations, axis=1, keepdims=True
        )
        local_covariance = np.einsum(
            'bnhws,bnhwt->bhwst', anomalies, anomalies
        ) / max(ensemble_size - 1, 1)
        p_h_transpose = np.einsum(
            'bhwst,bhwot->bhwso', local_covariance, masked_h
        )
        innovation_covariance = np.einsum(
            'bhwos,bhwst,bhwpt->bhwop',
            masked_h, local_covariance, masked_h
        )

        mask_outer = mask[..., :, None] * mask[..., None, :]
        identity = np.eye(obs_dim).reshape(1, 1, 1, obs_dim, obs_dim)
        effective_r = r_error * mask_outer + np.eye(obs_dim) * (1.0 - mask)[..., None, :]
        innovation_covariance = innovation_covariance + effective_r + 1e-6 * identity
        kalman_gain = np.linalg.solve(
            innovation_covariance,
            np.swapaxes(p_h_transpose, -1, -2)
        )
        kalman_gain = np.swapaxes(kalman_gain, -1, -2)
        return kalman_gain * mask[..., None, :]
    
    def _compute_analysis(self, background: np.ndarray, 
                         observations: np.ndarray,
                         kalman_gain: np.ndarray,
                         obs_operator: np.ndarray,
                         obs_mask: np.ndarray) -> np.ndarray:
        """使用与网络相同的显式 mask 和逐格点增益计算分析场。"""
        if obs_operator.ndim == 2:
            hx_background = np.einsum('os,bhws->bhwo', obs_operator, background)
        else:
            if obs_operator.ndim == 4:
                obs_operator = obs_operator[None]
            hx_background = np.einsum('bhwos,bhws->bhwo', obs_operator, background)
        innovation = obs_mask * (observations - hx_background)
        analysis_increment = np.einsum(
            'bhwso,bhwo->bhws', kalman_gain, innovation
        )
        return background + analysis_increment


# 训练函数
def train_kalman_model(model, data_generator, num_epochs=100, batch_size=32, lr=1e-3):
    """训练卡尔曼增益学习器"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = KalmanLoss()
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        # 生成训练数据
        data = data_generator.generate_batch(batch_size)
        
        # 转移到设备
        for key in data:
            data[key] = data[key].to(device)
        
        # 前向传播
        predicted_k = model(
            data['ensemble_perturbations'],
            obs_operator=data['obs_operator'],
            obs_error=data['obs_error'],
            obs_mask=data['obs_mask']
        )
        
        # 计算分析场
        predicted_analysis = model.compute_kalman_update(
            data['background'],
            data['observations'],
            data['ensemble_perturbations'],
            obs_operator=data['obs_operator'],
            obs_error=data['obs_error'],
            obs_mask=data['obs_mask']
        )
        
        # 计算损失
        loss, loss_dict = criterion(
            predicted_k, data['kalman_gain'],
            predicted_analysis, data['analysis']
        )
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}')
            print(f'  Breakdown - Analysis: {loss_dict["analysis"]:.4f}, '
                  f'Kalman: {loss_dict["kalman"]:.4f}, '
                  f'Reg: {loss_dict["regularization"]:.4f}')
    
    return model


# 使用示例
if __name__ == "__main__":
    # 设置参数
    GRID_SIZE = (64, 128)
    STATE_DIM = 1  # 例如，只同化温度场
    OBS_DIM = 1    # 温度观测
    ENSEMBLE_SIZE = 20
    BATCH_SIZE = 16
    
    # 创建模型
    model = KalmanGainLearner(
        state_dim=STATE_DIM,
        obs_dim=OBS_DIM,
        ensemble_size=ENSEMBLE_SIZE,
        hidden_channels=32,
        num_encoder_layers=2,
        use_attention=True
    )
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建数据生成器
    data_gen = DataGenerator(
        grid_size=GRID_SIZE,
        state_dim=STATE_DIM,
        obs_dim=OBS_DIM,
        ensemble_size=ENSEMBLE_SIZE
    )
    
    # 生成测试数据
    test_data = data_gen.generate_batch(2)
    
    # 测试前向传播
    with torch.no_grad():
        test_output = model(
            test_data['ensemble_perturbations'],
            obs_operator=test_data['obs_operator'],
            obs_error=test_data['obs_error'],
            obs_mask=test_data['obs_mask']
        )
        print(f"输入形状: {test_data['ensemble_perturbations'].shape}")
        print(f"输出K矩阵形状: {test_output.shape}")
        
        # 测试分析场计算
        analysis = model.compute_kalman_update(
            test_data['background'],
            test_data['observations'],
            test_data['ensemble_perturbations'],
            obs_operator=test_data['obs_operator'],
            obs_error=test_data['obs_error'],
            obs_mask=test_data['obs_mask']
        )
        print(f"分析场形状: {analysis.shape}")
    
    # 开始训练（注释掉以避免长时间运行）
    # trained_model = train_kalman_model(
    #     model, data_gen, num_epochs=50, batch_size=BATCH_SIZE
    # )
