# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved. 
# Maintainers:
#     Guowei Yang <471184555@qq.com>
#     Dun Liang <randonlang@gmail.com>. 
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import jittor as jt
from jittor import nn
from jittor import init
from jittor.contrib import concat

# 导入 Jittor 的多头注意力模块
from jittor.attention import MultiheadAttention

from PCT.misc.ops import FurthestPointSampler, knn_point, index_points


class TeLU(nn.Module):
    def __init__(self):
        super().__init__()

    def execute(self, x):        
        return x * jt.tanh(jt.exp(x))
class AttentionLayer(nn.Module):
    """
    自注意力层，只负责计算带有位置编码的注意力。
    归一化和残差连接交由外部的Transformer块处理。
    """
    def __init__(self, channels, num_heads=4):
        super().__init__()
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        
        # 定义多头注意力模块
        self.mha = MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=False)
        
        # 位置编码部分
        self.xyz_proj = nn.Conv1d(3, channels, 1, bias=False)
        self.bn_xyz = nn.BatchNorm1d(channels)
        self.act = TeLU()

    def execute(self, x, xyz):
        # x: [B, C, N] (注意：此处的x应为已在外部归一化后的特征)
        # xyz: [B, 3, N]
        
        # 1. 计算并添加位置编码
        xyz_feat = self.act(self.bn_xyz(self.xyz_proj(xyz))) # [B, C, N]
        x_with_pos = x + xyz_feat                           # [B, C, N]
        
        # 2. 准备 MHA 输入并执行
        # MultiheadAttention 期望的输入形状是 [N, B, C]
        q = k = v = x_with_pos.permute(2, 0, 1) # [N, B, C]
        attn_output, _ = self.mha(q, k, v, need_weights=False) # [N, B, C]
        
        # 3. 返回 MHA 的输出，变回原始形状，不进行残差连接或归一化 # <-- 调整/优化
        return attn_output.permute(1, 2, 0) # [B, C, N]


class Point_Transformer_Block_Standard(nn.Module):
    """
    标准的 "Pre-Norm" Transformer块 (Norm -> SA -> Res -> Norm -> FFN -> Res)。
    负责归一化和残差连接
    """
    def __init__(self, in_channels, ffn_dim_scale=2, num_heads=4): # <-- 调整/优化: ffn_dim_scale 通常为2或4
        super().__init__()
        # Self-Attention Layer (使用简化后的版本)
        self.attention = AttentionLayer(in_channels, num_heads)
        
        # Feed-Forward Network (FFN)
        self.ffn = nn.Sequential(
            nn.Conv1d(in_channels, in_channels * ffn_dim_scale, 1, bias=False),
            TeLU(), # <-- 优化: GELU在Transformer中通常表现更好
            nn.Conv1d(in_channels * ffn_dim_scale, in_channels, 1, bias=False)
        )
        self.norm1 = nn.LayerNorm(in_channels)
        self.norm2 = nn.LayerNorm(in_channels)

    def execute(self, x, xyz):
        # x: [B, C, N], xyz: [B, 3, N]
        
        # 1. Self-Attention 部分 (Pre-Norm 结构) 
        identity1 = x
        # LayerNorm 需要在特征维度上操作，所以需要置换维度
        # 输入x -> [B, N, C] -> Norm -> [B, C, N] -> Attention
        x_norm = self.norm1(x.permute(0, 2, 1)).permute(0, 2, 1)
        attn_out = self.attention(x_norm, xyz)
        # 残差连接
        x = identity1 + attn_out
        
        # 2. Feed-Forward Network 部分 (Pre-Norm 结构) 
        identity2 = x
        x_norm = self.norm2(x.permute(0, 2, 1)).permute(0, 2, 1)
        ffn_out = self.ffn(x_norm)
        # 残差连接
        x = identity2 + ffn_out
        
        return x


#====================================================
# 原始的、设计良好的模块 (无需修改)
#====================================================

class PointGrouper(nn.Module):
    """点云分组模块"""
    def __init__(self, nsample):
        super().__init__()
        self.nsample = nsample

    def execute(self, new_xyz, xyz, points):
        idx = knn_point(self.nsample, xyz, new_xyz)
        grouped_xyz = index_points(xyz, idx)
        grouped_xyz_norm = grouped_xyz - new_xyz.unsqueeze(2)
        if points is not None:
            grouped_points = index_points(points, idx)
            # Jittor的concat默认在dim=1上操作，对于点云特征[B, N, K, C]，通常在最后一个维度C上拼接
            new_features = concat([grouped_points, grouped_xyz_norm], dim=-1)
        else:
            new_features = grouped_xyz_norm
        return new_features


class MultiScaleSA(nn.Module):
    """融合PointNet++的多尺度特征提取模块"""
    def __init__(self, npoint, nsamples, in_channels, mlps):
        super().__init__()
        self.npoint = npoint
        self.groupers = nn.ModuleList()
        self.mlp_convs = nn.ModuleList()
        
        for i in range(len(nsamples)):
            # 这里的输入通道是上一层的特征通道+3个坐标通道
            group_in_channels = in_channels + 3 
            self.groupers.append(PointGrouper(nsamples[i]))
            mlp_spec = mlps[i]
            layers = []
            # 在PointNet++中，分组后的特征维度是 [B, N, K, C+3]
            # Conv2d期望输入 [B, C_in, H, W], 所以需要置换
            # 这里的 in_channels 是点特征维度，所以 group_in_channels = C + 3
            current_in = group_in_channels
            for out_channel in mlp_spec:
                layers.append(nn.Conv2d(current_in, out_channel, 1))
                layers.append(nn.BatchNorm2d(out_channel))
                layers.append(TeLU())
                current_in = out_channel
            self.mlp_convs.append(nn.Sequential(*layers))
        
        total_out_channels = sum(m[-1] for m in mlps)
        self.fuse_conv = nn.Sequential(
            nn.Conv1d(total_out_channels, total_out_channels, 1),
            nn.BatchNorm1d(total_out_channels),
            TeLU()
        )
        self.sampler = FurthestPointSampler(npoint)

    def execute(self, x, xyz):
        # x: [B, C, N], xyz: [B, 3, N]
        xyz_t = xyz.transpose(0, 2, 1) # [B, N, 3]
        # Jittor的FPS实现返回元组(distances, indices)
        _, fps_idx = self.sampler(xyz_t) # fps_idx: [B, npoint]
        new_xyz = index_points(xyz_t, fps_idx) # [B, npoint, 3]
        
        feature_list = []
        x_t = x.transpose(0, 2, 1) # [B, N, C]
        for i, grouper in enumerate(self.groupers):
            # grouped_points: [B, npoint, nsample, C+3]
            grouped_points = grouper(new_xyz, xyz_t, x_t)
            # Conv2d 需要 [B, C_in, H, W], 这里对应 [B, C+3, npoint, nsample]
            grouped_points = grouped_points.transpose(0, 3, 1, 2)
            features = self.mlp_convs[i](grouped_points) # [B, C_out, npoint, nsample]
            # Max pooling over neighbors
            features = jt.max(features, dim=3) # [B, C_out, npoint]
            feature_list.append(features)
        
        fused_features = concat(feature_list, dim=1) # [B, total_C_out, npoint]
        fused_features = self.fuse_conv(fused_features)
        new_xyz = new_xyz.transpose(0, 2, 1) # [B, 3, npoint]
        return fused_features, new_xyz

class JointPredictionHead(nn.Module):
    """骨骼点预测头"""
    def __init__(self, in_channels, num_joints=52):
        super().__init__()
        self.num_joints = num_joints
        self.joint_conv = nn.Sequential(
            nn.Conv1d(in_channels, 256, 1), nn.BatchNorm1d(256), TeLU(),
            nn.Conv1d(256, 128, 1), nn.BatchNorm1d(128), TeLU(),
            nn.Conv1d(128, num_joints * 3, 1)
        )
        self.weight_conv = nn.Sequential(
            nn.Conv1d(in_channels, 128, 1), nn.BatchNorm1d(128), TeLU(),
            nn.Conv1d(128, num_joints, 1), nn.Softmax(dim=2)
        )

    def execute(self, features, xyz):
        # features: [B, C, N], xyz: [B, 3, N]
        joint_offsets = self.joint_conv(features) # [B, num_joints*3, N]
        joint_offsets = joint_offsets.reshape(features.shape[0], self.num_joints, 3, -1) # [B, num_joints, 3, N]
        weights = self.weight_conv(features) # [B, num_joints, N]
        
        # 使用加权平均计算最终关节点
        xyz_expanded = xyz.unsqueeze(1) # [B, 1, 3, N]
        joint_positions = xyz_expanded + joint_offsets # [B, num_joints, 3, N]
        
        weights = weights.unsqueeze(2) # [B, num_joints, 1, N]
        # (Pos * Weight) -> sum over N points
        final_joints = jt.sum(joint_positions * weights, dim=3) # [B, num_joints, 3]
        return final_joints

#====================================================
# 最终的主模型 (现在使用重构后的 Transformer 块)
#====================================================


class EnhancedSkeletonModel(nn.Module):
    def __init__(self, output_channels=156, num_points=1024, num_heads=8):
        super().__init__()
        self.output_channels = output_channels
        self.num_points = num_points
        num_joints = output_channels // 3
        
        # 初始特征提取层
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        
        # 多尺度特征提取 (融合PointNet++思想)
        self.sa1 = MultiScaleSA(npoint=512, nsamples=[16, 32], in_channels=128, mlps=[[128, 128], [128, 256]]) # out: 384
        self.sa2 = MultiScaleSA(npoint=256, nsamples=[16, 32], in_channels=384, mlps=[[256, 256], [256, 512]]) # out: 768
        
        # Transformer特征增强 
        self.transformer1 = Point_Transformer_Block_Standard(in_channels=768, num_heads=num_heads)
        self.transformer2 = Point_Transformer_Block_Standard(in_channels=768, num_heads=num_heads)
        self.transformer3 = Point_Transformer_Block_Standard(in_channels=768, num_heads=num_heads)
        self.transformer4 = Point_Transformer_Block_Standard(in_channels=768, num_heads=num_heads)
        
        # 骨骼点预测头
        self.joint_head = JointPredictionHead(in_channels=768, num_joints=num_joints)
        
        self.relu = TeLU()

    def execute(self, vertices):
        # vertices: [B, 3, N]
        B, _, N = vertices.shape
        
        # 保存原始坐标用于后续处理
        xyz = vertices
        
        # 初始特征提取
        x = self.relu(self.bn1(self.conv1(vertices)))  # [B, 64, N]
        x = self.relu(self.bn2(self.conv2(x)))         # [B, 128, N]
        
        # 多尺度特征提取
        x, new_xyz1 = self.sa1(x, xyz)                 # [B, 384, 512]
        x, new_xyz2 = self.sa2(x, new_xyz1)            # [B, 768, 256]
        
        # Transformer特征增强
        x = self.transformer1(x, new_xyz2)             # [B, 768, 256]
        x = self.transformer2(x, new_xyz2)             # [B, 768, 256]
        x = self.transformer3(x, new_xyz2)             # [B, 768, 256]
        x = self.transformer4(x, new_xyz2)             # [B, 768, 256]
        
        # 骨骼点预测
        joints = self.joint_head(x, new_xyz2)          # [B, num_joints, 3]
        
        return joints.reshape(B, -1)                   # [B, num_joints * 3]


# 工厂函数保持不变
def create_model(model_name='pct', output_channels=156, **kwargs):
    if model_name == "pct":
        return EnhancedSkeletonModel(output_channels=output_channels, **kwargs)
    raise NotImplementedError()