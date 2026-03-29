"""
Prototype Refinement via Gradient Attribution Maps (原型加权)
============================================================
基于 Grad-CAM 风格的梯度归因图对 support 图像中目标框内的特征加权，
生成更具判别性的类别原型 P_c^*。

使用方式:
    from lib.proto_refine import PrototypeRefiner

消融开关 (cfg.DE.USE_PROTO_REFINEMENT):
    True  → 使用精化原型
    False → 原始均值原型（baseline）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeRefiner(nn.Module):
    """
    梯度归因图原型精化模块。

    支持两种模式：
    - gradient_mode=True : 使用余弦相似度分数对 ROI 池化特征做真实 Grad-CAM 求导
                           (适用于 use_one_shot 推理，support 特征可直接获取)
    - gradient_mode=False: 使用可学习的通道+空间注意力近似梯度加权
                           (适用于训练阶段的端到端优化)

    数学对应 Method.md:
        α_k = (1/wp*hp) Σ ∂y_c/∂F_pool_uv^k
        A_c  = ReLU(Σ_k α_k · F_pool^k)
        w_{i,mn} = A_c(m,n) / Σ A_c(m',n')
        P_c^* = (1/K) Σ_i Σ_{m,n} w_{i,mn} · F_{i,mn}
    """

    def __init__(
        self,
        feat_dim: int,
        pool_size: int = 7,
        gradient_mode: bool = False,
    ):
        """
        Args:
            feat_dim      : 特征通道维度 C（与 backbone 输出对齐）
            pool_size     : ROI Align 后的空间尺寸（wp = hp = pool_size）
            gradient_mode : True → 真实 Grad-CAM；False → 可学习近似
        """
        super().__init__()
        self.feat_dim = feat_dim
        self.pool_size = pool_size
        self.gradient_mode = gradient_mode

        # ── 可学习近似分支（gradient_mode=False 时使用）──────────────
        # 通道注意力：由原型向量引导，预测每个通道的重要性权重
        self.channel_attn = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 4),
            nn.ReLU(),
            nn.Linear(feat_dim // 4, feat_dim),
            nn.Sigmoid(),
        )
        # 空间注意力：预测 ROI 内各位置对分类的贡献强度
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(feat_dim // 4, 1, kernel_size=1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")

    # ──────────────────────────────────────────────────────────────────
    # Mode A: 真实 Grad-CAM（use_gradient=True）
    # ──────────────────────────────────────────────────────────────────
    def _gradcam_weights(
        self,
        f_pool: torch.Tensor,          # [N, C, H, W]  ROI 池化特征
        initial_proto: torch.Tensor,   # [C]           初始原型 P_c^(0)
    ) -> torch.Tensor:
        """
        对一个类别的 K 张 support 图像计算 Grad-CAM 空间权重。

        Returns:
            weights: [N, H*W]  归一化空间权重 w_{i,mn}
        """
        # 需要对 f_pool 求梯度
        f = f_pool.detach().requires_grad_(True)

        # 计算类别得分 y_c —— 用余弦相似度作为代理
        # 等价于检测头用原型做分类时的核心操作
        f_avg = f.mean(dim=[2, 3])                                   # [N, C]
        proto_n = F.normalize(initial_proto.unsqueeze(0), dim=-1)    # [1, C]
        f_n = F.normalize(f_avg, dim=-1)                             # [N, C]
        y_c = (f_n * proto_n).sum()                                  # scalar

        # ∂y_c / ∂F_pool
        grad = torch.autograd.grad(y_c, f, create_graph=False)[0]   # [N, C, H, W]

        # α_k = (1/wp*hp) * Σ_{u,v} ∂y_c/∂F_pool_uv^k
        alpha_k = grad.mean(dim=[2, 3])                              # [N, C]

        # A_c^(i) = ReLU(Σ_k α_k · F_pool^k)
        A = F.relu(
            (alpha_k[:, :, None, None] * f_pool.detach()).sum(dim=1)
        )                                                             # [N, H, W]

        # 归一化 → 空间权重 w_{i,mn}
        N, H, W = A.shape
        A_flat = A.view(N, -1)                                        # [N, H*W]
        A_sum = A_flat.sum(dim=1, keepdim=True).clamp(min=1e-8)
        weights = A_flat / A_sum                                      # [N, H*W]
        return weights

    def _refine_one_class_gradient(
        self,
        f_pool: torch.Tensor,         # [K, C, H, W]
        initial_proto: torch.Tensor,  # [C]
    ) -> torch.Tensor:
        """
        P_c^* = (1/K) Σ_i Σ_{m,n} w_{i,mn} · F_{i,mn}
        """
        weights = self._gradcam_weights(f_pool, initial_proto)  # [K, H*W]
        K, C, H, W = f_pool.shape
        f_flat = f_pool.detach().view(K, C, H * W)               # [K, C, H*W]
        refined = (f_flat * weights[:, None, :]).sum(dim=2)       # [K, C]
        return F.normalize(refined.mean(0), dim=-1)               # [C]

    # ──────────────────────────────────────────────────────────────────
    # Mode B: 可学习近似（gradient_mode=False，端到端训练）
    # ──────────────────────────────────────────────────────────────────
    def _refine_one_class_learned(
        self,
        f_pool: torch.Tensor,         # [K, C, H, W]
        initial_proto: torch.Tensor,  # [C]
    ) -> torch.Tensor:
        """
        使用可学习通道+空间注意力精化原型（近似 Grad-CAM）。
        """
        # 通道注意力（由原型向量引导）
        chan_w = self.channel_attn(initial_proto)                  # [C]

        # 空间注意力（由 ROI 特征自身决定）
        spat_logit = self.spatial_attn(f_pool)                     # [K, 1, H, W]
        spat_w = F.softmax(spat_logit.view(f_pool.shape[0], 1, -1), dim=-1)  # [K,1,H*W]

        K, C, H, W = f_pool.shape
        f_flat = f_pool.view(K, C, H * W)                         # [K, C, H*W]

        # 加权求和
        weighted = (f_flat * spat_w).sum(dim=2)                    # [K, C]
        weighted = weighted * chan_w[None, :]                       # 通道调制

        return F.normalize(weighted.mean(0), dim=-1)               # [C]

    # ──────────────────────────────────────────────────────────────────
    # 对外接口
    # ──────────────────────────────────────────────────────────────────
    def refine_prototypes(
        self,
        roi_features: torch.Tensor,        # [N, C, H, W]  ROI Align 输出（未 flatten）
        initial_prototypes: torch.Tensor,  # [Nc, C]        初始均值原型
        labels: torch.Tensor,              # [N]            每个 ROI 的类别标签
        num_classes: int,
    ) -> torch.Tensor:
        """
        批量精化所有类别的原型。

        Args:
            roi_features       : ROI Align 之后、flatten 之前的特征 [N, C, H, W]
            initial_prototypes : 初始均值原型矩阵 [Nc, C]
            labels             : 每个 ROI 对应的类别索引 [N]（背景不参与）
            num_classes        : 类别总数（不含背景）

        Returns:
            refined: [Nc, C]  精化后的类别原型（L2 normalized）
        """
        refined = initial_prototypes.clone()

        for c in range(num_classes):
            mask = (labels == c)
            if mask.sum() == 0:
                continue

            f_c = roi_features[mask]                    # [K, C, H, W]
            proto_c = initial_prototypes[c].detach()    # [C]

            if self.gradient_mode:
                refined[c] = self._refine_one_class_gradient(f_c, proto_c)
            else:
                refined[c] = self._refine_one_class_learned(f_c, proto_c)

        return refined

    def forward(
        self,
        roi_features: torch.Tensor,        # [N, C, H, W]
        initial_prototypes: torch.Tensor,  # [Nc, C]
        labels: torch.Tensor,              # [N]
        num_classes: int,
    ) -> torch.Tensor:
        """等同于 refine_prototypes，便于 nn.Module 调用。"""
        return self.refine_prototypes(roi_features, initial_prototypes, labels, num_classes)
