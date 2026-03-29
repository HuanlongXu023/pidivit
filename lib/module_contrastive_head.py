"""
IoU-Weighted Contrastive Learning Head (对比学习分支)
=====================================================
通过显式建模实例级类内相似性与类间差异性，
学习更具判别力的候选框特征表示。

使用方式:
    from lib.contrastive_head import ContrastiveHead, iou_weighted_contrastive_loss

消融开关 (cfg.DE.USE_CONTRASTIVE):
    True  → 在训练时附加对比损失 L_C
    False → 不使用对比损失（baseline）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveHead(nn.Module):
    """
    单隐层 MLP 投影头。

    将候选框 ROI 特征编码至对比嵌入空间（维度 proj_dim）。
    仅在训练时使用；推理时不产生额外开销。

    Args:
        in_dim   : 输入特征维度（ROI 特征 flatten 后的维度）
        proj_dim : 对比嵌入维度 D（默认 128）
    """

    def __init__(self, in_dim: int, proj_dim: int = 128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.ReLU(),
            nn.Linear(in_dim // 2, proj_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [N, in_dim]  ROI 池化特征（flatten + GAP）
        Returns:
            c: [N, proj_dim]  对比嵌入（未归一化）
        """
        return self.proj(x)


def iou_f(u: torch.Tensor) -> torch.Tensor:
    """
    IoU 加权函数 f(u_i)。

    IoU 越高（候选框与 GT 越吻合），该样本在对比学习中贡献的权重越大，
    确保高质量 proposal 对损失的贡献更大，低质量样本影响被压制。

    当前实现：f(u) = u（线性加权，简单且有效）
    可替换为 f(u) = u^2、1/(1-u) 等变体做消融。

    Args:
        u: [N]  每个 proposal 与对应 GT box 的 IoU
    Returns:
        weight: [N]  对比损失样本权重
    """
    return u.clamp(min=0.0, max=1.0)


def iou_weighted_contrastive_loss(
    features: torch.Tensor,    # [N, in_dim]  ROI 特征（未投影）
    labels: torch.Tensor,      # [N]          类别标签（背景用 num_classes 表示）
    ious: torch.Tensor,        # [N]          与对应 GT box 的 IoU
    contrastive_head: ContrastiveHead,
    num_classes: int,
    tau: float = 0.2,
    bg_label: int = -1,        # 用于标识背景的标签值；传入实际 num_classes
) -> torch.Tensor:
    """
    IoU 加权监督对比损失。

    数学对应 Method.md（公式 1-2）：

        L_C = -(1/M) * (1/(N-1)) * Σ_i f(u_i) · C_i

        C_i = Σ_{j≠i} 1(y_i==y_j, y_i≠bg) · log[
                exp(c̃_i · c̃_j / τ) / Σ_{k≠i} exp(c̃_i · c̃_k / τ)
              ]

    其中 c̃ = c / ‖c‖ 为 L2 归一化后的对比嵌入。

    Args:
        features         : [N, in_dim]  ROI 特征（将通过 contrastive_head 投影）
        labels           : [N]          类别索引（背景 = bg_label）
        ious             : [N]          每个 proposal 对应的 IoU
        contrastive_head : ContrastiveHead 实例
        num_classes      : 总类别数（不含背景）
        tau              : 温度超参数（默认 0.2）
        bg_label         : 背景类标签值

    Returns:
        loss: scalar Tensor
    """
    # ── 过滤背景：仅对前景 proposal 计算对比损失 ───────────────────
    fg_mask = labels != bg_label                               # [N]  bool
    if fg_mask.sum() < 2:
        # 前景样本不足，跳过（返回 0 避免 NaN）
        return features.sum() * 0.0

    fg_feats = features[fg_mask]                               # [M, in_dim]
    fg_labels = labels[fg_mask]                                # [M]
    fg_ious = ious[fg_mask]                                    # [M]
    M = fg_feats.shape[0]

    # ── 投影 + L2 归一化 ────────────────────────────────────────────
    c = contrastive_head(fg_feats)                             # [M, proj_dim]
    c_norm = F.normalize(c, dim=-1)                            # c̃

    # ── 相似度矩阵（余弦相似度 / τ）────────────────────────────────
    sim = c_norm @ c_norm.T / tau                              # [M, M]

    # ── IoU 加权 f(u_i) ─────────────────────────────────────────────
    weights = iou_f(fg_ious)                                   # [M]

    # ── 正样本掩码：同类且非自身 ─────────────────────────────────────
    label_eq = fg_labels.unsqueeze(0) == fg_labels.unsqueeze(1)  # [M, M]
    diag_mask = ~torch.eye(M, dtype=torch.bool, device=features.device)
    pos_mask = label_eq & diag_mask                              # [M, M]

    # ── 计算 C_i（对应 Method.md 公式 2）───────────────────────────
    # 分母：对所有 k≠i 的样本（不区分类别）求 logsumexp
    # sim 对角线置 -inf 排除自身
    sim_no_diag = sim.masked_fill(~diag_mask, float("-inf"))
    log_denom = torch.logsumexp(sim_no_diag, dim=1)            # [M]

    # 分子：log exp(sim_{ij}) = sim_{ij}
    # C_i = Σ_{j: 正样本} (sim_{ij} - log_denom)
    C_i = torch.zeros(M, device=features.device)
    num_pos = pos_mask.sum(dim=1).float().clamp(min=1.0)       # 每个 i 的正样本数

    for i in range(M):
        pos_j = pos_mask[i]
        if pos_j.sum() == 0:
            continue
        C_i[i] = (sim[i][pos_j] - log_denom[i]).sum() / num_pos[i]

    # ── 最终损失（Method.md 公式 1）───────────────────────────────
    loss = -(weights * C_i).mean()

    return loss


# ──────────────────────────────────────────────────────────────────────
# 向量化高效实现（可选，替代上方 for 循环，支持大 batch）
# ──────────────────────────────────────────────────────────────────────
def iou_weighted_contrastive_loss_vectorized(
    features: torch.Tensor,
    labels: torch.Tensor,
    ious: torch.Tensor,
    contrastive_head: ContrastiveHead,
    num_classes: int,
    tau: float = 0.2,
    bg_label: int = -1,
) -> torch.Tensor:
    """
    向量化版本，与上方函数等价但更高效，推荐在 batch 较大时使用。
    """
    fg_mask = labels != bg_label
    if fg_mask.sum() < 2:
        return features.sum() * 0.0

    fg_feats = features[fg_mask]
    fg_labels = labels[fg_mask]
    fg_ious = ious[fg_mask]
    M = fg_feats.shape[0]

    c = contrastive_head(fg_feats)
    c_norm = F.normalize(c, dim=-1)
    weights = iou_f(fg_ious)

    sim = c_norm @ c_norm.T / tau                               # [M, M]

    # 排除自身
    diag_mask = torch.eye(M, dtype=torch.bool, device=features.device)
    sim_no_diag = sim.masked_fill(diag_mask, float("-inf"))
    log_denom = torch.logsumexp(sim_no_diag, dim=1)            # [M]

    # 正样本掩码（同类 & 非自身）
    label_eq = fg_labels.unsqueeze(0) == fg_labels.unsqueeze(1)  # [M, M]
    pos_mask = label_eq & ~diag_mask                              # [M, M]

    # 对每个 i：Σ_{j∈pos} (sim_{ij} - log_denom[i])
    # 利用广播：sim - log_denom[:, None] → per-pair log-prob
    log_prob = sim - log_denom[:, None]                        # [M, M]
    log_prob_pos = log_prob * pos_mask.float()                 # 只保留正样本

    num_pos = pos_mask.sum(dim=1).float().clamp(min=1.0)
    C_i = log_prob_pos.sum(dim=1) / num_pos                   # [M]

    loss = -(weights * C_i).mean()
    return loss
