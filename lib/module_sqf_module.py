"""
Support-Query Fusion Module (支持查询融合模块)
==============================================
通过查询图像级描述子对类别原型进行门控筛选，
再以位置自适应方式将原型信息融合进查询特征图。

使用方式:
    from lib.sqf_module import SupportQueryFusionModule

消融开关 (cfg.DE.USE_SQF):
    True  → 使用 SQF 增强查询特征图
    False → 原始查询特征图（baseline）
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SupportQueryFusionModule(nn.Module):
    """
    查询引导的原型聚合模块（Support-Query Fusion, SQF）。

    完整数学流程（对应 Method.md 公式编号）：

    (1) S_{jhw}  = CosSim(F_{hw}, P_j)                    余弦相似度矩阵
    (2) g        = GAP(Reshape(T'))                        自注意力全局描述子
    (3) λ_j      = σ(CosSim(g, P_j))                      类别激活门控
    (4) S̃_{jhw} = λ_j · S_{jhw}                           门控调制
    (5) W_{jhw}  = softmax_j(S̃_{jhw})                     位置级归一化权重
    (6) p̂_{jhw} = W_{jhw} · P_j                           加权原型向量
    (7) p̂_{hw}  = Σ_j p̂_{jhw}                            原型特征图
    (8) Q̂       = F + α · P̂                               残差融合

    Args:
        feat_dim  : 特征通道维度 C（与 backbone 输出对齐）
        s         : 自注意力空间下采样尺寸（feature map → s×s tokens）
        num_heads : 多头自注意力头数
    """

    def __init__(
        self,
        feat_dim: int,
        s: int = 4,
        num_heads: int = 8,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.s = s
        self.num_heads = num_heads

        assert feat_dim % num_heads == 0, \
            f"feat_dim={feat_dim} 必须能被 num_heads={num_heads} 整除"

        head_dim = feat_dim // num_heads
        self.scale = head_dim ** -0.5

        # ── 自注意力投影矩阵 ─────────────────────────────────────────
        self.q_proj = nn.Linear(feat_dim, feat_dim, bias=False)
        self.k_proj = nn.Linear(feat_dim, feat_dim, bias=False)
        self.v_proj = nn.Linear(feat_dim, feat_dim, bias=False)
        self.out_proj = nn.Linear(feat_dim, feat_dim, bias=False)

        # ── 残差融合可学习权重 α（公式 8）────────────────────────────
        self.alpha = nn.Parameter(torch.zeros(1))

        self._init_weights()

    def _init_weights(self):
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(proj.weight)

    # ──────────────────────────────────────────────────────────────────
    # 内部：多头自注意力
    # ──────────────────────────────────────────────────────────────────
    def _multihead_self_attention(
        self,
        tokens: torch.Tensor,  # [B, L, C]
    ) -> torch.Tensor:
        """
        对 token 序列做 MHSA，返回残差叠加后的 T' [B, L, C]。
        """
        B, L, C = tokens.shape
        H = self.num_heads
        Hd = C // H

        def reshape_heads(x):
            return x.reshape(B, L, H, Hd).transpose(1, 2)   # [B, H, L, Hd]

        Q = reshape_heads(self.q_proj(tokens))
        K = reshape_heads(self.k_proj(tokens))
        V = reshape_heads(self.v_proj(tokens))

        # 注意力图 A ∈ R^{B × H × L × L}
        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, V)                           # [B, H, L, Hd]
        out = out.transpose(1, 2).reshape(B, L, C)            # [B, L, C]
        out = self.out_proj(out)

        return tokens + out                                    # T' with residual

    # ──────────────────────────────────────────────────────────────────
    # 主前向
    # ──────────────────────────────────────────────────────────────────
    def forward(
        self,
        F_query: torch.Tensor,  # [B, C, H, W]  查询特征图（backbone 输出）
        P: torch.Tensor,        # [Nc, C]        类别原型矩阵（L2 normalized）
    ) -> torch.Tensor:
        """
        Args:
            F_query : 查询图像特征图  [B, C, H, W]
            P       : 类别原型矩阵   [Nc, C]  （应已 L2-normalize）

        Returns:
            Q_hat   : 原型增强的查询特征图  [B, C, H, W]
        """
        B, C, H, W = F_query.shape
        Nc = P.shape[0]

        # ── Step 1: 余弦相似度矩阵 S（公式 1）─────────────────────────
        # F_n: [B, C, H, W]
        F_n = F_query / F_query.norm(dim=1, keepdim=True).clamp(min=1e-8)
        P_n = F.normalize(P, dim=-1)                           # [Nc, C]

        # F_flat: [B, H*W, C]
        F_flat = F_n.reshape(B, C, H * W).permute(0, 2, 1)

        # S: [B, H*W, Nc] → permute → [B, Nc, H, W]
        S = torch.einsum("bpc,nc->bpn", F_flat, P_n)          # [B, H*W, Nc]
        S = S.permute(0, 2, 1).reshape(B, Nc, H, W)           # [B, Nc, H, W]

        # ── Step 2: 自注意力全局描述子 g（公式 2）──────────────────────
        # 自适应平均池化压缩空间分辨率，避免大图上注意力开销过高
        F_small = F.adaptive_avg_pool2d(F_query, (self.s, self.s))  # [B, C, s, s]
        T = F_small.flatten(2).permute(0, 2, 1)                     # [B, s², C]

        T_prime = self._multihead_self_attention(T)                  # [B, s², C]

        # 全局描述子：GAP(Reshape(T'))
        g = T_prime.mean(dim=1)                                      # [B, C]

        # ── Step 3: 类别激活门控 λ（公式 3）────────────────────────────
        g_n = F.normalize(g, dim=-1)                                 # [B, C]
        # λ_j = σ(CosSim(g, P_j))   →  [B, Nc]
        lam = torch.sigmoid(g_n @ P_n.T)                             # [B, Nc]

        # ── Step 4: 门控调制 S̃（公式 4）────────────────────────────────
        # lam: [B, Nc] → [B, Nc, 1, 1]
        S_tilde = S * lam[:, :, None, None]                          # [B, Nc, H, W]

        # ── Step 5: 位置级 Softmax 归一化（公式 5）─────────────────────
        # 在类别维度（dim=1）做 Softmax，使每个空间位置的权重归一化
        W = F.softmax(S_tilde, dim=1)                                # [B, Nc, H, W]

        # ── Step 6-7: 加权原型聚合 → 原型特征图 P̂（公式 6-7）──────────
        # p̂_{hw} = Σ_j W_{jhw} · P_j
        # P_n: [Nc, C]
        # W  : [B, Nc, H*W]
        W_flat = W.reshape(B, Nc, H * W)                             # [B, Nc, H*W]
        # P_hat: [B, C, H*W] → [B, C, H, W]
        P_hat = torch.einsum("bnp,nc->bcp", W_flat, P_n)             # [B, C, H*W]
        P_hat = P_hat.reshape(B, C, H, W)                            # [B, C, H, W]

        # ── Step 8: 残差融合（公式 8）───────────────────────────────────
        # Q̂ = F + α · P̂
        Q_hat = F_query + self.alpha * P_hat

        return Q_hat
