"""JEPA Implementation for LeWorldModel (LeWM).

论文对应（见 paper/sections/3-method.tex）：
- Sec.3「Model Architecture」：编码器 E 将单帧观测 o_t 映射为 z_t；预测器在潜空间建模动力学，
  hat{z}_{t+1} = pred(z_t, a_t)。实现上观测嵌入经 projector，动作经 action_encoder 后以 AdaLN 注入预测器
  （module.py 中 ConditionalBlock）。
- 训练目标 L_LeWM = L_pred + λ·SIGReg(Z) 在 train.py 的 lejepa_forward 中计算；本文件实现前向结构与
  推理时的潜空间展开（Sec.3「Latent Planning」）。
"""

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


def detach_clone(v):
    return v.detach().clone() if torch.is_tensor(v) else v


class JEPA(nn.Module):
    """联合嵌入预测架构（JEPA）在 LeWM 中的实现：像素观测 → 潜表示，动作条件下预测下一时刻嵌入。"""

    def __init__(
        self,
        encoder,
        predictor,
        action_encoder,
        projector=None,
        pred_proj=None,
    ):
        super().__init__()

        # 论文：Encoder 为 ViT；Predictor 为 Transformer + AdaLN 条件于动作
        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        # 论文：将 [CLS] 经 Layer Norm 后需再经带 BN 的 MLP 投影到嵌入空间，便于 SIGReg 反坍塌优化
        self.projector = projector or nn.Identity()
        # 论文：预测器末端同样接 projector（pred_proj），与编码器侧 projector 结构一致
        self.pred_proj = pred_proj or nn.Identity()

    def encode(self, info):
        """Encode observations and actions into embeddings.
        info: dict with pixels and action keys
        """

        pixels = info["pixels"].float()
        b = pixels.size(0)
        # 论文 z_t = E(o_t)：按时间展平后批量过 ViT，等价于对序列中每一帧独立编码
        pixels = rearrange(pixels, "b t ... -> (b t) ...")  # flatten for encoding
        output = self.encoder(pixels, interpolate_pos_encoding=True)
        # 论文：观测嵌入取自最后一层 [CLS] token，再经 projector 得到 z_t
        pixels_emb = output.last_hidden_state[:, 0]  # cls token
        emb = self.projector(pixels_emb)
        info["emb"] = rearrange(emb, "(b t) d -> b t d", b=b)

        if "action" in info:
            # 论文：动作不直接进 ViT，而是编码为条件向量，供预测器内 AdaLN 使用（见 module.Embedder）
            info["act_emb"] = self.action_encoder(info["action"])

        return info

    def predict(self, emb, act_emb):
        """Predict next state embedding
        emb: (B, T, D)
        act_emb: (B, T, A_emb)
        """
        # 论文 hat{z}_{t+1} = pred(...)：自回归预测器在因果掩码下输出下一步潜变量（ARPredictor + pred_proj）
        preds = self.predictor(emb, act_emb)
        preds = self.pred_proj(rearrange(preds, "b t d -> (b t) d"))
        preds = rearrange(preds, "(b t) d -> b t d", b=emb.size(0))
        return preds

    ####################
    ## Inference only ##
    ####################

    def rollout(self, info, action_sequence, history_size: int = 3):
        """Rollout the model given an initial info dict and action sequence.
        pixels: (B, S, T, C, H, W)
        action_sequence: (B, S, T, action_dim)
         - S is the number of action plan samples
         - T is the time horizon
        """
        # 论文 Sec.3「Latent Planning」：在潜空间沿动作序列自回归展开，供 MPC/CEM 优化动作序列使用

        assert "pixels" in info, "pixels not in info_dict"
        H = info["pixels"].size(2)
        B, S, T = action_sequence.shape[:3]
        act_0, act_future = torch.split(action_sequence, [H, T - H], dim=2)
        info["action"] = act_0
        n_steps = T - H

        # hat{z}_1 = E(o_1)：仅用初始观测得到起始潜变量，再沿候选动作序列向前展开
        _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
        _init = self.encode(_init)
        emb = info["emb"] = _init["emb"].unsqueeze(1).expand(B, S, -1, -1)
        _init = {k: detach_clone(v) for k, v in _init.items()}

        # 论文：对多条采样轨迹 (S) 并行展开，便于 CEM 等算法评估候选动作序列
        emb = rearrange(emb, "b s ... -> (b s) ...").clone()
        act = rearrange(act_0, "b s ... -> (b s) ...")
        act_future = rearrange(act_future, "b s ... -> (b s) ...")

        # 论文：有限视界 H 下反复应用 hat{z}_{t+1}=pred(历史潜变量, 历史动作编码)；此处 HS 为历史长度 N
        HS = history_size
        for t in range(n_steps):
            act_emb = self.action_encoder(act)
            emb_trunc = emb[:, -HS:]  # (BS, HS, D)
            act_trunc = act_emb[:, -HS:]  # (BS, HS, A_emb)
            pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]  # (BS, 1, D)
            emb = torch.cat([emb, pred_emb], dim=1)  # (BS, T+1, D)

            next_act = act_future[:, t : t + 1, :]  # (BS, 1, action_dim)
            act = torch.cat([act, next_act], dim=1)  # (BS, T+1, action_dim)

        # 终端步再预测一次，对齐规划目标时刻的潜变量
        act_emb = self.action_encoder(act)  # (BS, T, A_emb)
        emb_trunc = emb[:, -HS:]  # (BS, HS, D)
        act_trunc = act_emb[:, -HS:]  # (BS, HS, A_emb)
        pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]  # (BS, 1, D)
        emb = torch.cat([emb, pred_emb], dim=1)

        pred_rollout = rearrange(emb, "(b s) ... -> b s ...", b=B, s=S)
        info["predicted_emb"] = pred_rollout

        return info

    def criterion(self, info_dict: dict):
        """Compute the cost between predicted embeddings and goal embeddings."""
        # 论文 Sec.3 Latent Planning：C(hat{z}_H)=||hat{z}_H - z_g||^2，z_g=E(o_g)；此处对候选轨迹终端嵌入与目标嵌入求 MSE
        pred_emb = info_dict["predicted_emb"]  # (B,S, T-1, dim)
        goal_emb = info_dict["goal_emb"]  # (B, S, T, dim)

        goal_emb = goal_emb[..., -1:, :].expand_as(pred_emb)

        cost = F.mse_loss(
            pred_emb[..., -1:, :],
            goal_emb[..., -1:, :].detach(),
            reduction="none",
        ).sum(dim=tuple(range(2, pred_emb.ndim)))  # (B, S)

        return cost

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor):
        """ Compute the cost of action candidates given an info dict with goal and initial state."""

        assert "goal" in info_dict, "goal not in info_dict"

        device = next(self.parameters()).device
        for k in list(info_dict.keys()):
            if torch.is_tensor(info_dict[k]):
                info_dict[k] = info_dict[k].to(device)

        goal = {k: v[:, 0] for k, v in info_dict.items() if torch.is_tensor(v)}
        # 目标观测 o_g 经编码器得到 z_g = E(o_g)（论文 Latent Planning 段）
        goal["pixels"] = goal["goal"]

        for k in info_dict:
            if k.startswith("goal_"):
                goal[k[len("goal_") :]] = goal.pop(k)

        goal.pop("action")
        goal = self.encode(goal)

        info_dict["goal_emb"] = goal["emb"]
        info_dict = self.rollout(info_dict, action_candidates)

        cost = self.criterion(info_dict)

        return cost
