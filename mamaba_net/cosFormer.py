import torch
import torch.nn.functional as F
import numpy as np

from torch import Tensor
from typing import Optional
from torch import nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


# def drop_path(x, drop_prob: float = 0., training: bool = False):
#     """
#     Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
#     This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
#     the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
#     See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
#     changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
#     'survival rate' as the argument.
#     """
#     if drop_prob.item() == 0. or not training:
#           return x
#     keep_prob = 1 - drop_prob
#     shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
#     random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
#     random_tensor.floor_()  # binarize
#     output = x.div(keep_prob) * random_tensor
#     return output
#
# class DropPath(nn.Module):
#     """
#     Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
#     """
#     def __init__(self, drop_prob=None):
#         super(DropPath, self).__init__()
#         self.drop_prob = drop_prob
#
#     def forward(self, x):
#         return drop_path(x, self.drop_prob, self.training)
class CosformerAttention(nn.Module):
    """
    cosformer attention in "cosFormer: Rethinking Softmax In Attention"
    https://arxiv.org/abs/2202.08791
    """

    def __init__(
            self,
            embed_dim,
            num_heads,
            kdim=None,
            vdim=None,
            # dropout_rate=0.0,
            causal=False,
            has_outproj=True,
            act_fun="relu",

    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if kdim is not None else embed_dim
        self.num_heads = num_heads
        self.has_outproj = has_outproj
        self.act_fun = self.get_act_fun(act_fun) #传递获取relu激活函数
        # q, k, v projection
        self.k_proj = nn.Linear(self.kdim, embed_dim)
        self.v_proj = nn.Linear(self.vdim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        # outprojection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        # dropout rate
        # self.dropout_rate = dropout_rate
        # causal
        self.causal = causal
        assert (self.embed_dim % self.num_heads == 0), "embed_dim must be divisible by num_heads"
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_index(self, seq_len):
        index = np.pi / 2 * torch.arange(1, seq_len + 1).reshape(1, -1, 1)
        #生成一个形状为 (1, seq_len, 1) 的张量，其中每个元素的值为 $\pi / 2$ 乘以相应的索引值
        return nn.Parameter(index, requires_grad=False)

    def get_act_fun(self, act_fun):
        if act_fun == "relu":
            return F.relu
        elif act_fun == "elu":
            return 1 + F.elu

    def forward(
            self,
            query: Tensor,
            # L 是目标序列的长度，表示序列中的时间步数或者单词数；
            # N 是批量大小，表示一次前向传播中输入的序列个数；
            # E 是嵌入维度，表示每个单词或时间步的特征维度。
            key: Optional[Tensor] = None,
            value: Optional[Tensor] = None,
            attn_mask: Optional[Tensor] = None,
            eps: Optional[float] = 1e-6,
            depths: int = 1,#新增代码用于循环输出
            drop_path_ratio = 0.1
    ):
        """Input shape: Sequence x Batch x Embedding
        Args:
            query (Tensor): `(L, N, E)` where L is the target sequence length, N is the batch size,
            E is the embedding dimension.
            key (Tensor): `(S, N, E)` where S is the source sequence length, N is the batch size,
            E is the embedding dimension.
            value (Tensor): `(S, N, E)` where S is the source sequence length, N is the batch size,
            E is the embedding dimension.
            attn_mask (Optional[Tensor], optional): typically used to implement causal attention,
            where the mask prevents the attention from looking forward in time (default: None).
        """
        num_heads = self.num_heads
        # tgt_len, bsz, embed_dim = query.size()#长度 tgt_len、批量大小 bsz 和嵌入维度 embed_dim
        batch_size, channels, height, width  = query.size()
        query = query.view(batch_size, channels, -1).permute(2, 0, 1)
        # print("query的形状为",query.shape)
        tgt_len, bsz, embed_dim = query.size()


        if key == None:
            key = query
        if value == None:
            value = query

        # src_len = key.size(0)
        head_dim = embed_dim // num_heads
        # m = max(src_len, tgt_len)  # 一样都是tgt_len
        # get index and send to cuda
        weight_index = self.get_index(tgt_len).to(query)
        # (N * h, L, 2 * d)
        q_pos_encodings_sin = torch.sin(weight_index[:, :tgt_len, :] / tgt_len)
        q_pos_encodings_cos = torch.cos(weight_index[:, :tgt_len, :] / tgt_len)
        k_pos_encodings_sin = torch.sin(weight_index[:, :tgt_len, :] / tgt_len)
        k_pos_encodings_cos = torch.cos(weight_index[:, :tgt_len, :] / tgt_len)
        # dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depths)]
        dpr = torch.linspace(0, drop_path_ratio, depths)

        for i in range(depths):
            drop_path = DropPath(dpr[i])
        # get q, k, v
        # (L, N, E)
            q = self.q_proj(query)
        # (S, N, E)
            k = self.k_proj(key)
        # (S, N, E)
            v = self.v_proj(value)

        # activation
            q = self.act_fun(q)
            k = self.act_fun(k)

        # multihead reshape
        # (N * h, L, d)
            q = q.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        # (N * h, S, d)
            k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        # (N * h, S, d)
            v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
            q_ = torch.cat([q * q_pos_encodings_sin, q * q_pos_encodings_cos], dim=-1)
            k_ = torch.cat([k * k_pos_encodings_sin, k * k_pos_encodings_cos], dim=-1)
        # cos transform
        # m = max(src_len, tgt_len)#一样都是tgt_len
        # # get index and send to cuda
        # weight_index = self.get_index(m).to(q)
        # # (N * h, L, 2 * d)
        # q_ = torch.cat(
        #     [q * torch.sin(weight_index[:, :tgt_len, :] / m), q * torch.cos(weight_index[:, :tgt_len, :] / m)], dim=-1)
        # # (N * h, S, 2 * d)
        # k_ = torch.cat(
        #     [k * torch.sin(weight_index[:, :src_len, :] / m), k * torch.cos(weight_index[:, :src_len, :] / m)], dim=-1)
            if self.causal:
            ## Need to improve speed!
            # (N * h, L, 2 * d) (N * h, L, d) -> (N * h, L, h, 2 * d, d)
                kv_ = torch.einsum("nld,nlm->nldm", k_, v)
            # (N * h, L, 2 * d, d) -> (N * h, L, 2 * d, d)
                kv_cum = torch.cumsum(kv_, dim=1)
            # (N * h, L, 2 * d) (N * h, L, 2 * d, d) -> (N * h, L, d)
                qkv = torch.einsum("nld,nldm->nlm", q_, kv_cum)
            # (N * h, L, 2 * d) -> (N * h, L, 2 * d)
                k_cum = torch.cumsum(k_, dim=1)
            # (N * h, L, 2 * d) (N * h, L, 2 * d) -> (N * h, L)
                denom = torch.clamp_min(torch.einsum("nlm,nlm->nl", q_, k_cum), eps)
            # (N * h, L, d) (N * h, L, 1) -> (N * h, L, d)
                attn_output = qkv / denom.unsqueeze(-1)
            # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
                attn_output = query + drop_path(attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1))
            else:
            # (N * h, L, 2 * d) (N * h, L, d) -> (N * h, 2 * d, d)
                kv_ = torch.einsum('nld,nlm->ndm', k_, v)
            # (N * h, L, 2 * d) (N * h, 2 * d) -> (N * h, L)
                z_ = 1 / torch.clamp_min(torch.einsum('nld,nd->nl', q_, torch.sum(k_, axis=1)), eps)
            # (N * h, L, 2 * d) (N * h, d, 2 * d) (N * h, L) -> (N * h, L, d)
                attn_output = torch.einsum('nld,ndm,nl->nlm', q_, kv_, z_)
            # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
                attn_output = query + drop_path(attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1))
        # L, N, E
        if self.has_outproj:
            attn_output = self.out_proj(attn_output)

        return attn_output

    # def left_product(
    #         self,
    #         query: Tensor,
    #         key: Optional[Tensor] = None,
    #         value: Optional[Tensor] = None,
    #         attn_mask: Optional[Tensor] = None,
    #         eps: Optional[float] = 1e-6,
    # ):
    #     """Input shape: Sequence x Batch x Embedding
    #     Args:
    #         query (Tensor): `(L, N, E)` where L is the target sequence length, N is the batch size,
    #         E is the embedding dimension.
    #         key (Tensor): `(S, N, E)` where S is the source sequence length, N is the batch size,
    #         E is the embedding dimension.
    #         value (Tensor): `(S, N, E)` where S is the source sequence length, N is the batch size,
    #         E is the embedding dimension.
    #         attn_mask (Optional[Tensor], optional): typically used to implement causal attention,
    #         where the mask prevents the attention from looking forward in time (default: None).
    #     """
    #     # test for the correctness of the program
    #     if key == None:
    #         key = query
    #     if value == None:
    #         value = query
    #
    #     num_heads = self.num_heads
    #     tgt_len, bsz, embed_dim = query.size()
    #     src_len = key.size(0)
    #     head_dim = embed_dim // num_heads
    #
    #     # get q, k, v
    #     # (L, N, E)
    #     q = self.q_proj(query)
    #     # (S, N, E)
    #     k = self.k_proj(key)
    #     # (S, N, E)
    #     v = self.v_proj(value)
    #
    #     # activation
    #     q = self.act_fun(q)
    #     k = self.act_fun(k)
    #
    #     # multihead reshape
    #     # (N * h, L, d)
    #     q = q.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    #     # (N * h, S, d)
    #     k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    #     # (N * h, S, d)
    #     v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    #
    #     # cos transform
    #     m = max(src_len, tgt_len)
    #     # get index and send to cuda
    #     weight_index = self.get_index(m).to(q)
    #     # (N * h, L, 2 * d)
    #     q_ = torch.cat(
    #         [q * torch.sin(weight_index[:, :tgt_len, :] / m), q * torch.cos(weight_index[:, :tgt_len, :] / m)], dim=-1)
    #     # (N * h, S, 2 * d)
    #     k_ = torch.cat(
    #         [k * torch.sin(weight_index[:, :src_len, :] / m), k * torch.cos(weight_index[:, :src_len, :] / m)], dim=-1)
    #
    #     # (N * h, L, d) (N * h, d, S) -> (N * h, L, S)
    #     weights = torch.bmm(q_, k_.transpose(1, 2))
    #     # mask
    #     if self.causal:
    #         weights = weights.masked_fill(attn_mask == float("-inf"), 0)
    #     # (N * h, L, S) -> (N * h, L, S)
    #     denom = torch.clamp_min(weights.sum(dim=-1, keepdim=True), eps)
    #     # (N * h, L, S) (N * h, L, S) -> (N * h, L, S)
    #     attn_weights = weights / denom
    #     # (N * h, L, S) (N * h, S, d) -> (N * h, L, d)
    #     attn_output = torch.bmm(attn_weights, v)
    #     # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
    #     attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
    #     # L, N, E
    #     if self.has_outproj:
    #         attn_output = self.out_proj(attn_output)
    #
    #     return attn_output

#
# def drop_path(x, drop_prob: float = 0., training: bool = False):
#     """
#     Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
#     This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
#     the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
#     See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
#     changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
#     'survival rate' as the argument.
#     """
#     if drop_prob == 0. or not training:
#         return x
#     keep_prob = 1 - drop_prob
#     shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
#     random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
#     random_tensor.floor_()  # binarize
#     output = x.div(keep_prob) * random_tensor
#     return output
#
# class DropPath(nn.Module):
#     """
#     Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
#     """
#     def __init__(self, drop_prob=None):
#         super(DropPath, self).__init__()
#         self.drop_prob = drop_prob
#
#     def forward(self, x):
#         return drop_path(x, self.drop_prob, self.training)


#图片形状的转换
# def convert_image(image_tensor):
#     batch_size, channels, height, width  = image_tensor.size()
#     reshaped_tensor = image_tensor.view(batch_size, channels, -1).permute(2, 0, 1)
#     return reshaped_tensor

#
# def execute_depths_times(model, query, key=None, value=None, attn_mask=None, eps=1e-6, depths=1,drop_path_ratio=0.1):
#     output = query
#     dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depths)]
#     for i in range(depths):
#         drop_path = DropPath(dpr[i])
#         output = output+ drop_path(model.forward(output, key, value, attn_mask, eps))
#     return output

# # 示例调用
# image = torch.randn(1, 1024, 14, 14)
# output = convert_image(image)
# print("Output shape:", output.shape)
# #Output shape: torch.Size([196, 1, 1024])


# Example usage:
# embed_dim = 1024
# num_heads = 8
# image = torch.randn(1, 1024, 14, 14) # Example query tensor shape
# model = CosformerAttention(num_heads = num_heads, embed_dim = embed_dim )
# query = convert_image(image)
# output = model(query)
# print(output.shape)#torch.Size([196, 1, 1024])
# # print("Output shape of CosformerAttention forward pass:", output_shape)






# def test(batch=2, tgt_len=10, src_len=20, embed_dim=128, num_heads=8, N=100, causal=False):
#     model = CosformerAttention(embed_dim=embed_dim, num_heads=num_heads, causal=causal)
#     diff = 0
#     if causal:
#         mask = (torch.triu(torch.ones(tgt_len, tgt_len)) == 1).transpose(0, 1)
#         mask = mask.float().masked_fill(mask == 0, float('-inf'))
#     else:
#         mask = None
#     for i in range(N):
#         query = torch.rand(tgt_len, batch, embed_dim)
#         key = torch.rand(src_len, batch, embed_dim)
#         value = torch.rand(src_len, batch, embed_dim)
#         left_res = model.left_product(query, key, value, mask)
#         right_res = model(query, key, value)
#         diff += torch.norm(left_res - right_res)
#     diff /= N
#
#     if causal:
#         print("Test result for causal model:")
#     else:
#         print("Test result for bidirectional model:")
#     print(f"The error of left multiplication and right multiplication is {diff}")
#
#
# def main():
#     test(tgt_len=10, src_len=20, causal=False)
#     test(tgt_len=10, src_len=10, causal=True)
#
#
# if __name__ == "__main__":
#    main()
#
# q_pos_encodings = torch.cat([torch.sin(weight_index[:, :tgt_len, :] / m), torch.cos(weight_index[:, :tgt_len, :] / m)], dim=-1)
# k_pos_encodings = torch.cat([torch.sin(weight_index[:, :src_len, :] / m), torch.cos(weight_index[:, :src_len, :] / m)], dim=-1)
# q_ = torch.cat([q * q_pos_encodings, q * q_pos_encodings], dim=-1)
# k_ = torch.cat([k * k_pos_encodings, k * k_pos_encodings], dim=-1)
# q_pos_encodings = torch.cat([torch.sin(weight_index[:, :tgt_len, :] / m), torch.cos(weight_index[:, :tgt_len, :] / m)], dim=-1)
# q_ = q * q_pos_encodings