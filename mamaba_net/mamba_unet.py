import time
import math
from functools import partial
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except:
    pass

# an alternative for mamba_ssm (in which causal_conv1d is needed)
try:
    from selective_scan import selective_scan_fn as selective_scan_fn_v1
except:
    pass



class PatchExpand2D(nn.Module):  # 高度和宽度会扩展为两倍，而通道数变为1/2
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim * dim_scale
        self.dim_scale = dim_scale
        self.expand = nn.Linear(self.dim, dim_scale * self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):
        B, H, W, C = x.shape  # 排列顺序中通道数放在最后了
        x = self.expand(x)

        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // self.dim_scale)
        x = self.norm(x)

        return x



class Final_PatchExpand2D(nn.Module):  # 高度和宽度会扩展为四倍，而通道数变为1/4
    def __init__(self, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        # self.dim = dim
        self.dim = dim * 4 #改动代码
        self.dim_scale = dim_scale
        self.expand = nn.Linear(self.dim, dim_scale * self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.expand(x)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // self.dim_scale)
        x = self.norm(x)

        return x


class SS2D(nn.Module):
    def __init__(
            self,
            d_model,  # dims_decoder=[768, 384, 192, 96]并非传入列表list而是一个个值输入先输入768，在输入384，在输入192等。
            d_state=16,
            # d_state="auto", # 20240109
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        # self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_model # 20240109
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)  # [768, 384, 192, 96]每个乘以2，变为[1536,768,384,192]
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank  # 设为self.d_model除以16的结果向上取整
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)  # 将维度变为4倍
        self.conv2d = nn.Conv2d(  # 此卷积不会改变输入图像的任何参数
            in_channels=self.d_inner,  # [1536,768,384,192]
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,  # conv_bias=True使用偏置
            kernel_size=d_conv,  # d_conv=3
            padding=(d_conv - 1) // 2,
            **factory_kwargs,  # factory_kwargs = {"device": device, "dtype": dtype}
        )
        self.act = nn.SiLU()  # 激活函数

        self.x_proj = (  # 当输入768时,d_inner->4536，dt_rank->48,d_state->16
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        # self.x_proj_weight = torch.stack([t.weight for t in self.x_proj], dim=0)#改变代码
        #print(self.x_proj_weight.shape)
        del self.x_proj
        # x_proj_weight.shape: torch.Size([4, 112, 1536])
        self.dt_projs = (
            # 当输入768时,dt_rank=48,d_inner=1536,dt_scale=1.0, dt_init="random",dt_min=0.001,dt_max=0.1, dt_init_floor=1e-4
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        #print(self.dt_projs_weight.shape)
        # self.dt_projs_weight = torch.stack([t.weight for t in self.dt_projs], dim=0)# (K=4, inner, rank)
        # self.dt_projs_bias = torch.stack([t.bias for t in self.dt_projs], dim=0) # (K=4, inner)
        del self.dt_projs
        # 如果输入768dt_projs_weight.shape: torch.Size([4, 1536, 48])
        # 如果输入768dt_projs_bias.shape: torch.Size([4, 1536])
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        # 如果输入768，d_inner=1536输出结果等于A_log shape: torch.Size([49152])
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        # 如果输入768，d_inner=1536，输出结果等于torch.Size([6144])
        self.selective_scan = selective_scan_fn
        self.forward_core = self.forward_corev0

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None
    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        # 当输入768时,dt_rank=48,d_inner=1536
        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale  # 初始化线性层的权重参数
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)  # 使用nn.init.uniform_函数对线性层的权重进行均匀分布随机初始化
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):  # 输入的copies=4，d_state=16，如果输入的768则d_inner为1536
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    # def selective_scan_easyv3(self,us, dts, As, Bs, Cs, Ds, delta_bias=None, delta_softplus=False, return_last_state=False,
    #                           chunksize=64):
    #     if chunksize < 0:
    #         chunksize = 64
    #     chunksize = min(chunksize, Bs.shape[-1])
    #     if len(Bs.shape) == 3:
    #         Bs = Bs.unsqueeze(1)
    #     if len(Cs.shape) == 3:
    #         Cs = Cs.unsqueeze(1)
    #
    #     B, GD, L = us.shape
    #     B, G, N, L = Bs.shape
    #     D = GD // G
    #
    #     # mask triu ==============
    #     _arange = torch.arange(0, chunksize, dtype=torch.int8, device=Bs.device)
    #     _row_arange = _arange[None, :]  # ((0, 1), (0, 1))
    #     _col_arange = _arange[:, None]  # ((0, 0), (1, 1))
    #     # _mask_triu = tl.where(_row_arange >= _col_arange, 1, 0)
    #     # _mask_tril = tl.where(_row_arange <= _col_arange, 1, 0)
    #     _mask_tril = (_col_arange >= _row_arange).float()
    #
    #     def cut_chunk(us, dts, Bs, Cs, chunksize=chunksize):
    #         B, H, L, D = us.shape
    #         B, H, L, N = Bs.shape
    #         NT = math.ceil(L / chunksize)
    #         to_pad = NT * chunksize - L
    #         _pad = lambda x: torch.nn.functional.pad(x.view(B * H, L, -1), (0, 0, 0, to_pad, 0, 0)).view(B * H, NT,
    #                                                                                                      chunksize,
    #                                                                                                      x.shape[-1])
    #         us, dts, Bs, Cs = _pad(us), _pad(dts), _pad(Bs), _pad(Cs)
    #         return us, dts, Bs, Cs
    #
    #     def ss_chunk_h1y1(qs, ks, vs, ws=None, As=None, ts=None, dts=None, mask=_mask_tril, scale=1):
    #         # C = n_chunks, M = B * H, E = B * H * C, T = L / C
    #         # MCTN, MCTN, MCTD; MCTND; HND, MCTD, MCTD;
    #         if ws is None:
    #             if ts is None:
    #                 ts = torch.cumsum(dts, dim=2)
    #             _ts = ts.view(-1, As.shape[0], *ts.shape[1:])[:, :, :, :, None, :]
    #             ws = (As[None, :, None, None, :, :] * _ts).exp().flatten(0, 1)  # MCND
    #         q_mul_w = qs[..., None] * ws * scale
    #         k_div_w = ks[..., None] / ws
    #         qwkw = torch.einsum("mctnd,mcrnd->mctrd", q_mul_w, k_div_w)
    #         qwkw = qwkw * mask[None, None, :, :, None]
    #         y1 = torch.einsum("mctrd,mcrd->mctd", qwkw, vs)
    #         ht1 = ws[:, :, -1, :, :] * (k_div_w * vs[..., None, :]).sum(dim=-3)
    #         cws = ws[:, :, -1, :, :]
    #         return ht1, y1, ws, cws, q_mul_w  # MCND, MCTD, MCTND, MCND, MCTND
    #
    #     def ss_chunk_h(cws, ht1):
    #         device, dtype = ht1.device, ht1.dtype
    #         M, C, N, D = ht1.shape
    #         hts = [torch.zeros((M, N, D), device=device, dtype=dtype)]
    #         inith = hts[0]
    #         for c in range(C):
    #             inith = cws[:, c] * inith + ht1[:, c]
    #             hts.append(inith)
    #         return torch.stack(hts, dim=1)  # M(C+1)ND
    #
    #     def ss_chunk_y(y1, hs, q_mul_w):
    #         iniths = hs[:, :-1, :, :].contiguous()
    #         y0 = torch.einsum("mctnd,mcnd->mctd", q_mul_w, iniths)
    #         y = y0 + y1
    #         return y
    #
    #     def ss_chunk_h1y1_dk1(qs, ks, vs, ws=None, As=None, ts=None, dts=None, mask=_mask_tril, scale=1):
    #         # C = n_chunks, M = B * H, E = B * H * C, T = L / C
    #         # MCTN, MCTN, MCTD; MCTND; HND, MCTD, MCTD;
    #         M, C, T, N = qs.shape
    #         assert N == 1
    #         if ws is None:
    #             if ts is None:
    #                 ts = torch.cumsum(dts, dim=2)
    #             _ts = ts.view(-1, As.shape[0], *ts.shape[1:])[:, :, :, :, None, :]
    #             ws = (As[None, :, None, None, :, :] * _ts).exp().flatten(0, 1)  # MCND
    #         q_mul_w = qs[..., None] * ws * scale
    #         # k_div_w = ks[...,None] / ws
    #         v_div_w = vs / ws[:, :, :, 0, :]  # MCTD
    #
    #         y1 = ws[:, :, :, 0, :] * torch.einsum("mctr,mcrd->mctd",
    #                                               qs[:, :, :, None, 0] * ks[:, :, None, :, 0] * mask[None, None, :, :],
    #                                               v_div_w)
    #         ht1 = (ws[:, :, -1, 0, :] * (ks * v_div_w).sum(dim=-2))[:, :, None, :]
    #         cws = ws[:, :, -1, :, :]
    #         return ht1, y1, ws, cws, q_mul_w  # MCND, MCTD, MCTND, MCND, MCTND
    #
    #     def ss_chunk_y_dk1(y1, hs, q_mul_w):
    #         iniths = hs[:, :-1, :, :].contiguous()
    #         y0 = q_mul_w[:, :, :, 0, :] * iniths
    #         y = y0 + y1
    #         return y
    #
    #     if N == 1:
    #         ss_chunk_h1y1 = ss_chunk_h1y1_dk1
    #         ss_chunk_y = ss_chunk_y_dk1
    #
    #     dtype = torch.float32
    #     # dtype = torch.float16
    #     inp_dtype = us.dtype
    #     has_D = Ds is not None
    #     dts = dts.to(dtype)
    #
    #     if delta_bias is not None:
    #         dts = dts + delta_bias.view(1, -1, 1).to(dtype)
    #     if delta_softplus:
    #         dts = torch.nn.functional.softplus(dts)
    #
    #     us = us.view(B, G, -1, L).permute(0, 1, 3, 2).to(dtype)
    #     dts = dts.view(B, G, -1, L).permute(0, 1, 3, 2).to(dtype)
    #     As = As.view(G, D, N).permute(0, 2, 1).to(dtype)
    #     Bs = Bs.permute(0, 1, 3, 2).to(dtype)
    #     Cs = Cs.permute(0, 1, 3, 2).to(dtype)
    #     Ds = Ds.view(G, -1).to(dtype) if has_D else None
    #
    #     _us, dts, Bs, Cs = cut_chunk(us, dts, Bs, Cs)
    #     ht1, y1, ws, cws, q_mul_w = ss_chunk_h1y1(Cs, Bs, _us * dts, None, As, None, dts)
    #     hts = ss_chunk_h(cws, ht1)  # M(C+1)ND
    #     oys = ss_chunk_y(y1, hts, q_mul_w)  # MCTD
    #     oys = oys.contiguous().view(B, G, -1, D)[:, :, :L, :].contiguous()
    #     hprefix = hts[:, -1, :, :].contiguous()  # MND
    #
    #     if has_D:
    #         oys = oys + Ds.view(1, G, 1, D) * us
    #     oys = oys.permute(0, 1, 3, 2).contiguous().view(B, -1, L)
    #     hprefix = hprefix.permute(0, 2, 1).contiguous().view(B, GD, N).float()
    #
    #     return oys.to(inp_dtype) if not return_last_state else (oys.to(inp_dtype), hprefix)
    #
    #
    # def selective_scan_easy(self, us, dts, As, Bs, Cs, Ds, delta_bias=None, delta_softplus=False,
    #                         return_last_state=False,
    #                         chunksize=64):
    #     """
    #     # B: batch_size, G: groups, D: dim, N: state dim, L: seqlen
    #     us: B, G * D, L
    #     dts: B, G * D, L
    #     As: G * D, N
    #     Bs: B, G, N, L
    #     Cs: B, G, N, L
    #     Ds: G * D
    #     delta_bias: G * D
    #     # chunksize can be any as you like. But as the chunksize raises, hs may get None, as exp(sum(delta) A) is really small
    #     """
    #
    #     def selective_scan_chunk(us, dts, As, Bs, Cs, hprefix):
    #         """
    #         partial(h) / partial(t) = Ah + Bu; y = Ch + Du;
    #         => partial(h*exp(-At)) / partial(t) = Bu*exp(-At);
    #         => h_t = h_0 + sum_{0}_{t}_{Bu*exp(A(t-v)) dv};
    #         => h_b = exp(A(dt_a + ... + dt_{b-1})) * (h_a + sum_{a}_{b-1}_{Bu*exp(-A(dt_a + ... + dt_i)) dt_i});
    #            y_i = C_i*h_i + D*u_i
    #         """
    #         """
    #         us, dts: (L, B, G, D) # L is chunk_size
    #         As: (G, D, N)
    #         Bs, Cs: (L, B, G, N)
    #         Ds: (G, D)
    #         hprefix: (B, G, D, N)
    #         """
    #         ts = dts.cumsum(dim=0)
    #         Ats = torch.einsum("gdn,lbgd->lbgdn", As, ts).exp()
    #         # scale = Ats[-1].detach()
    #         scale = 1
    #         rAts = Ats / scale
    #         duts = dts * us
    #         dtBus = torch.einsum("lbgd,lbgn->lbgdn", duts, Bs)
    #         hs_tmp = rAts * (dtBus / rAts).cumsum(dim=0)
    #         hs = hs_tmp + Ats * hprefix.unsqueeze(0)
    #         ys = torch.einsum("lbgn,lbgdn->lbgd", Cs, hs)
    #         return ys, hs
    #
    #     dtype = torch.float32
    #     # dtype = torch.float16
    #     inp_dtype = us.dtype
    #     has_D = Ds is not None
    #     if chunksize < 1:
    #         chunksize = Bs.shape[-1]
    #
    #     dts = dts.to(dtype)
    #     if delta_bias is not None:
    #         dts = dts + delta_bias.view(1, -1, 1).to(dtype)
    #     if delta_softplus:
    #         dts = torch.nn.functional.softplus(dts)
    #
    #     if len(Bs.shape) == 3:
    #         Bs = Bs.unsqueeze(1)
    #     if len(Cs.shape) == 3:
    #         Cs = Cs.unsqueeze(1)
    #     B, G, N, L = Bs.shape
    #     us = us.view(B, G, -1, L).permute(3, 0, 1, 2).to(dtype)
    #     dts = dts.view(B, G, -1, L).permute(3, 0, 1, 2).to(dtype)
    #     As = As.view(G, -1, N).to(dtype)
    #     Bs = Bs.permute(3, 0, 1, 2).to(dtype)
    #     Cs = Cs.permute(3, 0, 1, 2).to(dtype)
    #     Ds = Ds.view(G, -1).to(dtype) if has_D else None
    #     D = As.shape[1]
    #
    #     oys = []
    #     hprefix = us.new_zeros((B, G, D, N), dtype=dtype)
    #     for i in range(0, L, chunksize):
    #         ys, hs = selective_scan_chunk(
    #             us[i:i + chunksize], dts[i:i + chunksize],
    #             As, Bs[i:i + chunksize], Cs[i:i + chunksize], hprefix,
    #         )
    #         oys.append(ys)
    #         hprefix = hs[-1]
    #
    #     oys = torch.cat(oys, dim=0)
    #     if has_D:
    #         oys = oys + Ds * us
    #     oys = oys.permute(1, 2, 3, 0).view(B, -1, L)
    #
    #     # return oys, hprefix.view(B, G * D, N)
    #     return oys.to(inp_dtype) if not return_last_state else (oys.to(inp_dtype), hprefix.view(B, G * D, N).float())

    def forward_corev0(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn
        B, C, H, W = x.shape
        L = H * W
        K = 4
        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        #
        # if xs.is_cuda:
        #     print("xs 在 GPU 上")
        # else:
        #     print("xs 在 CPU 上")
        # if self.x_proj_weight.is_cuda:
        #     print("x_proj_weight 在 GPU 上")
        # else:
        #     print("x_proj_weight 在 CPU 上")

        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)
        #print(xs.shape)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state) d_state=16
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d) dt_projs_bias跟d_model=768有关
        #print(dt_projs_bias.shape)

        # xs:
        # torch.Size([1, 6144, 196])
        #
        # dts:
        # torch.Size([1, 6144, 196])
        #
        # As:
        # torch.Size([6144, 16])
        #
        # Bs:
        # torch.Size([1, 4, 16, 196])
        #
        # Cs:
        # torch.Size([1, 4, 16, 196])
        #
        # Ds:
        # torch.Size([6144])
        #
        # dt_projs_bias:
        # torch.Size([6144])
        # out_y = self.selective_scan_easyv3(
        #     xs, dts,
        #     As, Bs, Cs, Ds,
        #     # z=None,
        #     delta_bias=dt_projs_bias,
        #     delta_softplus=True,
        #     return_last_state=False,
        # ).view(B, K, -1, L)
        # out_y = self.selective_scan_easy(
        #     xs, dts,
        #     As, Bs, Cs, Ds,
        #     # z=None,
        #     delta_bias=dt_projs_bias,
        #     delta_softplus=True,
        #     return_last_state=False,
        # ).view(B, K, -1, L)
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float
        # 输入进去的数为input([1, 4, 1536, 196])，Output shape: torch.Size([1, 4, 1536, 196])
        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        # inv_y: torch.Size([1, 2, 1536, 196])
        # wh_y: torch.Size([1, 1536, 196])
        # invwh_y: torch.Size([1, 1536, 196])

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y  # 全部的形状都是: torch.Size([1, 1536, 196])

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)  # 将维度变为4倍
        x, z = xz.chunk(2, dim=-1)  # (b, h, w, d)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))  # (b, d, h, w)
        # d_model = 768，torch.randn(2, 32, 32, d_model)，截至到这一行输出结果为torch.Size([2, 1536, 32, 32])
        # d_model = 768，如果输入torch.randn(1, 14, 14, d_model)，Output shape after processing: torch.Size([1, 1536, 14, 14])
        y1, y2, y3, y4 = self.forward_core(x)  # y1，y2，y3，y4， shape: torch.Size([1, 1536, 196])
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        # 输出的shape等于torch.Size([1, 14, 14, 768])
        if self.dropout is not None:
            out = self.dropout(out)
        return out
class VSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)# hidden_dim=d_model
        self.self_attention = SS2D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)#d_model等于输入进去的通道数
        self.drop_path = DropPath(drop_path)

    def forward(self, input: torch.Tensor):
        x = input + self.drop_path(self.self_attention(self.ln_1(input)))
        return x


# class VSSLayer_up(nn.Module):
#     """ A basic Swin Transformer layer for one stage.
#     Args:
#         dim (int): Number of input channels.
#         depth (int): Number of blocks.
#         drop (float, optional): Dropout rate. Default: 0.0
#         attn_drop (float, optional): Attention dropout rate. Default: 0.0
#         drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
#         norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
#         downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
#         use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
#     """
#     def __init__(
#             self,
#             dim,
#             depth,  # depths_decoder=[2, 9, 2, 2]
#             attn_drop=0.,
#             drop_path=0.,  # dpr_decoder[sum(depths_decoder[:i_layer]):sum(depths_decoder[:i_layer + 1])]计算出一个列表
#             norm_layer=nn.LayerNorm,
#             upsample=None,
#             use_checkpoint=False,
#             d_state=16,
#             **kwargs,
#     ):
#         super().__init__()
#         self.dim = dim
#         self.use_checkpoint = use_checkpoint
#         print(drop_path)
#         self.blocks = nn.ModuleList([
#             VSSBlock(
#                 hidden_dim=dim,
#                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
#                 norm_layer=norm_layer,
#                 attn_drop_rate=attn_drop,
#                 d_state=d_state,
#             )
#             for i in range(depth)])
#
#         if True:  # is this really applied? Yes, but been overriden later in VSSM!
#             def _init_weights(module: nn.Module):
#                 for name, p in module.named_parameters():
#                     if name in ["out_proj.weight"]:
#                         p = p.clone().detach_()  # fake init, just to keep the seed ....
#                         nn.init.kaiming_uniform_(p, a=math.sqrt(5))
#
#             self.apply(_init_weights)
#
#         if upsample is not None:
#             self.upsample = upsample(dim=dim, norm_layer=norm_layer)
#         else:
#             self.upsample = None
#
#     def forward(self, x):
#         if self.upsample is not None:
#             x = self.upsample(x)  ##高度和宽度会扩展为两倍，而通道数变为1/2
#         for blk in self.blocks:
#             if self.use_checkpoint:
#                 x = checkpoint.checkpoint(blk, x)
#             else:
#                 x = blk(x)  # 输出经历过depth次调用vssblock后的结果
#         return x

class VSSLayer_up(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        upsample (nn.Module | None, optional): Upsample layer at the beginning of the layer. Default: None
    """
    def __init__(
            self,
            dim,
            depth,
            attn_drop=0.,
            #drop_path=0.,
            upsample=None,
            i_layer=0,#取值为0到3
            **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = False  # Set to False by default

        # Hardcode norm_layer, use_checkpoint, and d_state
        self.norm_layer = nn.LayerNorm
        self.use_checkpoint = False
        self.d_state = 16
        self.i_layer=i_layer
        depths_decoder = [3, 4, 6, 3]
        dpr_decoder = [x.item() for x in torch.linspace(0, 0.1, sum(depths_decoder))][::-1]
        drop_path = dpr_decoder[sum(depths_decoder[:self.i_layer]):sum(depths_decoder[:self.i_layer+ 1])]
        # print(drop_path)
        self.blocks = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=self.norm_layer,  # Use the hardcoded norm_layer
                attn_drop_rate=attn_drop,
                d_state=self.d_state,  # Use the hardcoded d_state
            )
            for i in range(depth)])

        if upsample is not None:
            self.upsample = upsample(dim=dim, norm_layer=self.norm_layer)
        else:
            self.upsample = None

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        """
        out_proj.weight which is previously initilized in VSSBlock, would be cleared in nn.Linear
        no fc.weight found in the any of the model parameters
        no nn.Embedding found in the any of the model parameters
        so the thing is, VSSBlock initialization is useless

        Conv2D is not intialized !!!
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        if self.upsample is not None:
            x = self.upsample(x)
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                # device = torch.device("cuda:0")
                # gpu_memory_allocated = torch.cuda.memory_allocated(device) / (1024 * 1024)
                # gpu_memory_cached = torch.cuda.memory_cached(device) / (1024 * 1024)
                #
                # print(f"已分配 GPU 内存: {gpu_memory_allocated:.2f} MB")
                # print(f"缓存的 GPU 内存: {gpu_memory_cached:.2f} MB")
                x = blk(x)
        return x

#
# layer = VSSLayer_up(
#     dim=512,  # 如果使用”upsample=PatchExpand2D“输入的必须是传入512的1/2（256）才正确，如果不使用则传入512就输入512
#     depth=2,  # depths=[2, 9, 2, 2]
#     # d_state=16,
#     # 如果 d_state 的值为 None，则返回 math.ceil(dims[0] / 6)，即 dims[0] 除以 6 的结果向上取整
#     # 如果 d_state 的值不为 None，则直接返回 d_state 的值 ，这里设置的d_state=16，返回不小于输入参数的最小整数
#     # drop=0.,
#     # attn_drop=0.,
#     #drop_path=dpr_decoder[sum(depths_decoder[:i_layer]):sum(depths_decoder[:i_layer + 1])],  # 设置参数，根据给定的 drop_path_rate 和 depths_decoder
#     #drop_path=dpr_decoder[sum(depths_decoder[:1]):sum(depths_decoder[:1 + 1])],
#     #drop_path=0,
#     i_layer=1,
#     #norm_layer=nn.LayerNorm,
#     #upsample=PatchExpand2D,
#     upsample=None,
#     #use_checkpoint=False
# )

# depths_decoder=[2, 9, 2, 2]
# dpr_decoder = [x.item() for x in torch.linspace(0, 0.1, sum(depths_decoder))][::-1]#设置参数
# layer = VSSLayer_up(
#     dim=512,  # 如果使用”upsample=PatchExpand2D“输入的必须是传入512的1/2（256）才正确，如果不使用则传入512就输入512
#     depth=2,  # depths=[2, 2, 9, 2]
#     d_state=16,
#     # 如果 d_state 的值为 None，则返回 math.ceil(dims[0] / 6)，即 dims[0] 除以 6 的结果向上取整
#     # 如果 d_state 的值不为 None，则直接返回 d_state 的值 ，这里设置的d_state=16，返回不小于输入参数的最小整数
#     drop=0.,
#     attn_drop=0.,
#     #drop_path=dpr_decoder[sum(depths_decoder[:i_layer]):sum(depths_decoder[:i_layer + 1])],  # 设置参数，根据给定的 drop_path_rate 和 depths_decoder
#     drop_path=dpr_decoder[sum(depths_decoder[:1]):sum(depths_decoder[:1 + 1])],
#     #drop_path=0,
#     norm_layer=nn.LayerNorm,
#     #upsample=PatchExpand2D,
#     upsample=None,
#     use_checkpoint=False
# )


def create_vsslayer_up(dim, depth,i_layer,upsample=PatchExpand2D):
    return VSSLayer_up(dim=dim, depth=depth, i_layer=i_layer,upsample=upsample)


#
# # 创建 VSSLayer_up 实例
# layer = create_vsslayer_up(dim=1024, depth=2,i_layer=0,upsample=PatchExpand2D)#输出结果的形状: torch.Size([1, 14, 14, 1024])
# x = torch.randn(1,2048, 7, 7)
# x = x.permute(0, 2, 3, 1)
# output = layer(x)
# print("输出结果的形状:", output.shape)#torch.Size([1, 28, 28, 512]). torch.Size([1, 56, 56, 256])
#
# if torch.cuda.is_available():
#     device = torch.device("cuda:0")  # 选择 GPU 设备
# else:
#     device = torch.device("cpu")  # 使用 CPU
# layer = create_vsslayer_up(dim=1024, depth=2,i_layer=1,upsample=Final_PatchExpand2D).to(device)#输出结果的形状: torch.Size([1, 14, 14, 1024])
# x = torch.randn(1,2048, 7, 7).to(device)
# x = x.permute(0, 2, 3, 1).to(device)




#
#
# import thop
# if __name__ == '__main__':
#     model =   create_vsslayer_up(dim=512, depth=2,i_layer=1,upsample=Final_PatchExpand2D)
#     device = torch.device('cuda')
#     model.to(device)
#     dump_input = torch.ones(1, 7, 7, 2048).to(device)
#     output = model(dump_input)
#     gpu_memory_allocated = torch.cuda.memory_allocated(device) / (1024 * 1024)
#     gpu_memory_cached = torch.cuda.memory_cached(device) / (1024 * 1024)
#
#     print(f"已分配 GPU 内存: {gpu_memory_allocated:.2f} MB")
#     print(f"缓存的 GPU 内存: {gpu_memory_cached:.2f} MB")
#     dump_input = torch.ones(1,7,7,2048).to(device)
#     flops, params = thop.profile(model, inputs=(dump_input,))
#     print(f"FLOPs: {flops / 1e9} G")  # 打印计算量（以十亿次浮点运算为单位）
#     print(f"Params: {params / 1e6} M")  # 打印参数量（以百万为单位）
#     # Warn-up
#     for _ in range(5):
#         start = time.time()
#         outputs = model(dump_input)
#         torch.cuda.synchronize()
#         end = time.time()
#         print('Time:{}ms'.format((end-start)*1000))
#
#     with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False, profile_memory=False) as prof:
#         outputs = model(dump_input)
#     print(prof.table())









# def model_structure(model):
#     blank = ' '
#     print('-' * 90)
#     print('|' + ' ' * 11 + 'weight name' + ' ' * 10 + '|' \
#           + ' ' * 15 + 'weight shape' + ' ' * 15 + '|' \
#           + ' ' * 3 + 'number' + ' ' * 3 + '|')
#     print('-' * 90)
#     num_para = 0
#     type_size = 1  # 如果是浮点数就是4
#
#     for index, (key, w_variable) in enumerate(model.named_parameters()):
#         if len(key) <= 30:
#             key = key + (30 - len(key)) * blank
#         shape = str(w_variable.shape)
#         if len(shape) <= 40:
#             shape = shape + (40 - len(shape)) * blank
#         each_para = 1
#         for k in w_variable.shape:
#             each_para *= k
#         num_para += each_para
#         str_num = str(each_para)
#         if len(str_num) <= 10:
#             str_num = str_num + (10 - len(str_num)) * blank
#
#         print('| {} | {} | {} |'.format(key, shape, str_num))
#     print('-' * 90)
#     print('The total number of parameters: ' + str(num_para))
#     print('The parameters of Model {}: {:4f}M'.format(model._get_name(), num_para * type_size / 1000 / 1000))
#     print('-' * 90)
#
# model_structure(create_vsslayer_up(dim=512, depth=2,i_layer=1,upsample=Final_PatchExpand2D))



#
# import torch
# from ptflops import get_model_complexity_info
# import re
# # device = torch.device('cuda')
# #Model thats already available
# net = create_vsslayer_up(dim=1024, depth=2,i_layer=0,upsample=PatchExpand2D)
# macs, params = get_model_complexity_info(net, (14, 14,2048), as_strings=True,
# print_per_layer_stat=True, verbose=True)
# # Extract the numerical value
# flops = eval(re.findall(r'([\d.]+)', macs)[0])*2
# # Extract the unit
# flops_unit = re.findall(r'([A-Za-z]+)', macs)[0][0]
#
# print('Computational complexity: {:<8}'.format(macs))
# print('Computational complexity: {} {}Flops'.format(flops, flops_unit))
# print('Number of parameters: {:<8}'.format(params))



# import torch
# device = torch.device("cuda:0")  # 选择 GPU 设备
# input_tensor = torch.randn(1, 7, 7, 2048).to(device)
# model = create_vsslayer_up(dim=1024, depth=2,i_layer=0,upsample=PatchExpand2D).to(device)
# output = model(input_tensor)#真正近期取得是（1，1024，14，14）出来的是torch.Size([1, 7, 7, 2048])
# gpu_memory_allocated = torch.cuda.memory_allocated(device) / (1024 * 1024)
# gpu_memory_cached = torch.cuda.memory_cached(device) / (1024 * 1024)
# print(f"已分配 GPU 内存: {gpu_memory_allocated:.2f} MB")
# print(f"缓存的 GPU 内存: {gpu_memory_cached:.2f} MB")
# # print(output.shape)
