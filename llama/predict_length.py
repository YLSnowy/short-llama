# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class CNN2D(nn.Module):
#     def __init__(self, **kwargs):
#         super(CNN2D, self).__init__()

#         self.MODEL = kwargs["MODEL"]
#         self.BATCH_SIZE = kwargs["BATCH_SIZE"]
#         self.MAX_SENT_LEN = kwargs["MAX_SENT_LEN"]
#         self.WORD_DIM = kwargs["WORD_DIM"]
#         self.VOCAB_SIZE = kwargs["VOCAB_SIZE"]
#         self.CLASS_SIZE = kwargs["CLASS_SIZE"]
#         self.FILTERS = kwargs["FILTERS"]
#         self.FILTER_NUM = kwargs["FILTER_NUM"]
#         self.DROPOUT_PROB = kwargs["DROPOUT_PROB"]
#         self.IN_CHANNEL = 2

#         assert (len(self.FILTERS) == len(self.FILTER_NUM))

#         self.embedding = nn.Identity()
#         self.embedding2 = nn.Identity()

#         for i in range(len(self.FILTERS)):
#             conv = nn.Conv2d(self.IN_CHANNEL, self.FILTER_NUM[i], (self.FILTERS[i], self.WORD_DIM * self.FILTERS[i]),
#                              stride=(1, self.WORD_DIM))
#             setattr(self, f'conv_{i}', conv)

#         for i in range(len(self.FILTERS)):
#             conv_outdim = 11 - self.FILTERS[i]
#             linear1 = nn.Sequential(
#                 nn.Linear(conv_outdim * conv_outdim, conv_outdim),
#                 nn.LeakyReLU()
#             )
#             linear2 = nn.Sequential(
#                 nn.Linear(conv_outdim, 1),
#                 nn.LeakyReLU()
#             )
#             setattr(self, f'linear1_{i}', linear1)
#             setattr(self, f'linear2_{i}', linear2)

#         for i in range(10):
#             fc = nn.Linear(sum(self.FILTER_NUM), self.CLASS_SIZE)
#             setattr(self, f'fc_{i}', fc)

#     def get_conv(self, i):
#         return getattr(self, f'conv_{i}')

#     def get_linear1(self, i):
#         return getattr(self, f'linear1_{i}')

#     def get_linear2(self, i):
#         return getattr(self, f'linear2_{i}')

#     def get_fc(self, i):
#         return getattr(self, f'fc_{i}')

#     def forward(self, inp):
#         bsz, channel, h, _, _ = inp.shape
#         x = self.embedding(inp).view(bsz, channel, h, -1)
#         x2 = self.embedding2(inp).view(bsz, channel, h, -1)
#         x = torch.cat((x, x2), 1)

#         conv_results = []
#         for i in range(len(self.FILTERS)):
#             conv_output = self.get_conv(i)(x)
#             conv_output = F.relu(conv_output)
#             conv_output = conv_output.view(bsz, 100, -1)
#             # print(conv_output.shape)

#             conv_output = self.get_linear1(i)(conv_output)
#             conv_output = conv_output.view(bsz, 100, -1)
#             conv_output = self.get_linear2(i)(conv_output)
#             conv_output = conv_output.view(bsz, 100)
#             # print(conv_output.shape)

#             conv_results.append(conv_output)

#         x = torch.cat(conv_results, 1)

#         del conv_results
#         x = F.dropout(x, p=self.DROPOUT_PROB, training=self.training)

#         pre = []
#         for i in range(10):
#             fc = self.get_fc(i)(x)
#             pre.append(fc.unsqueeze(1))
#             # print(fc.shape)

#         x = torch.cat(pre, 1)

#         return x


# def main():
#     params = {
#         "MODEL": "multichannel",
#         "MAX_SENT_LEN": 10,
#         "BATCH_SIZE": 50,
#         "WORD_DIM": 4096,
#         "VOCAB_SIZE": 32000,
#         "CLASS_SIZE": 3,
#         "FILTERS": [3, 4, 5],
#         "FILTER_NUM": [100, 100, 100],
#         "DROPOUT_PROB": 0.5,
#     }

#     # model = CNN2D(**params)



import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm


@dataclass
class ModelArgs:
    dim: int = 256
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    input_length: int = 80
    input_dim: int = 4096


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.

    
        

    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings.
        xk (torch.Tensor): Key tensor to apply rotary embeddings.
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.

        

    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    """Multi-head attention module."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        # model_parallel_size = fs_init.get_model_parallel_world_size()
        model_parallel_size = 1
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

    def forward(
            self,
            x: torch.Tensor,
            mask: Optional[torch.Tensor],
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        # xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        keys = xk
        values = xv

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(keys, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        values = repeat_kv(values, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
            self,
            dim: int,
            hidden_dim: int,
            multiple_of: int,
            ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
            self,
            x: torch.Tensor,
            mask: Optional[torch.Tensor],
    ):
        h = x + self.attention.forward(
            self.attention_norm(x), mask
        )
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.layers = torch.nn.ModuleList()

        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        # self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        # self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

    # @torch.inference_mode()
    def forward(self, tokens: torch.Tensor):
        seqlen = tokens.shape[1]
        h = tokens

        mask = None
        # if seqlen > 1:
        #     mask = torch.full(
        #         (1, 1, seqlen, seqlen), float("-inf"), device=tokens.device
        #     )
        #     mask = torch.triu(mask, diagonal=0 + 1).type_as(h)

        for layer in self.layers:
            h = layer(h, mask)

        return h

        
class BertPooler(nn.Module):
    def __init__(self, param):
        super().__init__()
        self.dense = nn.Linear(param.dim, param.dim)
        self.activation = nn.Tanh()

        self.l = nn.Sequential(
            nn.Conv1d(param.input_length, param.input_length//2, 1),
            nn.Tanh(), 
            nn.Conv1d(param.input_length//2, 1, 1),
            nn.Tanh()
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # first_token_tensor = hidden_states.transpose(1, 2)
        first_token_tensor = self.l(hidden_states)
        first_token_tensor = first_token_tensor.squeeze(1)
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class ClassifierHead(nn.Module):
    def __init__(self, param) -> None:
        super().__init__()
        self.pooler = BertPooler(param)
        self.l = nn.Sequential(
            nn.Linear(param.dim, param.dim//2),
            nn.Tanh(),
            nn.Linear(param.dim//2, 4),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.pooler(x)
        x = self.l(x)

        return x
    
class Classifier(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        # self.params = params
        self.transformer = Transformer(params)
        self.l = nn.Sequential(
            nn.Linear(params.input_dim, params.dim),
            nn.Tanh()
        )
        self.classifier = ClassifierHead(params)

    def forward(self, x):
        x = self.l(x)
        x = self.transformer(x)
        x = self.classifier(x)

        return x
    

class ClassifierHead_only(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.pooler = BertPooler()
        self.l = nn.Sequential(
            nn.Linear(768, 768//2),
            nn.Tanh(),
            nn.Linear(768//2, 4),
            nn.Tanh()
        )

        self.l1 = nn.Sequential(
            nn.Linear(4096, 768),
            nn.Tanh()
        )

    def forward(self, x):
        # x = self.l1(x)
        x = self.pooler(x)
        x = self.l(x)

        return x

class Classifier_Conbine(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.transformer = Transformer(params)
        self.l = nn.Sequential(
            nn.Linear(4096, 768),
            nn.Tanh()
        )
        self.l1 = nn.Sequential(
            nn.Linear(768, 768),
            nn.Tanh()
        )
        self.ch_down = nn.Sequential(
            nn.Conv1d(160, 80, 1),
            nn.Tanh()   
        )
        self.classifier = ClassifierHead()

    def forward(self, x, x1):
        x = self.l(x)
        x1 = self.l1(x1)
        x = torch.cat((x, x1), dim=1)
        x = self.ch_down(x)
        x = self.transformer(x)
        x = self.classifier(x)

        return x
    




# if __name__ == '__main__':
#     param = ModelArgs(dim=32, n_layers=1, n_heads=8, input_length=80)
#     model = Classifier(param)
#     i = torch.ones(2, 80, 4096)
#     o = model(i)
#     print(o.shape)  # torch.Size([2, 4])

    