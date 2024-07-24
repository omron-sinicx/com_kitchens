import torch
from torch import nn, Tensor

import math

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

def get_pad_attn_mask(q, k, masking_seq=False, padding_idx=0):
    '''
    create attention ctx_pad_attn_mask matrix (batch_size x len x len) 
    from input data (batch_size x len x model_dim).
    '''
    n_batch, len_q = q.size()
    n_batch, len_k = k.size()
    # set True for padding cell
    pad_attn_mask = k.data.eq(padding_idx)
    # transform single ctx_pad_attn_mask vector (batch_size x len) to
    # ctx_pad_attn_mask matrix (batch_size x len x len) by repeating the vector
    pad_attn_mask = pad_attn_mask.unsqueeze(1).expand(n_batch, len_q, len_k)

    if masking_seq == True:
        seq_mask = torch.ones(len_q, len_k, device=pad_attn_mask.device, dtype=torch.uint8)
        seq_mask = torch.triu(seq_mask, diagonal=1).bool()
        pad_attn_mask = pad_attn_mask | seq_mask

    return pad_attn_mask

def get_pad_mask(x, padding_idx=0):
    '''
    x: batch_size x len
    '''
    # assuming used by output.masked_fill(ret, 0)
    # output: batch_size x len x model_dim
    # ret   : batch_size x len x 1
    return x.eq(padding_idx).unsqueeze(-1)