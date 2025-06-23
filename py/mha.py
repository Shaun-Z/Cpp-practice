# %%
import math
from typing import Optional, List

import torch
from torch import nn

from labml import tracker

# %%
class PrepareForMultiHeadAttention(nn.Module):
    '''
    Prepares the input for multi-head attention by reshaping it.
    This module reshapes the input tensor to have an additional dimension for the number of heads.
    
    Args:
    - d_model (int): The dimension of the model.
    - n_heads (int): The number of attention heads.
    - d_heads (int): The dimension of each head.
    - bias (bool): Whether to include a bias term in the linear transformation.
    '''
    def __init__(self, d_model: int, n_heads: int, d_heads: int, bias: bool):
        super().__init__()
        # Linear layer for linear transformation
        self.linear = nn.Linear(d_model, n_heads * d_heads, bias=bias)
        # Number of heads
        self.n_heads = n_heads
        # Dimension of each head
        self.d_heads = d_heads
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass to reshape the input tensor for multi-head attention.
        Args:
        - x (torch.Tensor): Input tensor of shape (seq_len, batch_size, d_model) or (batch_size, d_model).
        Returns:
        - torch.Tensor: Reshaped tensor of shape (seq_len, batch_size, heads, d_heads) or (batch_size, heads, d_model).
        '''

        head_shape = x.shape[:-1]

        # Apply linear transformation
        x = self.linear(x)

        # Reshape to (batch_size, seq_len, n_heads, d_heads)
        x = x.view(*head_shape, self.n_heads, self.d_heads)

        # Output has shape [seq_len, batch_size, heads, d_k] or [batch_size, heads, d_model]
        return x 

# %%
class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads: int, d_model: int, dropout_prob: float = 0.1, bias: bool = True):
        super().__init__()
        # Number of features per head
        self.n_features = d_model // n_heads
        # Number of attention heads
        self.n_heads = n_heads
        # Prepare the query, key, and value transformations
        self.query = PrepareForMultiHeadAttention(d_model, n_heads, self.n_features, bias)
        self.key = PrepareForMultiHeadAttention(d_model, n_heads, self.n_features, bias)
        self.value = PrepareForMultiHeadAttention(d_model, n_heads, self.n_features, bias=True)

        self.softmax = nn.Softmax(dim=-1)
        self.output = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout_prob)
        self.scale = 1 / math.sqrt(self.n_features)

        self.attn = None

    def get_scores(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        '''
        Compute the attention scores.
        Args:
        - query (torch.Tensor): Query tensor of shape (seq_len_q, batch_size, n_heads, d_model).
        - key (torch.Tensor): Key tensor of shape (seq_len_k, batch_size, n_heads, d_model).
        Returns:
        - torch.Tensor: Attention scores of shape (seq_len_q, batch_size, n_heads, seq_len_k).
        '''
        return torch.einsum('ibhd,jbhd->ibhj', query, key)
    
    def prepare_mask(self, mask: torch.Tensor, query_shape: List[int], key_shape: List[int]) -> torch.Tensor:
        
        assert mask.shape[0] == 1 or mask.shape[0] == query_shape[0]
        assert mask.shape[1] == key_shape[0]
        assert mask.shape[2] == 1 or mask.shape[2] == query_shape[1]

        mask = mask.unsqueeze(-1)

        return mask
    
    def forward(self, *, query:torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        seq_len, batch_size, _ = query.shape
    
        if mask is not None:
            mask = self.prepare_mask(mask, query.shape, key.shape)
        
        # Reshape query, key, and value for multi-head attention (seq_len, batch_size, n_heads, d_model)
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        scores = self.get_scores(query, key) * self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = self.softmax(scores)

        tracker.add('attn', attn)

        attn = self.dropout(attn)

        x = torch.einsum('ijbh, jbhd->ibhd', attn, value)

        self.attn = attn.detach()

        x = x.reshape(seq_len, batch_size, -1)

        return self.output(x)
