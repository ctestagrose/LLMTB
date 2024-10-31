import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        x = self.dropout(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        # Transpose to match nn.MultiheadAttention input shape: (seq_len, batch_size, embed_dim)
        x = x.transpose(0, 1)  # Shape: (seq_len, batch_size, d_model)

        attn_output, attn_output_weights = self.attention(
            x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask
        )
        # Transpose back to original shape: (batch_size, seq_len, d_model)
        attn_output = attn_output.transpose(0, 1)

        return attn_output, attn_output_weights


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_head, ff_size, block_size, dropout=0.1, include_adjacent=True):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_head)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, ff_size),
            nn.GELU(),
            nn.Linear(ff_size, d_model)
        )
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.block_size = block_size
        self.include_adjacent = include_adjacent

    def generate_block_mask(self, seq_len, device):
        idxs = torch.arange(seq_len, device=device)
        distance = (idxs[None, :] - idxs[:, None]).abs()
        mask = (distance <= self.block_size)
    
        if not self.include_adjacent:
            mask &= (distance != self.block_size)
    
        return mask

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()
        device = x.device

        # Generate block mask of shape (seq_len, seq_len)
        block_mask = self.generate_block_mask(seq_len, device)  # Positions to keep are True

        # Invert the mask for nn.MultiheadAttention (positions to mask are True)
        attn_mask = ~block_mask  # Shape: (seq_len, seq_len)

        # Handle key padding mask (batch_size, seq_len)
        key_padding_mask = ~mask.bool() if mask is not None else None  # Positions to mask are True

        x_norm1 = self.layer_norm1(x)
        attention_out, attn_weights = self.attention(
            x_norm1, attn_mask=attn_mask, key_padding_mask=key_padding_mask
        )
        x = x + self.dropout(attention_out)
        x_norm2 = self.layer_norm2(x)
        feed_forward_out = self.feed_forward(x_norm2)
        x = x + self.dropout(feed_forward_out)
        return x, attn_weights

class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, x, mask=None):
        attn_scores = self.attention(x).squeeze(-1)

        if mask is not None:
            # attn_scores = attn_scores.masked_fill(~mask.bool(), -1e9)
            # attn_scores = attn_scores.masked_fill(~mask.bool(), -1e4)
            min_value = torch.finfo(attn_scores.dtype).min
            attn_scores = attn_scores.masked_fill(mask == 0, min_value)

        attn_weights = F.softmax(attn_scores, dim=-1)

        attn_weights = attn_weights.unsqueeze(-1)
        output = torch.sum(attn_weights * x, dim=1)

        return output
        
        
class BERT(nn.Module):
    def __init__(self, vocab_size, config):
        super(BERT, self).__init__()

        hidden_dim = config['hidden_dim']
        num_heads = config['num_heads']
        ff_dim = config['ff_dim']
        num_blocks = config['num_blocks']
        block_size = config['block_size']
        num_layers = config['num_layers']
        num_class = config.get('num_class', 1)
        dropout = config.get('dropout', 0.1)
        include_adjacent = config.get('include_adjacent', True)

        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim, dropout)

        self.layers = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, ff_dim, block_size, dropout, include_adjacent)
            for _ in range(num_layers)
        ])

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(p=0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(p=0.3),
            nn.Linear(hidden_dim, num_class)
        )

        self.pooling = AttentionPooling(hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.Tanh()
        

    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.positional_encoding(x)

        all_attn_weights = []

        for layer in self.layers:
            x, attn_weights = layer(x, mask)
            all_attn_weights.append(attn_weights)

        pooled_output = self.pooling(x, mask)
        ## next two lines were added
        pooled_output = self.dropout(pooled_output)
        pooled_output = self.activation(pooled_output)
        
        output = self.classifier(pooled_output)
        
        # output = self.classifier(x[:, 0, :])

        return output, all_attn_weights