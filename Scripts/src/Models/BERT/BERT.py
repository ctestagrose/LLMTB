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
        return self.dropout(x)


class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, max_len, d_model):
        super(LearnedPositionalEmbedding, self).__init__()
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x):
        positions = torch.arange(0, x.size(1), dtype=torch.long, device=x.device).unsqueeze(0)
        pos_embed = self.pe(positions)
        return x + pos_embed


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout
        )

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        x = x.transpose(0, 1)
        attn_output, attn_output_weights = self.attention(
            x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask
        )
        attn_output = attn_output.transpose(0, 1)

        return attn_output, attn_output_weights


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, ff_size, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, ff_size),
            nn.GELU(),
            nn.Linear(ff_size, d_model)
        )
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask=None):
        key_padding_mask = ~mask.bool() if mask is not None else None

        x_norm1 = self.layer_norm1(x)
        attn_out, attn_weights = self.attention(
            x_norm1, attn_mask=None, key_padding_mask=key_padding_mask
        )
        x = x + self.dropout(attn_out)

        x_norm2 = self.layer_norm2(x)
        ff_out = self.feed_forward(x_norm2)
        x = x + self.dropout(ff_out)

        return x, attn_weights


class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, x, mask=None):
        attn_scores = self.attention(x).squeeze(-1)

        if mask is not None:
            min_value = torch.finfo(attn_scores.dtype).min
            attn_scores = attn_scores.masked_fill(mask == 0, min_value)

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = attn_weights.unsqueeze(-1)
        output = torch.sum(attn_weights * x, dim=1)

        return output


class BERT(nn.Module):
    def __init__(self, vocab_size, config):
        super(BERT, self).__init__()

        # Read config
        hidden_dim = config['hidden_dim']
        num_heads = config['num_heads']
        ff_dim = config['ff_dim']
        num_layers = config['num_layers']
        num_class = config.get('num_class', 1)
        dropout = config.get('dropout', 0.1)
        self.grad_clip = 1.0

        # Token + Positional Embeddings
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.positional_encoding = LearnedPositionalEmbedding(max_len=5000, d_model=hidden_dim)
        self.embedding_dropout = nn.Dropout(dropout)
        self.embedding_norm = nn.LayerNorm(hidden_dim)

        # Multiple Transformer Layers
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=hidden_dim,
                num_heads=num_heads,
                ff_size=ff_dim,
                dropout=dropout
            ) for _ in range(num_layers)
        ])

        # Pooling Strategies
        self.attention_pooling = AttentionPooling(hidden_dim)
        self.max_pooling = nn.AdaptiveMaxPool1d(1)
        self.avg_pooling = nn.AdaptiveAvgPool1d(1)

        # Classification Head
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Dropout(p=0.3),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=0.3),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim // 2, num_class)
        )

        # Initialize Weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.embedding_dropout(x)
        x = self.embedding_norm(x)

        all_attn_weights = []
        for layer in self.layers:
            x, attn_weights = layer(x, mask=mask)
            all_attn_weights.append(attn_weights)

        attention_output = self.attention_pooling(x, mask=mask)

        x_transpose = x.transpose(1, 2)
        max_pooled = self.max_pooling(x_transpose).squeeze(-1)
        avg_pooled = self.avg_pooling(x_transpose).squeeze(-1)

        pooled_output = torch.cat([attention_output, max_pooled, avg_pooled], dim=-1)

        output = self.classifier(pooled_output)

        return output, all_attn_weights
