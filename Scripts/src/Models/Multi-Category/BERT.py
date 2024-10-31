import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import LongformerModel, LongformerTokenizer, AdamW

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        x = self.dropout(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.depth = d_model // num_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        self.fc = nn.Linear(d_model, d_model)

        self.scale = math.sqrt(self.depth)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.transpose(1, 2)

    def forward(self, x, mask=None):
        batch_size = x.size(0)
        seq_len = x.size(1)

        q = self.split_heads(self.wq(x), batch_size)
        k = self.split_heads(self.wk(x), batch_size)
        v = self.split_heads(self.wv(x), batch_size)

        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        # if mask is not None:
        #     print(f"Mask shape before unsqueeze: {mask.shape}")
        #     mask = mask.unsqueeze(1).unsqueeze(2)
        #     scores = scores.masked_fill(mask == 0, float('-inf'))
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            mask = mask.expand(batch_size, self.num_heads, seq_len, seq_len)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)  # Attention weights
        output = torch.matmul(attn, v)

        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        output = self.fc(output)

        return output, attn


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_head, ff_size, block_size, dropout=0.1, include_adjacent=False):
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

    def forward(self, x, mask=None):
        x_norm1 = self.layer_norm1(x)
        attention_out, attn_weights = self.attention(x_norm1, mask)  # Get attention weights
        x = x + self.dropout(attention_out)
        x_norm2 = self.layer_norm2(x)
        feed_forward_out = self.feed_forward(x_norm2)
        x = x + self.dropout(feed_forward_out)
        return x, attn_weights

class DeeperClassifier(nn.Module):
    def __init__(self, hidden_dim, num_class, dropout=0.3):
        super(DeeperClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim // 4, num_class)
        )

    def forward(self, x):
        return self.classifier(x)


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
        include_adjacent = config.get('include_adjacent', False)

        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim, dropout)

        self.blocks = nn.ModuleList([
            nn.ModuleList([
                TransformerBlock(hidden_dim, num_heads, ff_dim, block_size, dropout, include_adjacent)
                for _ in range(num_layers)
            ]) for _ in range(num_blocks)
        ])

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(p=0.3),
            nn.Linear(hidden_dim, num_class)
        )


    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.positional_encoding(x)

        all_attn_weights = []

        for block in self.blocks:
            for layer in block:
                if self.training and torch.rand(1).item() < 0.1:
                    continue
                x, attn_weights = layer(x, mask)
                all_attn_weights.append(attn_weights)

        output = self.classifier(x[:, 0, :])
        return output, all_attn_weights