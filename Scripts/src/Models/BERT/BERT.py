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
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_parameter('pe', nn.Parameter(pe, requires_grad=False))

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

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)
            # scores = scores.masked_fill(mask == 0, -1e9) 

        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, v)

        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        output = self.fc(output)

        return output, attn


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
        mask = torch.zeros((seq_len, seq_len), dtype=torch.bool, device=device)

        for i in range(seq_len):
            start = max(0, i - self.block_size)
            end = min(seq_len, i + self.block_size + 1)
            mask[i, start:end] = True
            if not self.include_adjacent:
                if i - self.block_size - 1 >= 0:
                    mask[i, i - self.block_size - 1] = False
                if i + self.block_size + 1 < seq_len:
                    mask[i, i + self.block_size + 1] = False
        return mask

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()
        device = x.device

        block_mask = self.generate_block_mask(seq_len, device)
        block_mask = block_mask.unsqueeze(0).repeat(batch_size, 1, 1)

        if mask is not None:
            mask = mask.bool()
            expanded_mask = mask.unsqueeze(1).repeat(1, seq_len, 1)
            block_mask = block_mask & expanded_mask

        attention_mask = block_mask.unsqueeze(1)

        x_norm1 = self.layer_norm1(x)
        attention_out, attn_weights = self.attention(x_norm1, mask=attention_mask)
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
            attn_scores = attn_scores.masked_fill(~mask.bool(), -1e4)

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

        self.pooling = AttentionPooling(hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        

    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.positional_encoding(x)

        all_attn_weights = []

        for block in self.blocks:
            for layer in block:
                x, attn_weights = layer(x, mask)
                all_attn_weights.append(attn_weights)

        pooled_output = self.pooling(x, mask)
        ## next two lines were added
        pooled_output = self.dropout(pooled_output)
        pooled_output = self.activation(pooled_output)
        
        output = self.classifier(pooled_output)
        
        # output = self.classifier(x[:, 0, :])

        return output, all_attn_weights