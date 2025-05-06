import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SelfAttentionHead(nn.Module):
    def __init__(self, embed_dim, head_dim):
        super().__init__()
        self.query = nn.Linear(embed_dim, head_dim)
        self.key = nn.Linear(embed_dim, head_dim)
        self.value = nn.Linear(embed_dim, head_dim)
        self.dk = head_dim ** 0.5

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        # (QK^T / sqrt(dk)) * V
        scores = (Q @ K.transpose(-2, -1)) / self.dk
        attention = F.softmax(scores, dim=-1)
        return attention @ V, attention
    
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.head_dim = embed_dim // num_heads
        self.heads = nn.ModuleList([SelfAttentionHead(embed_dim, self.head_dim) for _ in range(num_heads)])
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        head_outputs = []
        maps = []
        
        for head in self.heads:
            output, attn_map = head(x)
            head_outputs.append(output)
            maps.append(attn_map)
        
        out = torch.cat(head_outputs, dim=-1)
        return out, maps

class FeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, embed_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.linear2(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, ff_dim, num_heads):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.ff = FeedForward(embed_dim, ff_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attention, maps = self.attention(x)
        x = self.norm(x + attention)
        
        ff_out = self.ff(x)
        x = self.norm(x + ff_out)

        return x, maps

class Transformer(nn.Module):
    def __init__(self, vocab_size, n_embd, n_layer, n_hidden, n_output, block_size, num_heads):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, n_embd)
        self.pos_embed = nn.Embedding(block_size, n_embd)
        
        self.layers = nn.ModuleList([EncoderBlock(n_embd, n_hidden, num_heads) for _ in range(n_layer)])
        self.classifier = nn.Linear(n_embd, n_output)

    def forward(self, x):
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        x = self.token_embed(x) + self.pos_embed(positions)

        attention_maps = []
        for layer in self.layers:
            x, maps = layer(x)
            attention_maps.extend(maps)

        x = x.mean(dim=1)
        return self.classifier(x), attention_maps


#Part 2


class MaskedSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dk = self.head_dim ** 0.5

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, T, C = x.shape

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        scores = (Q @ K.transpose(-2, -1)) * self.dk
        mask = torch.triu(torch.full((T, T), float('-inf'), device=x.device), diagonal=1)
        scores += mask

        attention = F.softmax(scores, dim=-1)
        out = attention @ V

        return out, attention

class MultiHeadAttentionMasked(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.head_dim = embed_dim // num_heads
        self.heads = nn.ModuleList([MaskedSelfAttention(embed_dim, self.head_dim) for _ in range(num_heads)])
        self.proj = nn.Linear(embed_dim * num_heads, embed_dim)

    def forward(self, x):
        head_outputs = []
        attention_maps = []
        
        for head in self.heads:
            output, map = head(x)
            head_outputs.append(output)
            attention_maps.append(map)
        
        out = torch.cat(head_outputs, dim=-1)
        out = self.proj(out)
        return out, attention_maps

class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, ff_dim, num_heads):
        super().__init__()
        self.attention = MultiHeadAttentionMasked(embed_dim, num_heads)
        self.ff = FeedForward(embed_dim, ff_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_out, maps = self.attention(self.norm1(x))
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x, maps

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, n_embd, n_layer, block_size, ff_dim, num_heads):
        super().__init__()
        
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([DecoderBlock(n_embd, ff_dim, num_heads) for _ in range(n_layer)])
        self.norm = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        x = self.token_embedding(idx) + self.position_embedding(torch.arange(idx.size(1), device=idx.device))
        attention_maps = []
        for block in self.blocks:
            x, maps = block(x)
            attention_maps.extend(maps)
        x = self.norm(x)

        logits = self.lm_head(x)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return loss, attention_maps

        return logits, attention_maps