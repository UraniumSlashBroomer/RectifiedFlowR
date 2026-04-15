import torch.nn.functional as F
import torch.nn as nn
import torch
import math
import copy


class PositionalEncodingEmb(nn.Module):
    def __init__(self, n_patches, emb_dim, p_dropout=0.0):
        super().__init__()
        self.positional_emb = nn.Parameter(torch.zeros(n_patches, emb_dim))
        
        nn.init.trunc_normal_(self.positional_emb, std=0.02)
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x):
        """
        input: x [B, S, E]
        """
        x = self.dropout(x + self.positional_emb)

        return x


class PositionalEncodingSinCos(nn.Module):
    def __init__(self, emb_dim, p_dropout=0.0, max_length=5000):
        super().__init__()
        assert emb_dim % 2 == 0, "emb dim should be divisble by 2"
        
        self.dropout = nn.Dropout(p_dropout)

        pe = torch.zeros(1, max_length, emb_dim)

        for i in range(max_length):
            for j in range(emb_dim):
                pe[:, i, j] = math.sin(i * (10000 ** (-j / emb_dim))) if j % 2 == 0 else math.cos(i * (10000 ** (-(j - 1) / emb_dim)))

        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        input: x [B, S, E]
        """
        _, S, _ = x.shape
        x = self.dropout(x + self.pe[:, :S, :])
        
        return x


class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, img_size, in_channels, emb_dim):
        super().__init__()
        
        assert img_size % patch_size == 0, f"img_size should be divisable by patch_size"

        self.proj = nn.Conv2d(
                in_channels,
                emb_dim,
                kernel_size=patch_size,
                stride=patch_size
        )
    
    def forward(self, x):
        x = self.proj(x) # [B, E, num_patches_in_line, num_patches_in_line]
        x = x.flatten(2) # [B, E, num_patches]
        x = x.transpose(1, 2) # [B, S, E]
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, n_heads, p_dropout=0.0):
        super().__init__()
        assert emb_dim % n_heads == 0, "input dim must be divisible by number of heads"

        self.head_dim = emb_dim // n_heads
        self.n_heads = n_heads

        self.W_query = nn.Linear(emb_dim, emb_dim)
        self.W_value = nn.Linear(emb_dim, emb_dim)
        self.W_key = nn.Linear(emb_dim, emb_dim)
        self.W_out = nn.Linear(emb_dim, emb_dim)

        self.dropout = nn.Dropout(p_dropout)

    def forward(self, query, value, key):
        """
        input: query [B, S, E]
               value [B, T, E]
               key   [B, T, E]

        output: output [B, S, E]
        """

        B, S, E = query.shape
        B, T, E = value.shape
        
        query = self.W_query(query)
        value = self.W_value(value)
        key = self.W_key(key)

        query = query.reshape(B, S, self.n_heads, self.head_dim).swapaxes(1, 2)
        value = value.reshape(B, T, self.n_heads, self.head_dim).swapaxes(1, 2)
        key = key.reshape(B, T, self.n_heads, self.head_dim).swapaxes(1, 2)
        
        A = (query @ key.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        A = F.softmax(A, dim=-1)
        A = self.dropout(A)

        output = A @ value
        output = output.swapaxes(1, 2).reshape(B, S, E)
        output = self.W_out(output)
        
        return output


class FeedForwardNetwork(nn.Module):
    def __init__(self, emb_dim, ffn_dim, p_dropout=0.0):
        super().__init__()
        
        self.fc1 = nn.Linear(emb_dim, ffn_dim)
        self.GELU = nn.GELU()
        self.dropout = nn.Dropout(p_dropout)
        self.fc2 = nn.Linear(ffn_dim, emb_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.GELU(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, input_dim, n_heads, ffn_dim_ratio, p_dropout=0.0):
        super().__init__()
        
        ffn_dim = input_dim * ffn_dim_ratio
        self.self_attn = MultiHeadAttention(input_dim, n_heads, p_dropout)
        self.ffn = FeedForwardNetwork(input_dim, ffn_dim, p_dropout)
        
        self.norm_self = nn.LayerNorm(input_dim)
        self.norm_ffn = nn.LayerNorm(input_dim)

        self.dropout_self = nn.Dropout(p_dropout)
        self.dropout_ffn = nn.Dropout(p_dropout)

    def forward(self, x):
        shortcut = x
        x = self.self_attn(x, x, x)
        x = self.dropout_self(x)
        x += shortcut
        x = self.norm_self(x)

        shortcut = x
        x = self.ffn(x)
        x = self.dropout_ffn(x)
        x += shortcut
        x = self.norm_ffn(x)

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = clones(encoder_layer, num_layers)
        

    def forward(self, x):
        """
        input x [B, S, E]
        """

        for mod in self.layers:
            x = mod(x)

        return x 


class ViT(nn.Module):
    def __init__(self, img_size, in_channels, patch_size, emb_dim, ffn_dim_ratio,
                 n_heads, num_layers, positional_encoding='sincos', num_classes=10,
                 p_pos_encoding_dropout=0.0, p_encoder_dropout=0.0):
        super().__init__()
        
        self.img_size = img_size
        self.in_channels = in_channels
        self.patch_size = patch_size

        if positional_encoding == 'sincos':
            self.positional_encoding = PositionalEncodingSinCos(emb_dim, p_pos_encoding_dropout)
        elif positional_encoding == 'emb':
            n_patches = (img_size // patch_size) ** 2
            self.positional_encoding = PositionalEncodingEmb(n_patches=n_patches,
                                                             emb_dim=emb_dim,
                                                             p_dropout=p_pos_encoding_dropout)

        self.patch_embedding = PatchEmbedding(patch_size=patch_size,
                                              img_size=img_size,
                                              in_channels=in_channels,
                                              emb_dim=emb_dim)

        encoder_layer = TransformerEncoderLayer(input_dim=emb_dim,
                                                n_heads=n_heads,
                                                ffn_dim_ratio=ffn_dim_ratio,
                                                p_dropout=p_encoder_dropout)

        self.transformer = TransformerEncoder(encoder_layer, num_layers)
        
        self.head = nn.Linear(emb_dim, num_classes)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):
        """
        input: raw images [B, C, N, N]
        """
        x = self.patch_embedding(x) # [B, S, E]
        x = self.positional_encoding(x) # [B, S, E]
        x = self.transformer(x) # [B, S, E]
        x = x.mean(dim=1) # [B, E]
        logits = self.head(x) # [B, num_classes]

        return logits 


def clones(module, num_layers):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num_layers)])


if __name__ == '__main__':
    model = ViT(img_size=32, in_channels=3,
                patch_size=8, emb_dim=256,
                ffn_dim_ratio=4, n_heads=8, num_layers=3,
                positional_encoding='emb',
                num_classes=10,
                p_pos_encoding_dropout=0.2,
                p_encoder_dropout=0.2)

    img = torch.randn((3, 32, 32))
    img = img.unsqueeze(0)
    result = model(img)
    print(result)
    print(f"number of parameters: {sum([p.numel() for p in model.parameters()])}\nnumber of learnable parameters: {sum([p.numel() for p in model.parameters()])}")   
