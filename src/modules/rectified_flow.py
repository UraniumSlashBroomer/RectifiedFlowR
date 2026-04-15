from .modules import *
import torch
import torch.nn as nn
import math

class TimeEmbedding(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        half_dim = emb_dim // 2
        sincosemb = math.log(10000) / (half_dim - 1)
        sincosemb = torch.exp(torch.arange(half_dim) * -sincosemb)
        sincosemb = sincosemb.reshape(1, 1, half_dim)
        
        self.fc1 = nn.Linear(emb_dim, emb_dim)
        self.SiLU = nn.SiLU()
        self.fc2 = nn.Linear(emb_dim, emb_dim)

        self.register_buffer("sincosemb", sincosemb)

    def forward(self, t):
        """
        t: [B, 1, 1]
        output: [B, 1, E]
        """
        t = t * 1000 * self.sincosemb # [B, 1, E // 2]
        t = torch.cat((t.sin(), t.cos()), dim=-1) # [B, 1, E]
        t = self.fc1(t)
        t = self.SiLU(t)
        t = self.fc2(t)

        return t

class AdaLN_EncoderLayer(nn.Module):
    def __init__(self, input_dim, n_heads, ffn_dim_ratio, p_dropout=0.0):
        """
        I will add adaLN to default encoder layer and I'll make it pre-norm, not post-norm
        """
        super().__init__()
        
        ffn_dim = input_dim * ffn_dim_ratio
        self.self_attn = MultiHeadAttention(input_dim, n_heads, p_dropout)
        self.ffn = FeedForwardNetwork(input_dim, ffn_dim, p_dropout)
        
        self.norm_self = nn.LayerNorm(input_dim, elementwise_affine=False)
        self.norm_ffn = nn.LayerNorm(input_dim, elementwise_affine=False)

        self.dropout_self = nn.Dropout(p_dropout)
        self.dropout_ffn = nn.Dropout(p_dropout)
        
        self.adaLN_modulation = nn.Sequential( # for t embedding
                nn.SiLU(),
                nn.Linear(input_dim, 6 * input_dim),
                )
        
    def forward(self, x, t):
        """
        input: x [B, S, E]
               t [B, 1, E]

        output: x [B, S, E]
        """
        shift_a, scale_a, gate_a, shift_ffn, scale_ffn, gate_ffn = self.adaLN_modulation(t).chunk(6, dim=2) # after adaLN t [B, 1, 6*E]

        shortcut = x
        x = self.norm_self(x)
        x = (1 + scale_a) * x + shift_a
        x = self.self_attn(x, x, x)
        x *= gate_a
        x = self.dropout_self(x)
        x += shortcut

        shortcut = x
        x = self.norm_ffn(x)
        x = (1 + scale_ffn) * x + shift_ffn
        x = self.ffn(x)
        x *= gate_ffn
        x = self.dropout_ffn(x)
        x += shortcut

        return x

class RectifiedFlowTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = clones(encoder_layer, num_layers)
        

    def forward(self, x, t):
        """
        input x [B, S, E]
        """

        for mod in self.layers:
            x = mod(x, t)

        return x 

class Unpatchify(nn.Module):
    def __init__(self, emb_dim, patch_size, out_channels):
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.fc = nn.Linear(emb_dim, patch_size * patch_size * out_channels)

    def forward(self, x):
        """
        input: x [B, S, E] (S - num_patches)
        """
        B, num_patches, _ = x.shape
        x = self.fc(x) # [B, S, patch_size * patch_size * out_channels]
        num_patches_in_line = int(num_patches ** 0.5)
        x = x.reshape(B, num_patches_in_line, num_patches_in_line, self.patch_size, self.patch_size, self.out_channels)
        x = x.permute(0, 5, 1, 3, 2, 4)
        x = x.reshape(B, self.out_channels, num_patches_in_line * self.patch_size, num_patches_in_line * self.patch_size)
        
        return x

class RectifiedFlowViT(nn.Module):
    def __init__(self, img_size, in_channels, patch_size, emb_dim, ffn_dim_ratio,
                 n_heads, num_layers, positional_encoding='sincos',
                 p_pos_encoding_dropout=0.0, p_encoder_dropout=0.0):
        super().__init__()
        
        self.img_size = img_size
        self.in_channels = in_channels
        self.patch_size = patch_size
        
        self.n_heads = n_heads
    
        self.time_embedding = TimeEmbedding(emb_dim)

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

        encoder_layer = AdaLN_EncoderLayer(input_dim=emb_dim,
                                                n_heads=n_heads,
                                                ffn_dim_ratio=ffn_dim_ratio,
                                                p_dropout=p_encoder_dropout)

        self.transformer = RectifiedFlowTransformerEncoder(encoder_layer, num_layers)
        
        self.unpatchify = Unpatchify(emb_dim=emb_dim,
                                     patch_size=patch_size,
                                     out_channels=in_channels)

        self.final_norm = nn.LayerNorm(emb_dim)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, AdaLN_EncoderLayer):
            module.adaLN_modulation[-1].weight.data.zero_()
            module.adaLN_modulation[-1].bias.data.zero_()

    def forward(self, x, t):
        """
        input: images [B, C, N, N],
        t [B, 1]
        """
        t = self.time_embedding(t)
        x = self.patch_embedding(x) # [B, S, E]
        x = self.positional_encoding(x) # [B, S, E]
        x = self.transformer(x, t) # [B, S, E]
        x = self.final_norm(x)
        # x = x.mean(dim=1) # [B, E]
        imgs = self.unpatchify(x) # [B, C, N, N]

        return imgs
